"""
Golf ball detector (orange ball on grass) with GUI controls and snapshot annotation.
- Color masking: HSV + Lab (click to pick), optional green suppression.
- Shape validation: edge coverage, texture, circularity, axis ratio.
- GUI: trackbars for parameters; windows Original/Mask/Edges/Result.
- Snapshot mode: Shift+K to freeze frame, annotate ball region, auto-compute color params, resume live.
"""

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ---------- Camera helpers ----------
def open_camera_with_backends(device_index: int = 0) -> Optional[cv2.VideoCapture]:
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None


def get_candidate_indices(max_devices: int = 8) -> List[int]:
    env_idx = os.environ.get("CAM_INDEX")
    if env_idx is not None:
        try:
            return [int(env_idx)]
        except ValueError:
            pass
    preferred = list(range(0, 8))
    indices: List[int] = preferred.copy()
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int("".join(ch for ch in path if ch.isdigit()))
            indices.append(idx)
        except ValueError:
            continue
    if not indices:
        indices = list(range(0, max_devices + 1))
    seen = set()
    return [i for i in indices if not (i in seen or seen.add(i))]


def find_camera(max_devices: int = 8) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    for idx in get_candidate_indices(max_devices):
        cap = open_camera_with_backends(idx)
        if cap is not None:
            return cap, idx
    return None, None


# ---------- Config & GUI ----------
@dataclass
class Params:
    h_low: int = 5
    h_high: int = 25
    s_min: int = 60
    v_min: int = 60
    lab_tol: int = 40
    suppress_green: int = 0  # 先关闭绿色抑制，确保球能被捕获
    auto_exposure: int = 0  # 0=manual, 1=auto
    exposure: int = 200  # slider value; mapped to CAP_PROP_EXPOSURE

    morph_open: int = 1
    morph_close: int = 1
    blur_ksize: int = 3  # must be odd

    min_circularity: float = 0.60
    min_axis_ratio: float = 0.60
    min_fill: float = 0.50
    max_residual: float = 0.35
    min_arc_coverage: float = 0.50
    min_edge_cov: float = 0.35
    edge_band: float = 0.12
    patch_std_min: float = 3.5
    min_radius_px: int = 4
    max_radius_frac: float = 0.60

    scale: float = 1.0
    detect_every: int = 1
    score_thr: float = 0.5
    nms_iou: float = 0.35
    nms_keep: int = 10


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def create_trackbar_window(p: Params):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H low", "Controls", p.h_low, 179, lambda x: None)
    cv2.createTrackbar("H high", "Controls", p.h_high, 179, lambda x: None)
    cv2.createTrackbar("S min", "Controls", p.s_min, 255, lambda x: None)
    cv2.createTrackbar("V min", "Controls", p.v_min, 255, lambda x: None)
    cv2.createTrackbar("Lab tol", "Controls", p.lab_tol, 80, lambda x: None)
    cv2.createTrackbar("Suppress green (0/1)", "Controls", p.suppress_green, 1, lambda x: None)
    cv2.createTrackbar("Auto exp (0/1)", "Controls", p.auto_exposure, 1, lambda x: None)
    cv2.createTrackbar("Exposure slider", "Controls", p.exposure, 500, lambda x: None)

    cv2.createTrackbar("Morph open (0-2)", "Controls", p.morph_open, 2, lambda x: None)
    cv2.createTrackbar("Morph close (0-2)", "Controls", p.morph_close, 2, lambda x: None)
    cv2.createTrackbar("Blur ksize (odd)", "Controls", p.blur_ksize, 11, lambda x: None)

    cv2.createTrackbar("min circ x100", "Controls", int(p.min_circularity * 100), 100, lambda x: None)
    cv2.createTrackbar("min axis x100", "Controls", int(p.min_axis_ratio * 100), 100, lambda x: None)
    cv2.createTrackbar("min fill x100", "Controls", int(p.min_fill * 100), 100, lambda x: None)
    cv2.createTrackbar("max residual x100", "Controls", int(p.max_residual * 100), 100, lambda x: None)
    cv2.createTrackbar("min arc x100", "Controls", int(p.min_arc_coverage * 100), 100, lambda x: None)
    cv2.createTrackbar("min edge x100", "Controls", int(p.min_edge_cov * 100), 100, lambda x: None)
    cv2.createTrackbar("edge band x100", "Controls", int(p.edge_band * 100), 50, lambda x: None)
    cv2.createTrackbar("std min x10", "Controls", int(p.patch_std_min * 10), 200, lambda x: None)
    cv2.createTrackbar("min r px", "Controls", p.min_radius_px, 200, lambda x: None)
    cv2.createTrackbar("max r % short", "Controls", int(p.max_radius_frac * 100), 100, lambda x: None)

    cv2.createTrackbar("scale x100", "Controls", int(p.scale * 100), 100, lambda x: None)
    cv2.createTrackbar("detect every", "Controls", p.detect_every, 5, lambda x: None)
    cv2.createTrackbar("score thr x100", "Controls", int(p.score_thr * 100), 100, lambda x: None)
    cv2.createTrackbar("nms iou x100", "Controls", int(p.nms_iou * 100), 100, lambda x: None)
    cv2.createTrackbar("nms keep", "Controls", p.nms_keep, 20, lambda x: None)


def read_trackbar_params(p: Params) -> Params:
    q = Params()
    q.h_low = cv2.getTrackbarPos("H low", "Controls")
    q.h_high = cv2.getTrackbarPos("H high", "Controls")
    q.s_min = cv2.getTrackbarPos("S min", "Controls")
    q.v_min = cv2.getTrackbarPos("V min", "Controls")
    q.lab_tol = cv2.getTrackbarPos("Lab tol", "Controls")
    q.suppress_green = cv2.getTrackbarPos("Suppress green (0/1)", "Controls")
    q.auto_exposure = cv2.getTrackbarPos("Auto exp (0/1)", "Controls")
    q.exposure = cv2.getTrackbarPos("Exposure slider", "Controls")

    q.morph_open = cv2.getTrackbarPos("Morph open (0-2)", "Controls")
    q.morph_close = cv2.getTrackbarPos("Morph close (0-2)", "Controls")
    q.blur_ksize = ensure_odd(max(1, cv2.getTrackbarPos("Blur ksize (odd)", "Controls")))

    q.min_circularity = cv2.getTrackbarPos("min circ x100", "Controls") / 100.0
    q.min_axis_ratio = cv2.getTrackbarPos("min axis x100", "Controls") / 100.0
    q.min_fill = cv2.getTrackbarPos("min fill x100", "Controls") / 100.0
    q.max_residual = cv2.getTrackbarPos("max residual x100", "Controls") / 100.0
    q.min_arc_coverage = cv2.getTrackbarPos("min arc x100", "Controls") / 100.0
    q.min_edge_cov = cv2.getTrackbarPos("min edge x100", "Controls") / 100.0
    q.edge_band = cv2.getTrackbarPos("edge band x100", "Controls") / 100.0
    q.patch_std_min = cv2.getTrackbarPos("std min x10", "Controls") / 10.0
    q.min_radius_px = max(1, cv2.getTrackbarPos("min r px", "Controls"))
    q.max_radius_frac = cv2.getTrackbarPos("max r % short", "Controls") / 100.0

    q.scale = max(0.2, cv2.getTrackbarPos("scale x100", "Controls") / 100.0)
    q.detect_every = max(1, cv2.getTrackbarPos("detect every", "Controls"))
    q.score_thr = cv2.getTrackbarPos("score thr x100", "Controls") / 100.0
    q.nms_iou = cv2.getTrackbarPos("nms iou x100", "Controls") / 100.0
    q.nms_keep = max(1, cv2.getTrackbarPos("nms keep", "Controls"))
    return q


def apply_exposure(cap: cv2.VideoCapture, p: Params) -> None:
    """
    Try to force manual exposure on V4L2:
    - CAP_PROP_AUTO_EXPOSURE: 0.25 means manual, 0.75 means auto (per V4L2 backend约定)
    - CAP_PROP_EXPOSURE: typical range is negative (log2 seconds); here we pass slider directly,
      relying on driver to interpret. If无效，可在运行时调 slider。
    """
    if p.auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual
        cap.set(cv2.CAP_PROP_EXPOSURE, float(p.exposure))


# ---------- Masking ----------
def hsv_orange_mask(bgr: np.ndarray, p: Params) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_main = np.array([p.h_low, p.s_min, p.v_min], dtype=np.uint8)
    upper_main = np.array([p.h_high, 255, 255], dtype=np.uint8)
    lower_hi = np.array([p.h_low, max(0, p.s_min - 30), max(0, p.v_min - 20)], dtype=np.uint8)
    upper_hi = upper_main
    lower_shadow = np.array([p.h_low, p.s_min, max(0, p.v_min - 40)], dtype=np.uint8)
    upper_shadow = upper_main
    mask = cv2.inRange(hsv, lower_main, upper_main)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_hi, upper_hi))
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_shadow, upper_shadow))
    return mask


def lab_deltae_mask(bgr: np.ndarray, center_lab: np.ndarray, tol: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    dL = lab[:, :, 0] - int(center_lab[0])
    da = lab[:, :, 1] - int(center_lab[1])
    db = lab[:, :, 2] - int(center_lab[2])
    dist = np.sqrt(dL.astype(np.float32) ** 2 + da.astype(np.float32) ** 2 + db.astype(np.float32) ** 2)
    return (dist <= tol).astype(np.uint8) * 255


def suppress_green(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_g = np.array([35, 40, 40], dtype=np.uint8)
    upper_g = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_g, upper_g)
    return cv2.bitwise_not(green_mask)


# ---------- Geometry ----------
def ring_edge_coverage(edges: np.ndarray, center: Tuple[int, int], radius: int, band: float = 0.1) -> float:
    h, w = edges.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r_outer = int(radius * (1.0 + band))
    r_inner = max(1, int(radius * (1.0 - band)))
    if r_outer <= 0:
        return 0.0
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (cx, cy), r_outer, 255, thickness=-1)
    cv2.circle(ring, (cx, cy), r_inner, 0, thickness=-1)
    ring_edges = cv2.bitwise_and(edges, ring)
    ring_pixels = int(np.count_nonzero(ring))
    if ring_pixels == 0:
        return 0.0
    return float(np.count_nonzero(ring_edges)) / float(ring_pixels)


def patch_std(gray: np.ndarray, center: Tuple[int, int], radius: int) -> float:
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, thickness=-1)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        return 0.0
    return float(np.std(pixels))


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(dets: List[dict], iou_thr: float = 0.35, max_keep: int = 6) -> List[dict]:
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept: List[dict] = []
    for d in dets:
        if len(kept) >= max_keep:
            break
        if all(iou(d["bbox"], k["bbox"]) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def detect_candidates(mask: np.ndarray, edges: np.ndarray, gray: np.ndarray, p: Params) -> List[dict]:
    h, w = mask.shape[:2]
    min_r = max(p.min_radius_px, int(min(h, w) * 0.02))
    max_r = int(min(h, w) * p.max_radius_frac)
    min_area = max(1, int(h * w * 0.0003))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        cx_i, cy_i, r_i = int(cx), int(cy), int(radius)
        x, y, bw, bh = cv2.boundingRect(c)
        axis_ratio = float(min(bw, bh)) / float(max(bw, bh)) if max(bw, bh) > 0 else 0
        if axis_ratio < p.min_axis_ratio:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perim * perim))
        if circularity < p.min_circularity:
            continue
        fill = float(area) / float(np.pi * radius * radius + 1e-6)
        if fill < p.min_fill:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        residual = float(np.mean(np.abs(dists - radius)) / max(radius, 1e-3))
        if residual > p.max_residual:
            continue
        arc_band = np.sum(np.abs(dists - radius) <= (0.1 * radius))
        arc_cov = float(arc_band) / float(len(dists)) if len(dists) > 0 else 0.0
        if arc_cov < p.min_arc_coverage:
            continue
        edge_cov = ring_edge_coverage(edges, (cx_i, cy_i), r_i, band=p.edge_band)
        if edge_cov < p.min_edge_cov:
            continue
        std_val = patch_std(gray, (cx_i, cy_i), r_i)
        if std_val < p.patch_std_min:
            continue
        score = min(1.0, 0.5 * circularity + 0.25 * edge_cov + 0.25 * min(1.0, std_val / 12.0))
        dets.append(
            {
                "center": (cx_i, cy_i),
                "radius": r_i,
                "bbox": (x, y, bw, bh),
                "score": score,
            }
        )
    return dets


def draw_detections(frame: np.ndarray, circles: List[dict], score_thr: float = 0.7) -> None:
    for c in circles:
        if c["score"] < score_thr:
            continue
        cx, cy, r = c["center"][0], c["center"][1], c["radius"]
        color = (0, 200, 0)
        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.rectangle(frame, (c["bbox"][0], c["bbox"][1]),
                      (c["bbox"][0] + c["bbox"][2], c["bbox"][1] + c["bbox"][3]), color, 1)
        label = f"{int(round(c['score'] * 100))}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (cx, cy - th - bl - 6), (cx + tw + 6, cy), color, -1)
        cv2.putText(frame, label, (cx + 3, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


# ---------- Snapshot annotation ----------
class Annotator:
    def __init__(self):
        self.drawing = False
        self.radius = 8
        self.mask = None
        self.img = None
        self.window = "Annotate"

    def set_image(self, img: np.ndarray):
        self.img = img.copy()
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)

    def on_mouse(self, event, x, y, flags, param):
        if self.img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.radius, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.mask, (x, y), self.radius, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.mask, (x, y), self.radius, 255, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.mask, (x, y), self.radius, 0, -1)

    def show(self):
        if self.img is None:
            return False
        overlay = self.img.copy()
        overlay[self.mask > 0] = (0, 165, 255)
        cv2.imshow(self.window, overlay)
        return True

    def compute_params(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Compute Lab center and radius guess from painted mask."""
        if self.img is None or self.mask is None:
            return None, (0, 0)
        if np.count_nonzero(self.mask) == 0:
            return None, (0, 0)
        # Lab mean
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        pixels = lab[self.mask > 0]
        lab_center = np.mean(pixels, axis=0)
        # radius guess from mask area
        area = np.count_nonzero(self.mask)
        r = int(np.sqrt(area / np.pi))
        return lab_center, (r, r)


# ---------- Main loop ----------
def main() -> None:
    params = Params()
    create_trackbar_window(params)

    cap, cam_idx = find_camera()
    if cap is None:
        print("无法打开摄像头。用 CAM_INDEX=编号 指定 /dev/videoX，并确保未被 pipewire 等占用。")
        return
    print(f"已打开摄像头 /dev/video{cam_idx if cam_idx is not None else '?'}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Initial exposure setup (manual by default)
    apply_exposure(cap, params)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    lab_center = np.array([170, 140, 80], dtype=np.int16)
    clicked_lab = [lab_center]
    mode = "live"  # "live" or "snapshot"
    snapshot_frame = None
    annotator = Annotator()

    def on_mouse_original(event, x, y, flags, userdata):
        frame_bgr = userdata.get("frame")
        if frame_bgr is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < 0 or y >= frame_bgr.shape[0] or x < 0 or x >= frame_bgr.shape[1]:
                return
            bgr = frame_bgr[y, x].reshape(1, 1, 3)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3,).astype(np.int16)
            clicked_lab[0] = lab
            print(f"Picked Lab center: {lab}")

    cv2.setMouseCallback("Original", on_mouse_original, {"frame": None})

    frame_idx = 0
    last_dets: List[dict] = []

    try:
        while True:
            params = read_trackbar_params(params)
            # Update exposure each loop (cheap; some drivers need this)
            apply_exposure(cap, params)

            if mode == "live":
                cap.grab()
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    print("从摄像头读取失败")
                    break
                source_frame = frame
            else:
                source_frame = snapshot_frame
                if source_frame is None:
                    mode = "live"
                    continue

            scale = params.scale
            frame_small = cv2.resize(source_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # update mouse callback data
            cv2.setMouseCallback("Original", on_mouse_original, {"frame": frame_small})

            # Color masks
            mask_hsv = hsv_orange_mask(frame_small, params)
            mask_lab = lab_deltae_mask(frame_small, clicked_lab[0], params.lab_tol)
            mask_color = cv2.bitwise_or(mask_hsv, mask_lab)
            if params.suppress_green:
                mask_color = cv2.bitwise_and(mask_color, suppress_green(frame_small))

            # Morph
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = mask_color
            if params.morph_open > 0:
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=params.morph_open)
            if params.morph_close > 0:
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=params.morph_close)

            # Grayscale for edges
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            k = ensure_odd(params.blur_ksize)
            gray_blur = cv2.GaussianBlur(gray_eq, (k, k), 0)
            edges = cv2.Canny(gray_blur, 50, 140)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            edges_masked = cv2.bitwise_and(edges, mask_clean)

            run_detect = (frame_idx % params.detect_every == 0) or (mode == "snapshot")
            dets = last_dets
            if run_detect:
                dets = detect_candidates(mask_clean, edges_masked, gray_eq, params)
                dets = nms(dets, iou_thr=params.nms_iou, max_keep=params.nms_keep)
                if dets and scale != 1.0:
                    mapped = []
                    inv = 1.0 / scale
                    for d in dets:
                        cx, cy = d["center"]
                        r = d["radius"]
                        x, y, bw, bh = d["bbox"]
                        mapped.append(
                            {
                                "center": (int(cx * inv), int(cy * inv)),
                                "radius": int(r * inv),
                                "bbox": (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)),
                                "score": d["score"],
                            }
                        )
                    dets = mapped
                last_dets = dets

            frame_idx += 1

            result = source_frame.copy()
            draw_detections(result, dets, score_thr=params.score_thr)

            cv2.imshow("Original", source_frame)
            cv2.imshow("Mask", cv2.resize(mask_clean, (source_frame.shape[1], source_frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Edges", cv2.resize(edges_masked, (source_frame.shape[1], source_frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("k") and (cv2.getWindowProperty("Original", 0) >= 0):
                # Take snapshot
                snapshot_frame = source_frame.copy()
                mode = "snapshot"
                annotator.set_image(cv2.resize(snapshot_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                cv2.namedWindow(annotator.window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(annotator.window, annotator.on_mouse)
                # Pause in annotate loop until window closed or Enter pressed
                while True:
                    if not annotator.show():
                        break
                    k2 = cv2.waitKey(1) & 0xFF
                    if k2 == 13:  # Enter to compute
                        lab_center_new, r_guess = annotator.compute_params()
                        if lab_center_new is not None:
                            clicked_lab[0] = lab_center_new.astype(np.int16)
                            print(f"[Annotate] Lab center set to {lab_center_new}, r_guess={r_guess}")
                        break
                    if k2 == ord("c"):
                        if annotator.mask is not None:
                            annotator.mask[:] = 0
                    if k2 == ord("q") or k2 == 27:
                        break
                cv2.destroyWindow(annotator.window)
                mode = "live"

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

