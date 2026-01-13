"""
Golf ball detector for orange balls on green grass.
Uses color masking (HSV + Lab) with green suppression, then shape validation.
"""

import glob
import os
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


# ---------- Masking helpers ----------
def hsv_orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Base orange range
    lower_main = np.array([5, 80, 80], dtype=np.uint8)
    upper_main = np.array([25, 255, 255], dtype=np.uint8)
    # Highlight-tolerant (lower saturation)
    lower_hi = np.array([5, 50, 70], dtype=np.uint8)
    upper_hi = upper_main
    # Shadow-tolerant (lower value)
    lower_shadow = np.array([5, 90, 40], dtype=np.uint8)
    upper_shadow = upper_main
    mask = cv2.inRange(hsv, lower_main, upper_main)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_hi, upper_hi))
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_shadow, upper_shadow))
    return mask


def lab_deltae_mask(bgr: np.ndarray, center_lab: np.ndarray, tol: float = 28.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    dL = lab[:, :, 0] - int(center_lab[0])
    da = lab[:, :, 1] - int(center_lab[1])
    db = lab[:, :, 2] - int(center_lab[2])
    dist = np.sqrt(dL.astype(np.float32) ** 2 + da.astype(np.float32) ** 2 + db.astype(np.float32) ** 2)
    return (dist <= tol).astype(np.uint8) * 255


def suppress_green(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Green range (typical grass) to suppress
    lower_g = np.array([35, 40, 40], dtype=np.uint8)
    upper_g = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_g, upper_g)
    return cv2.bitwise_not(green_mask)


# ---------- Geometry helpers ----------
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


def nms(dets: List[dict], iou_thr: float = 0.3, max_keep: int = 5) -> List[dict]:
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


# ---------- Detection ----------
def detect_candidates_from_mask(mask: np.ndarray, edges: np.ndarray, gray: np.ndarray) -> List[dict]:
    h, w = mask.shape[:2]
    min_r = max(6, min(h, w) // 40)
    max_r = int(min(h, w) * 0.45)
    min_area = max(1, int(h * w * 0.0003))

    dets: List[dict] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        cx_i, cy_i, r_i = int(cx), int(cy), int(radius)

        # Geometry quick checks
        x, y, bw, bh = cv2.boundingRect(c)
        axis_ratio = float(min(bw, bh)) / float(max(bw, bh)) if max(bw, bh) > 0 else 0
        if axis_ratio < 0.75:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perim * perim))
        if circularity < 0.75:
            continue
        # Edge/texture checks
        edge_cov = ring_edge_coverage(edges, (cx_i, cy_i), r_i, band=0.10)
        if edge_cov < 0.55:
            continue
        std_val = patch_std(gray, (cx_i, cy_i), r_i)
        if std_val < 5.0:
            continue
        score = min(1.0, 0.6 * circularity + 0.25 * edge_cov + 0.15 * min(1.0, std_val / 12.0))
        dets.append(
            {
                "center": (cx_i, cy_i),
                "radius": r_i,
                "bbox": (x, y, bw, bh),
                "score": score,
            }
        )
    return dets


# ---------- Drawing ----------
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


# ---------- Main ----------
def main() -> None:
    cap, cam_idx = find_camera()
    if cap is None:
        print("无法打开摄像头。用 CAM_INDEX=编号 指定 /dev/videoX，并确保未被 pipewire 等占用。")
        return
    print(f"已打开摄像头 /dev/video{cam_idx if cam_idx is not None else '?'}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # Default Lab center for orange; can be updated by clicking if needed
    default_lab_center = np.array([170, 140, 80], dtype=np.int16)  # coarse orange center
    lab_center = default_lab_center

    # Simple click-to-pick Lab center
    clicked_lab = [lab_center]

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_bgr = userdata["frame"]
            if frame_bgr is None:
                return
            if y < 0 or y >= frame_bgr.shape[0] or x < 0 or x >= frame_bgr.shape[1]:
                return
            bgr = frame_bgr[y, x].reshape(1, 1, 3)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3,).astype(np.int16)
            clicked_lab[0] = lab
            print(f"Picked Lab center: {lab}")

    cv2.setMouseCallback("Original", on_mouse, {"frame": None})

    try:
        while True:
            cap.grab()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                print("从摄像头读取失败")
                break

            # Downscale for speed
            scale = 0.75
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Update mouse callback data
            cv2.setMouseCallback("Original", on_mouse, {"frame": frame_small})

            # Green suppression mask
            not_green = suppress_green(frame_small)

            # HSV orange mask
            mask_hsv = hsv_orange_mask(frame_small)

            # Lab deltaE mask around picked color
            lab_center = clicked_lab[0]
            mask_lab = lab_deltae_mask(frame_small, lab_center, tol=28.0)

            # Combined mask (color OR) then suppress green
            mask_color = cv2.bitwise_or(mask_hsv, mask_lab)
            mask_color = cv2.bitwise_and(mask_color, not_green)

            # Morph clean
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Edges (on grayscale)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray_blur, 50, 140)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

            # Mask edges to color region to reduce false positives
            edges_masked = cv2.bitwise_and(edges, mask_clean)

            # Detect from mask + edges
            dets = detect_candidates_from_mask(mask_clean, edges_masked, gray)

            # Map back to full-res
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

            dets = nms(dets, iou_thr=0.3, max_keep=5)

            result = frame.copy()
            draw_detections(result, dets, score_thr=0.7)

            cv2.imshow("Original", frame)
            cv2.imshow("Mask", cv2.resize(mask_clean, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Edges", cv2.resize(edges_masked, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

