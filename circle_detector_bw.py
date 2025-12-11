import glob
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ---------- Camera utilities ----------
def open_camera_with_backends(device_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Try V4L2 first, then default."""
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None


def get_candidate_indices(max_devices: int = 6) -> List[int]:
    """
    Build candidate camera indices.
    - If CAM_INDEX is set, only use it.
    - Else, prefer the known Arducam OV9281 nodes (/dev/video8, /dev/video9),
      then discovered /dev/video* up to max_devices.
    """
    env_idx = os.environ.get("CAM_INDEX")
    if env_idx is not None:
        try:
            return [int(env_idx)]
        except ValueError:
            pass
    preferred = [8, 9]
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


def find_camera(max_devices: int = 6) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    for idx in get_candidate_indices(max_devices):
        cap = open_camera_with_backends(idx)
        if cap is not None:
            return cap, idx
    return None, None


# ---------- Geometry utilities ----------
def compute_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def compute_fill_ratio(contour: np.ndarray, min_enclosing_circle) -> float:
    (cx, cy), radius = min_enclosing_circle
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius * radius
    if circle_area <= 0:
        return 0.0
    return float(area / circle_area)


def compute_axis_ratio(bbox: Tuple[int, int, int, int]) -> float:
    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(w, h)) / float(max(w, h))


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


def nms(dets: List[dict], iou_thr: float = 0.35, max_keep: int = 8) -> List[dict]:
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
def detect_circles_mask(
    mask: np.ndarray,
    min_area_frac: float = 0.0005,
    min_circularity: float = 0.78,
    min_axis_ratio: float = 0.78,
    min_fill: float = 0.65,
) -> List[dict]:
    h, w = mask.shape[:2]
    min_area = max(1, int(h * w * min_area_frac))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = compute_axis_ratio((x, y, bw, bh))
        if ar < min_axis_ratio:
            continue
        circ = compute_circularity(c)
        if circ < min_circularity:
            continue
        mec = cv2.minEnclosingCircle(c)
        fill = compute_fill_ratio(c, mec)
        if fill < min_fill:
            continue
        (cx, cy), radius = mec
        score = float(0.5 * circ + 0.2 * ar + 0.3 * fill)
        dets.append(
            {
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "center": (int(cx), int(cy)),
                "radius": int(radius),
                "score": min(1.0, max(0.0, score)),
            }
        )
    return dets


def hough_fallback(
    gray: np.ndarray,
    mask: np.ndarray,
    min_dist: int,
    param1: int = 140,
    param2: int = 32,
    min_radius: int = 6,
    max_radius: int = 0,
    coverage_thr: float = 0.6,
) -> List[dict]:
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    h, w = mask.shape[:2]
    dets: List[dict] = []
    circles = np.round(circles[0, :]).astype(int)
    for (cx, cy, r) in circles[:12]:
        if r <= 0:
            continue
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            continue
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, thickness=-1)
        orange_pix = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
        total_pix = cv2.countNonZero(circle_mask)
        if total_pix == 0:
            continue
        coverage = orange_pix / float(total_pix)
        if coverage < coverage_thr:
            continue
        dets.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "center": (cx, cy),
                "radius": r,
                "score": min(1.0, max(0.0, coverage)),
            }
        )
    return dets


# ---------- Drawing ----------
def draw_detections(img_bgr: np.ndarray, detections: List[dict]) -> None:
    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        r = d["radius"]
        score = d["score"]
        color = (0, 200, 0) if score >= 0.8 else (0, 200, 200)
        cv2.circle(img_bgr, (cx, cy), r, color, 2)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 1)
        label = f"{int(round(score * 100))}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x, y - th - baseline - 4), (x + tw + 6, y), color, -1)
        cv2.putText(
            img_bgr,
            label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


# ---------- Main loop ----------
def main() -> None:
    cap, cam_idx = find_camera()
    if cap is None:
        print(
            "无法打开摄像头。可尝试：\n"
            "- 确认设备已连接并在 /dev/video* 出现。\n"
            "- 将当前用户加入 video 组；不要用 sudo 运行。\n"
            "- 使用环境变量 CAM_INDEX=数字 指定正确的 /dev/videoX。\n"
        )
        return
    print(f"已打开摄像头 /dev/video{cam_idx if cam_idx is not None else '?'}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Reduce latency: grab then retrieve latest frame
            cap.grab()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                print("从摄像头读取失败。")
                break

            # Ensure grayscale
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Preprocess
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morph clean
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Contour-based detection
            dets = detect_circles_mask(
                mask_clean,
                min_area_frac=0.0005,
                min_circularity=0.78,
                min_axis_ratio=0.78,
                min_fill=0.65,
            )

            # Hough fallback when none found
            if not dets:
                min_dist = max(12, min(gray.shape[:2]) // 12)
                hough_dets = hough_fallback(
                    gray_blur,
                    mask_clean,
                    min_dist=min_dist,
                    param1=140,
                    param2=30,
                    min_radius=6,
                    max_radius=0,
                    coverage_thr=0.6,
                )
                dets.extend(hough_dets)

            dets = nms(dets, iou_thr=0.35, max_keep=6)

            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if dets:
                draw_detections(result, dets)

            cv2.imshow("Original", gray)
            cv2.imshow("Mask", mask_clean)
            cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


