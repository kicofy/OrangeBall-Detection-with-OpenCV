import glob
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ---------- Camera helpers ----------
def open_camera_with_backends(device_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Try V4L2 first, then default."""
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
    """Prefer explicit CAM_INDEX; else prefer /dev/video0-7, then all /dev/video*."""
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


# ---------- Circle validation ----------
def ring_edge_coverage(edges: np.ndarray, center: Tuple[int, int], radius: int, band: float = 0.08) -> float:
    """Compute edge coverage within a ring [r*(1-band), r*(1+band)]."""
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


# ---------- Drawing ----------
def draw_detections(frame: np.ndarray, circles: List[dict], score_thr: float = 0.6) -> None:
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
        print("无法打开摄像头。请用 CAM_INDEX=编号 指定正确的 /dev/videoX，或检查是否有进程占用。")
        return
    print(f"已打开摄像头 /dev/video{cam_idx if cam_idx is not None else '?'}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    try:
        while True:
            cap.grab()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                print("从摄像头读取失败")
                break

            # Optionally downscale for speed
            scale = 0.75
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Convert to gray and enhance contrast
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Blur + edges
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray_blur, 50, 140)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

            # Hough candidates on edges
            h, w = gray.shape[:2]
            min_dist = max(16, min(h, w) // 10)
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=140,
                param2=18,  # low to get candidates; we validate later
                minRadius=6,
                maxRadius=int(min(h, w) * 0.5),
            )

            dets: List[dict] = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                for (cx, cy, r) in circles:
                    if r <= 0:
                        continue
                    edge_cov = ring_edge_coverage(edges, (cx, cy), r, band=0.10)
                    if edge_cov < 0.55:
                        continue
                    std_val = patch_std(gray, (cx, cy), r)
                    if std_val < 4.0:
                        continue
                    # Score combines edge coverage and local texture
                    score = min(1.0, 0.7 * edge_cov + 0.3 * min(1.0, std_val / 15.0))
                    x1, y1 = max(0, cx - r), max(0, cy - r)
                    x2, y2 = min(w, cx + r), min(h, cy + r)
                    dets.append(
                        {
                            "center": (cx, cy),
                            "radius": r,
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "score": score,
                        }
                    )

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

            result = frame.copy()
            draw_detections(result, dets, score_thr=0.6)

            cv2.imshow("Original", frame)
            cv2.imshow("Edges", cv2.resize(edges, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

