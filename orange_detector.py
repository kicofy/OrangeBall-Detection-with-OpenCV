import cv2
import numpy as np
import glob

latest_frame_bgr = None  # updated each frame for mouse picking
latest_picked_lab = None  # persists last picked color in Lab for ΔE mask


def create_trackbar_window(
    initial_lower_hsv: tuple[int, int, int] = (5, 100, 100),
    initial_upper_hsv: tuple[int, int, int] = (25, 255, 255),
) -> None:
    """
    Create a control window with HSV range trackbars.

    Defaults target an orange hue in OpenCV HSV space:
    - Hue range is [0, 179] in OpenCV (not [0, 360]).
    - Saturation/Value are [0, 255].
    """
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 480, 320)

    lower_h, lower_s, lower_v = initial_lower_hsv
    upper_h, _, _ = initial_upper_hsv

    # Simple controls: center/width for H, minimums for S/V
    # Defaults tuned for your current setup
    default_h_center = 7
    default_h_width = 2
    default_s_min = 180
    default_v_min = 215

    cv2.createTrackbar("H center", "Controls", default_h_center, 179, lambda x: None)
    cv2.createTrackbar("H width", "Controls", default_h_width, 60, lambda x: None)
    cv2.createTrackbar("S min", "Controls", default_s_min, 255, lambda x: None)
    cv2.createTrackbar("V min", "Controls", default_v_min, 255, lambda x: None)

    # Optional morphology strength (0 disables)
    cv2.createTrackbar("Morph (0-5)", "Controls", 1, 5, lambda x: None)

    # Lighting tolerance to handle highlights/shadows on glossy spheres
    cv2.createTrackbar("Light tol (0-3)", "Controls", 2, 3, lambda x: None)
    # LAB distance mask controls
    cv2.createTrackbar("DE tol (10-50)", "Controls", 25, 50, lambda x: None)
    cv2.createTrackbar("Use LAB (0/1)", "Controls", 1, 1, lambda x: None)

    # Picker mode (fallback for mac trackpad modifier issues)
    cv2.createTrackbar("Pick mode (0/1)", "Controls", 0, 1, lambda x: None)

    # Toggle detection for performance tuning
    cv2.createTrackbar("Detect (0/1)", "Controls", 1, 1, lambda x: None)


def read_trackbar_hsv_bounds() -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read simplified HSV bounds and morphology strength from trackbars.
    Returns (lower, upper, morph_strength).
    """
    h_center = cv2.getTrackbarPos("H center", "Controls")
    h_width = cv2.getTrackbarPos("H width", "Controls")
    s_min = cv2.getTrackbarPos("S min", "Controls")
    v_min = cv2.getTrackbarPos("V min", "Controls")

    h_low = max(0, h_center - h_width)
    h_high = min(179, h_center + h_width)

    morph_strength = cv2.getTrackbarPos("Morph (0-5)", "Controls")

    lower = np.array([h_low, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    return lower, upper, morph_strength








def set_hsv_bounds_from_center(center_hsv: np.ndarray, dh: int, ds: int, dv: int) -> None:
    """
    Given a center HSV and deltas, update the lower/upper HSV trackbars.
    If the picked color is already inside current bounds, shrink the window
    toward the picked color by intersecting with [center - delta, center + delta].
    """
    h, s, v = int(center_hsv[0]), int(center_hsv[1]), int(center_hsv[2])

    # Current simple parameters
    h_center = cv2.getTrackbarPos("H center", "Controls")
    h_width = cv2.getTrackbarPos("H width", "Controls")
    s_min = cv2.getTrackbarPos("S min", "Controls")
    v_min = cv2.getTrackbarPos("V min", "Controls")

    curr_h_low = max(0, h_center - h_width)
    curr_h_high = min(179, h_center + h_width)

    # Candidate window around picked pixel
    cand_h_low = max(0, h - dh)
    cand_h_high = min(179, h + dh)

    inside = (curr_h_low <= h <= curr_h_high) and (s_min <= s <= 255) and (v_min <= v <= 255)

    if inside:
        # Intersect hue range; move center to picked hue
        inter_low = max(curr_h_low, cand_h_low)
        inter_high = min(curr_h_high, cand_h_high)
        new_h_center = int((inter_low + inter_high) // 2)
        new_h_width = int(max(1, (inter_high - inter_low) // 2))
        # Tighten S/V minimums toward picked value
        new_s_min = int(max(s_min, max(0, s - ds)))
        new_v_min = int(max(v_min, max(0, v - dv)))
    else:
        # Recenter around picked hue with width=dh, and set S/V mins near picked
        new_h_center = int(h)
        new_h_width = int(max(1, dh))
        new_s_min = int(max(0, s - ds))
        new_v_min = int(max(0, v - dv))

    cv2.setTrackbarPos("H center", "Controls", new_h_center)
    cv2.setTrackbarPos("H width", "Controls", new_h_width)
    cv2.setTrackbarPos("S min", "Controls", new_s_min)
    cv2.setTrackbarPos("V min", "Controls", new_v_min)


def on_mouse_original(event: int, x: int, y: int, flags: int, userdata) -> None:
    """
    Shift + Left Click on Original window: center HSV thresholds on clicked pixel.
    """
    global latest_frame_bgr, latest_picked_lab

    if event == cv2.EVENT_LBUTTONUP:
        shift_down = (flags & cv2.EVENT_FLAG_SHIFTKEY) != 0
        pick_mode_enabled = cv2.getTrackbarPos("Pick mode (0/1)", "Controls") > 0
        if not (shift_down or pick_mode_enabled):
            return
        if latest_frame_bgr is None:
            return
        if y < 0 or x < 0 or y >= latest_frame_bgr.shape[0] or x >= latest_frame_bgr.shape[1]:
            return

        bgr = latest_frame_bgr[y, x].reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(3,)
        # persist LAB center for ΔE mask
        latest_picked_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3,).astype(np.int16)
        # Use half of current width as shrink step for hue, and fixed S/V deltas
        current_width = max(1, cv2.getTrackbarPos("H width", "Controls"))
        dh = max(2, current_width // 2)
        ds, dv = 40, 40
        set_hsv_bounds_from_center(hsv, dh, ds, dv)

def compute_circularity(contour: np.ndarray) -> float:
    """
    Compute circularity metric in [0,1], where 1 is a perfect circle.
    circularity = 4*pi*area / perimeter^2
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return 0.0
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    # Clamp to [0,1] for numerical stability
    if circularity < 0.0:
        return 0.0
    if circularity > 1.0:
        return 1.0
    return float(circularity)


def find_circle_like_regions(
    mask_binary: np.ndarray,
    min_area_px: int,
    min_circularity: float,
    min_fill_ratio: float = 0.6,
    min_axis_ratio: float = 0.75,
) -> list[dict]:
    """
    Find circle-like regions from a binary mask using several geometric tests:
    - circularity (4*pi*A/P^2)
    - fill ratio versus minimum enclosing circle area
    - axis ratio from bounding box (w/h close to 1)
    Returns list of detections with bbox and composite score in [0,1].
    """
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(0, int(min_area_px)):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        axis_ratio = float(min(w, h)) / float(max(w, h))
        if axis_ratio < min_axis_ratio:
            continue

        circularity = compute_circularity(contour)
        if circularity < min_circularity:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = float(np.pi * radius * radius)
        if circle_area <= 0.0:
            continue
        fill_ratio = float(area) / circle_area
        if fill_ratio < min_fill_ratio:
            continue

        # Composite score: weighted average
        score = float(0.5 * circularity + 0.3 * fill_ratio + 0.2 * axis_ratio)
        detections.append(
            {
                "bbox": (int(x), int(y), int(w), int(h)),
                "score": max(0.0, min(1.0, score)),
                "contour": contour,
            }
        )
    return detections


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> None:
    """
    Draw bounding boxes and similarity score (%) for each detection on frame_bgr.
    """
    for det in detections:
        x, y, w, h = det["bbox"]
        score = det["score"]
        score_percent = int(round(score * 100))

        # Color by score
        if score >= 0.8:
            color = (0, 200, 0)  # green
        elif score >= 0.6:
            color = (0, 215, 255)  # gold-ish
        else:
            color = (0, 140, 255)  # orange

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        label = f"{score_percent}%"
        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x, y - th - baseline - 4), (x + tw + 6, y), color, -1)
        cv2.putText(
            frame_bgr,
            label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def apply_morphology(mask: np.ndarray, strength: int) -> np.ndarray:
    """
    Apply simple morphology to clean up noise in the binary mask.
    'strength' controls kernel size (0 disables morphology).
    """
    if strength <= 0:
        return mask

    kernel_size = 2 * strength + 1  # odd sizes: 3,5,7,9,11
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Open to remove small noise, then close to fill small holes
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def build_lighting_robust_mask(frame_hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Union of three HSV masks to cover highlights and shadows on glossy objects:
    - main:   [H_low, S_min, V_min] .. [H_high, 255, 255]
    - hilite: relax saturation lower bound
    - shadow: relax value(lower V) bound
    Relaxation amounts depend on "Light tol (0-3)".
    """
    h_low, s_min, v_min = int(lower[0]), int(lower[1]), int(lower[2])
    h_high = int(upper[0])

    tol = cv2.getTrackbarPos("Light tol (0-3)", "Controls")
    # Map tolerance to relax amounts
    s_relax = [0, 25, 45, 65][min(max(0, tol), 3)]
    v_relax = [0, 25, 45, 65][min(max(0, tol), 3)]

    # Base range
    lower_main = np.array([h_low, s_min, v_min], dtype=np.uint8)
    upper_main = np.array([h_high, 255, 255], dtype=np.uint8)
    mask_main = cv2.inRange(frame_hsv, lower_main, upper_main)

    # Highlight (desaturated bright spots)
    lower_hilite = np.array([h_low, max(0, s_min - s_relax), v_min], dtype=np.uint8)
    upper_hilite = np.array([h_high, 255, 255], dtype=np.uint8)
    mask_hilite = cv2.inRange(frame_hsv, lower_hilite, upper_hilite)

    # Shadow (darker but still orange)
    lower_shadow = np.array([h_low, s_min, max(0, v_min - v_relax)], dtype=np.uint8)
    upper_shadow = np.array([h_high, 255, 255], dtype=np.uint8)
    mask_shadow = cv2.inRange(frame_hsv, lower_shadow, upper_shadow)

    mask = cv2.bitwise_or(mask_main, mask_hilite)
    mask = cv2.bitwise_or(mask, mask_shadow)
    return mask


def build_lab_deltae_mask(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Build a ΔE (CIE76) mask around the last picked Lab color.
    Returns None if no color has been picked or Use LAB is disabled.
    """
    if cv2.getTrackbarPos("Use LAB (0/1)", "Controls") == 0:
        return None
    global latest_picked_lab
    if latest_picked_lab is None:
        return None
    tol = cv2.getTrackbarPos("DE tol (10-50)", "Controls")
    tol = max(1, int(tol))

    lab_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    # compute Euclidean distance in Lab space
    dL = lab_img[:, :, 0] - int(latest_picked_lab[0])
    da = lab_img[:, :, 1] - int(latest_picked_lab[1])
    db = lab_img[:, :, 2] - int(latest_picked_lab[2])
    dist = np.sqrt(dL.astype(np.float32) ** 2 + da.astype(np.float32) ** 2 + db.astype(np.float32) ** 2)
    mask = (dist <= float(tol)).astype(np.uint8) * 255
    return mask


def _compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms_detections(detections: list[dict], iou_threshold: float = 0.3, max_keep: int = 20) -> list[dict]:
    """
    Greedy Non-Maximum Suppression to reduce duplicate/overlapping boxes.
    """
    if not detections:
        return detections
    detections_sorted = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for det in detections_sorted:
        if len(kept) >= max_keep:
            break
        if all(_compute_iou(det["bbox"], k["bbox"]) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


def open_camera_with_fallbacks(device_index: int = 0) -> cv2.VideoCapture | None:
    """
    Try multiple backends to open the camera, useful on Raspberry Pi / Linux when GStreamer fails.
    """
    candidates = [
        (cv2.CAP_V4L2, "CAP_V4L2"),
        (cv2.CAP_ANY, "CAP_ANY"),
    ]
    for backend, name in candidates:
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # try a test read to verify frames are available
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None


def find_first_working_camera(max_devices: int = 10) -> tuple[cv2.VideoCapture | None, int | None]:
    """
    Scan /dev/video* (up to max_devices) and return the first working camera capture and its index.
    """
    indices: list[int] = []
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int("".join(ch for ch in path if ch.isdigit()))
            indices.append(idx)
        except ValueError:
            continue
    # fallback to 0..max_devices if none found
    if not indices:
        indices = list(range(0, max_devices + 1))
    seen = set()
    unique_indices = [i for i in indices if not (i in seen or seen.add(i))]

    for idx in unique_indices:
        cap = open_camera_with_fallbacks(idx)
        if cap is not None:
            return cap, idx
    return None, None


def main() -> None:
    """
    打开电脑摄像头，使用HSV阈值识别画面中橙色区域，并显示阈值内的画面。

    操作方法：
    - 简化参数（推荐）：在“Controls”窗口仅调整：
      H center（色相中心）、H width（半宽度）、S min、V min、Morph。
      Light tol (0-3) 用于增强对高光/阴影的容忍度。
    - 在“Original”窗口按住 Shift 并左键点击（或开启 Pick mode 直接左键），
      将以点击像素为中心自动设置/收紧阈值；若点击点已在范围内将与
      [中心±步长] 求交集，向点击颜色靠拢。
      若 mac 触摸板 Shift 修饰键无效，可打开“Pick mode (0/1)”或按 'p' 切换拾色模式，
      这时左键点击即可拾色。
    - 若调参卡顿，可将“Detect (0/1)”设为 0 关闭检测，仅看阈值效果；调好后再打开。
      同时可将“Morph”设小一些。程序会尝试将摄像头分辨率设置为 640x480 以减轻负载。
    - 按 'q' 或 ESC 退出。
    """
    # Default range roughly for orange
    default_lower = (5, 100, 100)
    default_upper = (25, 255, 255)
    create_trackbar_window(default_lower, default_upper)

    video_capture, cam_idx = find_first_working_camera()
    if video_capture is None:
        print(
            "无法打开摄像头。请检查：\n"
            "1) 摄像头是否连接、启用（树莓派需在 raspi-config 打开摄像头）。\n"
            "2) 当前用户是否在 video 组，可用 `groups` 检查，若无执行 `sudo usermod -aG video $(whoami)` 并重启。\n"
            "3) 若仍失败，可安装/修复 v4l2 与 gstreamer，或尝试指定其他 /dev/videoX。"
        )
        return
    else:
        print(f"已打开摄像头 /dev/video{cam_idx if cam_idx is not None else '?'}")
    # Reduce processing load by lowering capture resolution
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result (In-Range)", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("Original", on_mouse_original)

    try:
        while True:
            captured, frame_bgr = video_capture.read()
            if not captured or frame_bgr is None:
                print("从摄像头读取失败。")
                break

            # update latest frame for picker usage
            global latest_frame_bgr
            latest_frame_bgr = frame_bgr

            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower_hsv, upper_hsv, morph_strength = read_trackbar_hsv_bounds()

            # Lighting-robust union mask for orange (handles highlights/shadows)
            mask_hsv = build_lighting_robust_mask(frame_hsv, lower_hsv, upper_hsv)
            # Optional LAB ΔE mask centered at picked color (handles bright/dark sides)
            mask_lab = build_lab_deltae_mask(frame_bgr)
            mask = mask_hsv if mask_lab is None else cv2.bitwise_or(mask_hsv, mask_lab)
            mask = apply_morphology(mask, morph_strength)

            result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

            # Circle-like detection based on the orange mask (can be disabled)
            result_to_show = result
            if cv2.getTrackbarPos("Detect (0/1)", "Controls") > 0:
                # Early exit if mask is near-empty to avoid false positives in Hough
                frame_h, frame_w = mask.shape[:2]
                num_foreground = int(cv2.countNonZero(mask))
                frame_area = int(frame_h * frame_w)
                fg_ratio = num_foreground / max(1, frame_area)
                if fg_ratio < 0.002:  # less than 0.2% pixels in-range → skip detection
                    cv2.imshow("Original", frame_bgr)
                    cv2.imshow("Mask", mask)
                    cv2.imshow("Result (In-Range)", result_to_show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("p"):
                        pick_val = cv2.getTrackbarPos("Pick mode (0/1)", "Controls")
                        cv2.setTrackbarPos("Pick mode (0/1)", "Controls", 0 if pick_val > 0 else 1)
                    continue

                # Adaptive thresholds based on frame size
                min_area_px = max(150, int(frame_area * 0.0005))  # ~0.05% of frame
                min_circularity = 0.7

                detections = find_circle_like_regions(
                    mask,
                    min_area_px=min_area_px,
                    min_circularity=min_circularity,
                    min_fill_ratio=0.65,
                    min_axis_ratio=0.8,
                )

                # Fallback: try HoughCircles if contour-based finds nothing
                if len(detections) == 0 and fg_ratio > 0.01:
                    # Work on masked grayscale; use tighter params
                    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
                    min_dist = max(12, min(frame_h, frame_w) // 12)
                    circles = cv2.HoughCircles(
                        gray,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=min_dist,
                        param1=140,
                        param2=45,
                        minRadius=6,
                        maxRadius=0,
                    )
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype(int)
                        # Validate by orange coverage inside the circle
                        for (cx, cy, r) in circles[:12]:
                            x = max(0, cx - r)
                            y = max(0, cy - r)
                            w = min(frame_w - x, 2 * r)
                            h = min(frame_h - y, 2 * r)
                            if w <= 0 or h <= 0:
                                continue
                            circle_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                            cv2.circle(circle_mask, (int(cx), int(cy)), int(r), 255, thickness=-1)
                            # coverage of orange pixels in this circle
                            orange_in_circle = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
                            circle_area_px = cv2.countNonZero(circle_mask)
                            if circle_area_px == 0:
                                continue
                            coverage = orange_in_circle / float(circle_area_px)
                            if coverage < 0.55:
                                continue
                            score = max(0.0, min(1.0, coverage))
                            detections.append({"bbox": (x, y, w, h), "score": score, "contour": None})

                # De-duplicate boxes
                detections = nms_detections(detections, iou_threshold=0.35, max_keep=6)

                result_with_boxes = result.copy()
                if detections:
                    draw_detections(result_with_boxes, detections)
                result_to_show = result_with_boxes

            cv2.imshow("Original", frame_bgr)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result (In-Range)", result_to_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 is ESC
                break
            if key == ord("p"):
                # toggle pick mode
                pick_val = cv2.getTrackbarPos("Pick mode (0/1)", "Controls")
                cv2.setTrackbarPos("Pick mode (0/1)", "Controls", 0 if pick_val > 0 else 1)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


