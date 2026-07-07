# OrangeBall Detection with OpenCV

[中文说明](README.zh-CN.md)

OrangeBall Detection with OpenCV is a camera-based orange golf ball detector built for a Wardlaw Hartridge School Robotics Club robot. It estimates where the ball appears in the camera frame so the robot code can use that position as a visual cue.

The detector uses OpenCV color segmentation with simple circle-like shape checks. The main script includes live controls for HSV thresholds, LAB color picking, morphology, lighting tolerance, and detection toggles, which makes field tuning easier under changing lighting.

## Project Goals

- Detect orange golf balls from a live camera feed.
- Estimate the ball position in image coordinates for robot movement decisions.
- Tune color thresholds interactively instead of recompiling or editing constants repeatedly.
- Handle highlights and shadows on glossy orange balls using HSV and optional LAB color distance.
- Reject non-ball orange regions by checking circularity, fill ratio, and bounding-box shape.
- Keep older detector experiments available for comparison and future tuning.

## Repository Structure

```text
src/
  orange_detector.py          Recommended live detector with GUI controls.

experiments/
  golf_ball_detector_gui.py   More advanced GUI experiment with snapshot annotation.
  golf_ball_detector_grass.py Orange-ball-on-grass detector experiment.
  golf_ball_detector_color.py Color-focused detector experiment.
  circle_detector_bw.py       Black/white circle detector experiment.

requirements.txt             Python dependencies.
README.md                    English documentation.
README.zh-CN.md              Chinese documentation.
```

## Main Detector

Run:

```bash
python src/orange_detector.py
```

The program opens a camera feed and displays:

- `Original`: live camera image.
- `Mask`: binary mask for the selected orange range.
- `Result`: annotated output with detected circle-like regions.
- `Controls`: sliders for threshold and detection tuning.

Recommended starting parameters are already built into the script:

```text
H center = 7
H width  = 2
S min    = 180
V min    = 215
Morph    = 1
Light tol = 2
Use LAB  = 1
Detect   = 1
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Camera Selection

By default, the detector tries to open a camera automatically. You can force a camera index with:

```bash
# Windows PowerShell
$env:CAM_INDEX="0"
python src/orange_detector.py

# macOS/Linux
CAM_INDEX=0 python src/orange_detector.py
```

Some experiment scripts also support camera index discovery for Linux `/dev/video*` devices.

## Controls

- `H center` / `H width`: orange hue center and half-width in OpenCV HSV space.
- `S min` / `V min`: minimum saturation and brightness.
- `Morph`: morphology strength for noise cleanup.
- `Light tol`: tolerance for highlights and shadows.
- `Use LAB` / `DE tol`: optional LAB color-distance mask after picking a sample color.
- `Pick mode`: when enabled, left-click the ball in `Original` to update the picked color.
- `Detect`: enable or disable circle-like region detection.

Shortcut keys:

- `p`: toggle pick mode.
- `q` or `Esc`: quit.

## Tuning Workflow

1. Start with `python src/orange_detector.py`.
2. Put the orange golf ball in the robot camera view.
3. If the mask misses part of the ball, enable `Pick mode` and click a well-lit area of the ball.
4. If the ball appears fragmented, increase `Light tol`, `DE tol`, or `Morph`.
5. If the background leaks into the mask, reduce `H width` or increase `S min` / `V min`.
6. Temporarily set `Detect = 0` while tuning the mask, then turn it back on for shape validation.

## Robot Integration Notes

This project currently provides visual detection and image-space position. A robot controller can use the detected ball center as a steering signal:

```text
camera frame -> orange mask -> circle-like detection -> ball center (x, y) -> robot alignment/movement logic
```

A typical control strategy is:

- If no ball is detected, rotate or scan.
- If the ball center is left/right of the image center, turn toward it.
- If the ball is centered, drive forward.
- Use the detected radius or bounding-box size as a rough distance cue.

## License

No license file is currently included.
