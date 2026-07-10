<div align="center">
  <h1>OrangeBall Detection with OpenCV</h1>
  <p>A live-tunable OpenCV detector for orange golf balls in robotics camera feeds.</p>

  <p>
    <a href="README.zh-CN.md">Chinese</a>
    &middot;
    <a href="#quickstart">Quickstart</a>
    &middot;
    <a href="#features">Features</a>
    &middot;
    <a href="#tech-stack">Tech Stack</a>
  </p>

  <p>
    <img alt="Python: OpenCV" src="https://img.shields.io/badge/Python-OpenCV-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Vision: color detection" src="https://img.shields.io/badge/Vision-color%20detection-287866?style=for-the-badge" />
    <img alt="Robotics: camera cue" src="https://img.shields.io/badge/Robotics-camera%20cue-7d73b7?style=for-the-badge" />
  </p>
</div>

<p align="center">
  <img src=".github/assets/readme-hero.svg" alt="OrangeBall Detection with OpenCV overview image" width="100%" />
</p>

## Overview

This is the classical computer-vision path for orange-ball detection. It favors fast live tuning over training a model.

The detector exposes color and shape parameters so field lighting changes can be handled quickly during robotics tests.

## Features

- Live camera detector with camera-index fallback logic.
- HSV/LAB and morphology tuning for lighting changes.
- Circularity, fill-ratio, and bounding-box filters to reject non-ball blobs.
- Experiment scripts for grass, color, grayscale, and GUI detector variants.
- Useful as a lightweight camera cue before heavier learned detection.

## How It Works

1. Open a camera feed with OpenCV.
2. Build a color mask for likely orange regions.
3. Clean the mask and extract contours.
4. Score candidate blobs by geometry.
5. Return/visualize the ball center for downstream robot logic.

## Quickstart

Run the project locally with the commands below.

```bash
git clone https://github.com/Ha22yX/OrangeBall-Detection-with-OpenCV.git
cd OrangeBall-Detection-with-OpenCV
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/orange_detector.py
```

Set `CAM_INDEX=1` or another index if the automatic camera scan picks the wrong device.

## Configuration

| Item | Purpose |
| --- | --- |
| `CAM_INDEX` | Force a specific `/dev/videoX` or camera index. |
| Thresholds | Tune color/morphology constants for the test environment. |
| Experiments | Use scripts under `experiments/` to compare alternate detectors. |

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Vision | OpenCV | Color segmentation and contour filtering. |
| Math | NumPy | Mask and coordinate operations. |
| Runtime | Camera feed | Live tuning and annotated output. |
| Robotics | Image-space center | Steering cue for downstream logic. |

## Project Layout

```text
src/orange_detector.py        recommended live detector
experiments/                   older detector experiments
requirements.txt               Python dependencies
.github/assets/                README overview asset
```

## Status

Classical CV experiment. For learned detection, use the YOLO Orange Ball Detection repo.

## Related Projects

- [Yolo-Orange-Ball-detection](https://github.com/Ha22yX/Yolo-Orange-Ball-detection) - YOLO training and inference path for the same object class.

## License

No project-wide open-source license has been declared yet.
