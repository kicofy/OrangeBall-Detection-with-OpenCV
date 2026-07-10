<div align="center">
  <h1>OrangeBall Detection with OpenCV</h1>
  <p>A live-tunable OpenCV detector for orange golf balls in robotics camera feeds.</p>

  <p>
    <a href="README.zh-CN.md">Chinese</a>
    &middot;
    <a href="#quickstart">Quickstart</a>
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

## Why This Exists

Robotics vision often fails because lighting changes faster than constants can be edited. This detector exposes HSV/LAB and shape controls live so a team can tune the ball mask in the field.

## Quickstart

```bash
git clone https://github.com/Ha22yX/OrangeBall-Detection-with-OpenCV.git
cd OrangeBall-Detection-with-OpenCV
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/orange_detector.py
```

Set `CAM_INDEX` if your camera is not index 0.

## Features

- Live camera detection for orange golf balls.
- HSV controls, optional LAB color picking, morphology, and light tolerance sliders.
- Shape checks for circularity, fill ratio, and bounding boxes.
- Older experiments preserved for comparison and tuning.

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Vision | OpenCV | Color segmentation and contour filtering. |
| Math | NumPy | Mask and image coordinate operations. |
| Runtime | Camera feed | Live tuning and annotated output. |
| Robot use | Image-space center | Steering cue for downstream robot logic. |


## Project Notes

This is a classical computer-vision detector. For learned detection experiments, see [Yolo-Orange-Ball-detection](https://github.com/Ha22yX/Yolo-Orange-Ball-detection).
