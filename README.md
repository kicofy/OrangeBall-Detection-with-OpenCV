## OrangeBall-Detection-with-OpenCV

### Overview
OpenCV-based webcam app that segments orange pixels and performs circle-likeness detection (via circularity). Includes a simplified control panel and a color picker to quickly tune thresholds.

### Install
```bash
pip install opencv-python numpy
```

### Run
```bash
python3 orange_detector.py
```

### Controls (window: "Controls")
- H center / H width: hue center and half-width (range = center ± width)
- S min / V min: minimum saturation / minimum value (upper bounds fixed at 255)
- Morph (0-5): morphology denoise strength (0 disables)
- Pick mode (0/1): enable to pick colors with a plain left click (recommended on mac trackpads)
- Detect (0/1): enable/disable circle-likeness detection and overlays

### Color Picking & Hotkeys
- Pick color: in the "Original" window, Shift + Left Click; or enable Pick mode and just Left Click
- If the clicked pixel is already inside the current range, the bounds shrink toward that color
- p: toggle Pick mode
- q or ESC: quit

### Tuning Tips
- For smoother tuning, set Detect to 0 (off) while you adjust thresholds; turn it back on afterward
- If performance is slow, reduce Morph; the app also attempts 640×480 capture to lower processing cost
