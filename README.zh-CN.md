# OrangeBall Detection with OpenCV

[English](README.md)

OrangeBall Detection with OpenCV 是为 Wardlaw Hartridge School Robotics Club 机器人制作的 OpenCV 橙色高尔夫球检测项目。它估计球在摄像头画面中的位置，供机器人程序作为视觉参考。

检测流程基于 OpenCV 颜色分割和简单的近似圆形判断。主程序提供实时控件，可调整 HSV 阈值、LAB 拾色、形态学降噪、光照容忍度和检测开关，方便在不同光照下现场调参。

## 项目目标

- 从实时摄像头画面中检测橙色高尔夫球。
- 输出球在图像中的位置，为机器人移动决策提供视觉信号。
- 通过 GUI 实时调参，避免频繁修改代码常量。
- 使用 HSV 和可选 LAB 色差增强，应对橙色球表面的高光和阴影。
- 通过圆度、填充率和外接框比例过滤非球形橙色区域。
- 保留旧版检测实验脚本，方便之后对比和继续优化。

## 项目结构

```text
src/
  orange_detector.py          推荐使用的实时检测主程序，带 GUI 控件。

experiments/
  golf_ball_detector_gui.py   带截图标注功能的高级 GUI 实验。
  golf_ball_detector_grass.py 草地背景下的橙色球检测实验。
  golf_ball_detector_color.py 以颜色分割为主的检测实验。
  circle_detector_bw.py       黑白圆形检测实验。

requirements.txt             Python 依赖。
README.md                    英文说明。
README.zh-CN.md              中文说明。
```

## 主检测程序

运行：

```bash
python src/orange_detector.py
```

程序会打开摄像头，并显示：

- `Original`：原始摄像头画面。
- `Mask`：当前橙色阈值生成的二值掩码。
- `Result`：带检测框和分数的结果画面。
- `Controls`：用于调参的滑块窗口。

脚本内置的推荐起始参数：

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

## 安装

推荐使用 Python 3.10 或更新版本。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 摄像头选择

程序默认会自动尝试打开摄像头。也可以通过环境变量指定摄像头编号：

```bash
# Windows PowerShell
$env:CAM_INDEX="0"
python src/orange_detector.py

# macOS/Linux
CAM_INDEX=0 python src/orange_detector.py
```

部分实验脚本也支持 Linux `/dev/video*` 摄像头自动发现。

## 控件说明

- `H center` / `H width`：OpenCV HSV 空间中的橙色色相中心和半宽度。
- `S min` / `V min`：最小饱和度和最小亮度。
- `Morph`：形态学降噪强度。
- `Light tol`：对高光和阴影的容忍度。
- `Use LAB` / `DE tol`：拾色后启用 LAB 色差掩码。
- `Pick mode`：开启后，在 `Original` 窗口左键点击球面即可拾色。
- `Detect`：开启或关闭近似圆形区域检测。

快捷键：

- `p`：切换拾色模式。
- `q` 或 `Esc`：退出程序。

## 调参流程

1. 运行 `python src/orange_detector.py`。
2. 把橙色高尔夫球放到机器人摄像头画面中。
3. 如果掩码没有覆盖完整球体，开启 `Pick mode` 并点击球上光照较稳定的区域。
4. 如果球在掩码中断裂，可以增大 `Light tol`、`DE tol` 或 `Morph`。
5. 如果背景进入掩码，可以减小 `H width`，或提高 `S min` / `V min`。
6. 调阈值时可以先把 `Detect = 0`，只看掩码；调好后再开启形状检测。

## 机器人集成说明

当前项目提供视觉检测和图像坐标中的球位置。机器人控制器可以把检测到的球心作为转向和移动信号：

```text
摄像头画面 -> 橙色掩码 -> 近似圆形检测 -> 球心坐标 (x, y) -> 机器人对准/移动逻辑
```

一种常见控制策略：

- 如果没有检测到球，机器人旋转或扫描。
- 如果球心在画面中心左侧/右侧，机器人向对应方向转向。
- 如果球心接近画面中心，机器人向前移动。
- 可以用检测半径或外接框大小粗略估计距离。

## License

当前仓库暂未包含 license 文件。
