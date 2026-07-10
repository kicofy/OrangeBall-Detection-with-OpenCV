<div align="center">
  <h1>OrangeBall Detection with OpenCV</h1>
  <p>一个可实时调参的 OpenCV 橙色高尔夫球检测器，面向机器人摄像头画面。</p>

  <p>
    <a href="README.md">English</a>
    &middot;
    <a href="#快速开始">快速开始</a>
    &middot;
    <a href="#技术栈">技术栈</a>
  </p>

  <p>
    <img alt="Python: OpenCV" src="https://img.shields.io/badge/Python-OpenCV-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Vision: color detection" src="https://img.shields.io/badge/Vision-color%20detection-287866?style=for-the-badge" />
    <img alt="Robotics: camera cue" src="https://img.shields.io/badge/Robotics-camera%20cue-7d73b7?style=for-the-badge" />
  </p>
</div>

<p align="center">
  <img src=".github/assets/readme-hero.svg" alt="OrangeBall Detection with OpenCV 项目概览图" width="100%" />
</p>

## 项目价值

机器人视觉常常因为光照变化而失效。本检测器把 HSV/LAB 和形状参数做成实时控制，方便在现场调出稳定的橙色球掩膜。

## 工作流

- 打开实时摄像头画面。
- 调节 HSV 范围、饱和度、亮度、形态学和可选 LAB 色差。
- 用圆度、填充率和边界框形状验证候选区域。
- 把检测到的球心作为机器人转向参考。
- 保留旧实验脚本用于对比。

## 核心功能

- 面向橙色高尔夫球的实时摄像头检测器。
- 用于阈值和光照调节的交互控制。
- 通过形状筛选排除非球体橙色区域。
- 保留草地、颜色、圆形检测等实验变体脚本。

## 快速开始

```bash
git clone https://github.com/Ha22yX/OrangeBall-Detection-with-OpenCV.git
cd OrangeBall-Detection-with-OpenCV
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/orange_detector.py
```

如果摄像头不是 0 号设备，请设置 `CAM_INDEX`。

## 技术栈

| 层级 | 技术 | 作用 |
| --- | --- | --- |
| 视觉 | OpenCV | 颜色分割和轮廓筛选。 |
| 数学 | NumPy | 掩膜和图像坐标运算。 |
| 运行 | Camera feed | 实时调参和标注输出。 |
| 机器人 | Image-space center | 给下游机器人逻辑提供转向参考。 |

## 项目结构

```text
src/orange_detector.py        recommended live detector
experiments/                   older detector experiments
requirements.txt               Python dependencies
README.zh-CN.md                Chinese documentation
```

## 项目说明

这是传统计算机视觉检测器。学习式检测实验可参考 YOLO Orange Ball Detection。
