## OrangeBall-Detection-with-OpenCV

简单的 OpenCV 程序：打开摄像头，提取画面里的橙色区域，并检测“接近圆形”的目标（用于橙色高尔夫球）。已内置抗高光/阴影与 LAB 拾色增强。

### 安装
```bash
pip install opencv-python numpy
```

### 运行
```bash
python3 orange_detector.py
```

### 快速使用（推荐顺序）
1) 打开程序后，窗口会出现三个画面：Original / Mask / Result (In-Range) 以及一个 Controls 控制窗口。  
2) 在 Controls 中，默认参数已适配常见橙色球：
   - H center=7, H width=2, S min=180, V min=215
   - Morph=1, Light tol=2, Use LAB=1, Detect=1
3) 开启 Pick mode（或按 p），在 Original 窗口“左键点击”球面（选中等亮的区域）进行拾色；若用 Shift+左键也可拾色。  
4) 观察 Mask/Result 是否把球完整连成一块：
   - 如果球碎裂：增大 Light tol（2→3），或把 DE tol 从 25 调大到 30-40（Use LAB=1）。
   - 如果背景进来太多：适当减小 H width，或提高 S min / V min。  
5) 打开 Detect=1，会在 Result 里给接近圆的区域画框与分数。  
6) 退出：按 q 或 ESC。

### 控件说明（Controls）
- H center / H width：色相中心与半宽度（范围 = center ± width）
- S min / V min：最小饱和度 / 最小亮度（上限固定 255）
- Morph (0-5)：形态学降噪（0 关闭）
- Light tol (0-3)：对高光/阴影的容忍度（越大越容易连成整球）
- Use LAB (0/1) + DE tol (10-50)：基于 LAB ΔE 的拾色增强，越大越宽容
- Pick mode (0/1)：拾色模式（开后直接左键点击即可拾色；mac 触摸板建议开启）
- Detect (0/1)：开启/关闭圆形检测与标注

### 性能/调参小贴士
- 调阈值时可临时把 Detect 设为 0，只看 Mask/Result，不卡后再打开。
- 若仍卡顿，降低 Morph；程序会尝试将摄像头分辨率设置为 640×480。*** End Patch*** End Patch }```  }``` md code fence? The tool expects patch only. The above content ends with *** End Patch twice; ensure format correct. We'll fix.
