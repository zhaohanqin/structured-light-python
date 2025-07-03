# 改进版相位解包裹模块

本模块是结构光3D扫描系统中的相位解包裹核心组件，基于原始unwrap_phase.py进行了多线程支持改进。

## 主要特性

- 支持多线程环境执行，避免matplotlib显示冲突
- 设置Matplotlib后端为'Agg'，防止在非主线程中显示GUI
- 添加中文字体支持，优化可视化效果
- 可控制是否显示图像（show_plots参数）
- 实现多种相位解包裹算法

## 相位解包裹方法

### 1. 质量引导法（Quality-guided Unwrapping）

这是一种空间解包裹方法，通过相位质量图引导解包裹路径。算法从质量最高的点开始，按照质量降序进行广度优先搜索，计算相邻像素间的相位差异。这种方法适用于单频率相位图，对噪声和不连续区域有较好的鲁棒性。

### 2. 多频率法（Multi-frequency Unwrapping）

利用不同频率的包裹相位进行解包裹，从最低频率开始，逐步利用高频率相位提高分辨率。通过计算相邻频率之间的频率比和相位差异，实现从低频到高频的解包裹过程。多频率法能有效解决单频率相位图中的包裹歧义问题，提高解包裹的准确性。

### 3. 时序相位解包裹（Temporal Phase Unwrapping）

时序解包裹综合了多频率方法和质量加权策略，它不仅考虑不同频率的相位关系，还结合相位质量图进行加权融合。该算法计算高频相位的预测值与实际值之间的差异，确定相位偏移量，并根据质量图调整不同频率解包裹结果的权重。这种方法在处理复杂场景时能获得更平滑和准确的结果。

## 相位计算方法

模块支持多种相位计算算法：

- **四步相移算法**：通过四张相位相差90°的图像计算包裹相位，这是最常用的相位计算方法。
- **三步相移算法**：使用三张相位相差120°的图像计算包裹相位，图像数量更少但精度略低。
- **N步相移算法**：使用任意数量的均匀分布相移图像计算相位，采用最小二乘法提高精度。

## 多线程支持改进

与原始版本相比，本模块主要进行了以下改进：

1. **设置Matplotlib后端**：

   ```python
   matplotlib.use('Agg')  # 设置不使用GUI后端
   ```

2. **中文字体支持**：

   ```python
   chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'SimSun' in f.name or 'Microsoft YaHei' in f.name]
   if chinese_fonts:
       plt.rcParams['font.family'] = chinese_fonts[0]
   ```

3. **可视化函数改进**：

   ```python
   def visualize_wrapped_phase(..., show_plots: bool = True):
       # ...
       if show_plots:
           plt.show()
       else:
           plt.close()
   ```

## 使用方法

### 安装依赖

```bash
pip install numpy opencv-python matplotlib
```

### 基本用法

```python
from unwrap_phase_modified import process_four_step_images

# 处理四步相移图像（不在线程中显示图形）
image_paths = ["image1.png", "image2.png", "image3.png", "image4.png"]
unwrapped_phase = process_four_step_images(
    image_paths, 
    output_dir="results",
    show_plots=False  # 多线程环境中设为False
)
```

### 命令行使用

```bash
python unwrap_phase_modified.py --images image1.png image2.png image3.png image4.png --output results --method quality_guided --no-display
```

参数说明：

- `--images`: 四张相移图像的路径，按照相位偏移0°, 90°, 180°, 270°的顺序
- `--output`: 输出结果的目录 (默认为"output")
- `--method`: 解包裹方法，可选值为"quality_guided", "temporal", "multi_freq"
- `--no-display`: 不显示图像，只保存结果（多线程环境中使用）

## 输出结果

程序会在指定的输出目录中生成以下文件：

- `wrapped_phase.png`: 包裹相位可视化图像
- `unwrapped_phase.png`: 解包裹相位可视化图像
- `wrapped_phase.npy`: 包裹相位数据 (NumPy数组)
- `unwrapped_phase.npy`: 解包裹相位数据 (NumPy数组)
- `quality_map.npy`: 相位质量图数据 (NumPy数组)

## 注意事项

- 在多线程环境中使用时，始终将`show_plots`参数设置为`False`
- 使用`plt.close()`代替`plt.show()`以避免线程阻塞
- 对于需要显示图形的场景，建议在主线程中调用可视化函数
- 当使用"temporal"或"multi_freq"方法处理单频率数据时，程序会自动使用"quality_guided"方法代替
