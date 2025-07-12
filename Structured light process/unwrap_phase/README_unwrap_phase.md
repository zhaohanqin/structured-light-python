# 改进版相位解包裹模块

本模块是结构光3D扫描系统中的相位解包裹核心组件。它经过专门设计，支持在多线程环境中安全运行，避免了在后端服务或图形界面中调用时出现的常见问题。

## 主要特性

- **多线程安全**：通过设置Matplotlib后端为'Agg'，防止在非主线程中创建GUI窗口，确保在线程中稳定运行。
- **灵活的相移算法**：根据输入图像的数量自动选择合适的相移算法（三步、四步或N步）。
- **两种解包裹方法**：提供高质量的自定义引导算法和标准的Scikit-image算法。
- **丰富的结果可视化**：不仅生成2D相位图，还包括相位质量图、纯净结果图以及3D表面图。
- **可控的图像显示**：通过`show_plots`参数控制是否显示图像，便于在服务器或自动化脚本中运行。

## 相位解包裹方法

### 1. 质量引导法（Quality-guided Unwrapping）

一种稳健的空间解包裹方法。它首先计算一幅相位质量图，然后从质量最高的点开始，通过广度优先搜索的方式，沿着高质量路径逐步展开相位。这种方法对噪声和不连续区域有很好的鲁棒性。

### 2. Scikit-image 解包裹法

利用 `skimage.restoration.unwrap_phase` 函数进行解包裹。这是一种基于图中像素路径上积分的算法，也是一种快速且广泛使用的标准解包裹方法。

## 相位计算方法

模块支持多种相移算法，并根据输入图像数量自动选择：

- **N步相移算法**：通用的离散傅里叶变换法，适用于任意N(N>=3)张均匀相移的图像。
- **四步相移算法**：N步算法在N=4时的优化特例，计算速度快。
- **三步相移算法**：N步算法在N=3时的优化特例。

## 多线程支持改进

1. **设置Matplotlib后端**：

   ```python
   matplotlib.use('Agg')  # 设置不使用GUI后端
   ```

2. **可控的绘图显示**：
   所有可视化函数都包含 `show_plots: bool = True` 参数。当在线程中调用时，应将其设置为 `False`，此时图像将直接保存到文件而不尝试在屏幕上显示。

   ```python
   def visualize_wrapped_phase(..., show_plots: bool = True):
       # ...
       if show_plots:
           plt.show()
       else:
           plt.close()
   ```

## 使用方法

该模块主要作为库被其他脚本（如UI程序或自动化流程）调用。核心函数是 `process_single_frequency_images`。

### 安装依赖

```bash
pip install numpy opencv-python matplotlib scikit-image
```

### 基本用法

```python
import glob
import os
from unwrap_phase import process_single_frequency_images

# 1. 准备图像路径
image_folder = "path/to/your/single_frequency_images"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

if not image_paths:
    print(f"在 '{image_folder}' 中未找到图像。请更新路径。")
else:
    # 2. 设置输出目录
    output_dir = "output/single_freq_test"
    
    # 3. 调用处理函数
    # 在非GUI脚本或多线程环境中，将 show_plots 设为 False
    results = process_single_frequency_images(
        image_paths=image_paths,
        output_dir=output_dir,
        method="quality_guided",  # 可选 "quality_guided" 或 "skimage"
        show_plots=False
    )
    
    if results:
        print("处理完成。解包裹后的相位数据可通过 'results[\"unwrapped_phase\"]' 访问。")

```

### 命令行使用

本模块当前没有提供直接的命令行接口。请通过Python脚本调用 `process_single_frequency_images` 函数来使用。

## 输出结果

成功执行后，程序会在指定的输出目录中生成以下文件：

- `wrapped_phase.png`: 包裹相位图（带坐标轴）。
- `wrapped_phase_and_quality.png`: 包裹相位与质量图的组合图。
- `unwrapped_phase.png`: 解包裹相位图（带坐标轴）。
- `unwrapped_phase_clean.png`: 纯净的解包裹相位图（不含坐标轴和文字）。
- `unwrapped_phase_3d.png`: 解包裹相位的3D表面可视化图。
- `unwrapped_phase.npy`: 解包裹相位数据 (NumPy数组)。

## 注意事项

- 在多线程环境中使用时，始终将 `show_plots` 参数设置为 `False`。
- 输入的所有图像必须具有相同的尺寸。
- 该模块专注于处理单频条纹图案，不直接支持多频或时序解包裹。
