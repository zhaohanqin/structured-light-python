import numpy as np
import cv2
import matplotlib
# 设置Matplotlib不使用GUI后端，避免线程问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import os
from typing import List, Dict, Tuple, Optional
from enum import Enum
import glob

# 尝试设置中文字体
try:
    # 检查系统是否有支持中文的字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'SimSun' in f.name or 'Microsoft YaHei' in f.name]
    if chinese_fonts:
        plt.rcParams['font.family'] = chinese_fonts[0]
    else:
        # 如果没有中文字体，使用默认字体，但不显示中文标题
        plt.rcParams['font.family'] = 'sans-serif'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# 解决在中文环境下保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


class PhaseShiftingAlgorithm(Enum):
    """相移算法类型枚举"""
    three_step = 0      # 三步相移
    four_step = 1       # 四步相移
    n_step = 2          # N步相移


def compute_phasor_and_phase(images: List[np.ndarray], algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据相移图像计算包裹相位和对应的复数形式（相量）
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")
    
    float_images = [img.astype(np.float32) for img in images]
    
    # 统一使用 sum(I*sin) 和 sum(I*cos) 的形式
    sin_sum = 0
    cos_sum = 0
    n = len(float_images)

    # 假设相移是 +k*delta 的形式
    if algorithm == PhaseShiftingAlgorithm.three_step:
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
        # sin_sum = I1*sin(0) + I2*sin(2pi/3) + I3*sin(4pi/3) = I2*(sqrt(3)/2) - I3*(sqrt(3)/2)
        sin_sum = (np.sqrt(3)/2) * (I2 - I3)
        # cos_sum = I1*cos(0) + I2*cos(2pi/3) + I3*cos(4pi/3) = I1 - 0.5*I2 - 0.5*I3
        cos_sum = I1 - 0.5 * (I2 + I3)
    elif algorithm == PhaseShiftingAlgorithm.four_step:
        I1, I2, I3, I4 = float_images[0], float_images[1], float_images[2], float_images[3]
        # sin_sum = I1*sin(0)+I2*sin(pi/2)+I3*sin(pi)+I4*sin(3pi/2) = I2 - I4
        sin_sum = I2 - I4
        # cos_sum = I1*cos(0)+I2*cos(pi/2)+I3*cos(pi)+I4*cos(3pi/2) = I1 - I3
        cos_sum = I1 - I3
    elif algorithm == PhaseShiftingAlgorithm.n_step:
        delta = 2 * np.pi / n
        sin_sum = sum(float_images[i] * np.sin(i * delta) for i in range(n))
        cos_sum = sum(float_images[i] * np.cos(i * delta) for i in range(n))
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    # 【根本性修正】统一符号约定。
    # 根据标准N步相移算法的离散傅里叶变换推导，
    # 对于 I(phi + k*delta) 的信号，其相位 phi = atan2(-sum(I*sin), sum(I*cos))
    wrapped_phase = np.arctan2(-sin_sum, cos_sum)
    # 对应的相量为 sum(I*cos) - j*sum(I*sin)
    phasor = cos_sum - 1j * sin_sum
    return wrapped_phase, phasor


def compute_phase_quality(images: List[np.ndarray]) -> np.ndarray:
    """
    计算相位质量图，用于评估相位的可靠性
    
    参数:
        images: 相移图像列表
    
    返回:
        quality_map: 相位质量图，值越大表示质量越高
    """
    # 计算强度调制
    n = len(images)
    float_images = [img.astype(np.float32) for img in images]
    
    # 计算平均强度
    avg_intensity = sum(float_images) / n
    
    # 计算相移步长
    delta = 2 * np.pi / n
    
    # 计算正弦和余弦分量
    sin_sum = 0
    cos_sum = 0
    
    for i in range(n):
        phase_shift = i * delta
        sin_sum += float_images[i] * np.sin(phase_shift)
        cos_sum += float_images[i] * np.cos(phase_shift)
    
    # 计算调制幅度
    modulation = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    
    # 计算质量图 (调制幅度除以平均强度，避免除零)
    eps = 1e-10  # 小值防止除零
    quality_map = modulation / (avg_intensity + eps)
    
    return quality_map


def visualize_wrapped_phase(wrapped_phase: np.ndarray, quality_map: Optional[np.ndarray] = None, 
                           title: str = "Wrapped Phase", save_path: Optional[str] = None,
                           show_plots: bool = True):
    """
    可视化包裹相位图
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图 (可选)
        title: 图像标题
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
    """
    plt.figure(figsize=(12, 9))
    
    # 如果有质量图，创建一个2x1的子图
    if quality_map is not None:
        plt.subplot(2, 1, 1)
    
    # 显示包裹相位
    phase_img = plt.imshow(wrapped_phase, cmap='jet')
    plt.colorbar(phase_img, label='Phase (rad)')
    plt.title(title)
    
    # 如果有质量图，在第二个子图中显示
    if quality_map is not None:
        plt.subplot(2, 1, 2)
        quality_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(quality_img, label='Quality (Modulation/Mean)')
        plt.title("Phase Quality Map")
    
    plt.tight_layout()
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    else:
        plt.close()


def quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    使用基于优先队列（堆）的洪水填充算法，进行稳健的质量图引导空间相位解包裹。
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    
    # 创建输出的解包裹相位图
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 创建质量排序索引
    quality_flat = quality_map.flatten()
    indices = np.argsort(-quality_flat)  # 按质量降序排序的索引
    
    # 找到质量最高的点作为种子点
    seed_idx = indices[0]
    seed_y, seed_x = np.unravel_index(seed_idx, (height, width))
    
    # 标记种子点为已访问
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
        
    # 创建队列，从种子点开始
    queue = [(seed_y, seed_x)]
        
    # 定义邻域方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    # 基于质量引导的广度优先搜索
    while queue:
        y, x = queue.pop(0)
            
        # 对当前点的四个邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                # 计算相位差异
                phase_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 调整相位差异到 [-pi, pi] 范围
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算解包裹相位
                unwrapped_phase[ny, nx] = unwrapped_phase[y, x] + phase_diff
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到队列
                queue.append((ny, nx))
                
    return unwrapped_phase


def visualize_unwrapped_phase(unwrapped_phase: np.ndarray, title: str = "Unwrapped Phase", 
                             save_path: Optional[str] = None, show_plots: bool = True) -> None:
    """
    可视化解包裹相位图

    参数:
        unwrapped_phase: 解包裹相位图
        title: 图像标题
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
    """
    plt.figure(figsize=(10, 8))
    
    # 显示解包裹相位
    img = plt.imshow(unwrapped_phase, cmap='jet')
    plt.colorbar(img, label='Phase (rad)')
    plt.title(title)
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    else:
        plt.close()


def save_unwrapped_phase_raw(unwrapped_phase: np.ndarray, save_path: str):
    """
    将解包裹后的相位保存为纯净的彩色图像，不含任何坐标轴或文字。
    
    参数:
        unwrapped_phase: 解包裹相位图
        save_path: 保存路径
    """
    if unwrapped_phase is None:
        print("没有可保存的解包裹相位数据")
        return

    # 归一化相位数据到0-255范围，并转换为8位图像
    img_normalized = cv2.normalize(unwrapped_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用伪彩色映射
    img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
    
    # 保存图像
    cv2.imwrite(save_path, img_color)
    print(f"纯净的解包裹相位图已保存至: {save_path}")


def generate_combined_phase(h_unwrapped: np.ndarray, v_unwrapped: np.ndarray, 
                           title: str = "水平和垂直方向相位组合图", 
                           save_path: Optional[str] = None, 
                           show_plots: bool = True) -> np.ndarray:
    """
    生成水平和垂直方向相位组合图
    
    参数:
        h_unwrapped: 水平方向解包裹相位
        v_unwrapped: 垂直方向解包裹相位
        title: 图像标题
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
    
    返回:
        combined_rgb: 组合的RGB图像
    """
    if h_unwrapped is None or v_unwrapped is None:
        print("需要水平和垂直方向的相位数据才能生成组合图")
        return None
        
    # 确保两个相位图具有相同的大小
    if h_unwrapped.shape != v_unwrapped.shape:
        print("水平和垂直方向相位图尺寸不一致，无法生成组合图")
        return None
        
    height, width = h_unwrapped.shape
    
    # 归一化两个相位图
    h_norm = (h_unwrapped - np.min(h_unwrapped)) / (np.max(h_unwrapped) - np.min(h_unwrapped))
    v_norm = (v_unwrapped - np.min(v_unwrapped)) / (np.max(v_unwrapped) - np.min(v_unwrapped))
    
    # 组合两个方向的相位图得到伪彩色图像
    combined_rgb = np.zeros((height, width, 3), dtype=np.float32)
    combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
    combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
    combined_rgb[:,:,2] = (h_norm + v_norm) / 2  # 蓝色通道为两者平均
    
    # 创建并显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_rgb)
    plt.title(title)
    plt.colorbar(label='归一化相位值')
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return combined_rgb


def _quantize_for_pairing(values: np.ndarray, precision: float) -> np.ndarray:
    """
    将连续相位按给定精度量化为非负整数，便于使用配对函数。
    """
    # 平移到非负并量化
    min_val = float(np.min(values))
    if min_val < 0:
        values = values - min_val
    quantized = np.rint(values / precision).astype(np.int64)
    # 保证非负
    quantized[quantized < 0] = 0
    return quantized


def compute_unique_combined_phase(
    h_unwrapped: np.ndarray,
    v_unwrapped: np.ndarray,
    precision: float = 1e-3,
) -> np.ndarray:
    """
    将 (H,V) 两个连续相位对映射为单一且在该精度下唯一的标量值。

    方法：先把两个相位用 precision 量化为整数，再用 Cantor 配对函数：
        pi(a,b) = (a+b)(a+b+1)/2 + b

    返回：int64 的“组合相位ID”矩阵（数值越大不代表物理量，仅用于唯一标识）。
    """
    if h_unwrapped is None or v_unwrapped is None:
        raise ValueError("需要提供水平与垂直相位数据")
    if h_unwrapped.shape != v_unwrapped.shape:
        raise ValueError("水平与垂直相位尺寸必须一致")

    a = _quantize_for_pairing(h_unwrapped.astype(np.float64), precision)
    b = _quantize_for_pairing(v_unwrapped.astype(np.float64), precision)

    s = a + b
    unique_id = (s * (s + 1)) // 2 + b
    return unique_id.astype(np.int64)


def combine_pair_scalar(h_value: float, v_value: float, precision: float = 1e-3) -> int:
    """
    对单个像素 (h,v) 计算在给定精度下唯一的组合相位ID（Cantor配对）。
    """
    # 量化（移到非负）
    if h_value < 0:
        v_value = v_value - h_value
        h_value = 0.0
    if v_value < 0:
        h_value = h_value - v_value
        v_value = 0.0
    a = int(round(h_value / precision))
    b = int(round(v_value / precision))
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    s = a + b
    return (s * (s + 1)) // 2 + b


def visualize_3d_surface(unwrapped_phase: np.ndarray, 
                         title: str = "解包裹相位 3D 表面", 
                         cmap: str = 'viridis',
                         save_path: Optional[str] = None, 
                         show_plots: bool = True) -> None:
    """
    将解包裹相位可视化为3D表面
    
    参数:
        unwrapped_phase: 解包裹相位图
        title: 图像标题
        cmap: 颜色映射方案
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
    """
    if unwrapped_phase is None:
        print("没有可显示的解包裹相位数据")
        return
    
    # 获取图像尺寸
    height, width = unwrapped_phase.shape
    
    # 创建坐标网格
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # 创建3D图形
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    # 绘制3D表面
    surf = ax.plot_surface(xx, yy, unwrapped_phase, cmap=cmap, edgecolor='none')
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('相位值')
    
    # 添加颜色条
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='相位值')
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    else:
        plt.close()


def visualize_combined_3d_surface(h_unwrapped: np.ndarray, 
                                  v_unwrapped: np.ndarray, 
                                  title: str = "Combined 3D Surface",
                                  save_path: Optional[str] = None, 
                                  show_plots: bool = True) -> None:
    """
    将组合的水平和垂直解包裹相位可视化为3D表面。
    
    参数:
        h_unwrapped: 水平方向解包裹相位
        v_unwrapped: 垂直方向解包裹相位
        title: 图像标题
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
    """
    if h_unwrapped is None or v_unwrapped is None:
        print("需要水平和垂直相位数据才能显示组合3D表面")
        return

    # 确保两个相位图具有相同的大小
    if h_unwrapped.shape != v_unwrapped.shape:
        print("水平和垂直方向相位图尺寸不一致，无法生成组合图")
        return

    # 将两个方向的相位组合成一个标量场 (例如，使用幅值)
    combined_phase = np.sqrt(h_unwrapped**2 + v_unwrapped**2)
    
    # 使用现有的3D可视化函数进行绘制
    visualize_3d_surface(
        combined_phase,
        title=title,
        save_path=save_path,
        show_plots=show_plots
    )


def process_single_frequency_images(image_paths: List[str], output_dir: str, method: str, show_plots: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """
    处理单频条纹图像，执行完整的解包裹流程
    
    返回一个包含解包裹相位和包裹相位的字典，或者在失败时返回 None
    """
    if not image_paths:
        print("错误: 未提供图像文件路径。")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    if any(img is None for img in images):
        print("错误: 一个或多个图像文件无法加载。")
        return None
    
    # 确定相移算法
    num_images = len(images)
    if num_images == 3:
        algorithm = PhaseShiftingAlgorithm.three_step
    elif num_images == 4:
        algorithm = PhaseShiftingAlgorithm.four_step
    else:
        algorithm = PhaseShiftingAlgorithm.n_step
    
    print(f"使用 {num_images}-步 相移算法。")
    
    # 计算包裹相位
    wrapped_phase, _ = compute_phasor_and_phase(images, algorithm=algorithm)
    
    # 1. 始终计算相位质量图
    quality_map = compute_phase_quality(images)
    
    # 2. 保存单独的包裹相位图
    output_path_wrapped = os.path.join(output_dir, "wrapped_phase.png")
    visualize_wrapped_phase(wrapped_phase,
                            quality_map=None, #不显示质量
                            title="包裹相位图",
                            save_path=output_path_wrapped,
                            show_plots=show_plots)

    # 3. 保存包含质量图的包裹相位图
    output_path_quality = os.path.join(output_dir, "wrapped_phase_and_quality.png")
    visualize_wrapped_phase(wrapped_phase,
                            quality_map=quality_map,
                            title="包裹相位和质量图",
                            save_path=output_path_quality,
                            show_plots=show_plots)

    # 解包裹
    print(f"使用解包裹方法: {method}")
    if method == "quality_guided":
        # 此处已经计算过 quality_map，直接使用即可
        unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map)
    elif method == "skimage":
        # 使用skimage进行解包裹
        from skimage.restoration import unwrap_phase as skimage_unwrap
        unwrapped_phase = skimage_unwrap(wrapped_phase)
    else:
        raise ValueError(f"未知的解包裹方法: {method}")

    # 后处理：平移相位值，使最小值为0（所有值为非负数）
    min_phase = np.min(unwrapped_phase)
    if min_phase < 0:
        print(f"平移相位值：{min_phase:.2f} -> 0")
        unwrapped_phase = unwrapped_phase - min_phase

    # 可视化解包裹相位
    output_path = os.path.join(output_dir, "unwrapped_phase.png")
    visualize_unwrapped_phase(unwrapped_phase,
                              title=f"解包裹相位 ({method})",
                              save_path=output_path,
                              show_plots=show_plots)

    # 另外保存一幅不带文字和坐标轴的纯净结果图
    clean_output_path = os.path.join(output_dir, "unwrapped_phase_clean.png")
    save_unwrapped_phase_raw(unwrapped_phase, clean_output_path)

    # 保存解包裹后的npy数据
    npy_output_path = os.path.join(output_dir, "unwrapped_phase.npy")
    np.save(npy_output_path, unwrapped_phase)
    print(f"解包裹相位数据已保存至: {npy_output_path}")
    
    # 可视化3D表面
    output_3d_path = os.path.join(output_dir, "unwrapped_phase_3d.png")
    visualize_3d_surface(unwrapped_phase,
                         title=f"解包裹相位 3D 表面 ({method})",
                         save_path=output_3d_path,
                         show_plots=show_plots)

    return {
        "unwrapped_phase": unwrapped_phase,
        "wrapped_phase": wrapped_phase
    }


def main():
    """
    主函数，用于命令行测试
    """
    # 示例：处理一组单频图像
    # 准备你的图像路径
    # image_folder = "path/to/your/single_frequency_images"
    # image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    # if not image_paths:
    #     print(f"在 '{image_folder}' 中未找到图像。请更新路径。")
    #     return
        
    # # 设置输出目录
    # output_dir = "output/single_freq_test"
    
    # # 调用处理函数
    # process_single_frequency_images(
    #     image_paths=image_paths,
    #     output_dir=output_dir,
    #     method="quality_guided",
    #     show_plots=False  # 在非GUI脚本中设为False
    # )
    
    print("单频解包裹模块。请通过UI或其他脚本调用 'process_single_frequency_images' 函数。")


if __name__ == '__main__':
    main() 