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


class PhaseShiftingAlgorithm(Enum):
    """相移算法类型枚举"""
    three_step = 0      # 三步相移
    four_step = 1       # 四步相移
    n_step = 2          # N步相移


def compute_wrapped_phase(images: List[np.ndarray], algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> np.ndarray:
    """
    根据相移图像计算包裹相位
    
    参数:
        images: 相移图像列表
        algorithm: 使用的相移算法类型
    
    返回:
        wrapped_phase: 包裹相位图 (-pi到pi范围)
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")
    
    # 将图像转换为浮点数类型
    float_images = [img.astype(np.float32) for img in images]
    
    # 根据不同算法计算相位
    if algorithm == PhaseShiftingAlgorithm.three_step:
        # 三步相移算法
        # 假设三张图像的相移分别为0°, 120°, 240°
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
        
        # 计算相位
        numerator = np.sqrt(3) * (I2 - I3)
        denominator = 2 * I1 - I2 - I3
        wrapped_phase = np.arctan2(numerator, denominator)
        
    elif algorithm == PhaseShiftingAlgorithm.four_step:
        # 四步相移算法
        # 假设四张图像的相移分别为0°, 90°, 180°, 270°
        I1, I2, I3, I4 = float_images[0], float_images[1], float_images[2], float_images[3]
        
        # 计算相位
        numerator = I4 - I2
        denominator = I1 - I3
        wrapped_phase = np.arctan2(numerator, denominator)
        
    elif algorithm == PhaseShiftingAlgorithm.n_step:
        # N步相移算法 (最小二乘法)
        n = len(float_images)
        
        # 计算相移步长
        delta = 2 * np.pi / n
        
        # 计算分子和分母
        numerator = 0
        denominator = 0
        
        for i in range(n):
            phase_shift = i * delta
            numerator += float_images[i] * np.sin(phase_shift)
            denominator += float_images[i] * np.cos(phase_shift)
            
        wrapped_phase = np.arctan2(numerator, denominator)
    
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    return wrapped_phase


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


def multi_frequency_unwrap(wrapped_phases: Dict[int, np.ndarray], frequencies: List[int]) -> np.ndarray:
    """
    使用多频率解包裹方法解包裹相位
    
    参数:
        wrapped_phases: 不同频率的包裹相位字典，键为频率，值为包裹相位图
        frequencies: 频率列表，与wrapped_phases的键对应
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    # 确保频率按照从低到高排序
    sorted_freqs = sorted(frequencies)
    
    # 从最低频率开始
    lowest_freq = sorted_freqs[0]
    unwrapped_phase = wrapped_phases[lowest_freq].copy()
    
    # 对更高频率进行解包裹
    for i in range(1, len(sorted_freqs)):
        current_freq = sorted_freqs[i]
        
        # 当前频率的包裹相位
        current_wrapped = wrapped_phases[current_freq]
        
        # 上一频率的解包裹相位 (缩放到当前频率)
        freq_ratio = current_freq / sorted_freqs[i-1]
        scaled_prev_unwrapped = unwrapped_phase * freq_ratio
        
        # 计算相位差异
        phase_diff = current_wrapped - (scaled_prev_unwrapped % (2 * np.pi))
        
        # 调整相位差异到 [-pi, pi] 范围
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # 更新当前频率的解包裹相位
        unwrapped_phase = scaled_prev_unwrapped + phase_diff
    
    return unwrapped_phase


def temporal_phase_unwrap(wrapped_phases: Dict[int, np.ndarray], 
                         frequencies: List[int], 
                         quality_maps: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    """
    使用时序相位解包裹方法，结合质量图加权
    
    参数:
        wrapped_phases: 不同频率的包裹相位字典
        frequencies: 频率列表
        quality_maps: 不同频率的相位质量图字典 (可选)
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    # 确保频率按照从低到高排序
    frequency_indices = np.argsort(frequencies)
    sorted_freqs = [frequencies[i] for i in frequency_indices]
    
    # 初始化解包裹相位 (使用最低频率的包裹相位)
    lowest_freq = sorted_freqs[0]
    unwrapped_phase = wrapped_phases[lowest_freq].copy()
    
    # 如果有质量图，获取最低频率的质量图
    current_quality = None
    if quality_maps is not None and lowest_freq in quality_maps:
        current_quality = quality_maps[lowest_freq].copy()
    
    # 对每个更高频率进行解包裹
    for i in range(1, len(sorted_freqs)):
        current_freq = sorted_freqs[i]
        current_wrapped = wrapped_phases[current_freq]
        
        # 计算频率比例
        freq_ratio = current_freq / sorted_freqs[i-1]
        
        # 预测高频的解包裹相位
        predicted_phase = unwrapped_phase * freq_ratio
        
        # 计算包裹相位和预测相位之间的差异
        wrapped_predicted = np.angle(np.exp(1j * predicted_phase))
        phase_error = np.angle(np.exp(1j * (current_wrapped - wrapped_predicted)))
        
        # 计算相位偏移量 (整数倍的2pi)
        phase_offset = np.round((predicted_phase - current_wrapped - phase_error) / (2 * np.pi))
        
        # 解包裹当前频率的相位
        current_unwrapped = current_wrapped + 2 * np.pi * phase_offset
        
        # 根据质量图合并解包裹结果
        if quality_maps is not None and current_freq in quality_maps:
            # 当前频率的质量图
            new_quality = quality_maps[current_freq]
            
            # 计算加权因子
            if current_quality is not None:
                # 使用质量图加权合并
                weight = new_quality / (new_quality + current_quality + 1e-10)
                unwrapped_phase = (1 - weight) * unwrapped_phase + weight * current_unwrapped
                current_quality = np.maximum(current_quality, new_quality)
            else:
                unwrapped_phase = current_unwrapped
                current_quality = new_quality
        else:
            # 没有质量图，直接更新为当前频率的解包裹相位
            unwrapped_phase = current_unwrapped
    
    return unwrapped_phase


def quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    使用质量图引导的空间相位解包裹
    
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


def process_four_step_images(image_paths: List[str], output_dir: str = "output", 
                           method: str = "quality_guided", show_plots: bool = True):
    """
    处理四步相移图像并生成解包裹相位
    
    参数:
        image_paths: 四张相移图像的路径列表
        output_dir: 输出目录
        method: 解包裹方法 ('quality_guided', 'temporal', 'multi_freq')
        show_plots: 是否显示图形 (在线程中应设为False)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        images.append(img)
    
    if len(images) != 4:
        raise ValueError(f"需要4张相移图像，但收到了{len(images)}张")
    
    # 计算包裹相位
    wrapped_phase = compute_wrapped_phase(images, PhaseShiftingAlgorithm.four_step)
    
    # 计算相位质量图
    quality_map = compute_phase_quality(images)
    
    # 可视化并保存包裹相位
    wrapped_path = os.path.join(output_dir, "wrapped_phase.png")
    visualize_wrapped_phase(wrapped_phase, quality_map, "Wrapped Phase", wrapped_path, show_plots)
    
    # 解包裹相位
    if method == "quality_guided":
        # 质量引导解包裹
        unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map)
    else:
        # 对于temporal和multi_freq方法，我们只有单频率数据，所以直接用质量引导法
        print(f"警告: 只有单频率数据，不能使用{method}方法。使用质量引导法代替。")
        unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map)
    
    # 可视化并保存解包裹相位
    unwrapped_path = os.path.join(output_dir, "unwrapped_phase.png")
    visualize_unwrapped_phase(unwrapped_phase, "Unwrapped Phase", unwrapped_path, show_plots)
    
    # 生成并保存3D表面可视化
    surface_3d_path = os.path.join(output_dir, "unwrapped_phase_3d.png")
    visualize_3d_surface(unwrapped_phase, "解包裹相位 3D 表面", 'viridis', surface_3d_path, show_plots)
    
    # 保存相位数据
    np.save(os.path.join(output_dir, "wrapped_phase.npy"), wrapped_phase)
    np.save(os.path.join(output_dir, "unwrapped_phase.npy"), unwrapped_phase)
    np.save(os.path.join(output_dir, "quality_map.npy"), quality_map)
    
    print(f"相位处理完成。结果已保存到 {output_dir} 目录。")
    return unwrapped_phase


def process_dual_direction_images(h_image_paths: List[str], v_image_paths: List[str], 
                                output_dir: str = "output", method: str = "quality_guided", 
                                show_plots: bool = True):
    """
    同时处理水平和垂直方向的四步相移图像
    
    参数:
        h_image_paths: 水平方向的四张相移图像路径
        v_image_paths: 垂直方向的四张相移图像路径
        output_dir: 输出目录
        method: 解包裹方法 ('quality_guided', 'temporal', 'multi_freq')
        show_plots: 是否显示图形 (在线程中应设为False)
    
    返回:
        result_dict: 包含水平和垂直方向解包裹相位的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理水平方向
    print("\n处理水平方向相移图像...")
    h_output_dir = os.path.join(output_dir, "horizontal")
    h_unwrapped = process_four_step_images(
        h_image_paths, 
        h_output_dir, 
        method, 
        show_plots
    )
    
    # 处理垂直方向
    print("\n处理垂直方向相移图像...")
    v_output_dir = os.path.join(output_dir, "vertical")
    v_unwrapped = process_four_step_images(
        v_image_paths, 
        v_output_dir, 
        method, 
        show_plots
    )
    
    # 生成组合图
    if h_unwrapped is not None and v_unwrapped is not None:
        print("\n生成水平和垂直方向相位组合图...")
        
        # 生成2D组合相位图
        combined_path = os.path.join(output_dir, "combined_phase.png")
        combined_rgb = generate_combined_phase(
            h_unwrapped, 
            v_unwrapped,
            "水平和垂直方向相位组合图", 
            combined_path,
            show_plots
        )
        
        # 保存组合相位数据
        np.save(os.path.join(output_dir, "combined_phase.npy"), combined_rgb)
        
        # 为水平和垂直方向生成3D表面可视化
        h_3d_path = os.path.join(output_dir, "horizontal_3d.png")
        visualize_3d_surface(
            h_unwrapped,
            "水平方向解包裹相位 3D 表面",
            'viridis',
            h_3d_path,
            show_plots
        )
        
        v_3d_path = os.path.join(output_dir, "vertical_3d.png")
        visualize_3d_surface(
            v_unwrapped,
            "垂直方向解包裹相位 3D 表面",
            'plasma',
            v_3d_path,
            show_plots
        )
    
    # 返回结果
    return {
        "horizontal": h_unwrapped,
        "vertical": v_unwrapped
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="四步相移图像解包裹处理")
    parser.add_argument("--images", nargs=4, type=str, 
                        help="四张相移图像的路径，相位分别为0°, 90°, 180°, 270°")
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--method", type=str, choices=["quality_guided", "temporal", "multi_freq"],
                        default="quality_guided", help="解包裹方法")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示图形，只保存结果")
    parser.add_argument("--folder", action="store_true",
                        help="使用交互式选择文件夹模式")
    parser.add_argument("--dual", action="store_true",
                        help="处理双方向（水平和垂直）的相位图")
    
    args = parser.parse_args()
    
    # 交互式选择文件夹模式
    if args.folder or not args.images:
        try:
            # 提示用户输入文件夹路径
            print("\n=== 交互式相位解包裹 ===")
            print("注意: 四步相移法需要4张图像，相位偏移分别为0°, 90°, 180°, 270°")
            print("推荐的文件命名方式: phase_0.png, phase_90.png, phase_180.png, phase_270.png")
            folder_path = input("\n请输入包含相移图像的文件夹路径 (直接回车使用当前目录): ")
            
            # 如果用户直接回车，使用当前目录
            if not folder_path:
                folder_path = "."
            
            # 确保路径存在
            if not os.path.exists(folder_path):
                print(f"错误: 路径 '{folder_path}' 不存在")
                exit(1)
            
            # 查找文件夹中的图像文件
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            image_files = []
            
            print(f"\n正在扫描 '{folder_path}' 中的图像文件...")
            
            # 使用不区分大小写的方式查找图像文件
            for ext in image_extensions:
                # 使用不区分大小写的模式进行查找
                pattern = os.path.join(folder_path, f"*{ext}")
                found_files = glob.glob(pattern, recursive=False)
                # 也匹配大写扩展名
                found_files.extend(glob.glob(pattern.replace(ext, ext.upper()), recursive=False))
                image_files.extend(found_files)
            
            # 确保路径格式统一 (使用标准化的绝对路径)
            normalized_files = [os.path.normcase(os.path.abspath(f)) for f in image_files]
            
            # 使用字典保留唯一的文件路径，同时保持顺序
            unique_files_dict = {}
            for norm_path, orig_path in zip(normalized_files, image_files):
                if norm_path not in unique_files_dict:
                    unique_files_dict[norm_path] = orig_path
            
            # 获取去重后的文件列表
            image_files = list(unique_files_dict.values())
            
            # 对文件按名称排序 (更可靠的排序方式)
            image_files.sort(key=lambda x: os.path.basename(x).lower())
            
            if len(image_files) < 4:
                print(f"错误: 在 '{folder_path}' 中找到的图像文件少于4个，无法进行四步相移处理")
                exit(1)
            elif len(image_files) > 4 and len(image_files) < 8 and args.dual:
                print(f"警告: 在 '{folder_path}' 中只找到 {len(image_files)} 个图像文件，无法进行双方向解包裹（需要至少8张图像）")
                print("将使用单方向解包裹代替")
                args.dual = False
            
            # 显示找到的图像文件
            print(f"\n在 '{folder_path}' 中找到 {len(image_files)} 个唯一图像文件:")
            for i, img_path in enumerate(image_files):
                # 显示图像文件的基本名称和大小信息，帮助用户识别
                file_size = os.path.getsize(img_path) / 1024  # 转换为KB
                file_name = os.path.basename(img_path)
                print(f"{i+1}. {file_name} ({file_size:.1f} KB)")
            
            # 询问是否需要查看更多图像信息
            show_details = input("\n是否需要查看图像的详细信息？这将显示图像尺寸等属性 (y/n, 默认n): ").lower()
            if show_details.startswith('y'):
                try:
                    for i, img_path in enumerate(image_files):
                        # 尝试读取图像并显示其尺寸信息
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            h, w = img.shape
                            print(f"{i+1}. {os.path.basename(img_path)}: {w}x{h} 像素")
                        else:
                            print(f"{i+1}. {os.path.basename(img_path)}: 无法读取图像尺寸")
                except Exception as e:
                    print(f"读取图像信息时出错: {e}")
            
            # 询问处理模式
            if not args.dual and len(image_files) >= 8:
                process_mode = input("\n检测到8个或更多图像文件。是否要处理双方向 (水平和垂直)? (y/n): ").lower()
                args.dual = process_mode.startswith('y')
            
            # 选择解包裹方法
            method_mapping = {
                "1": "quality_guided",
                "2": "temporal",
                "3": "multi_freq"
            }
            method_choice = input("\n选择解包裹方法:\n"
                                 "1. 质量引导法 (默认)\n"
                                 "2. 时序相位解包裹\n"
                                 "3. 多频率法\n"
                                 "请输入选项 (1-3): ")
            
            if method_choice in method_mapping:
                args.method = method_mapping[method_choice]
            
            # 设置输出目录
            output_dir = input(f"\n请输入输出目录 (直接回车使用默认值 '{args.output}'): ")
            if output_dir:
                args.output = output_dir
            
            # 是否显示图像
            show_plots = input("\n是否在处理过程中显示图像? (y/n, 默认y): ").lower()
            args.no_display = not (not show_plots or show_plots.startswith('y'))
            
            # 如果是双方向处理
            if args.dual:
                if len(image_files) < 8:
                    print(f"警告: 双方向处理需要至少8张图像，但只找到了 {len(image_files)} 张")
                    print("将使用单方向解包裹代替")
                    args.dual = False
                
                # 默认前4张用于水平方向，后4张用于垂直方向
                print("\n默认使用前4张图像作为水平方向，后4张图像作为垂直方向")
                change_order = input("是否要更改这个默认顺序? (y/n, 默认n): ").lower()
                
                if change_order.startswith('y'):
                    print("\n请选择水平方向使用的4张图像:")
                    print("方法1: 输入序号，如 '1,2,3,4'")
                    print("方法2: 输入范围，如 '1-4'")
                    print("注意: 相位偏移应按照0°, 90°, 180°, 270°的顺序排列")
                    
                    h_selection = input("请输入: ").strip()
                    
                    # 处理范围输入 (如 '1-4')
                    if '-' in h_selection:
                        try:
                            start, end = map(int, h_selection.split('-'))
                            if 1 <= start <= end <= len(image_files):
                                h_indices = list(range(start-1, end))
                            else:
                                print(f"错误: 范围应在1到{len(image_files)}之间")
                                exit(1)
                        except ValueError:
                            print("错误: 无法解析范围，格式应为 '开始-结束'")
                            exit(1)
                    else:
                        # 处理逗号分隔的输入 (如 '1,2,3,4')
                        try:
                            h_indices = [int(idx.strip()) - 1 for idx in h_selection.split(',') if idx.strip()]
                        except ValueError:
                            print("错误: 无法解析序号，应为以逗号分隔的数字")
                            exit(1)
                    
                    if len(h_indices) != 4:
                        print("错误: 水平方向需要恰好4张图像")
                        exit(1)
                    
                    # 确保所有索引都在有效范围内
                    if not all(0 <= i < len(image_files) for i in h_indices):
                        print(f"错误: 索引超出范围，应在1到{len(image_files)}之间")
                        exit(1)
                    
                    print("\n请选择垂直方向使用的4张图像:")
                    print("方法1: 输入序号，如 '5,6,7,8'")
                    print("方法2: 输入范围，如 '5-8'")
                    print("注意: 相位偏移应按照0°, 90°, 180°, 270°的顺序排列")
                    
                    v_selection = input("请输入: ").strip()
                    
                    # 处理范围输入 (如 '5-8')
                    if '-' in v_selection:
                        try:
                            start, end = map(int, v_selection.split('-'))
                            if 1 <= start <= end <= len(image_files):
                                v_indices = list(range(start-1, end))
                            else:
                                print(f"错误: 范围应在1到{len(image_files)}之间")
                                exit(1)
                        except ValueError:
                            print("错误: 无法解析范围，格式应为 '开始-结束'")
                            exit(1)
                    else:
                        # 处理逗号分隔的输入 (如 '5,6,7,8')
                        try:
                            v_indices = [int(idx.strip()) - 1 for idx in v_selection.split(',') if idx.strip()]
                        except ValueError:
                            print("错误: 无法解析序号，应为以逗号分隔的数字")
                            exit(1)
                    
                    if len(v_indices) != 4:
                        print("错误: 垂直方向需要恰好4张图像")
                        exit(1)
                    
                    # 确保所有索引都在有效范围内
                    if not all(0 <= i < len(image_files) for i in v_indices):
                        print(f"错误: 索引超出范围，应在1到{len(image_files)}之间")
                        exit(1)
                    
                    # 检查水平和垂直方向是否有重复的图像
                    common_indices = set(h_indices) & set(v_indices)
                    if common_indices:
                        common_files = [os.path.basename(image_files[i]) for i in common_indices]
                        print(f"警告: 水平和垂直方向选择了相同的图像: {', '.join(common_files)}")
                        confirm = input("确定要继续吗? (y/n, 默认n): ").lower()
                        if not confirm.startswith('y'):
                            print("操作已取消")
                            exit(0)
                    
                    h_image_paths = [image_files[i] for i in h_indices]
                    v_image_paths = [image_files[i] for i in v_indices]
                    
                    # 显示选择的图像
                    print("\n您选择的水平方向图像:")
                    for i, img_path in enumerate(h_image_paths):
                        print(f"{i+1}. {os.path.basename(img_path)} (相位偏移: {90*i}°)")
                    
                    print("\n您选择的垂直方向图像:")
                    for i, img_path in enumerate(v_image_paths):
                        print(f"{i+1}. {os.path.basename(img_path)} (相位偏移: {90*i}°)")
                else:
                    h_image_paths = image_files[:4]
                    v_image_paths = image_files[4:8]
                    
                    # 显示选择的图像
                    print("\n使用默认选择:")
                    print("水平方向图像:")
                    for i, img_path in enumerate(h_image_paths):
                        print(f"{i+1}. {os.path.basename(img_path)} (相位偏移: {90*i}°)")
                    
                    print("\n垂直方向图像:")
                    for i, img_path in enumerate(v_image_paths):
                        print(f"{i+1}. {os.path.basename(img_path)} (相位偏移: {90*i}°)")
                
                print("\n=== 开始双方向解包裹 ===")
                print(f"水平方向图像: {[os.path.basename(p) for p in h_image_paths]}")
                print(f"垂直方向图像: {[os.path.basename(p) for p in v_image_paths]}")
                print(f"解包裹方法: {args.method}")
                print(f"输出目录: {args.output}")
                
                # 处理双方向图像
                process_dual_direction_images(
                    h_image_paths,
                    v_image_paths,
                    args.output,
                    args.method,
                    not args.no_display
                )
            else:
                # 单方向处理
                if len(image_files) > 4:
                    print(f"\n发现 {len(image_files)} 张图像，但只需要4张用于四步相移")
                    use_first_four = input("使用前4张图像? (y/n, 默认y): ").lower()
                    
                    if not use_first_four or use_first_four.startswith('y'):
                        selected_images = image_files[:4]
                    else:
                        print("\n请选择要使用的4张图像:")
                        print("方法1: 输入序号，如 '1,2,3,4'")
                        print("方法2: 输入范围，如 '1-4'")
                        print("注意: 相位偏移应按照0°, 90°, 180°, 270°的顺序排列")
                        
                        selection = input("请输入: ").strip()
                        
                        # 处理范围输入 (如 '1-4')
                        if '-' in selection:
                            try:
                                start, end = map(int, selection.split('-'))
                                if 1 <= start <= end <= len(image_files):
                                    indices = list(range(start-1, end))
                                else:
                                    print(f"错误: 范围应在1到{len(image_files)}之间")
                                    exit(1)
                            except ValueError:
                                print("错误: 无法解析范围，格式应为 '开始-结束'")
                                exit(1)
                        else:
                            # 处理逗号分隔的输入 (如 '1,2,3,4')
                            try:
                                indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
                            except ValueError:
                                print("错误: 无法解析序号，应为以逗号分隔的数字")
                                exit(1)
                        
                        if len(indices) != 4:
                            print("错误: 需要选择恰好4张图像")
                            exit(1)
                        
                        # 确保所有索引都在有效范围内
                        if not all(0 <= i < len(image_files) for i in indices):
                            print(f"错误: 索引超出范围，应在1到{len(image_files)}之间")
                            exit(1)
                        
                        selected_images = [image_files[i] for i in indices]
                        
                        # 显示选择的图像
                        print("\n您选择的图像:")
                        for i, img_path in enumerate(selected_images):
                            print(f"{i+1}. {os.path.basename(img_path)} (相位偏移: {90*i}°)")
                else:
                    selected_images = image_files
                
                print("\n=== 开始单方向解包裹 ===")
                print(f"选择的图像: {[os.path.basename(p) for p in selected_images]}")
                print(f"解包裹方法: {args.method}")
                print(f"输出目录: {args.output}")
                
                # 处理单方向图像
                process_four_step_images(
                    selected_images,
                    args.output,
                    args.method,
                    not args.no_display
                )
                
        except KeyboardInterrupt:
            print("\n处理已取消")
            exit(0)
    else:
        # 命令行参数模式
        if args.dual:
            if len(args.images) < 8:
                print("错误: 双方向处理需要至少8张图像")
                exit(1)
            
            # 使用前4个参数作为水平方向，后4个作为垂直方向
            h_image_paths = args.images[:4]
            v_image_paths = args.images[4:8]
            
            # 处理双方向图像
            process_dual_direction_images(
                h_image_paths,
                v_image_paths,
                args.output,
                args.method,
                not args.no_display
            )
        else:
            # 处理单方向图像
            unwrapped_phase = process_four_step_images(
                args.images, 
                args.output, 
                args.method,
                show_plots=not args.no_display
            )
    
    print("\n处理完成。") 