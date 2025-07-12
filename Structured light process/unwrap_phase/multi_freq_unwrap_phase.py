import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import os
from typing import List, Dict, Tuple, Optional
from enum import Enum
import glob

try:
    from scipy.signal import medfilt2d
except ImportError:
    raise ImportError("本功能需要 SciPy 库。请运行 'pip install scipy' 来安装。")
try:
    from scipy.ndimage import binary_dilation
except ImportError:
    raise ImportError("本功能需要 SciPy 库。请运行 'pip install scipy' 来安装。")
try:
    from scipy.interpolate import griddata
except ImportError:
    raise ImportError("本功能需要 SciPy 库。请运行 'pip install scipy' 来安装。")


# 尝试设置中文字体
try:
    # 检查系统是否有支持中文的字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'SimSun' in f.name or 'Microsoft YaHei' in f.name]
    if chinese_fonts:
        plt.rcParams['font.family'] = chinese_fonts[0]
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    else:
        # 如果没有中文字体，使用默认字体，但不显示中文标题
        plt.rcParams['font.family'] = 'sans-serif'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# --- 从其他模块集成的函数 ---

class PhaseShiftingAlgorithm(Enum):
    """相移算法类型枚举"""
    three_step = 0
    four_step = 1
    n_step = 2

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

    # 假设相移是 + k*delta 的形式
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
    计算相位质量图
    """
    n = len(images)
    float_images = [img.astype(np.float32) for img in images]
    avg_intensity = sum(float_images) / n
    delta = 2 * np.pi / n
    sin_sum = sum(float_images[i] * np.sin(i * delta) for i in range(n))
    cos_sum = sum(float_images[i] * np.cos(i * delta) for i in range(n))
    modulation = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    quality_map = modulation / (avg_intensity + 1e-10)
    return quality_map

def visualize_wrapped_phase(wrapped_phase: np.ndarray, quality_map: Optional[np.ndarray] = None, 
                           title: str = "包裹相位图", save_path: Optional[str] = None, show_plots: bool = True):
    """
    可视化包裹相位图
    """
    if not show_plots:
        return
    plt.figure(figsize=(12, 9))
    if quality_map is not None:
        plt.subplot(2, 1, 1)
    img = plt.imshow(wrapped_phase, cmap='jet')
    plt.colorbar(img, label='相位 (弧度)')
    plt.title(title)
    if quality_map is not None:
        plt.subplot(2, 1, 2)
        q_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(q_img, label='质量')
        plt.title("相位质量图")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

def multi_frequency_unwrap(
    wrapped_phases: Dict[int, np.ndarray],
    phasors: Dict[int, np.ndarray],
    frequencies: List[int],
    quality_maps: Optional[Dict[int, np.ndarray]] = None,
    filter_kernel_size: int = 9,
    direction: str = 'horizontal',
    direction_output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用多频外差法（Hierarchical Heterodyne）解包裹相位。
    该方法通过对相邻频率的相位进行逐级外差，生成频率更低的等效相位，
    直到获得一个频率最低的绝对相位。然后，利用这个绝对相位
    反向逐级展开更高频率的相位，最终得到原始最高频相位的绝对展开结果。
    
    参考原理:
    1. 逐级外差: Φ_equiv = (Φ_high - Φ_low) mod 2π, f_equiv = f_high - f_low
    2. 逐级展开: k = round((Φ_unwrapped_low * (f_high / f_low) - Φ_wrapped_high) / (2π))
                  Φ_unwrapped_high = Φ_wrapped_high + 2π * k
    """
    print("\n--- 开始执行多频外差解包裹 ---")
    
    if len(frequencies) < 2:
        raise ValueError("多频外差法至少需要两个频率。")

    # 1. 频率从高到低排序
    sorted_freqs = sorted(frequencies, reverse=True)
    print(f"输入频率 (从高到低排序): {sorted_freqs}")

    # 存储各级外差结果
    level_phases = [wrapped_phases[f] for f in sorted_freqs]
    level_phasors = [phasors[f] for f in sorted_freqs]
    levels = [{'freqs': sorted_freqs, 'phases': level_phases, 'phasors': level_phasors}]

    # 2. 逐级外差，生成等效低频相位
    print("\n步骤 1: 逐级外差生成等效相位")
    current_freqs = sorted_freqs
    current_phasors = level_phasors
    
    level_num = 1
    while len(current_freqs) > 1:
        next_freqs = []
        next_phases = []
        next_phasors = []
        print(f"  外差等级 {level_num}:")
        
        for i in range(len(current_freqs) - 1):
            f_high, f_low = current_freqs[i], current_freqs[i+1]
            C_high, C_low = current_phasors[i], current_phasors[i+1]
            
            f_equiv = f_high - f_low
            if f_equiv <= 0:
                raise ValueError(f"频率差为非正数({f_equiv})，请检查频率选择。应选择递减的频率序列。")

            # 【关键修正】使用复数相量进行正确的外差操作
            C_equiv = C_high * np.conj(C_low)
            p_equiv_wrapped = np.angle(C_equiv)

            print(f"    - {f_high} Hz - {f_low} Hz -> 等效频率: {f_equiv} Hz")
            
            next_freqs.append(f_equiv)
            next_phases.append(p_equiv_wrapped)
            next_phasors.append(C_equiv)

        levels.append({'freqs': next_freqs, 'phases': next_phases, 'phasors': next_phasors})
        current_freqs = next_freqs
        current_phasors = next_phasors
        level_num += 1

    # 3. 逐级展开，恢复绝对相位
    print("\n步骤 2: 逐级展开恢复绝对相位")
    
    # 获取频率为1的基准相位
    base_unwrapped_phase = levels[-1]['phases'][0].copy()
    print(f"  使用频率为 {levels[-1]['freqs'][0]} Hz 的等效相位作为展开基准。")

    # 自动检测相位方向
    # 为提高方向判断的稳定性，我们对基准相位进行一次强力滤波，并仅将该滤波结果用于方向判断
    base_phase_for_orient = medfilt2d(base_unwrapped_phase, kernel_size=15)
    _, was_flipped = _auto_orient_phase(base_phase_for_orient, direction)

    # 根据判断结果，修正原始基准相位的方向
    if was_flipped:
        print("  基准相位方向已被自动修正。")
        unwrapped_phase = -base_unwrapped_phase
    else:
        unwrapped_phase = base_unwrapped_phase
    
    # 获取图像尺寸
    height, width = unwrapped_phase.shape
    
    # 创建质量图，用于指导展开过程
    # 如果没有提供质量图，则创建一个基于相位梯度的质量图
    if quality_maps is None or len(quality_maps) == 0:
        quality_map = np.ones_like(unwrapped_phase)
    else:
        # 使用最高频率的质量图
        highest_freq = sorted_freqs[0]
        quality_map = quality_maps[highest_freq]
    
    # 初始化掩码，所有像素都是有效的
    valid_mask = np.ones((height, width), dtype=bool)
    
    # 初始化k_map字典，用于存储每个频率的条纹阶数
    k_maps = {}
    
    # 【新改进】对基准相位执行更强的平滑和非负化处理
    # 首先确保无负值
    unwrapped_phase = np.maximum(unwrapped_phase, 0)
    
    # 应用更强的平滑，确保低频相位足够平滑
    unwrapped_phase = medfilt2d(unwrapped_phase, kernel_size=11)
    
    # 【新改进】对基准相位的梯度方向进行更稳健的处理
    # 计算梯度
    gy, gx = np.gradient(unwrapped_phase)
    
    # 根据预期方向，确保梯度主体是正向的
    if direction == 'horizontal':
        # 水平方向，应该主要是x方向的正梯度
        if np.median(gx) < 0:
            print("  基准相位梯度方向不符合预期，进行翻转。")
            unwrapped_phase = -unwrapped_phase + np.max(-unwrapped_phase)  # 翻转并保持非负
    else:  # vertical
        # 垂直方向，应该主要是y方向的正梯度
        if np.median(gy) < 0:
            print("  基准相位梯度方向不符合预期，进行翻转。")
            unwrapped_phase = -unwrapped_phase + np.max(-unwrapped_phase)  # 翻转并保持非负
    
    # 【新改进】对基准相位进行零点对齐
    # 对于水平方向，使左侧为起始点；对于垂直方向，使顶部为起始点
    if direction == 'horizontal':
        # 水平方向，使用左侧10%区域的最小值作为零点
        left_region = unwrapped_phase[:, :int(width * 0.1)]
        zero_ref = np.percentile(left_region, 5)  # 使用低百分位数，避免异常值影响
    else:
        # 垂直方向，使用顶部10%区域的最小值作为零点
        top_region = unwrapped_phase[:int(height * 0.1), :]
        zero_ref = np.percentile(top_region, 5)  # 使用低百分位数，避免异常值影响
    
    # 对齐零点，确保基准相位从接近0开始
    unwrapped_phase -= zero_ref
    unwrapped_phase = np.maximum(unwrapped_phase, 0)  # 再次确保非负
    
    print(f"  基准相位已优化：进行了平滑处理、方向校正和零点对齐 (零点参考值: {zero_ref:.2f})。")
    
    # 逐级展开，使用更稳健的策略
    for i in range(len(levels) - 2, -1, -1):
        # 选择展开链条的第一个相位进行处理
        freq_low = levels[i+1]['freqs'][0]
        freq_high = levels[i]['freqs'][0]
        phase_high_wrapped = levels[i]['phases'][0].copy()
        phasor_high = levels[i]['phasors'][0]
        
        # 将低频展开结果作为当前展开的基准
        unwrapped_phase_low = unwrapped_phase.copy()
        
        print(f"  展开等级 {len(levels) - 1 - i}: 使用 {freq_low} Hz 展开 {freq_high} Hz")
        
        freq_ratio = freq_high / freq_low
        
        # 计算初步的k值估计
        k_float = (unwrapped_phase_low * freq_ratio - phase_high_wrapped) / (2 * np.pi)
        
        # 对k值进行平滑，但保留更多细节
        k_smooth = k_float.copy()
        if filter_kernel_size > 1:
            k_smooth = medfilt2d(k_float, kernel_size=filter_kernel_size)
        
        # 初始化k值图，使用平滑后的k值的四舍五入作为初始估计
        k = np.round(k_smooth).astype(np.int32)
        
        # 使用区域生长法修正k值
        # 创建一个置信度图，表示我们对k值估计的置信程度
        confidence = 1.0 - np.abs(k_float - k_smooth)
        
        # 找到置信度最高的点作为种子
        # 为了更稳健，我们选择质量高且置信度高的区域作为种子
        combined_quality = confidence * quality_map
        
        # 【改进】不再使用随机种子点或纯粹基于质量的种子点
        # 而是根据解包裹方向，选择起始边缘中心附近的高质量点作为种子
        if direction == 'horizontal':
            # 水平方向，从左侧边缘开始解包裹
            # 在左侧5%区域内寻找高质量点
            left_edge = int(width * 0.05)
            edge_region_quality = combined_quality[:, :left_edge]
            
            # 找出质量较高的点（超过中值的点）
            threshold = np.median(edge_region_quality)
            high_quality_mask = edge_region_quality > threshold
            
            if np.any(high_quality_mask):
                # 在高质量点中找一个接近中心的点作为种子
                y_indices, x_indices = np.where(high_quality_mask)
                center_y = height // 2
                
                # 找出离中心y最近的点
                closest_idx = np.argmin(np.abs(y_indices - center_y))
                seed_y = y_indices[closest_idx]
                seed_x = x_indices[closest_idx]
                print(f"  选择左侧边缘的高质量种子点: ({seed_x}, {seed_y})")
            else:
                # 如果没有高质量点，选择左侧边缘中心点
                seed_y = height // 2
                seed_x = left_edge // 2
                print(f"  使用左侧边缘中心点作为种子: ({seed_x}, {seed_y})")
        else:  # vertical
            # 垂直方向，从上侧边缘开始解包裹
            # 在上侧5%区域内寻找高质量点
            top_edge = int(height * 0.05)
            edge_region_quality = combined_quality[:top_edge, :]
            
            # 找出质量较高的点（超过中值的点）
            threshold = np.median(edge_region_quality)
            high_quality_mask = edge_region_quality > threshold
            
            if np.any(high_quality_mask):
                # 在高质量点中找一个接近中心的点作为种子
                y_indices, x_indices = np.where(high_quality_mask)
                center_x = width // 2
                
                # 找出离中心x最近的点
                closest_idx = np.argmin(np.abs(x_indices - center_x))
                seed_y = y_indices[closest_idx]
                seed_x = x_indices[closest_idx]
                print(f"  选择上侧边缘的高质量种子点: ({seed_x}, {seed_y})")
            else:
                # 如果没有高质量点，选择上侧边缘中心点
                seed_y = top_edge // 2
                seed_x = width // 2
                print(f"  使用上侧边缘中心点作为种子: ({seed_x}, {seed_y})")
        
        # 【改进13】初始化已访问掩码
        visited = np.zeros_like(k, dtype=bool)
        visited[seed_y, seed_x] = True
        
        # 【改进14】使用队列进行广度优先搜索
        from collections import deque
        queue = deque([(seed_y, seed_x)])
        
        # 【改进15】定义8-邻域，增加连通性
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # 【改进16】区域生长
        while queue:
            y, x = queue.popleft()
            k_center = k[y, x]
            
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # 检查边界
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                
                # 如果已访问，跳过
                if visited[ny, nx]:
                    continue
                
                # 【改进17】计算该点的k值应该是多少
                # 假设相邻点的k值差异不超过1
                k_expected = k_center
                
                # 如果k值差异较大，检查是否需要调整
                k_diff = k_float[ny, nx] - k_float[y, x]
                if abs(k_diff) > 0.5:
                    # 如果差异接近整数，可能是需要调整k值
                    k_adjustment = round(k_diff)
                    k_expected = k_center + k_adjustment
                
                # 更新k值
                k[ny, nx] = k_expected
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到队列
                queue.append((ny, nx))
        
        # 【改进18】检查是否有未访问的区域
        if not np.all(visited):
            print(f"  警告: 有 {np.sum(~visited)} 个像素未被区域生长法访问，将使用初始k值估计。")
            # 对于未访问的区域，我们保留初始的k值估计
        
        # 【改进19】使用修正后的k值进行相位展开
        unwrapped_phase = phase_high_wrapped + 2 * np.pi * k
        
        # 【改进20】保存当前频率的k_map
        k_maps[freq_high] = k
    
    # 【改进21】最终的k_map是最高频率的k_map
    k_map = k_maps[sorted_freqs[0]] if sorted_freqs else None
    
    # 【改进22】全局优化 - 应用全局平滑以减少噪声
    # 使用中值滤波对最终结果进行轻微平滑，保留边缘
    if filter_kernel_size > 1:
        unwrapped_phase = medfilt2d(unwrapped_phase, kernel_size=3)
    
    # 【改进23】额外检查 - 确保没有明显的不连续性
    # 计算相位梯度
    gy, gx = np.gradient(unwrapped_phase)
    
    # 检测异常大的梯度（可能是不连续点）
    if direction == 'horizontal':
        # 水平方向，主要关注x方向的梯度
        gradient = gx
    else:
        # 垂直方向，主要关注y方向的梯度
        gradient = gy
    
    # 计算梯度的均值和标准差
    mean_grad = np.median(gradient)  # 使用中位数而不是均值，对异常值更稳健
    std_grad = np.std(gradient)
    
    # 标记梯度异常的点
    abnormal_points = np.abs(gradient - mean_grad) > 5 * std_grad
    
    # 【改进24】如果存在异常点，尝试修复
    if np.any(abnormal_points):
        print(f"  检测到 {np.sum(abnormal_points)} 个梯度异常点，尝试修复...")
        
        # 使用膨胀操作扩大异常区域，确保完全覆盖不连续区域
        from scipy.ndimage import binary_dilation
        abnormal_region = binary_dilation(abnormal_points, iterations=3)
        
        # 在异常区域的两侧找到参考点
        if direction == 'horizontal':
            # 找到异常区域的左右边界
            x_indices = np.where(np.any(abnormal_region, axis=0))[0]
            if len(x_indices) > 0:
                x_left = max(0, x_indices[0] - 5)
                x_right = min(width - 1, x_indices[-1] + 5)
                
                # 【改进25】检测是否是整个右半部分出现问题
                if x_left < width * 0.6 and x_right > width * 0.6:
                    print("  检测到右半部分可能存在解包裹失败，尝试修复...")
                    
                    # 创建一个新的掩码，标记需要修复的区域
                    repair_mask = np.zeros((height, width), dtype=bool)
                    repair_mask[:, x_left:] = True
                    
                    # 使用左侧区域的平均梯度进行线性外推
                    left_region = unwrapped_phase[:, :x_left]
                    
                    # 【修复】确保左侧区域足够大，以便计算梯度
                    if x_left > 10:  # 确保至少有10列数据
                        # 计算左侧区域的平均梯度
                        left_gradients = np.gradient(left_region)[1]
                        # 使用左侧边界附近的梯度，但确保不会越界
                        boundary_size = min(10, left_gradients.shape[1])
                        avg_gradient = np.median(left_gradients[:, -boundary_size:])
                    else:
                        # 如果左侧区域太小，使用一个默认的梯度值
                        print("  左侧区域太小，使用默认梯度值")
                        avg_gradient = 1.0  # 默认梯度值
                    
                    # 从左边界开始，逐列外推
                    for x in range(x_left, width):
                        dx = x - x_left + 1
                        unwrapped_phase[:, x] = unwrapped_phase[:, x_left-1] + dx * avg_gradient
        
        else:  # vertical
            # 类似的处理垂直方向
            y_indices = np.where(np.any(abnormal_region, axis=1))[0]
            if len(y_indices) > 0:
                y_top = max(0, y_indices[0] - 5)
                y_bottom = min(height - 1, y_indices[-1] + 5)
                
                if y_top < height * 0.6 and y_bottom > height * 0.6:
                    print("  检测到下半部分可能存在解包裹失败，尝试修复...")
                    
                    repair_mask = np.zeros((height, width), dtype=bool)
                    repair_mask[y_top:, :] = True
                    
                    top_region = unwrapped_phase[:y_top, :]
                    
                    # 【修复】确保上部区域足够大，以便计算梯度
                    if y_top > 10:  # 确保至少有10行数据
                        # 计算上部区域的平均梯度
                        top_gradients = np.gradient(top_region)[0]
                        # 使用上部边界附近的梯度，但确保不会越界
                        boundary_size = min(10, top_gradients.shape[0])
                        avg_gradient = np.median(top_gradients[-boundary_size:, :])
                    else:
                        # 如果上部区域太小，使用一个默认的梯度值
                        print("  上部区域太小，使用默认梯度值")
                        avg_gradient = 1.0  # 默认梯度值
                    
                    for y in range(y_top, height):
                        dy = y - y_top + 1
                        unwrapped_phase[y, :] = unwrapped_phase[y_top-1, :] + dy * avg_gradient

    # 【改进26】确保相位是连续的
    # 检查是否有突变点
    gy, gx = np.gradient(unwrapped_phase)
    if direction == 'horizontal':
        gradient = gx
    else:
        gradient = gy
    
    # 计算梯度的统计特性
    median_grad = np.median(gradient)
    mad = np.median(np.abs(gradient - median_grad))  # 中位数绝对偏差，比标准差更稳健
    
    # 检测异常大的梯度变化
    threshold = median_grad + 10 * mad
    jump_points = gradient > threshold
    
    if np.any(jump_points):
        print(f"  检测到 {np.sum(jump_points)} 个相位跳变点，尝试修复...")
        
        # 使用最小二乘法拟合一个平滑的相位表面
        y_indices, x_indices = np.mgrid[:height, :width]
        
        # 排除跳变点
        valid_points = ~jump_points
        y_valid = y_indices[valid_points]
        x_valid = x_indices[valid_points]
        z_valid = unwrapped_phase[valid_points]
        
        # 拟合一个二次曲面
        from scipy.interpolate import griddata
        
        # 对跳变点进行插值
        y_jump, x_jump = np.where(jump_points)
        points = np.column_stack((y_valid, x_valid))
        
        try:
            # 使用自然邻居插值
            interpolated = griddata(points, z_valid, (y_jump, x_jump), method='nearest')
            unwrapped_phase[jump_points] = interpolated
        except Exception as e:
            print(f"  插值失败: {e}")
    
    # 【改进27】最终检查 - 确保没有负值
    if np.any(unwrapped_phase < 0):
        min_val = np.min(unwrapped_phase)
        print(f"  检测到负相位值 (最小值: {min_val:.2f})，进行偏移校正...")
        unwrapped_phase -= min_val
    
    # 【改进28】最终的零点校准
    if direction == 'horizontal':
        # 水平方向，以最左侧20%区域的中位数为基准
        left_region = unwrapped_phase[:, :int(width * 0.2)]
        zero_ref = np.median(left_region)
    else:
        # 垂直方向，以最顶部20%区域的中位数为基准
        top_region = unwrapped_phase[:int(height * 0.2), :]
        zero_ref = np.median(top_region)
    
    # 偏移整个相位图，使得起始边缘的相位接近于0
    unwrapped_phase -= zero_ref
    print(f"  最终相位修正：已将相位零点对齐到图像起始边缘 (参考值: {zero_ref:.2f})。")
    
    # 确保没有负值
    unwrapped_phase[unwrapped_phase < 0] = 0

    # 【新增改进29】应用额外平滑以确保结果更加平滑
    # 使用更小的滤波核进行最终平滑，保留边缘细节
    if filter_kernel_size > 1:
        smoothed_phase = medfilt2d(unwrapped_phase, kernel_size=3)
        # 不直接替换，而是加权混合，保留细节
        unwrapped_phase = 0.8 * smoothed_phase + 0.2 * unwrapped_phase
    
    # 【新增改进30】再次检查并确保没有负值
    unwrapped_phase = np.maximum(unwrapped_phase, 0)
    
    # 【新增改进31】检查并修复局部不连续区域
    # 计算局部梯度的方差
    gy, gx = np.gradient(unwrapped_phase)
    grad_mag = np.sqrt(gy**2 + gx**2)
    local_var = cv2.boxFilter(grad_mag, -1, (5, 5), normalize=True)
    
    # 找出局部梯度方差异常大的区域
    high_var_mask = local_var > np.mean(local_var) + 3 * np.std(local_var)
    if np.any(high_var_mask):
        print(f"  检测到 {np.sum(high_var_mask)} 个局部不连续点，进行修复...")
        # 对这些区域进行局部中值滤波
        for _ in range(3):  # 迭代几次以确保平滑
            # 创建一个临时数组以避免边界效应
            temp = unwrapped_phase.copy()
            # 在高方差区域应用局部中值滤波
            for i in range(2, height-2):
                for j in range(2, width-2):
                    if high_var_mask[i, j]:
                        temp[i, j] = np.median(unwrapped_phase[i-2:i+3, j-2:j+3])
            unwrapped_phase = temp

    print("\n--- 多频外差解包裹完成 ---")
    return unwrapped_phase, k_map


def visualize_unwrapped_phase(unwrapped_phase: np.ndarray, title: str = "Unwrapped Phase", 
                             save_path: Optional[str] = None, show_plots: bool = True, raw_save: bool = False) -> None:
    """
    可视化解包裹相位图

    参数:
        unwrapped_phase: 解包裹相位图
        title: 图像标题
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形 (在线程中应设为False)
        raw_save: 是否只保存无边框的纯图像
    """
    # 如果是纯图像保存模式，使用OpenCV直接保存，效率更高且无边框
    if save_path and raw_save:
        try:
            # 归一化到0-255以便应用颜色图
            img_normalized = cv2.normalize(unwrapped_phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 应用伪彩色
            img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
            # 保存图像
            cv2.imwrite(save_path, img_color)
        except Exception as e:
            print(f"使用OpenCV保存原始图像时出错: {e}")
        
        # 如果不需要显示，则直接返回
        if not show_plots:
            return

    # --- Matplotlib 可视化 (用于带边框的保存或屏幕显示) ---
    plt.figure(figsize=(10, 8))
    
    # 显示解包裹相位
    img = plt.imshow(unwrapped_phase, cmap='jet')
    plt.colorbar(img, label='Phase (rad)')
    plt.title(title)
    
    # 如果指定了保存路径且不是纯图像模式，则保存
    if save_path and not raw_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    
    # 总是关闭以释放内存，特别是当 show_plots 为 False 时
    plt.close()


def _auto_orient_phase(unwrapped_phase: np.ndarray, direction: str) -> Tuple[np.ndarray, bool]:
    """
    根据梯度的中值自动判断并修正解包裹相位的方向。

    参数:
        unwrapped_phase: 解包裹相位图
        direction: 解包裹的方向 ('horizontal' 或 'vertical')

    返回:
        (修正方向后的解包裹相位图, 是否进行了翻转)
    """
    if direction not in ['horizontal', 'vertical']:
        return unwrapped_phase, False # 如果方向未知，不作处理

    # 计算梯度
    gy, gx = np.gradient(unwrapped_phase)

    was_flipped = False
    # 【新改进】使用更稳健的方法判断相位方向
    if direction == 'horizontal':
        # 水平方向，相位应从左到右增加，即x梯度为正
        # 使用中值而不是均值，对异常值更稳健
        median_gradient = np.median(gx)
        
        # 【新改进】计算正梯度和负梯度的比例
        positive_ratio = np.sum(gx > 0) / gx.size
        negative_ratio = np.sum(gx < 0) / gx.size
        
        # 如果中值为负或者负梯度明显多于正梯度，则认为方向相反
        if median_gradient < 0 or (negative_ratio > 0.6 and positive_ratio < 0.4):
            print("自动方向修正：检测到水平相位方向相反，已进行翻转。")
            print(f"  梯度中值: {median_gradient:.4f}, 正梯度比例: {positive_ratio:.2f}, 负梯度比例: {negative_ratio:.2f}")
            was_flipped = True
            return -unwrapped_phase, was_flipped
    elif direction == 'vertical':
        # 垂直方向，相位应从上到下增加，即y梯度为正
        median_gradient = np.median(gy)
        
        # 【新改进】计算正梯度和负梯度的比例
        positive_ratio = np.sum(gy > 0) / gy.size
        negative_ratio = np.sum(gy < 0) / gy.size
        
        # 如果中值为负或者负梯度明显多于正梯度，则认为方向相反
        if median_gradient < 0 or (negative_ratio > 0.6 and positive_ratio < 0.4):
            print("自动方向修正：检测到垂直相位方向相反，已进行翻转。")
            print(f"  梯度中值: {median_gradient:.4f}, 正梯度比例: {positive_ratio:.2f}, 负梯度比例: {negative_ratio:.2f}")
            was_flipped = True
            return -unwrapped_phase, was_flipped
    
    # 【新改进】记录方向判断的依据，便于调试
    if direction == 'horizontal':
        print(f"相位方向判断：水平方向梯度中值为 {np.median(gx):.4f}，保持原方向。")
    else:
        print(f"相位方向判断：垂直方向梯度中值为 {np.median(gy):.4f}，保持原方向。")
    
    return unwrapped_phase, was_flipped


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
    combined_rgb = np.zeros((height, width, 3), dtype=np.float64)
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


def process_multi_frequency_images(
    frequency_data: Dict[int, List[str]],
    output_dir: str = "output",
    method: str = "multi_freq",
    show_plots: bool = True,
    filter_kernel_size: int = 9,
    direction: str = 'horizontal'
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """
    处理多频率相移图像并生成解包裹相位

    参数:
        frequency_data: 频率数据字典 {frequency: [image_paths]}
        output_dir: 输出目录
        method: 解包裹方法 (当前仅支持 'multi_freq')
        show_plots: 是否显示图形
        filter_kernel_size: 中值滤波器的边长
        direction: 解包裹方向 ('horizontal' 或 'vertical')

    返回:
        unwrapped_phase: 最终的解包裹相位图
        k_map: 最终的条纹阶数图
        unwrapped_phase_path: 保存的最终解包裹相位图路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    wrapped_phases = {}
    phasors = {}
    quality_maps = {}
    frequencies = sorted(frequency_data.keys())

    print("正在计算各个频率的包裹相位...")
    for freq in frequencies:
        image_paths = frequency_data[freq]
        print(f"  处理频率: {freq} ({len(image_paths)} 张图像)")
        
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        if any(img is None for img in images):
            raise ValueError(f"无法加载频率 {freq} 的一张或多张图像")
        
        # 计算包裹相位、复数相量和质量图（使用N步算法）
        wrapped_phase, phasor = compute_phasor_and_phase(images, PhaseShiftingAlgorithm.n_step)

        # 【关键修正】自动检测并校正相位符号约定。
        # 算法内部假设相位是沿梯度方向增加的(+k*delta)，但输入数据可能是按(-k*delta)生成的。
        # 此步骤通过检查包裹相位的梯度来统一符号，确保后续处理的正确性。
        gy, gx = np.gradient(wrapped_phase)
        # 根据解包裹方向（条纹方向的垂直方向）选择对应的梯度
        median_grad = np.median(gx) if direction == 'horizontal' else np.median(gy)
        
        if median_grad < 0:
            print(f"  频率 {freq}: 检测到负相位梯度，已自动修正符号约定。")
            wrapped_phase = -wrapped_phase
            phasor = np.conj(phasor)
        
        quality_map = compute_phase_quality(images)
        
        # 【重要修正】在将包裹相位送入多频算法前，不再需要修正方向。
        # 方向一致性将在 multi_frequency_unwrap 内部通过更稳健的方式保证。
        # wrapped_phase = _auto_orient_phase(wrapped_phase, direction)
        
        wrapped_phases[freq] = wrapped_phase
        phasors[freq] = phasor
        quality_maps[freq] = quality_map

        # 可选：保存每个频率的中间结果
        freq_output_dir = os.path.join(output_dir, f"freq_{freq}")
        os.makedirs(freq_output_dir, exist_ok=True)
        visualize_wrapped_phase(
            wrapped_phase, quality_map,
            title=f"频率 {freq} 的包裹相位",
            save_path=os.path.join(freq_output_dir, "wrapped_phase.png"),
            show_plots=show_plots
        )
        np.save(os.path.join(freq_output_dir, "wrapped_phase.npy"), wrapped_phase)
        np.save(os.path.join(freq_output_dir, "quality_map.npy"), quality_map)

    print("\n开始进行多频解包裹...")
    if method == "multi_freq":
        unwrapped_phase, k_map = multi_frequency_unwrap(
            wrapped_phases, phasors, frequencies, quality_maps, filter_kernel_size, direction,
            direction_output_dir=output_dir
        )
    elif method == "temporal":
        print("警告: 'temporal' 方法已被弃用，将使用 'multi_freq' 方法执行。")
        unwrapped_phase, k_map = multi_frequency_unwrap(
            wrapped_phases, phasors, frequencies, quality_maps, filter_kernel_size, direction,
            direction_output_dir=output_dir
        )
    else:
        raise ValueError(f"不支持的多频解包裹方法: {method}")

    # 【最终修正 V2 - "边缘对齐"零点校准】
    # 1. 以图像最左侧或最上侧的平均值作为零点参考，强制将相位起点对齐到0
    if direction == 'horizontal':
        # 水平方向，以最左侧一列为基准
        # 【改进】使用左侧20%区域的平均值，而不是仅第一列，以增强稳定性
        left_region = unwrapped_phase[:, :int(unwrapped_phase.shape[1] * 0.2)]
        zero_ref = np.mean(left_region[left_region > 0]) if np.any(left_region > 0) else 0
    else: # vertical
        # 垂直方向，以最顶侧一行为基准
        # 【改进】使用顶部20%区域的平均值，而不是仅第一行，以增强稳定性
        top_region = unwrapped_phase[:int(unwrapped_phase.shape[0] * 0.2), :]
        zero_ref = np.mean(top_region[top_region > 0]) if np.any(top_region > 0) else 0
    
    unwrapped_phase -= zero_ref
    print(f"最终相位修正：已将相位零点对齐到图像起始边缘 (参考值: {zero_ref:.2f})。")

    # 2. 确保没有负值（由于噪声可能产生极小的负数）
    unwrapped_phase[unwrapped_phase < 0] = 0
    
    # 【新增全局改进A】添加一个可选的全局平滑操作
    # 这里使用高斯滤波而非中值滤波，以保留更多细节
    smoothed_phase = cv2.GaussianBlur(unwrapped_phase, (5, 5), 0.5)
    # 以80%原始相位与20%平滑相位混合，保留细节的同时增强平滑性
    unwrapped_phase = 0.8 * unwrapped_phase + 0.2 * smoothed_phase
    
    # 【新增全局改进B】使用梯度信息检查并修复全局平滑性
    gy, gx = np.gradient(unwrapped_phase)
    if direction == 'horizontal':
        # 对于水平方向，主要关注x方向的梯度
        gradient_mag = np.abs(gx)
    else:
        # 对于垂直方向，主要关注y方向的梯度
        gradient_mag = np.abs(gy)
    
    # 检测异常大的梯度（可能是跳变点）
    median_grad = np.median(gradient_mag)
    std_grad = np.std(gradient_mag)
    outliers = gradient_mag > median_grad + 5 * std_grad
    
    if np.any(outliers):
        print(f"全局处理：检测到 {np.sum(outliers)} 个梯度异常点，进行平滑修复...")
        # 对异常点应用局部中值滤波
        outliers_dilated = binary_dilation(outliers, iterations=2)  # 稍微扩大区域以确保覆盖问题区域
        for i in range(2, unwrapped_phase.shape[0]-2):
            for j in range(2, unwrapped_phase.shape[1]-2):
                if outliers_dilated[i, j]:
                    unwrapped_phase[i, j] = np.median(unwrapped_phase[i-2:i+3, j-2:j+3])
    
    # 【新增全局改进C】最终确保没有负值和极值点
    unwrapped_phase = np.maximum(unwrapped_phase, 0)
    
    # 检测并处理孤立的极值点
    local_max = unwrapped_phase > np.roll(unwrapped_phase, 1, axis=0)
    local_max &= unwrapped_phase > np.roll(unwrapped_phase, -1, axis=0)
    local_max &= unwrapped_phase > np.roll(unwrapped_phase, 1, axis=1)
    local_max &= unwrapped_phase > np.roll(unwrapped_phase, -1, axis=1)
    local_min = unwrapped_phase < np.roll(unwrapped_phase, 1, axis=0)
    local_min &= unwrapped_phase < np.roll(unwrapped_phase, -1, axis=0)
    local_min &= unwrapped_phase < np.roll(unwrapped_phase, 1, axis=1)
    local_min &= unwrapped_phase < np.roll(unwrapped_phase, -1, axis=1)
    
    # 找出明显偏离周围平均值的极值点
    extrema = np.zeros_like(unwrapped_phase, dtype=bool)
    for i in range(2, unwrapped_phase.shape[0]-2):
        for j in range(2, unwrapped_phase.shape[1]-2):
            if local_max[i, j] or local_min[i, j]:
                neighborhood = unwrapped_phase[i-2:i+3, j-2:j+3]
                neighborhood_mean = np.mean(neighborhood)
                neighborhood_std = np.std(neighborhood)
                if abs(unwrapped_phase[i, j] - neighborhood_mean) > 3 * neighborhood_std:
                    extrema[i, j] = True
    
    # 修复极值点
    if np.any(extrema):
        print(f"全局处理：检测到 {np.sum(extrema)} 个异常极值点，进行修复...")
        for i in range(2, unwrapped_phase.shape[0]-2):
            for j in range(2, unwrapped_phase.shape[1]-2):
                if extrema[i, j]:
                    neighborhood = unwrapped_phase[i-2:i+3, j-2:j+3]
                    # 排除当前点本身
                    neighborhood_flat = neighborhood.flatten()
                    neighborhood_flat = neighborhood_flat[neighborhood_flat != unwrapped_phase[i, j]]
                    unwrapped_phase[i, j] = np.median(neighborhood_flat)

    # 3. 基于最终的、边缘对齐的非负相位，重新计算k值，确保一致性
    if k_map is not None:
        # 注意：此时 frequencies 是升序的，最高频是最后一个
        highest_freq = frequencies[-1]
        p_high_final_wrapped = wrapped_phases[highest_freq]
        k_map = np.round((unwrapped_phase - p_high_final_wrapped) / (2 * np.pi))
        # 同样确保k_map非负
        k_map[k_map < 0] = 0
        print("最终k值已根据边缘对齐的非负相位重新计算，确保数据一致性。")
    
    # 可视化并保存最终结果
    final_unwrapped_path = os.path.join(output_dir, "unwrapped_phase_final.png")
    # 以纯图像模式保存，无边框
    visualize_unwrapped_phase(unwrapped_phase, f"最终解包裹相位 ({method})", final_unwrapped_path, show_plots, raw_save=True)
    
    final_3d_path = os.path.join(output_dir, "unwrapped_phase_final_3d.png")
    visualize_3d_surface(unwrapped_phase, f"最终解包裹相位 3D 表面 ({method})", 'viridis', final_3d_path, show_plots)

    np.save(os.path.join(output_dir, "unwrapped_phase_final.npy"), unwrapped_phase)

    # 保存由解包裹过程生成的原始k_map
    if k_map is not None:
        np.save(os.path.join(output_dir, "k_map_final.npy"), k_map)
    
    print(f"多频解包裹完成。结果已保存到 {output_dir} 目录。")
    return unwrapped_phase, k_map, final_unwrapped_path


def process_multi_frequency_dual_direction(
    h_freq_data: Dict[int, List[str]], 
    v_freq_data: Dict[int, List[str]], 
    output_dir: str = "output", 
    method: str = "multi_freq", 
    show_plots: bool = True,
    filter_kernel_size: int = 9):
    """
    同时处理水平和垂直方向的多频率N步相移图像
    
    参数:
        h_freq_data: 水平方向的频率数据字典 {frequency: [image_paths]}
        v_freq_data: 垂直方向的频率数据字典 {frequency: [image_paths]}
        output_dir: 输出目录
        method: 解包裹方法 ('multi_freq' or 'temporal')
        show_plots: 是否显示图形
        filter_kernel_size: 中值滤波器的边长
    
    返回:
        result_dict: 包含水平和垂直方向解包裹结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h_unwrapped, v_unwrapped = None, None
    h_k_map, v_k_map = None, None
    h_output_dir, v_output_dir = None, None
    
    # Process horizontal direction
    if h_freq_data:
        print("\n处理水平方向多频相移图像...")
        h_output_dir = os.path.join(output_dir, "horizontal")
        h_unwrapped, h_k_map, _ = process_multi_frequency_images(
            h_freq_data, 
            h_output_dir, 
            method, 
            show_plots,
            filter_kernel_size,
            direction='horizontal'
        )

    # Process vertical direction
    if v_freq_data:
        print("\n处理垂直方向多频相移图像...")
        v_output_dir = os.path.join(output_dir, "vertical")
        v_unwrapped, v_k_map, _ = process_multi_frequency_images(
            v_freq_data, 
            v_output_dir, 
            method, 
            show_plots,
            filter_kernel_size,
            direction='vertical'
        )
        
    # Generate combined visualizations if both are available
    if h_unwrapped is not None and v_unwrapped is not None:
        print("\n生成水平和垂直方向多频相位组合图...")
        
        combined_path = os.path.join(output_dir, "combined_phase_multi_freq.png")
        combined_rgb = generate_combined_phase(
            h_unwrapped, 
            v_unwrapped,
            "多频解包裹相位组合图", 
            combined_path,
            show_plots
        )
        np.save(os.path.join(output_dir, "combined_phase_multi_freq.npy"), combined_rgb)

    result_dict = {}
    if h_unwrapped is not None:
        result_dict["horizontal"] = {"unwrapped_phase": h_unwrapped, "k_map": h_k_map, "output_dir": h_output_dir}
    if v_unwrapped is not None:
        result_dict["vertical"] = {"unwrapped_phase": v_unwrapped, "k_map": v_k_map, "output_dir": v_output_dir}
        
    return result_dict 