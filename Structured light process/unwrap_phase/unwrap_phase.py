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
from skimage import morphology
import skimage.filters as filters

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


def compute_phasor_and_phase_masked(images: List[np.ndarray], mask: np.ndarray, algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> Tuple[np.ndarray, np.ndarray]:
    """
    在掩膜约束下计算包裹相位和对应的复数形式（相量）
    只在掩膜区域内进行计算，掩膜外区域设为0
    
    参数:
        images: 相移图像列表
        mask: 投影区域掩膜，True表示需要计算的区域
        algorithm: 相移算法类型
    
    返回:
        wrapped_phase: 包裹相位图，掩膜外区域为0
        phasor: 相量图，掩膜外区域为0
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")
    
    # 确保掩膜是布尔类型
    mask = mask.astype(bool)
    
    # 创建掩膜约束的图像
    masked_images = []
    for img in images:
        masked_img = img.copy().astype(np.float32)
        masked_img[~mask] = 0  # 掩膜外区域设为0
        masked_images.append(masked_img)
    
    # 使用掩膜约束的图像计算包裹相位
    float_images = masked_images
    n = len(float_images)
    
    # 统一使用 sum(I*sin) 和 sum(I*cos) 的形式
    sin_sum = 0
    cos_sum = 0

    # 假设相移是 +k*delta 的形式
    if algorithm == PhaseShiftingAlgorithm.three_step:
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
        # 三步相移算法的标准实现
        sin_sum = (np.sqrt(3)/2) * (I2 - I3)
        cos_sum = I1 - 0.5 * (I2 + I3)
    elif algorithm == PhaseShiftingAlgorithm.four_step:
        I1, I2, I3, I4 = float_images[0], float_images[1], float_images[2], float_images[3]
        sin_sum = I2 - I4
        cos_sum = I1 - I3
    elif algorithm == PhaseShiftingAlgorithm.n_step:
        delta = 2 * np.pi / n
        sin_sum = sum(float_images[i] * np.sin(i * delta) for i in range(n))
        cos_sum = sum(float_images[i] * np.cos(i * delta) for i in range(n))
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    # 计算包裹相位和相量
    wrapped_phase = np.arctan2(-sin_sum, cos_sum)
    phasor = cos_sum - 1j * sin_sum
    
    # 确保掩膜外区域为0
    wrapped_phase[~mask] = 0
    phasor[~mask] = 0
    
    return wrapped_phase, phasor


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
        # 三步相移算法的标准实现
        # 相移角度为 0, 2π/3, 4π/3
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


def compute_amplitude_from_images(images: List[np.ndarray], algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> np.ndarray:
    """
    从相移图像计算振幅（调制度），用于生成投影区域掩膜
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
    
    返回:
        amplitude: 振幅图，值越大表示投影强度越高
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")
    
    float_images = [img.astype(np.float32) for img in images]
    n = len(float_images)
    
    # 计算振幅（调制度）
    if algorithm == PhaseShiftingAlgorithm.three_step:
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
        # 三步相移的振幅计算
        sin_sum = (np.sqrt(3)/2) * (I2 - I3)
        cos_sum = I1 - 0.5 * (I2 + I3)
    elif algorithm == PhaseShiftingAlgorithm.four_step:
        I1, I2, I3, I4 = float_images[0], float_images[1], float_images[2], float_images[3]
        # 四步相移的振幅计算
        sin_sum = I2 - I4
        cos_sum = I1 - I3
    elif algorithm == PhaseShiftingAlgorithm.n_step:
        delta = 2 * np.pi / n
        sin_sum = sum(float_images[i] * np.sin(i * delta) for i in range(n))
        cos_sum = sum(float_images[i] * np.cos(i * delta) for i in range(n))
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    # 计算振幅（调制度）
    amplitude = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    
    return amplitude


def generate_projection_mask(images: List[np.ndarray], 
                           algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                           method: str = 'otsu', 
                           thresh_rel: Optional[float] = None, 
                           min_area: int = 500,
                           confidence: float = 0.5) -> np.ndarray:
    """
    基于相移图像生成投影区域掩膜，用于区分投影区域和背景
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
        method: 阈值化方法 ('otsu', 'adaptive', 'relative')
        thresh_rel: 相对阈值（仅用于relative方法）
        min_area: 最小连通区域面积
        confidence: 掩膜置信度阈值 (0.1-0.9)，控制掩膜的严格程度
    
    返回:
        mask: 二值掩膜，True表示投影区域
    """
    # 计算振幅图
    amplitude = compute_amplitude_from_images(images, algorithm)
    
    # 归一化振幅到0-255范围
    a_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-9)
    a8 = (a_norm * 255).astype(np.uint8)
    
    # 根据选择的方法进行阈值化，并应用置信度调节
    if method == 'otsu':
        th = filters.threshold_otsu(a8)
        # 使用置信度调节阈值：低置信度使阈值更宽松，高置信度使阈值更严格
        th_adjusted = th * (2 - confidence)  # confidence=0.5时保持原阈值
        mask = a8 >= th_adjusted
    elif method == 'adaptive':
        # 对于自适应阈值，调整偏移参数
        offset = int(-10 * confidence)  # confidence=0.5时offset=-5
        mask = cv2.adaptiveThreshold(a8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, offset) > 0
    elif method == 'relative':
        # 相对百分位阈值：使用置信度调节保留比例
        if thresh_rel is None: 
            thresh_rel = 0.2
        # 置信度越高，保留的像素比例越少（更严格）
        adjusted_thresh_rel = thresh_rel * (2 - confidence)
        th = np.percentile(a8, 100 * (1 - adjusted_thresh_rel))
        mask = a8 >= th
    else:
        raise ValueError(f"不支持的阈值化方法: {method}")
    
    print(f"掩膜生成参数: 方法={method}, 置信度={confidence:.2f}")
    
    # 形态学清理
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = mask.astype(np.uint8)
    
    # 保留最大连通分量（可选）
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        if len(areas) > 0:
            keep = np.argmax(areas) + 1
            mask = (labels == keep).astype(np.uint8)
    
    return mask.astype(bool)


def compute_phase_quality_masked(images: List[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """
    在掩膜约束下计算相位质量图，用于评估相位的可靠性
    只在掩膜区域内进行计算，掩膜外区域设为0
    
    参数:
        images: 相移图像列表
        mask: 投影区域掩膜，True表示需要计算的区域
    
    返回:
        quality_map: 相位质量图，掩膜外区域为0
    """
    # 确保掩膜是布尔类型
    mask = mask.astype(bool)
    
    # 创建掩膜约束的图像
    masked_images = []
    for img in images:
        masked_img = img.copy().astype(np.float32)
        masked_img[~mask] = 0  # 掩膜外区域设为0
        masked_images.append(masked_img)
    
    # 计算强度调制
    n = len(masked_images)
    
    # 计算平均强度
    avg_intensity = sum(masked_images) / n
    
    # 计算相移步长
    delta = 2 * np.pi / n
    
    # 计算正弦和余弦分量
    sin_sum = 0
    cos_sum = 0
    
    for i in range(n):
        phase_shift = i * delta
        sin_sum += masked_images[i] * np.sin(phase_shift)
        cos_sum += masked_images[i] * np.cos(phase_shift)
    
    # 计算调制幅度
    modulation = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    
    # 计算质量图 (调制幅度除以平均强度，避免除零)
    eps = 1e-10  # 小值防止除零
    quality_map = modulation / (avg_intensity + eps)
    
    # 确保掩膜外区域为0
    quality_map[~mask] = 0
    
    return quality_map


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


def quality_guided_unwrap_with_strict_mask(wrapped_phase: np.ndarray, quality_map: np.ndarray, is_three_step: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用严格掩膜约束的质量引导解包裹算法，确保只在掩膜区域内进行解包裹
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
        is_three_step: 是否为三步相移算法数据
        mask: 投影区域掩膜，True表示需要解包裹的区域
    
    返回:
        unwrapped_phase: 解包裹后的相位图，掩膜外区域为0
    """
    import heapq
    
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 如果没有提供掩膜，创建一个全True的掩膜
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        # 确保掩膜是布尔类型且尺寸匹配
        mask = mask.astype(bool)
        if mask.shape != (height, width):
            raise ValueError(f"掩膜尺寸 {mask.shape} 与相位图尺寸 {(height, width)} 不匹配")
    
    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    
    # 创建输出的解包裹相位图，初始化为0
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 在掩膜外的区域保持为0（黑色）
    # 掩膜内的区域将在后续处理中填充
    
    # 计算相位梯度，用于检测相位跳跃和增强质量图
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 增强质量图：结合原始质量图和相位梯度
    # 只在掩膜区域内计算增强质量图
    enhanced_quality = np.zeros_like(quality_map)
    if np.any(mask):
        mask_grad = phase_gradient_magnitude[mask]
        if np.max(mask_grad) > 0:
            if is_three_step:
                # 对于三步相移，更强调相位连续性
                enhanced_quality[mask] = quality_map[mask] * (1 + 0.8 * (1 - phase_gradient_magnitude[mask] / np.max(mask_grad)))
            else:
                # 对于其他相移算法，保持原有质量图
                enhanced_quality[mask] = quality_map[mask]
    
    # 创建质量排序索引，只考虑掩膜内的像素
    mask_indices = np.where(mask)
    if len(mask_indices[0]) == 0:
        print("警告：掩膜区域内没有有效像素，返回零相位图")
        return unwrapped_phase
    
    # 只对掩膜内的像素进行质量排序
    mask_quality = enhanced_quality[mask]
    quality_indices = np.argsort(-mask_quality)
    
    # 找到掩膜区域内质量最高的点作为种子点
    best_idx = quality_indices[0]
    seed_y, seed_x = mask_indices[0][best_idx], mask_indices[1][best_idx]
    
    # 标记种子点为已访问
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    
    # 使用优先队列（堆）进行质量引导的解包裹
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 定义邻域方向
    if is_three_step:
        # 对于三步相移，使用4邻域以减少噪声传播
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 对于其他相移算法，使用8邻域
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # 相位跳跃阈值
    if is_three_step:
        # 对于三步相移算法，使用更宽松的阈值
        phase_jump_threshold = np.pi * 1.8
        # 计算动态阈值：基于相位梯度的中值和标准差
        median_grad = np.median(phase_gradient_magnitude[mask])
        std_grad = np.std(phase_gradient_magnitude[mask])
        dynamic_threshold = min(np.pi * 1.8, median_grad + 2 * std_grad)
        # 使用动态阈值和固定阈值的较小值
        phase_jump_threshold = min(phase_jump_threshold, dynamic_threshold)
        print(f"三步相移算法使用动态相位跳跃阈值: {phase_jump_threshold:.3f} rad")
    else:
        # 对于其他相移算法，使用标准阈值
        phase_jump_threshold = np.pi * 1.5
    
    while heap:
        neg_quality, y, x, current_phase = heapq.heappop(heap)
        
        # 对当前点的邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            
            # 严格检查：必须在掩膜区域内
            if not mask[ny, nx]:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                # 计算包裹相位差异
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 将相位差异调整到 [-π, π] 范围
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算可能的解包裹相位值
                candidate_phase = current_phase + wrapped_diff
                
                # 检查相位跳跃是否合理
                phase_jump = abs(candidate_phase - current_phase)
                if phase_jump > phase_jump_threshold:
                    # 如果相位跳跃过大，尝试添加或减去2π的整数倍
                    k = round((candidate_phase - current_phase) / (2 * np.pi))
                    candidate_phase = current_phase + wrapped_diff - k * 2 * np.pi
                
                # 计算新的相位跳跃
                new_phase_jump = abs(candidate_phase - current_phase)
                
                # 如果相位跳跃仍然过大，跳过这个点
                if new_phase_jump > phase_jump_threshold:
                    continue
                
                # 设置解包裹相位
                unwrapped_phase[ny, nx] = candidate_phase
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到优先队列
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))
    
    # 确保掩膜外的区域为0
    unwrapped_phase[~mask] = 0
    
    # 最终检查：确保所有掩膜外的像素都为0
    assert np.all(unwrapped_phase[~mask] == 0), "掩膜外区域应该全为0"
    
    return unwrapped_phase


def quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray, is_three_step: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用改进的质量引导解包裹算法，更好地保留物体的高度调制信息。
    针对三步相移算法进行了特殊优化。
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
        is_three_step: 是否为三步相移算法数据
        mask: 投影区域掩膜，True表示需要解包裹的区域
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    import heapq
    
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 如果没有提供掩膜，创建一个全True的掩膜
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        # 确保掩膜是布尔类型且尺寸匹配
        mask = mask.astype(bool)
        if mask.shape != (height, width):
            raise ValueError(f"掩膜尺寸 {mask.shape} 与相位图尺寸 {(height, width)} 不匹配")
    
    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    
    # 创建输出的解包裹相位图
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 在掩膜外的区域设置为0，表示无效区域（后续会处理为黑色）
    unwrapped_phase[~mask] = 0
    
    # 计算相位梯度，用于检测相位跳跃和增强质量图
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 增强质量图：结合原始质量图和相位梯度
    # 只在掩膜区域内计算增强质量图
    enhanced_quality = quality_map.copy()
    if np.any(mask):
        mask_grad = phase_gradient_magnitude[mask]
        if np.max(mask_grad) > 0:
            if is_three_step:
                # 对于三步相移，更强调相位连续性
                enhanced_quality[mask] = quality_map[mask] * (
                    1 + 0.8 * (1 - phase_gradient_magnitude[mask] / np.max(mask_grad))
                )
            else:
                # 对于其他相移算法，保持原有质量图
                enhanced_quality[mask] = quality_map[mask]
    
    # 在掩膜外的区域设置质量为0
    enhanced_quality[~mask] = 0
    
    # 创建质量排序索引
    quality_flat = enhanced_quality.flatten()
    indices = np.argsort(-quality_flat)  # 按质量降序排序的索引
    
    # 找到掩膜区域内质量最高的点作为种子点
    valid_indices = []
    for idx in indices:
        y, x = np.unravel_index(idx, (height, width))
        if mask[y, x]:
            valid_indices.append(idx)
            break
    
    if not valid_indices:
        print("警告：掩膜区域内没有有效的种子点，返回原始包裹相位")
        return wrapped_phase.copy()
    
    seed_idx = valid_indices[0]
    seed_y, seed_x = np.unravel_index(seed_idx, (height, width))
    
    # 标记种子点为已访问
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    
    # 使用优先队列（堆）进行质量引导的解包裹
    # 队列元素：(负质量值, y, x, 当前相位值)
    # 使用负质量值是因为heapq是最小堆
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 定义邻域方向
    if is_three_step:
        # 对于三步相移，使用4邻域以减少噪声传播
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 对于其他相移算法，使用8邻域
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # 相位跳跃阈值
    if is_three_step:
        # 对于三步相移算法，使用更宽松的阈值
        phase_jump_threshold = np.pi * 1.8
        # 计算动态阈值：基于相位梯度的中值和标准差
        median_grad = np.median(phase_gradient_magnitude)
        std_grad = np.std(phase_gradient_magnitude)
        dynamic_threshold = min(np.pi * 1.8, median_grad + 2 * std_grad)
        # 使用动态阈值和固定阈值的较小值
        phase_jump_threshold = min(phase_jump_threshold, dynamic_threshold)
        print(f"三步相移算法使用动态相位跳跃阈值: {phase_jump_threshold:.3f} rad")
    else:
        # 对于其他相移算法，使用标准阈值
        phase_jump_threshold = np.pi * 1.5
    
    while heap:
        neg_quality, y, x, current_phase = heapq.heappop(heap)
        
        # 对当前点的8个邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                # 计算包裹相位差异
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 将相位差异调整到 [-π, π] 范围
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算可能的解包裹相位值
                candidate_phase = current_phase + wrapped_diff
                
                # 检查相位跳跃是否合理
                phase_jump = abs(candidate_phase - current_phase)
                if phase_jump > phase_jump_threshold:
                    # 如果相位跳跃过大，尝试添加或减去2π的整数倍
                    k = round((candidate_phase - current_phase) / (2 * np.pi))
                    candidate_phase = current_phase + wrapped_diff - k * 2 * np.pi
                
                # 计算新的相位跳跃
                new_phase_jump = abs(candidate_phase - current_phase)
                
                # 如果相位跳跃仍然过大，跳过这个点
                if new_phase_jump > phase_jump_threshold:
                    continue
                
                # 设置解包裹相位
                unwrapped_phase[ny, nx] = candidate_phase
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到优先队列
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))
    
    # 对于三步相移，进行后处理以提高质量
    if is_three_step:
        # 检测并修复剩余的相位跳跃
        grad_y, grad_x = np.gradient(unwrapped_phase)
        phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # 检测异常大的梯度
        median_grad = np.median(phase_gradient_magnitude)
        mad = np.median(np.abs(phase_gradient_magnitude - median_grad))
        threshold = median_grad + 3 * mad
        
        jump_mask = phase_gradient_magnitude > threshold
        
        if np.any(jump_mask):
            print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
            # 对跳跃点进行局部中值滤波
            from scipy.ndimage import median_filter
            # 创建一个临时数组，只对跳跃点进行滤波
            temp = unwrapped_phase.copy()
            temp[jump_mask] = median_filter(unwrapped_phase, size=5)[jump_mask]
            unwrapped_phase = temp
        
        # 应用轻微的高斯滤波以平滑结果
        from scipy.ndimage import gaussian_filter
        unwrapped_phase = gaussian_filter(unwrapped_phase, sigma=0.8)
    
    # 对于三步相移，进行后处理以提高质量
    if is_three_step:
        # 检测并修复剩余的相位跳跃
        grad_y, grad_x = np.gradient(unwrapped_phase)
        phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # 检测异常大的梯度
        median_grad = np.median(phase_gradient_magnitude)
        mad = np.median(np.abs(phase_gradient_magnitude - median_grad))
        threshold = median_grad + 3 * mad
        
        jump_mask = phase_gradient_magnitude > threshold
        
        if np.any(jump_mask):
            print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
            # 对跳跃点进行局部中值滤波
            from scipy.ndimage import median_filter
            # 创建一个临时数组，只对跳跃点进行滤波
            temp = unwrapped_phase.copy()
            temp[jump_mask] = median_filter(unwrapped_phase, size=5)[jump_mask]
            unwrapped_phase = temp
        
        # 应用轻微的高斯滤波以平滑结果
        from scipy.ndimage import gaussian_filter
        unwrapped_phase = gaussian_filter(unwrapped_phase, sigma=0.8)
    
    # 对于三步相移，进行后处理以提高质量
    if is_three_step:
        # 检测并修复剩余的相位跳跃
        grad_y, grad_x = np.gradient(unwrapped_phase)
        phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # 检测异常大的梯度
        median_grad = np.median(phase_gradient_magnitude)
        mad = np.median(np.abs(phase_gradient_magnitude - median_grad))
        threshold = median_grad + 3 * mad
        
        jump_mask = phase_gradient_magnitude > threshold
        
        if np.any(jump_mask):
            print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
            # 对跳跃点进行局部中值滤波
            from scipy.ndimage import median_filter
            # 创建一个临时数组，只对跳跃点进行滤波
            temp = unwrapped_phase.copy()
            temp[jump_mask] = median_filter(unwrapped_phase, size=5)[jump_mask]
            unwrapped_phase = temp
        
        # 应用轻微的高斯滤波以平滑结果
        from scipy.ndimage import gaussian_filter
        unwrapped_phase = gaussian_filter(unwrapped_phase, sigma=0.8)
    
    return unwrapped_phase


def improved_quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray, is_three_step: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用更高级的质量引导解包裹算法，专门针对保留高度调制信息进行优化。
    此版本结合了相位梯度和相位跳跃来生成更可靠的质量图，并使用稳健的相位跳跃校正方法。
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
        is_three_step: 是否为三步相移算法数据
        mask: 投影区域掩膜，True表示需要解包裹的区域

    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    import heapq
    
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 创建访问标记和解包裹相位数组
    visited = np.zeros((height, width), dtype=bool)
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 1. 计算相位梯度
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 2. 检测相位跳跃
    phase_jumps = np.zeros_like(wrapped_phase)
    for y in range(height - 1):
        for x in range(width - 1):
            diff_h = wrapped_phase[y, x+1] - wrapped_phase[y, x]
            phase_jumps[y, x] += abs(np.mod(diff_h + np.pi, 2 * np.pi) - np.pi)
            diff_v = wrapped_phase[y+1, x] - wrapped_phase[y, x]
            phase_jumps[y, x] += abs(np.mod(diff_v + np.pi, 2 * np.pi) - np.pi)

    # 3. 增强质量图: 结合原始质量、相位梯度和相位跳跃（仅掩膜内）
    max_grad = np.max(phase_gradient_magnitude)
    if mask is None:
        grad_term = (1 - phase_gradient_magnitude / max_grad) if max_grad > 0 else np.ones_like(quality_map)
    else:
        grad_term = np.zeros_like(quality_map)
        if max_grad > 0:
            grad_term[mask] = 1 - (phase_gradient_magnitude[mask] / max_grad)
    
    max_jumps = np.max(phase_jumps)
    if mask is None:
        jump_term = (1 - phase_jumps / max_jumps) if max_jumps > 0 else np.ones_like(quality_map)
    else:
        jump_term = np.zeros_like(quality_map)
        if max_jumps > 0:
            jump_term[mask] = 1 - (phase_jumps[mask] / max_jumps)

    if mask is None:
        enhanced_quality = quality_map * (1 + 0.7 * grad_term) * (1 + 0.3 * jump_term)
    else:
        enhanced_quality = np.zeros_like(quality_map)
        enhanced_quality[mask] = quality_map[mask] * (1 + 0.7 * grad_term[mask]) * (1 + 0.3 * jump_term[mask])
    
    # 4. 找到质量最高的点作为种子点
    if mask is None:
        seed_idx = np.argmax(enhanced_quality)
        seed_y, seed_x = np.unravel_index(seed_idx, (height, width))
    else:
        inds = np.argwhere(mask)
        if inds.size == 0:
            return unwrapped_phase
        sub = enhanced_quality[mask]
        best = np.argmax(sub)
        seed_y, seed_x = inds[best][0], inds[best][1]
    
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    
    # 5. 使用优先队列进行质量引导的解包裹
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 6. 设置算法参数
    if is_three_step:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4邻域
        median_grad = np.median(phase_gradient_magnitude)
        std_grad = np.std(phase_gradient_magnitude)
        phase_jump_threshold = min(np.pi * 1.8, median_grad + 2.5 * std_grad)
        print(f"三步相移算法使用动态相位跳跃阈值: {phase_jump_threshold:.3f} rad")
    else:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] # 8邻域
        phase_jump_threshold = np.pi * 1.2

    # 7. 主循环
    while heap:
        _, y, x, current_phase = heapq.heappop(heap)
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                if mask is not None and not mask[ny, nx]:
                    continue
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                k = -np.round(wrapped_diff / (2 * np.pi))
                unwrapped_diff = wrapped_diff + k * 2 * np.pi

                if abs(unwrapped_diff) > phase_jump_threshold:
                    continue
                
                candidate_phase = current_phase + unwrapped_diff
                unwrapped_phase[ny, nx] = candidate_phase
                
                visited[ny, nx] = True
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))
    
    if mask is not None:
        unwrapped_phase[~mask] = 0
    return unwrapped_phase


def robust_phase_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray, is_three_step: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用鲁棒的相位解包裹算法，专门设计用于保留物体的高度调制信息。
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    import heapq
    
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    
    # 创建输出的解包裹相位图
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 计算相位梯度
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 检测相位跳跃：计算相邻像素的相位差异
    phase_jumps = np.zeros_like(wrapped_phase)
    for y in range(height-1):
        for x in range(width-1):
            # 水平方向相位跳跃
            diff_h = wrapped_phase[y, x+1] - wrapped_phase[y, x]
            diff_h = np.mod(diff_h + np.pi, 2 * np.pi) - np.pi
            phase_jumps[y, x] += abs(diff_h)
            
            # 垂直方向相位跳跃
            diff_v = wrapped_phase[y+1, x] - wrapped_phase[y, x]
            diff_v = np.mod(diff_v + np.pi, 2 * np.pi) - np.pi
            phase_jumps[y, x] += abs(diff_v)
    
    # 改进的质量图：结合原始质量图、相位梯度和相位跳跃
    if is_three_step:
        # 对于三步相移，更强调相位连续性
        enhanced_quality = quality_map * (1 + 0.8 * (1 - phase_gradient_magnitude / np.max(phase_gradient_magnitude)))
        # 对于三步相移，减少相位跳跃的影响
        enhanced_quality = enhanced_quality * (1 + 0.1 * (1 - phase_jumps / np.max(phase_jumps)))
    else:
        # 对于其他相移算法，在相位变化大的区域给予更高的权重
        enhanced_quality = quality_map * (1 + 0.3 * phase_gradient_magnitude / np.max(phase_gradient_magnitude))
        enhanced_quality = enhanced_quality * (1 + 0.2 * phase_jumps / np.max(phase_jumps))
    
    # 在掩膜内找到质量最高的点作为种子点
    if mask is not None:
        enhanced_quality[~mask] = 0
    seed_idx = np.argmax(enhanced_quality)
    seed_y, seed_x = np.unravel_index(seed_idx, (height, width))
    
    # 标记种子点为已访问
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    
    # 使用优先队列（堆）进行质量引导的解包裹
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 定义邻域方向
    if is_three_step:
        # 对于三步相移，使用4邻域以减少噪声传播
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 对于其他相移算法，使用8邻域以提高连通性
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 设置相位跳跃阈值
    if is_three_step:
        # 对于三步相移，使用更宽松的阈值
        # 计算相位梯度的统计特性
        median_grad = np.median(phase_gradient_magnitude)
        std_grad = np.std(phase_gradient_magnitude)
        # 动态阈值：基于相位梯度的中值和标准差
        phase_jump_threshold = np.pi * 2.0 + median_grad * 0.5
        print(f"三步相移动态阈值: {phase_jump_threshold}")
    else:
        # 对于其他相移算法，使用标准阈值
        phase_jump_threshold = np.pi * 1.2  # 相位跳跃阈值，可以根据需要调整
    
    while heap:
        neg_quality, y, x, current_phase = heapq.heappop(heap)
        
        # 对当前点的4个邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                if mask is not None and not mask[ny, nx]:
                    continue
                # 计算包裹相位差异
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 将相位差异调整到 [-π, π] 范围
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算可能的解包裹相位值
                candidate_phase = current_phase + wrapped_diff
                
                # 检查相位跳跃是否合理
                phase_jump = abs(candidate_phase - current_phase)
                
                # 如果相位跳跃过大，尝试添加或减去2π的整数倍
                if phase_jump > phase_jump_threshold:
                    k = round((candidate_phase - current_phase) / (2 * np.pi))
                    candidate_phase = current_phase + wrapped_diff - k * 2 * np.pi
                    phase_jump = abs(candidate_phase - current_phase)
                
                # 如果相位跳跃仍然过大，跳过这个点
                if phase_jump > phase_jump_threshold:
                    continue
                
                # 设置解包裹相位
                unwrapped_phase[ny, nx] = candidate_phase
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到优先队列
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))
    
    if mask is not None:
        unwrapped_phase[~mask] = 0
    return unwrapped_phase


def edge_preserving_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    边缘保持的相位解包裹算法，专门用于保留物体的高度调制信息。
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图，值越大表示质量越高
        mask: 投影区域掩膜，True 表示需要解包裹的区域
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    import heapq
    
    # 图像尺寸
    height, width = wrapped_phase.shape
    
    # 掩膜准备
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.shape != (height, width):
            raise ValueError(f"掩膜尺寸 {mask.shape} 与相位图尺寸 {(height, width)} 不匹配")
    
    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    # 掩膜外标记为已访问，避免出队
    visited[~mask] = True
    
    # 创建输出的解包裹相位图
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 计算相位梯度
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 使用Canny边缘检测来识别相位边缘
    # 只对掩膜内数据进行归一化
    temp_for_edges = np.zeros_like(wrapped_phase, dtype=np.float32)
    temp_for_edges[mask] = wrapped_phase[mask].astype(np.float32)
    phase_uint8 = cv2.normalize(temp_for_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(phase_uint8, 50, 150)
    edges[~mask] = 0
    
    # 计算边缘强度
    edge_strength = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 1.0)
    edge_strength[~mask] = 0
    
    # 向量化检测相位跳跃
    diff_h = np.diff(wrapped_phase, axis=1)
    diff_h = (diff_h + np.pi) % (2 * np.pi) - np.pi
    diff_v = np.diff(wrapped_phase, axis=0)
    diff_v = (diff_v + np.pi) % (2 * np.pi) - np.pi
    phase_jumps = np.zeros_like(wrapped_phase, dtype=np.float32)
    phase_jumps[:, :-1] += np.abs(diff_h)
    phase_jumps[:-1, :] += np.abs(diff_v)
    phase_jumps[~mask] = 0
    
    # 创建增强的质量图，特别关注边缘区域
    enhanced_quality = np.zeros_like(quality_map)
    enhanced_quality[mask] = quality_map[mask]
    
    # 在边缘区域增加权重
    max_edge = float(np.max(edge_strength[mask])) if np.any(mask) else 0.0
    if max_edge > 0:
        edge_weight = np.ones_like(edge_strength)
        edge_weight[mask] = 1 + 2.0 * edge_strength[mask] / max_edge
    enhanced_quality *= edge_weight
    
    # 在相位跳跃大的区域增加权重
    max_jump = float(np.max(phase_jumps[mask])) if np.any(mask) else 0.0
    if max_jump > 0:
        jump_weight = np.ones_like(phase_jumps)
        jump_weight[mask] = 1 + 1.5 * phase_jumps[mask] / max_jump
    enhanced_quality *= jump_weight
    
    # 在相位梯度大的区域增加权重
    max_grad = float(np.max(phase_gradient_magnitude[mask])) if np.any(mask) else 0.0
    if max_grad > 0:
        grad_weight = np.ones_like(phase_gradient_magnitude)
        grad_weight[mask] = 1 + 1.0 * phase_gradient_magnitude[mask] / max_grad
    enhanced_quality *= grad_weight
    
    # 掩膜外不参与
    enhanced_quality[~mask] = 0
    
    # 创建质量排序索引
    quality_flat = enhanced_quality.flatten()
    indices = np.argsort(-quality_flat)  # 按质量降序排序的索引
    
    # 找到掩膜内质量最高的点作为种子点
    seed_y, seed_x = None, None
    for idx in indices:
        y, x = np.unravel_index(idx, (height, width))
        if mask[y, x]:
            seed_y, seed_x = y, x
            break
    if seed_y is None:
        # 掩膜为空，直接返回全零
        return unwrapped_phase
    
    # 若掩膜区域过大，使用快速回退策略（最小二乘法）
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels > 500000:
        print(f"掩膜像素 {mask_pixels} 过大，边缘保持法回退为质量引导最小二乘解包裹以避免长时间运行…")
        fast_unwrap = quality_guided_least_squares_unwrap(wrapped_phase, quality_map)
        fast_unwrap[~mask] = 0
        return fast_unwrap
    
    # 标记种子点为已访问
    visited[seed_y, seed_x] = True
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    
    # 使用优先队列（堆）进行质量引导的解包裹
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 定义4邻域方向（更稳健且更快）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 动态相位跳跃阈值
    base_threshold = np.pi * 0.9  # 稍微放宽阈值以保留更多细节
    
    processed = 0
    max_to_process = mask_pixels
    while heap:
        neg_quality, y, x, current_phase = heapq.heappop(heap)
        
        # 对当前点的4个邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
                
            # 必须在掩膜内
            if not mask[ny, nx]:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                # 计算包裹相位差异
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 将相位差异调整到 [-π, π] 范围
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算可能的解包裹相位值
                candidate_phase = current_phase + wrapped_diff
                
                # 检查相位跳跃是否合理
                phase_jump = abs(candidate_phase - current_phase)
                
                # 动态调整阈值：在边缘区域使用更宽松的阈值
                local_edge_strength = edge_strength[ny, nx]
                denom_edge = float(np.max(edge_strength[mask])) if np.any(mask) else 0.0
                local_threshold = base_threshold * (1 + 0.5 * (local_edge_strength / denom_edge if denom_edge > 0 else 0.0))
                
                # 如果相位跳跃过大，尝试添加或减去2π的整数倍
                if phase_jump > local_threshold:
                    k = round((candidate_phase - current_phase) / (2 * np.pi))
                    candidate_phase = current_phase + wrapped_diff - k * 2 * np.pi
                    phase_jump = abs(candidate_phase - current_phase)
                
                # 如果相位跳跃仍然过大，跳过这个点
                if phase_jump > local_threshold:
                    continue
                
                # 设置解包裹相位
                unwrapped_phase[ny, nx] = candidate_phase
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 添加到优先队列
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))

                processed += 1
                if processed >= max_to_process:
                    print("达到掩膜内像素上限，提前结束以避免长时间运行。")
                    heap = []
                    break
    
    # 掩膜外区域置零
    unwrapped_phase[~mask] = 0
    
    return unwrapped_phase


def quality_guided_least_squares_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    基于权重最小二乘(Weighted Least Squares, WLS)的Poisson相位解包裹。
    思路：用包裹相位的一阶差分(经[-π,π]回卷)作为目标梯度，
    以质量图作为权重，解泊松方程获得连续相位。
    该方法能很好保留物体的细节调制信息，并避免路径依赖与过度平滑。

    参数:
        wrapped_phase: 包裹相位图 (H×W)
        quality_map: 相位质量图 (H×W)，建议为[0,1]范围或任意正数

    返回:
        unwrapped_phase: 连续相位 (H×W)
    """
    from scipy.fft import dctn, idctn

    height, width = wrapped_phase.shape
    print(f"开始WLS-Poisson解包裹，图像尺寸: {height}x{width}")

    # 归一化权重，避免为0
    w = quality_map.astype(np.float64)
    w = w / (np.max(w) + 1e-12)
    w[w < 0] = 0
    w = 0.05 + 0.95 * w  # 保证最小权重，防止孤岛

    # 计算包裹相位梯度，并回卷到[-π, π]
    def wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    gx = wrap_to_pi(wrapped_phase[:, 1:] - wrapped_phase[:, :-1])
    gy = wrap_to_pi(wrapped_phase[1:, :] - wrapped_phase[:-1, :])

    # 与权重相乘（边界对齐）
    wx = 0.5 * (w[:, 1:] + w[:, :-1])
    wy = 0.5 * (w[1:, :] + w[:-1, :])

    gx_w = wx * gx
    gy_w = wy * gy

    # 计算散度 b = Dx^T(gx_w) + Dy^T(gy_w)
    b = np.zeros_like(wrapped_phase, dtype=np.float64)
    # x方向散度
    b[:, :-1] -= gx_w
    b[:, 1:] += gx_w
    # y方向散度
    b[:-1, :] -= gy_w
    b[1:, :] += gy_w

    # 用 Neumann 边界条件的DCT解 Poisson：Δphi = b
    # 参考: https://en.wikipedia.org/wiki/Discrete_Poisson_equation#Discrete_cosine_transform
    B = dctn(b, type=2, norm='ortho')

    yy = np.arange(height)
    xx = np.arange(width)
    cos_y = np.cos(np.pi * yy / height)
    cos_x = np.cos(np.pi * xx / width)
    denom = (2 - 2 * cos_x)[None, :] + (2 - 2 * cos_y)[:, None]

    # (0,0) 处的特征值为0，对应自由增益；将其强制为0，等价固定平均值
    B[0, 0] = 0.0
    denom[0, 0] = 1.0

    Phi = B / denom
    unwrapped = idctn(Phi, type=3, norm='ortho')

    # 平移使最小值为0，便于显示
    min_val = np.min(unwrapped)
    if min_val < 0:
        unwrapped = unwrapped - min_val

    print("WLS-Poisson解包裹完成")
    return unwrapped


def _apply_global_smoothing(unwrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    应用全局平滑优化，确保相位连续性
    """
    from scipy.ndimage import gaussian_filter
    
    height, width = unwrapped_phase.shape
    
    # 计算相位梯度
    grad_y, grad_x = np.gradient(unwrapped_phase)
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 检测异常大的梯度
    median_grad = np.median(gradient_magnitude)
    mad = np.median(np.abs(gradient_magnitude - median_grad))
    threshold = median_grad + 2 * mad
    
    # 对异常区域进行平滑
    abnormal_mask = gradient_magnitude > threshold
    
    if np.any(abnormal_mask):
        print(f"检测到 {np.sum(abnormal_mask)} 个异常梯度点，进行平滑处理...")
        
        # 使用自适应高斯滤波
        sigma = 1.5
        smoothed_phase = gaussian_filter(unwrapped_phase, sigma=sigma)
        
        # 根据质量进行加权混合
        quality_normalized = quality_map / (np.max(quality_map) + 1e-10)
        
        # 在异常区域使用更多平滑
        smooth_weight = np.where(abnormal_mask, 0.8, 0.3)
        unwrapped_phase = (1 - smooth_weight) * unwrapped_phase + smooth_weight * smoothed_phase
    
    # 应用全局轻微平滑
    sigma = 0.8
    smoothed_phase = gaussian_filter(unwrapped_phase, sigma=sigma)
    
    # 根据质量进行加权混合
    quality_normalized = quality_map / (np.max(quality_map) + 1e-10)
    unwrapped_phase = 0.7 * unwrapped_phase + 0.3 * smoothed_phase
    
    # 确保相位非负
    min_phase = np.min(unwrapped_phase)
    if min_phase < 0:
        unwrapped_phase = unwrapped_phase - min_phase
        print(f"相位已偏移，最小值为: {min_phase:.2f}")
    
    return unwrapped_phase


def _iterative_phase_optimization(wrapped_phase: np.ndarray, initial_unwrapped: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    使用迭代方法优化解包裹相位
    """
    from scipy.ndimage import gaussian_filter
    
    height, width = initial_unwrapped.shape
    optimized_phase = initial_unwrapped.copy()
    
    print("开始迭代优化...")
    
    # 迭代次数
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}/{max_iterations}")
        
        # 计算当前相位的梯度
        grad_y, grad_x = np.gradient(optimized_phase)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # 检测需要优化的区域（梯度变化大的区域）
        median_grad = np.median(gradient_magnitude)
        mad = np.median(np.abs(gradient_magnitude - median_grad))
        threshold = median_grad + 2 * mad
        
        # 找到需要优化的像素
        optimize_mask = gradient_magnitude > threshold
        
        if not np.any(optimize_mask):
            print("没有需要优化的像素，提前结束迭代")
            break
        
        print(f"优化 {np.sum(optimize_mask)} 个像素")
        
        # 对需要优化的像素进行局部优化
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if not optimize_mask[y, x]:
                    continue
                
                # 计算与邻域像素的相位关系
                neighbors = []
                weights = []
                
                # 检查4邻域
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        # 计算包裹相位差
                        wrapped_diff = wrapped_phase[y, x] - wrapped_phase[ny, nx]
                        wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                        
                        # 计算期望的相位值
                        expected_phase = optimized_phase[ny, nx] + wrapped_diff
                        neighbors.append(expected_phase)
                        
                        # 使用质量作为权重
                        weight = np.sqrt(quality_map[y, x] * quality_map[ny, nx])
                        weights.append(weight)
                
                if len(neighbors) > 0:
                    # 使用加权平均更新相位
                    weights = np.array(weights)
                    weights = weights / (np.sum(weights) + 1e-10)
                    
                    new_phase = np.sum(np.array(neighbors) * weights)
                    
                    # 平滑更新
                    alpha = 0.3  # 更新步长
                    optimized_phase[y, x] = (1 - alpha) * optimized_phase[y, x] + alpha * new_phase
        
        # 应用轻微的高斯平滑
        if iteration < max_iterations - 1:  # 最后一次迭代不进行平滑
            sigma = 0.5
            smoothed_phase = gaussian_filter(optimized_phase, sigma=sigma)
            
            # 根据质量进行加权混合
            quality_normalized = quality_map / (np.max(quality_map) + 1e-10)
            optimized_phase = 0.8 * optimized_phase + 0.2 * smoothed_phase
    
    print("迭代优化完成")
    return optimized_phase


def _global_least_squares_optimization(wrapped_phase: np.ndarray, initial_unwrapped: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    使用全局最小二乘法优化解包裹相位
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    
    height, width = initial_unwrapped.shape
    total_pixels = height * width
    
    print(f"构建全局最小二乘系统，像素总数: {total_pixels}")
    
    # 构建线性系统 Ax = b
    # 其中 A 是约束矩阵，x 是解包裹相位，b 是包裹相位差
    row_indices = []
    col_indices = []
    data = []
    rhs = []
    
    equation_count = 0
    
    # 为每个像素建立相位连续性约束
    for y in range(height):
        for x in range(width):
            # 与右邻域的约束
            if x < width - 1:
                # 约束: phi[y,x+1] - phi[y,x] = wrapped_diff
                wrapped_diff = wrapped_phase[y, x+1] - wrapped_phase[y, x]
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 使用质量作为权重
                weight = np.sqrt(quality_map[y, x] * quality_map[y, x+1])
                
                if weight > 0.1:  # 只考虑质量足够高的约束
                    row_indices.append(equation_count)
                    col_indices.append(y * width + x)
                    data.append(-weight)
                    
                    row_indices.append(equation_count)
                    col_indices.append(y * width + (x + 1))
                    data.append(weight)
                    
                    rhs.append(weight * wrapped_diff)
                    equation_count += 1
            
            # 与下邻域的约束
            if y < height - 1:
                # 约束: phi[y+1,x] - phi[y,x] = wrapped_diff
                wrapped_diff = wrapped_phase[y+1, x] - wrapped_phase[y, x]
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 使用质量作为权重
                weight = np.sqrt(quality_map[y, x] * quality_map[y+1, x])
                
                if weight > 0.1:  # 只考虑质量足够高的约束
                    row_indices.append(equation_count)
                    col_indices.append(y * width + x)
                    data.append(-weight)
                    
                    row_indices.append(equation_count)
                    col_indices.append((y + 1) * width + x)
                    data.append(weight)
                    
                    rhs.append(weight * wrapped_diff)
                    equation_count += 1
    
    print(f"构建了 {equation_count} 个约束方程")
    
    if equation_count == 0:
        print("没有足够的约束方程，返回初始解包裹结果")
        return initial_unwrapped
    
    # 添加初始值约束（防止解发散）
    for i in range(min(100, total_pixels)):  # 只约束前100个像素
        y = i // width
        x = i % width
        row_indices.append(equation_count)
        col_indices.append(i)
        data.append(1.0)
        rhs.append(initial_unwrapped[y, x])
        equation_count += 1
    
    # 构建稀疏矩阵
    A = csr_matrix((data, (row_indices, col_indices)), 
                  shape=(equation_count, total_pixels))
    
    # 求解最小二乘问题
    try:
        print("求解最小二乘问题...")
        # 使用正则化防止过拟合
        regularization = 0.01
        ATA = A.T @ A
        ATA += regularization * csr_matrix((np.ones(total_pixels), 
                                          (np.arange(total_pixels), np.arange(total_pixels))), 
                                         shape=(total_pixels, total_pixels))
        
        solution = spsolve(ATA, A.T @ np.array(rhs))
        optimized_phase = solution.reshape((height, width))
        
        print("全局最小二乘法优化成功")
        return optimized_phase
        
    except Exception as e:
        print(f"全局最小二乘法优化失败: {e}")
        print("返回初始解包裹结果")
        return initial_unwrapped


def _final_smoothing(unwrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    最终平滑处理，确保相位连续性
    """
    from scipy.ndimage import gaussian_filter
    
    # 计算相位梯度
    grad_y, grad_x = np.gradient(unwrapped_phase)
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 检测异常大的梯度
    median_grad = np.median(gradient_magnitude)
    mad = np.median(np.abs(gradient_magnitude - median_grad))
    threshold = median_grad + 3 * mad
    
    # 对异常区域进行平滑
    abnormal_mask = gradient_magnitude > threshold
    
    if np.any(abnormal_mask):
        print(f"检测到 {np.sum(abnormal_mask)} 个异常梯度点，进行平滑处理...")
        
        # 使用自适应高斯滤波
        sigma = 1.0
        smoothed_phase = gaussian_filter(unwrapped_phase, sigma=sigma)
        
        # 根据质量进行加权混合
        quality_normalized = quality_map / (np.max(quality_map) + 1e-10)
        
        # 在异常区域使用更多平滑
        smooth_weight = np.where(abnormal_mask, 0.7, 0.2)
        unwrapped_phase = (1 - smooth_weight) * unwrapped_phase + smooth_weight * smoothed_phase
    
    # 确保相位非负
    min_phase = np.min(unwrapped_phase)
    if min_phase < 0:
        unwrapped_phase = unwrapped_phase - min_phase
        print(f"相位已偏移，最小值为: {min_phase:.2f}")
    
    return unwrapped_phase


def _block_based_quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    分块处理大图像的质量引导解包裹
    """
    height, width = wrapped_phase.shape
    block_size = 256  # 分块大小
    
    # 计算分块数量
    blocks_y = (height + block_size - 1) // block_size
    blocks_x = (width + block_size - 1) // block_size
    
    print(f"将图像分为 {blocks_y}x{blocks_x} 个块进行处理")
    
    # 初始化结果
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 处理每个块
    for by in range(blocks_y):
        for bx in range(blocks_x):
            print(f"处理块 ({by+1}/{blocks_y}, {bx+1}/{blocks_x})")
            
            # 计算块边界
            y_start = by * block_size
            y_end = min((by + 1) * block_size, height)
            x_start = bx * block_size
            x_end = min((bx + 1) * block_size, width)
            
            # 提取块
            block_wrapped = wrapped_phase[y_start:y_end, x_start:x_end]
            block_quality = quality_map[y_start:y_end, x_start:x_end]
            
            # 对块进行解包裹
            block_unwrapped = improved_quality_guided_unwrap(block_wrapped, block_quality)
            
            # 存储结果
            unwrapped_phase[y_start:y_end, x_start:x_end] = block_unwrapped
    
    # 块间平滑处理
    print("进行块间平滑处理...")
    unwrapped_phase = _smooth_block_boundaries(unwrapped_phase, block_size)
    
    return unwrapped_phase


def _local_least_squares_refinement(wrapped_phase: np.ndarray, initial_unwrapped: np.ndarray, 
                                   quality_map: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    使用局部最小二乘法优化解包裹相位
    """
    height, width = initial_unwrapped.shape
    refined_phase = initial_unwrapped.copy()
    
    # 计算相位梯度
    grad_y, grad_x = np.gradient(initial_unwrapped)
    
    # 检测需要优化的区域（梯度变化大的区域）
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    median_grad = np.median(gradient_magnitude)
    mad = np.median(np.abs(gradient_magnitude - median_grad))
    threshold = median_grad + 2 * mad
    
    # 找到需要优化的像素
    optimize_mask = gradient_magnitude > threshold
    
    if not np.any(optimize_mask):
        return refined_phase
    
    print(f"对 {np.sum(optimize_mask)} 个像素进行局部优化")
    
    # 对需要优化的像素进行局部最小二乘法优化
    half_window = window_size // 2
    
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            if not optimize_mask[y, x]:
                continue
            
            # 提取局部窗口
            y_start = y - half_window
            y_end = y + half_window + 1
            x_start = x - half_window
            x_end = x + half_window + 1
            
            local_wrapped = wrapped_phase[y_start:y_end, x_start:x_end]
            local_unwrapped = initial_unwrapped[y_start:y_end, x_start:x_end]
            local_quality = quality_map[y_start:y_end, x_start:x_end]
            
            # 对中心像素进行优化
            center_y, center_x = half_window, half_window
            
            # 计算与邻域像素的相位关系
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < window_size and 0 <= nx < window_size:
                        neighbors.append((ny, nx))
            
            if len(neighbors) < 2:
                continue
            
            # 使用加权最小二乘法优化中心像素
            total_weight = 0
            weighted_sum = 0
            
            for ny, nx in neighbors:
                # 计算包裹相位差
                wrapped_diff = local_wrapped[center_y, center_x] - local_wrapped[ny, nx]
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 使用质量作为权重
                weight = local_quality[ny, nx]
                total_weight += weight
                weighted_sum += weight * (local_unwrapped[ny, nx] + wrapped_diff)
            
            if total_weight > 0:
                # 更新中心像素
                refined_phase[y, x] = weighted_sum / total_weight
    
    return refined_phase


def _smooth_block_boundaries(unwrapped_phase: np.ndarray, block_size: int) -> np.ndarray:
    """
    平滑分块边界
    """
    height, width = unwrapped_phase.shape
    
    # 对垂直边界进行平滑
    for x in range(block_size, width, block_size):
        if x < width - 1:
            # 获取边界两侧的像素
            left_col = unwrapped_phase[:, x-1]
            right_col = unwrapped_phase[:, x]
            
            # 计算相位偏移
            phase_diff = right_col - left_col
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # 应用平滑
            smooth_width = min(5, block_size // 4)
            for i in range(smooth_width):
                if x + i < width:
                    alpha = i / smooth_width
                    unwrapped_phase[:, x + i] = (1 - alpha) * left_col + alpha * right_col
    
    # 对水平边界进行平滑
    for y in range(block_size, height, block_size):
        if y < height - 1:
            # 获取边界两侧的像素
            top_row = unwrapped_phase[y-1, :]
            bottom_row = unwrapped_phase[y, :]
            
            # 计算相位偏移
            phase_diff = bottom_row - top_row
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # 应用平滑
            smooth_width = min(5, block_size // 4)
            for i in range(smooth_width):
                if y + i < height:
                    alpha = i / smooth_width
                    unwrapped_phase[y + i, :] = (1 - alpha) * top_row + alpha * bottom_row
    
    return unwrapped_phase


def _post_process_unwrapped_phase(unwrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    后处理解包裹相位，确保连续性和非负性
    """
    from scipy.ndimage import median_filter, gaussian_filter
    
    # 1. 检测并修复相位跳跃
    grad_y, grad_x = np.gradient(unwrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 检测异常大的梯度
    median_grad = np.median(phase_gradient_magnitude)
    mad = np.median(np.abs(phase_gradient_magnitude - median_grad))
    threshold = median_grad + 3 * mad
    
    jump_mask = phase_gradient_magnitude > threshold
    
    if np.any(jump_mask):
        print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
        
        # 对跳跃点进行局部中值滤波
        unwrapped_phase = median_filter(unwrapped_phase, size=3)
    
    # 2. 确保相位非负
    min_phase = np.min(unwrapped_phase)
    if min_phase < 0:
        unwrapped_phase = unwrapped_phase - min_phase
        print(f"相位已偏移，最小值为: {min_phase:.2f}")
    
    # 3. 最终平滑
    # 使用自适应高斯滤波进行最终平滑
    sigma = 0.5
    smoothed_phase = gaussian_filter(unwrapped_phase, sigma=sigma)
    
    # 根据质量进行加权混合
    quality_normalized = quality_map / (np.max(quality_map) + 1e-10)
    unwrapped_phase = 0.8 * unwrapped_phase + 0.2 * smoothed_phase
    
    return unwrapped_phase


def three_step_optimized_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
    """
    专门针对三步相移算法优化的相位解包裹方法
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图
    
    返回:
        unwrapped_phase: 解包裹后的相位图
    """
    print("使用三步相移专用解包裹算法...")
    
    # 1. 预处理包裹相位 - 使用更小的高斯核以保留更多细节
    wrapped_phase = cv2.GaussianBlur(wrapped_phase.astype(np.float32), (3, 3), 0.6)
    
    # 2. 计算相位梯度，用于增强质量图和动态阈值
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 计算梯度统计特性，用于动态阈值
    median_grad = np.median(phase_gradient_magnitude)
    std_grad = np.std(phase_gradient_magnitude)
    
    # 3. 增强质量图 - 结合原始质量和相位连续性
    enhanced_quality = quality_map * (1 + 0.9 * (1 - phase_gradient_magnitude / np.max(phase_gradient_magnitude)))
    
    # 4. 使用改进的质量引导解包裹算法，但使用更宽松的相位跳跃阈值
    height, width = wrapped_phase.shape
    unwrapped_phase = np.zeros_like(wrapped_phase)
    visited = np.zeros((height, width), dtype=bool)
    
    # 找到质量最高的点作为种子点
    seed_y, seed_x = np.unravel_index(np.argmax(enhanced_quality), enhanced_quality.shape)
    
    # 初始化种子点
    unwrapped_phase[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    visited[seed_y, seed_x] = True
    
    # 使用优先队列进行解包裹
    import heapq
    
    # 队列元素：(负质量值, y, x, 当前相位值)
    heap = [(-enhanced_quality[seed_y, seed_x], seed_y, seed_x, unwrapped_phase[seed_y, seed_x])]
    
    # 定义4邻域方向（不包括对角线）- 对于三步相移，4邻域更稳定
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    # 动态相位跳跃阈值 - 基于相位梯度的统计特性
    base_threshold = np.pi * 1.8  # 基础阈值
    dynamic_factor = 1.0 + 0.5 * (median_grad / np.pi)  # 动态因子
    phase_jump_threshold = base_threshold * dynamic_factor
    
    print(f"三步相移动态阈值: {phase_jump_threshold}")
    
    while heap:
        neg_quality, y, x, current_phase = heapq.heappop(heap)
        
        # 对当前点的4个邻域进行检查
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
                
            # 检查边界
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
                
            # 如果邻域点未访问
            if not visited[ny, nx]:
                # 计算包裹相位差异
                wrapped_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                
                # 将相位差异调整到 [-π, π] 范围
                wrapped_diff = np.mod(wrapped_diff + np.pi, 2 * np.pi) - np.pi
                
                # 计算可能的解包裹相位值
                candidate_phase = current_phase + wrapped_diff
                
                # 检查相位跳跃是否合理
                phase_jump = abs(candidate_phase - current_phase)
                
                # 局部动态阈值 - 根据局部梯度调整
                local_gradient = phase_gradient_magnitude[ny, nx]
                local_threshold = phase_jump_threshold
                
                # 在梯度较大的区域使用更宽松的阈值
                if local_gradient > median_grad * 1.5:
                    local_threshold = phase_jump_threshold * 1.2
                
                if phase_jump > local_threshold:
                    # 如果相位跳跃过大，尝试添加或减去2π的整数倍
                    k = round((candidate_phase - current_phase) / (2 * np.pi))
                    candidate_phase = current_phase + wrapped_diff - k * 2 * np.pi
                    phase_jump = abs(candidate_phase - current_phase)
                    
                    # 如果相位跳跃仍然过大，跳过这个点
                    if phase_jump > local_threshold:
                        continue
                
                # 设置解包裹相位
                unwrapped_phase[ny, nx] = candidate_phase
                
                # 标记为已访问
                visited[ny, nx] = True
                
                # 将邻域点加入队列 - 使用增强质量图
                heapq.heappush(heap, (-enhanced_quality[ny, nx], ny, nx, candidate_phase))
    
    # 4. 后处理 - 增强版本
    from scipy.ndimage import gaussian_filter, median_filter
    
    # 检测并修复相位跳跃
    grad_y, grad_x = np.gradient(unwrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 检测异常大的梯度
    median_grad_post = np.median(phase_gradient_magnitude)
    mad = np.median(np.abs(phase_gradient_magnitude - median_grad_post))
    threshold = median_grad_post + 3 * mad
    
    jump_mask = phase_gradient_magnitude > threshold
    
    if np.any(jump_mask):
        print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
        # 创建修复掩码的扩展版本（包括周围像素）
        from scipy.ndimage import binary_dilation
        repair_mask = binary_dilation(jump_mask, iterations=2)
        
        # 保存原始相位值
        original_phase = unwrapped_phase.copy()
        
        # 对跳跃区域进行局部中值滤波
        filtered_phase = median_filter(unwrapped_phase, size=5)
        
        # 只替换需要修复的区域
        unwrapped_phase[repair_mask] = filtered_phase[repair_mask]
        
        # 平滑修复区域的边界
        boundary_mask = binary_dilation(repair_mask, iterations=1) & ~repair_mask
        if np.any(boundary_mask):
            # 在边界区域进行加权平均
            alpha = 0.5
            unwrapped_phase[boundary_mask] = (alpha * unwrapped_phase[boundary_mask] + 
                                             (1-alpha) * filtered_phase[boundary_mask])
    
    # 自适应平滑处理
    # 1. 先进行中值滤波去除离群点
    unwrapped_phase = cv2.medianBlur(unwrapped_phase.astype(np.float32), 5)
    
    # 2. 计算局部噪声水平
    local_std = cv2.boxFilter(np.abs(unwrapped_phase - cv2.GaussianBlur(unwrapped_phase, (5,5), 0)), 
                             -1, (15,15), normalize=True)
    
    # 3. 根据局部噪声水平自适应调整高斯滤波的强度
    noise_level = np.clip(local_std / np.mean(local_std), 0.5, 2.0)
    
    # 创建自适应高斯滤波的结果
    smoothed_phase = gaussian_filter(unwrapped_phase, sigma=1.5)
    
    # 在噪声较大的区域使用更强的平滑
    high_noise_mask = noise_level > 1.2
    if np.any(high_noise_mask):
        strong_smoothed = gaussian_filter(unwrapped_phase, sigma=2.5)
        smoothed_phase[high_noise_mask] = strong_smoothed[high_noise_mask]
    
    # 4. 质量加权混合原始解包裹相位和平滑后的相位
    # 使用质量图作为权重，质量高的区域保留更多原始细节
    normalized_quality = enhanced_quality / np.max(enhanced_quality)
    weight_map = np.clip(normalized_quality, 0.3, 0.8)  # 限制权重范围
    
    # 加权混合
    unwrapped_phase = weight_map * unwrapped_phase + (1 - weight_map) * smoothed_phase
    
    # 5. 最后进行轻微的中值滤波去除可能的伪影
    unwrapped_phase = cv2.medianBlur(unwrapped_phase.astype(np.float32), 3)
    
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


def save_unwrapped_phase_raw(unwrapped_phase: np.ndarray, save_path: str, mask: Optional[np.ndarray] = None):
    """
    将解包裹后的相位保存为纯净的彩色图像，不含任何坐标轴或文字。
    掩膜外的区域将显示为纯黑色。
    
    参数:
        unwrapped_phase: 解包裹相位图
        save_path: 保存路径
        mask: 投影区域掩膜（可选）
    """
    if unwrapped_phase is None:
        print("没有可保存的解包裹相位数据")
        return

    # 创建掩膜，如果未提供则使用非零区域
    if mask is None:
        mask = unwrapped_phase != 0
    
    # 创建输出图像，初始化为黑色
    height, width = unwrapped_phase.shape
    img_color = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 只在掩膜区域内应用伪彩色映射
    if np.any(mask):
        # 创建临时相位图，掩膜外区域设为0
        temp_phase = unwrapped_phase.copy()
        temp_phase[~mask] = 0
        # 归一化相位数据到0-255范围
        if np.max(temp_phase) > 0:
            img_normalized = cv2.normalize(temp_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_normalized = np.zeros_like(unwrapped_phase, dtype=np.uint8)
        # 应用伪彩色映射
        img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        # 确保掩膜外的区域为纯黑色
        img_color[~mask] = [0, 0, 0]
    
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


def process_single_frequency_images(image_paths: List[str], output_dir: str, method: str, show_plots: bool = True, 
                                  use_mask: bool = True, mask_method: str = 'otsu', min_area: int = 500, 
                                  mask_confidence: float = 0.5) -> Optional[Dict[str, np.ndarray]]:
    """
    处理单频条纹图像，执行完整的解包裹流程
    
    参数:
        image_paths: 相移图像路径列表
        output_dir: 输出目录
        method: 解包裹方法
        show_plots: 是否显示图形
        use_mask: 是否使用投影区域掩膜
        mask_method: 掩膜生成方法 ('otsu', 'adaptive', 'relative')
        min_area: 最小连通区域面积
        mask_confidence: 掩膜置信度阈值 (0.1-0.9)
    
    返回一个包含解包裹相位和包裹相位的字典，或者在失败时返回 None
    """
    if not image_paths:
        print("错误: 未提供图像文件路径。")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    images = []
    for p in image_paths:
        try:
            # 使用 imdecode 来处理可能包含非 ASCII 字符的路径
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            images.append(img)
        except Exception as e:
            print(f"加载图像时出错 '{p}': {e}")
            images.append(None) # 添加None以触发下面的错误检查

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
    
    # 1. 首先生成投影区域掩膜（如果启用）
    mask = None
    if use_mask:
        print(f"生成投影区域掩膜，方法: {mask_method}")
        try:
            mask = generate_projection_mask(images, algorithm=algorithm, method=mask_method, min_area=min_area, confidence=mask_confidence)
            print(f"掩膜生成完成，投影区域像素数: {np.sum(mask)}")
            
            # 保存掩膜图像
            mask_path = os.path.join(output_dir, "projection_mask.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
            print(f"掩膜已保存至: {mask_path}")
        except Exception as e:
            print(f"掩膜生成失败: {e}，将不使用掩膜")
            mask = None
    else:
        # 如果不使用掩膜，创建一个全True的掩膜
        mask = np.ones((images[0].shape[0], images[0].shape[1]), dtype=bool)
        print("未使用掩膜，将在整个图像区域进行处理")
    
    # 2. 在掩膜约束下计算包裹相位
    print("在掩膜约束下计算包裹相位...")
    wrapped_phase, _ = compute_phasor_and_phase_masked(images, mask, algorithm=algorithm)
    
    # 3. 在掩膜约束下计算相位质量图
    print("在掩膜约束下计算相位质量图...")
    quality_map = compute_phase_quality_masked(images, mask)
    
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
    
    # 对于三步相移算法，需要特殊处理
    is_three_step = (num_images == 3)
    if is_three_step:
        print("检测到三步相移算法，使用特殊处理...")
        # 对于三步相移，增强质量图以提高解包裹效果
        quality_map = quality_map * 2.0  # 增加质量图增强系数
        # 对于三步相移，使用更宽松的相位跳跃阈值
        print("为三步相移设置更宽松的相位跳跃阈值...")
        # 预处理包裹相位，减少噪声（只在掩膜区域内）
        if np.any(mask):
            # 先进行高斯模糊减少噪声
            wrapped_phase = cv2.GaussianBlur(wrapped_phase.astype(np.float32), (3, 3), 0.5)
            # 然后进行中值滤波去除离群点
            wrapped_phase = cv2.medianBlur(wrapped_phase.astype(np.float32), 3)
            # 确保掩膜外区域仍为0
            wrapped_phase[~mask] = 0
    
    # 根据解包裹方法选择相应的算法
    if method == "quality_guided":
        # 使用严格掩膜约束的质量引导解包裹算法
        unwrapped_phase = quality_guided_unwrap_with_strict_mask(wrapped_phase, quality_map, is_three_step=is_three_step, mask=mask)
    elif method == "improved_quality_guided":
        # 使用改进的质量引导解包裹算法
        unwrapped_phase = improved_quality_guided_unwrap(wrapped_phase, quality_map, is_three_step=is_three_step, mask=mask)
    elif method == "robust":
        # 使用鲁棒的相位解包裹算法
        unwrapped_phase = robust_phase_unwrap(wrapped_phase, quality_map, is_three_step=is_three_step, mask=mask)
    elif method == "three_step_optimized":
        # 使用三步相移专用解包裹方法
        unwrapped_phase = three_step_optimized_unwrap(wrapped_phase, quality_map)
        # 确保掩膜外区域为0
        unwrapped_phase[~mask] = 0
    else:
        raise ValueError(f"未知的解包裹方法: {method}")
        
    # 对于三步相移，需要额外的平滑处理
    if is_three_step:
        from scipy.ndimage import gaussian_filter, median_filter
        # 增强平滑以减少三步相移的噪声
        print("对三步相移解包裹结果进行增强平滑处理...")
        # 只在掩膜区域内进行平滑处理
        if np.any(mask):
            # 检测并修复相位跳跃
            grad_y, grad_x = np.gradient(unwrapped_phase)
            phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            # 检测异常大的梯度
            median_grad = np.median(phase_gradient_magnitude[mask])
            mad = np.median(np.abs(phase_gradient_magnitude[mask] - median_grad))
            threshold = median_grad + 3 * mad
            jump_mask = phase_gradient_magnitude > threshold
            if np.any(jump_mask):
                print(f"检测到 {np.sum(jump_mask)} 个相位跳跃点，进行修复...")
                # 对跳跃点进行局部中值滤波
                unwrapped_phase = median_filter(unwrapped_phase, size=5)
            # 先进行中值滤波去除离群点
            unwrapped_phase = cv2.medianBlur(unwrapped_phase.astype(np.float32), 5)
            # 然后进行高斯平滑
            unwrapped_phase = gaussian_filter(unwrapped_phase, sigma=1.5)
            # 再次进行中值滤波去除可能的伪影
            unwrapped_phase = cv2.medianBlur(unwrapped_phase.astype(np.float32), 3)
            # 确保掩膜外区域仍为0
            unwrapped_phase[~mask] = 0

    # 后处理：平移相位值，使最小值为0（所有值为非负数）
    # 只在掩膜区域内计算最小值
    if np.any(mask):
        masked_phase = unwrapped_phase[mask]
        min_phase = np.min(masked_phase)
        if min_phase < 0:
            print(f"平移相位值：{min_phase:.2f} -> 0")
            unwrapped_phase = unwrapped_phase - min_phase
            # 确保掩膜外区域仍为0
            unwrapped_phase[~mask] = 0

    # 可视化解包裹相位
    output_path = os.path.join(output_dir, "unwrapped_phase.png")
    visualize_unwrapped_phase(unwrapped_phase,
                              title=f"解包裹相位 ({method})",
                              save_path=output_path,
                              show_plots=show_plots)

    # 另外保存一幅不带文字和坐标轴的纯净结果图
    clean_output_path = os.path.join(output_dir, "unwrapped_phase_clean.png")
    save_unwrapped_phase_raw(unwrapped_phase, clean_output_path, mask)

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


def test_all_unwrap_methods(image_paths: List[str], output_base_dir: str, show_plots: bool = True) -> Dict[str, np.ndarray]:
    """
    测试所有可用的解包裹方法，比较它们的效果
    
    参数:
        image_paths: 相移图像路径列表
        output_base_dir: 输出基础目录
        show_plots: 是否显示图形
    
    返回:
        results: 包含所有方法结果的字典
    """
    methods = ["quality_guided", "improved_quality_guided", "robust", "adaptive"]
    results = {}
    
    for method in methods:
        print(f"\n=== 测试解包裹方法: {method} ===")
        output_dir = os.path.join(output_base_dir, method)
        
        try:
            result = process_single_frequency_images(
                image_paths=image_paths,
                output_dir=output_dir,
                method=method,
                show_plots=show_plots
            )
            
            if result is not None:
                results[method] = result
                print(f"方法 {method} 处理完成")
            else:
                print(f"方法 {method} 处理失败")
                
        except Exception as e:
            print(f"方法 {method} 出现错误: {e}")
    
    return results


def compare_unwrap_methods(results: Dict[str, np.ndarray], save_path: Optional[str] = None, show_plots: bool = True):
    """
    比较不同解包裹方法的结果
    
    参数:
        results: 包含不同方法结果的字典
        save_path: 保存路径 (可选)
        show_plots: 是否显示图形
    """
    if not results:
        print("没有可比较的结果")
        return
    
    # 计算子图布局
    n_methods = len(results)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, (method, result) in enumerate(results.items()):
        if result is None or "unwrapped_phase" not in result:
            continue
            
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 显示解包裹相位
        img = plt.imshow(result["unwrapped_phase"], cmap='jet')
        plt.colorbar(img, label='Phase (rad)')
        plt.title(f"解包裹方法: {method}")
    
    plt.tight_layout()
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 只有在主线程中且需要显示时才调用plt.show()
    if show_plots:
        plt.show()
    else:
        plt.close()


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
    print("可用的解包裹方法:")
    print("  - quality_guided: 原始质量引导解包裹")
    print("  - improved_quality_guided: 改进的质量引导解包裹")
    print("  - robust: 鲁棒的相位解包裹")
    print("  - three_step_optimized: 三步相移专用解包裹")
    print("\n掩膜功能:")
    print("  - use_mask: 是否使用投影区域掩膜（默认True）")
    print("  - mask_method: 掩膜生成方法 ('otsu', 'adaptive', 'relative')")
    print("  - min_area: 最小连通区域面积（默认500）")
    print("  - mask_confidence: 掩膜置信度阈值（0.1-0.9，默认0.5）")
    print("     * 0.1-0.3: 宽松掩膜，包含更多区域但可能有噪声")
    print("     * 0.4-0.6: 推荐范围，平衡的掩膜质量")
    print("     * 0.7-0.9: 严格掩膜，只包含高置信度区域")


if __name__ == '__main__':
    main()