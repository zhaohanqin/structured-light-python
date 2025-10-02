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
from dataclasses import dataclass
import glob
from skimage import morphology
import skimage.filters as filters

# 导入掩膜生成模块
try:
    from .Mask_generation import (
        PhaseShiftingAlgorithm,
        generate_projection_mask,
        get_or_create_mask,
        save_mask_visualization,
        load_mask_from_file
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入（用于直接运行此文件）
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from Mask_generation import (
        PhaseShiftingAlgorithm,
        generate_projection_mask,
        get_or_create_mask,
        save_mask_visualization,
        load_mask_from_file
    )

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


@dataclass
class UnwrapConfig:
    """相位解包裹配置参数"""
    gradient_weight: float = 0.3  # 相位梯度权重（0-1），越小对高梯度区域惩罚越小
    base_threshold: float = np.pi * 2.5  # 基础相位跳跃阈值
    dynamic_threshold_factor: float = 3.0  # 动态阈值的标准差系数
    use_4_neighbors: bool = True  # 是否使用4邻域（False则使用8邻域）
    
    @classmethod
    def for_algorithm(cls, algorithm: PhaseShiftingAlgorithm) -> 'UnwrapConfig':
        """根据相移算法类型返回推荐配置"""
        if algorithm == PhaseShiftingAlgorithm.three_step:
            # 三步相移：信噪比较低，使用更宽松的参数
            return cls(
                gradient_weight=0.3,
                base_threshold=np.pi * 2.5,
                dynamic_threshold_factor=3.0,
                use_4_neighbors=True
            )
        else:
            # 四步及以上：信噪比较高，可以使用更严格的参数
            return cls(
                gradient_weight=0.5,
                base_threshold=np.pi * 1.8,
                dynamic_threshold_factor=2.0,
                use_4_neighbors=False
            )


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


def quality_guided_unwrap(
    wrapped_phase: np.ndarray, 
    quality_map: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    config: Optional[UnwrapConfig] = None
) -> np.ndarray:
    """
    统一的质量引导相位解包裹算法
    
    这是唯一的解包裹函数，通过配置参数适配不同的相移算法。
    包裹相位就是包裹相位（范围[-π,π]），解包裹算法不应该关心它来自几步相移。
    
    参数:
        wrapped_phase: 包裹相位图（范围 [-π, π]）
        quality_map: 相位质量图，值越大表示质量越高
        mask: 投影区域掩膜，True表示需要解包裹的区域（可选）
        config: 解包裹配置参数（可选，默认使用标准配置）
    
    返回:
        unwrapped_phase: 解包裹后的相位图，掩膜外区域为0
    """
    import heapq
    
    # 使用默认配置（如果未提供）
    if config is None:
        config = UnwrapConfig()
    
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
    
    # 计算相位梯度，用于增强质量图
    grad_y, grad_x = np.gradient(wrapped_phase)
    phase_gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 增强质量图：结合原始质量图和相位梯度
    # 使用配置的gradient_weight来控制对高梯度区域的惩罚力度
    enhanced_quality = np.zeros_like(quality_map)
    if np.any(mask):
        mask_grad = phase_gradient_magnitude[mask]
        if np.max(mask_grad) > 0:
            # 统一的质量增强策略，通过config.gradient_weight参数控制
            enhanced_quality[mask] = quality_map[mask] * (
                1 + config.gradient_weight * (1 - phase_gradient_magnitude[mask] / np.max(mask_grad))
            )
        else:
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
    
    # 定义邻域方向（根据配置）
    if config.use_4_neighbors:
        # 4邻域：更稳定，适合低信噪比
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 8邻域：更连通，适合高信噪比
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # 计算相位跳跃阈值（基于配置和数据统计）
    median_grad = np.median(phase_gradient_magnitude[mask])
    std_grad = np.std(phase_gradient_magnitude[mask])
    dynamic_threshold = median_grad + config.dynamic_threshold_factor * std_grad
    phase_jump_threshold = max(config.base_threshold, dynamic_threshold)
    print(f"解包裹相位跳跃阈值: {phase_jump_threshold:.3f} rad")
    
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


# 【已删除】以下冗余的解包裹函数已被删除：
# - improved_quality_guided_unwrap()
# - robust_phase_unwrap()
# - three_step_optimized_unwrap()
#
# 现在统一使用 quality_guided_unwrap() 函数
# 通过 UnwrapConfig 参数适配不同的相移算法


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


def save_wrapped_phase_raw(wrapped_phase: np.ndarray, save_path: str, mask: Optional[np.ndarray] = None):
    """
    将包裹相位保存为纯净的彩色图像（无坐标轴）。掩膜外区域设为黑色。
    """
    if wrapped_phase is None:
        print("没有可保存的包裹相位数据")
        return
    if mask is None:
        mask = wrapped_phase != 0
    height, width = wrapped_phase.shape
    img_color = np.zeros((height, width, 3), dtype=np.uint8)
    if np.any(mask):
        temp = wrapped_phase.copy()
        temp[~mask] = 0
        # wrap range is [-pi, pi]; normalize to 0-255
        temp_min, temp_max = -np.pi, np.pi
        temp_norm = (np.clip(temp, temp_min, temp_max) - temp_min) / (temp_max - temp_min + 1e-12)
        gray = (temp_norm * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        img_color[~mask] = [0, 0, 0]
    cv2.imwrite(save_path, img_color)
    print(f"纯净的包裹相位图已保存至: {save_path}")


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
                                  mask_confidence: float = 0.5,
                                  use_shared_mask: bool = True,
                                  shared_mask_name: str = 'mask/final_mask.png') -> Optional[Dict[str, np.ndarray]]:
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
    
    # 1. 首先生成或复用投影区域掩膜（如果启用）
    mask = None
    parent_dir = os.path.abspath(os.path.join(output_dir, os.pardir))
    mask_assets_dir = os.path.join(parent_dir, 'mask')
    
    if use_mask:
        # 计算共享掩膜路径：若 output_dir 为 .../horizontal 或 .../vertical，则共享目录为其父目录
        shared_mask_path = None
        if use_shared_mask:
            shared_mask_path = os.path.join(parent_dir, shared_mask_name)

        # 使用封装好的函数获取或创建掩膜
        try:
            mask = get_or_create_mask(
                images=images,
                algorithm=algorithm,
                use_shared_mask=use_shared_mask,
                shared_mask_path=shared_mask_path,
                mask_method=mask_method,
                thresh_rel=None,
                min_area=min_area,
                confidence=mask_confidence,
                border_trim_px=10,
                save_visualization=True,
                visualization_dir=mask_assets_dir
            )
        except Exception as e:
            print(f"掩膜生成失败: {e}，将使用全图掩膜")
            mask = np.ones((images[0].shape[0], images[0].shape[1]), dtype=bool)
    
    # 确保掩膜不为None，并且尺寸正确
    if mask is None or mask.shape != (images[0].shape[0], images[0].shape[1]):
        print("警告：掩膜无效或尺寸不匹配，使用全图掩膜")
        mask = np.ones((images[0].shape[0], images[0].shape[1]), dtype=bool)
    
    # 2. 计算包裹相位与质量图（仅用于解包裹；不再在各方向文件夹保存额外可视化图）
    print("在掩膜约束下计算包裹相位...")
    wrapped_phase, _ = compute_phasor_and_phase_masked(images, mask, algorithm=algorithm)

    print("在掩膜约束下计算相位质量图...")
    quality_map = compute_phase_quality_masked(images, mask)

    # 解包裹（统一使用quality_guided_unwrap，通过配置适配不同算法）
    print(f"开始相位解包裹...")
    
    # 根据相移算法自动选择最佳配置
    unwrap_config = UnwrapConfig.for_algorithm(algorithm)
    
    # 可选：根据method参数微调配置（保留与UI的兼容性）
    if method == "robust":
        # 鲁棒模式：更保守的参数
        unwrap_config.base_threshold = unwrap_config.base_threshold * 0.8
    
    # 调用统一的解包裹函数
    unwrapped_phase = quality_guided_unwrap(
        wrapped_phase=wrapped_phase,
        quality_map=quality_map,
        mask=mask,
        config=unwrap_config
    )

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

    # 保存一幅不带文字和坐标轴的纯净结果图（改名：unwrapped_phase.png）
    unwrapped_img_path = os.path.join(output_dir, "unwrapped_phase.png")
    save_unwrapped_phase_raw(unwrapped_phase, unwrapped_img_path, mask)

    # 保存解包裹后的npy数据
    npy_output_path = os.path.join(output_dir, "unwrapped_phase.npy")
    np.save(npy_output_path, unwrapped_phase)
    print(f"解包裹相位数据已保存至: {npy_output_path}")

    # 保存包裹相位到当前方向文件夹（PNG + NPY）
    wrapped_img_path = os.path.join(output_dir, "wrapped_phase.png")
    save_wrapped_phase_raw(wrapped_phase, wrapped_img_path, mask)
    wrapped_npy_path = os.path.join(output_dir, "wrapped_phase.npy")
    np.save(wrapped_npy_path, wrapped_phase)
    print(f"包裹相位数据已保存至: {wrapped_npy_path}")
    
    # 不再在方向文件夹中生成 wrapped/quality/3D 可视化图

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
    print("可用的解包裹方法:")
    print("  - quality_guided: 原始质量引导解包裹")
    print("  - improved_quality_guided: 改进的质量引导解包裹（推荐）")
    print("  - robust: 鲁棒的相位解包裹")
    print("  - three_step_optimized: 三步相移专用解包裹")
    print("\n掩膜功能:")
    print("  - use_mask: 是否使用投影区域掩膜（默认True）")
    print("  - mask_method: 掩膜生成方法（固定为 'otsu'）")
    print("  - min_area: 最小连通区域面积（默认500）")
    print("  - mask_confidence: 掩膜置信度阈值（0.1-0.9，对Otsu方法影响较小）")


if __name__ == '__main__':
    main()