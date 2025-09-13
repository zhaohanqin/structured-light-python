import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os

# 导入包裹相位计算模块
from 相位计算 import compute_wrapped_phase, compute_phase_quality, PhaseShiftingAlgorithm, visualize_wrapped_phase


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


def phase_to_depth(unwrapped_phase_x: np.ndarray, unwrapped_phase_y: np.ndarray, 
                  camera_matrix: np.ndarray, projector_matrix: np.ndarray, 
                  R: np.ndarray, T: np.ndarray, 
                  projector_width: int, projector_height: int) -> np.ndarray:
    """
    根据X和Y方向的解包裹相位，计算深度图
    
    参数:
        unwrapped_phase_x: X方向的解包裹相位
        unwrapped_phase_y: Y方向的解包裹相位
        camera_matrix: 相机内参矩阵
        projector_matrix: 投影仪内参矩阵
        R: 从投影仪到相机的旋转矩阵
        T: 从投影仪到相机的平移向量
        projector_width: 投影仪宽度 (像素)
        projector_height: 投影仪高度 (像素)
    
    返回:
        depth_map: 深度图
    """
    # 获取图像尺寸
    height, width = unwrapped_phase_x.shape
    
    # 计算投影仪坐标 (归一化)
    proj_x = unwrapped_phase_x * projector_width / (2 * np.pi)
    proj_y = unwrapped_phase_y * projector_height / (2 * np.pi)
    
    # 相机内参
    fx_cam = camera_matrix[0, 0]
    fy_cam = camera_matrix[1, 1]
    cx_cam = camera_matrix[0, 2]
    cy_cam = camera_matrix[1, 2]
    
    # 投影仪内参
    fx_proj = projector_matrix[0, 0]
    fy_proj = projector_matrix[1, 1]
    cx_proj = projector_matrix[0, 2]
    cy_proj = projector_matrix[1, 2]
    
    # 初始化深度图
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # 构建重投影矩阵
    P_cam = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_proj = projector_matrix @ np.hstack((R, T))
    
    # 为每个像素计算深度 (注意：这是简化版，实际实现可能需要更复杂的计算)
    for v in range(height):
        for u in range(width):
            # 相机和投影仪的像素坐标
            cam_pixel = np.array([u, v])
            proj_pixel = np.array([proj_x[v, u], proj_y[v, u]])
            
            # 构建线性方程组 (最小二乘问题)
            A = np.zeros((4, 3))
            b = np.zeros(4)
            
            # 相机投影方程
            A[0, :] = P_cam[0, :3] - u * P_cam[2, :3]
            A[1, :] = P_cam[1, :3] - v * P_cam[2, :3]
            b[0] = u * P_cam[2, 3] - P_cam[0, 3]
            b[1] = v * P_cam[2, 3] - P_cam[1, 3]
            
            # 投影仪投影方程
            A[2, :] = P_proj[0, :3] - proj_pixel[0] * P_proj[2, :3]
            A[3, :] = P_proj[1, :3] - proj_pixel[1] * P_proj[2, :3]
            b[2] = proj_pixel[0] * P_proj[2, 3] - P_proj[0, 3]
            b[3] = proj_pixel[1] * P_proj[2, 3] - P_proj[1, 3]
            
            # 解线性方程组
            X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # 获取深度值 (到相机的距离)
            depth_map[v, u] = np.linalg.norm(X)
    
    return depth_map


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


def unwrap_x_y_phases(x_wrapped_phases: Dict[int, np.ndarray], 
                    y_wrapped_phases: Dict[int, np.ndarray], 
                    frequencies: List[int],
                    x_quality_maps: Optional[Dict[int, np.ndarray]] = None,
                    y_quality_maps: Optional[Dict[int, np.ndarray]] = None,
                    method: str = 'temporal') -> Tuple[np.ndarray, np.ndarray]:
    """
    解包裹X和Y方向的相位
    
    参数:
        x_wrapped_phases: X方向不同频率的包裹相位字典
        y_wrapped_phases: Y方向不同频率的包裹相位字典
        frequencies: 频率列表
        x_quality_maps: X方向相位质量图字典 (可选)
        y_quality_maps: Y方向相位质量图字典 (可选)
        method: 解包裹方法 ('temporal', 'multi_freq', 'quality_guided')
    
    返回:
        x_unwrapped_phase: X方向解包裹相位
        y_unwrapped_phase: Y方向解包裹相位
    """
    # X方向解包裹
    if method == 'temporal':
        x_unwrapped_phase = temporal_phase_unwrap(x_wrapped_phases, frequencies, x_quality_maps)
    elif method == 'multi_freq':
        x_unwrapped_phase = multi_frequency_unwrap(x_wrapped_phases, frequencies)
    elif method == 'quality_guided':
        # 对于质量引导法，使用最高频率的相位
        highest_freq = max(frequencies)
        x_unwrapped_phase = quality_guided_unwrap(
            x_wrapped_phases[highest_freq], 
            x_quality_maps[highest_freq] if x_quality_maps else np.ones_like(x_wrapped_phases[highest_freq])
        )
    else:
        raise ValueError(f"不支持的解包裹方法: {method}")
    
    # Y方向解包裹
    if method == 'temporal':
        y_unwrapped_phase = temporal_phase_unwrap(y_wrapped_phases, frequencies, y_quality_maps)
    elif method == 'multi_freq':
        y_unwrapped_phase = multi_frequency_unwrap(y_wrapped_phases, frequencies)
    elif method == 'quality_guided':
        highest_freq = max(frequencies)
        y_unwrapped_phase = quality_guided_unwrap(
            y_wrapped_phases[highest_freq], 
            y_quality_maps[highest_freq] if y_quality_maps else np.ones_like(y_wrapped_phases[highest_freq])
        )
    
    return x_unwrapped_phase, y_unwrapped_phase


def visualize_unwrapped_phase(unwrapped_phase: np.ndarray, title: str = "解包裹相位图", 
                             save_path: Optional[str] = None) -> None:
    """
    可视化解包裹相位图
    
    参数:
        unwrapped_phase: 解包裹相位图
        title: 图像标题
        save_path: 保存路径 (可选)
    """
    plt.figure(figsize=(10, 8))
    
    # 显示解包裹相位
    img = plt.imshow(unwrapped_phase, cmap='jet')
    plt.colorbar(img, label='相位 (弧度)')
    plt.title(title)
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def simulate_wrapped_phases_for_test():
    """生成模拟的多频率包裹相位数据用于测试"""
    # 图像尺寸
    height, width = 480, 640
    
    # 模拟的频率
    frequencies = [1, 4, 16]
    
    # 创建真实相位场 (线性变化)
    x = np.linspace(0, 20 * np.pi, width)
    y = np.linspace(0, 16 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    
    # 创建圆形物体
    cx, cy = width // 2, height // 2
    r = min(width, height) // 3
    mask = ((np.arange(width) - cx)**2 + (np.arange(height)[:, np.newaxis] - cy)**2) < r**2
    
    # 创建真实相位场 (包含物体)
    true_phase_x = xx.copy()
    true_phase_y = yy.copy()
    
    # 为物体区域添加高度变化
    phase_height = 10 * np.pi
    true_phase_x[mask] += phase_height * np.exp(-((xx[mask] - x[cx])**2 + (yy[mask] - y[cy])**2) / (r**2))
    true_phase_y[mask] += phase_height * np.exp(-((xx[mask] - x[cx])**2 + (yy[mask] - y[cy])**2) / (r**2))
    
    # 创建不同频率的包裹相位
    x_wrapped_phases = {}
    y_wrapped_phases = {}
    x_quality_maps = {}
    y_quality_maps = {}
    
    for freq in frequencies:
        # 创建包裹相位
        x_wrapped = np.angle(np.exp(1j * freq * true_phase_x))
        y_wrapped = np.angle(np.exp(1j * freq * true_phase_y))
        
        # 添加噪声
        noise_level = 0.05
        x_wrapped += np.random.normal(0, noise_level, x_wrapped.shape)
        y_wrapped += np.random.normal(0, noise_level, y_wrapped.shape)
        
        # 重新包裹到 [-pi, pi]
        x_wrapped = np.angle(np.exp(1j * x_wrapped))
        y_wrapped = np.angle(np.exp(1j * y_wrapped))
        
        # 存储包裹相位
        x_wrapped_phases[freq] = x_wrapped
        y_wrapped_phases[freq] = y_wrapped
        
        # 创建简单的质量图 (边缘质量较低)
        quality_x = np.ones_like(x_wrapped)
        quality_y = np.ones_like(y_wrapped)
        
        # 边缘质量降低
        border = 20
        quality_x[:border, :] *= 0.5
        quality_x[-border:, :] *= 0.5
        quality_x[:, :border] *= 0.5
        quality_x[:, -border:] *= 0.5
        
        quality_y[:border, :] *= 0.5
        quality_y[-border:, :] *= 0.5
        quality_y[:, :border] *= 0.5
        quality_y[:, -border:] *= 0.5
        
        # 存储质量图
        x_quality_maps[freq] = quality_x
        y_quality_maps[freq] = quality_y
    
    return x_wrapped_phases, y_wrapped_phases, frequencies, x_quality_maps, y_quality_maps, true_phase_x, true_phase_y


if __name__ == "__main__":
    # 示例1: 使用模拟数据测试解包裹算法
    print("生成模拟包裹相位数据...")
    x_wrapped_phases, y_wrapped_phases, frequencies, x_quality_maps, y_quality_maps, true_phase_x, true_phase_y = simulate_wrapped_phases_for_test()
    
    # 可视化最高频率的包裹相位
    highest_freq = max(frequencies)
    visualize_wrapped_phase(
        x_wrapped_phases[highest_freq], 
        x_quality_maps[highest_freq], 
        title=f"X方向频率 {highest_freq} 的包裹相位",
        save_path="x_wrapped_phase.png"
    )
    
    visualize_wrapped_phase(
        y_wrapped_phases[highest_freq], 
        y_quality_maps[highest_freq], 
        title=f"Y方向频率 {highest_freq} 的包裹相位",
        save_path="y_wrapped_phase.png"
    )
    
    # 测试不同的解包裹方法
    methods = ['temporal', 'multi_freq', 'quality_guided']
    
    for method in methods:
        print(f"\n使用 {method} 方法进行相位解包裹...")
        
        # 解包裹X和Y方向的相位
        x_unwrapped, y_unwrapped = unwrap_x_y_phases(
            x_wrapped_phases, y_wrapped_phases, 
            frequencies, 
            x_quality_maps, y_quality_maps,
            method=method
        )
        
        # 可视化解包裹结果
        visualize_unwrapped_phase(
            x_unwrapped, 
            title=f"X方向解包裹相位 (方法: {method})",
            save_path=f"x_unwrapped_{method}.png"
        )
        
        visualize_unwrapped_phase(
            y_unwrapped, 
            title=f"Y方向解包裹相位 (方法: {method})",
            save_path=f"y_unwrapped_{method}.png"
        )
        
        # 计算与真实相位的误差
        x_error = x_unwrapped - true_phase_x
        y_error = y_unwrapped - true_phase_y
        
        # 调整相位偏移 (因为解包裹结果可能有常数相位偏移)
        x_error -= np.mean(x_error)
        y_error -= np.mean(y_error)
        
        # 计算RMS误差
        x_rms_error = np.sqrt(np.mean(x_error**2))
        y_rms_error = np.sqrt(np.mean(y_error**2))
        
        print(f"X方向RMS误差: {x_rms_error:.6f}")
        print(f"Y方向RMS误差: {y_rms_error:.6f}")
        
        # 可视化误差
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(x_error, cmap='jet')
        plt.colorbar(label='误差 (弧度)')
        plt.title(f"X方向相位误差 (RMS: {x_rms_error:.6f})")
        
        plt.subplot(122)
        plt.imshow(y_error, cmap='jet')
        plt.colorbar(label='误差 (弧度)')
        plt.title(f"Y方向相位误差 (RMS: {y_rms_error:.6f})")
        
        plt.tight_layout()
        plt.savefig(f"phase_error_{method}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n完成相位解包裹测试。") 