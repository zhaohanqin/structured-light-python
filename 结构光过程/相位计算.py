import numpy as np
import cv2
import matplotlib.pyplot as plt
from enum import Enum
import os
from typing import List, Tuple, Dict, Any, Optional


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
                           title: str = "包裹相位图", save_path: Optional[str] = None):
    """
    可视化包裹相位图
    
    参数:
        wrapped_phase: 包裹相位图
        quality_map: 相位质量图 (可选)
        title: 图像标题
        save_path: 保存路径 (可选)
    """
    plt.figure(figsize=(12, 9))
    
    # 如果有质量图，创建一个2x1的子图
    if quality_map is not None:
        plt.subplot(2, 1, 1)
    
    # 显示包裹相位
    phase_img = plt.imshow(wrapped_phase, cmap='jet')
    plt.colorbar(phase_img, label='相位 (弧度)')
    plt.title(title)
    
    # 如果有质量图，在第二个子图中显示
    if quality_map is not None:
        plt.subplot(2, 1, 2)
        quality_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(quality_img, label='质量 (调制幅度/平均强度)')
        plt.title("相位质量图")
    
    plt.tight_layout()
    
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def process_phase_shifting_images(image_folder: str, pattern_type: str = 'vertical', 
                                algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                                frequencies: List[int] = [1, 4, 12, 48]) -> Dict[int, np.ndarray]:
    """
    处理相移图像文件夹，计算不同频率的包裹相位
    
    参数:
        image_folder: 图像文件夹路径
        pattern_type: 条纹类型 ('vertical' 或 'horizontal')
        algorithm: 相移算法
        frequencies: 使用的频率列表
    
    返回:
        wrapped_phases: 字典，键为频率，值为对应的包裹相位图
    """
    wrapped_phases = {}
    
    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"找不到图像文件夹: {image_folder}")
    
    # 处理每个频率
    for freq in frequencies:
        # 获取该频率的所有相移图像
        phase_images = []
        
        # 查找对应文件 (假设文件名格式为 "freq{freq}_shift{shift}_{pattern_type}.jpg")
        pattern = f"freq{freq}_shift*_{pattern_type}.jpg"
        image_files = sorted([f for f in os.listdir(image_folder) if f.startswith(f"freq{freq}_shift") and f.endswith(f"_{pattern_type}.jpg")])
        
        if not image_files:
            print(f"警告: 未找到频率 {freq} 的图像，跳过")
            continue
        
        print(f"处理频率 {freq} 的 {len(image_files)} 张图像...")
        
        # 加载图像
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"无法加载图像: {img_path}")
            phase_images.append(img)
        
        # 计算包裹相位
        wrapped_phase = compute_wrapped_phase(phase_images, algorithm)
        
        # 计算质量图
        quality_map = compute_phase_quality(phase_images)
        
        # 可视化当前频率的包裹相位
        save_path = os.path.join(image_folder, f"wrapped_phase_freq{freq}_{pattern_type}.png")
        visualize_wrapped_phase(wrapped_phase, quality_map, 
                               title=f"频率 {freq} 的包裹相位 ({pattern_type}方向)", 
                               save_path=save_path)
        
        # 存储包裹相位
        wrapped_phases[freq] = wrapped_phase
    
    return wrapped_phases


def simulate_phase_shifting_images(height: int = 480, width: int = 640, frequencies: List[int] = [1, 4, 12], 
                                  shifts: int = 4, noise_level: float = 0.02) -> Dict[int, List[np.ndarray]]:
    """
    生成模拟的相移图像用于测试
    
    参数:
        height: 图像高度
        width: 图像宽度
        frequencies: 使用的频率列表
        shifts: 每个频率的相移数量
        noise_level: 添加的噪声水平
    
    返回:
        images: 字典，键为频率，值为对应频率的相移图像列表
    """
    images = {}
    
    # 创建网格坐标
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # 对每个频率生成相移图像
    for freq in frequencies:
        freq_images = []
        
        # 计算相移步长
        delta = 2 * np.pi / shifts
        
        # 生成每个相移的图像
        for i in range(shifts):
            phase_shift = i * delta
            
            # 创建垂直条纹图案 (沿x轴变化)
            pattern = 128 + 127 * np.cos(2 * np.pi * freq * xx / width + phase_shift)
            
            # 添加噪声
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, pattern.shape)
                pattern = np.clip(pattern + noise, 0, 255)
            
            # 转换为8位无符号整数
            pattern = pattern.astype(np.uint8)
            
            freq_images.append(pattern)
        
        images[freq] = freq_images
    
    return images


if __name__ == "__main__":
    # 示例1: 使用模拟数据
    print("生成模拟相移图像...")
    simulated_images = simulate_phase_shifting_images(
        height=480, width=640, 
        frequencies=[1, 4, 12], 
        shifts=4, 
        noise_level=0.02
    )
    
    # 处理模拟数据
    for freq, images in simulated_images.items():
        print(f"处理频率 {freq} 的模拟图像...")
        wrapped_phase = compute_wrapped_phase(images, PhaseShiftingAlgorithm.four_step)
        quality_map = compute_phase_quality(images)
        
        # 可视化
        visualize_wrapped_phase(
            wrapped_phase, quality_map,
            title=f"模拟数据 - 频率 {freq} 的包裹相位",
            save_path=f"simulated_wrapped_phase_freq{freq}.png"
        )
    
    # 示例2: 处理实际图像 (需要提供图像文件夹)
    # 如果有实际数据，取消下面的注释并修改路径
    '''
    image_folder = "../measurement_data/camera1"
    if os.path.exists(image_folder):
        print(f"\n处理实际相移图像文件夹: {image_folder}")
        wrapped_phases = process_phase_shifting_images(
            image_folder,
            pattern_type='vertical',
            algorithm=PhaseShiftingAlgorithm.four_step,
            frequencies=[1, 4, 12, 48]
        )
    ''' 