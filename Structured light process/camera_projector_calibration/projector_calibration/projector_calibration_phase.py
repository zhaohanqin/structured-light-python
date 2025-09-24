#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影仪标定程序 (基于相位解包裹)

该程序允许用户使用相移条纹图案和相位解包裹方法对投影仪进行标定，获取投影仪的内参矩阵、畸变系数
以及投影仪与相机的相对位置关系。此方法提供了亚像素级别的精度，适用于高精度的结构光3D扫描系统。

【像素点匹配方法】
本程序使用基于相位解包裹的高精度方法进行投影仪标定：
1. 投影仪投射相移条纹图案（水平和垂直方向）到标定板上
2. 相机拍摄投影图案的图像
3. 使用四步相移算法计算包裹相位
4. 通过质量引导的相位解包裹算法获得连续相位图
5. 通过相位值与投影仪坐标的线性映射建立亚像素级精度的对应关系
6. 使用双线性插值技术获取标定板角点处的精确相位值
7. 根据这些高精度对应关系计算投影仪参数

此方法实现复杂，但可提供亚像素级精度，适用于高精度3D重建应用。

使用方法:
1. 确保已经完成相机标定，并有相机标定结果文件
2. 投影相移条纹图案（水平和垂直方向）到标定板上
3. 用相机拍摄投影图案的图像
4. 运行程序，按照提示输入相关参数
5. 程序将自动进行相位解包裹和投影仪标定，并保存结果

作者: [Your Name]
日期: [Current Date]
"""

import os
import sys
import numpy as np
import cv2
import glob
import json
import argparse
from datetime import datetime
import matplotlib
# 设置Matplotlib不使用GUI后端，避免在某些环境下（如线程中）出现问题
try:
    matplotlib.use('Agg')
except Exception as e:
    print(f"无法设置matplotlib后端: {e}")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from enum import Enum
from typing import List, Dict, Tuple, Optional
from skimage import morphology
import skimage.filters as filters

try:
    from tqdm import tqdm  # 导入进度条库
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告：未找到tqdm库，将不显示进度条。可通过pip install tqdm安装。")

# 配置matplotlib支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    print("已配置matplotlib支持中文显示")
except ImportError:
    print("警告: matplotlib配置失败，图像中的中文可能无法正常显示")

# --- 从 unwrap_phase.py 集成的功能 ---

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
        # N步相移算法 (傅里叶变换法)
        n = len(float_images)
        if n == 0:
            raise ValueError("用于计算包裹相位的图像列表不能为空。")

        # 计算相移步长
        delta = 2 * np.pi / n
        
        # 使用向量化操作计算分子和分母
        indices = np.arange(n)
        phase_shifts = indices * delta
        
        sin_coeffs = np.sin(phase_shifts).reshape(n, 1, 1)
        cos_coeffs = np.cos(phase_shifts).reshape(n, 1, 1)

        image_stack = np.stack(float_images, axis=0)

        numerator = np.sum(image_stack * sin_coeffs, axis=0)
        denominator = np.sum(image_stack * cos_coeffs, axis=0)
            
        wrapped_phase = np.arctan2(-numerator, denominator)
    
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    return wrapped_phase


def _compute_amplitude_from_images(images: List[np.ndarray], algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> np.ndarray:
    """
    从相移图像计算振幅（调制度），用于生成投影区域掩膜
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")

    float_images = [img.astype(np.float32) for img in images]
    n = len(float_images)

    if algorithm == PhaseShiftingAlgorithm.three_step:
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
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

    amplitude = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    return amplitude


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


def generate_projection_mask(images: List[np.ndarray], 
                             algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                             method: str = 'adaptive', 
                             thresh_rel: Optional[float] = None, 
                             min_area: int = 500,
                             confidence: float = 0.5,
                             border_trim_px: int = 10) -> np.ndarray:
    """
    基于相移图像生成投影区域掩膜（True为有效投影区域），支持置信度调节。
    改进版本：增加了自适应阈值化、噪声抑制、智能边界处理等功能
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
        method: 阈值化方法 ('otsu', 'relative', 'adaptive')
        thresh_rel: 相对阈值（仅用于relative方法）
        min_area: 最小连通区域面积
        confidence: 掩膜置信度阈值 (0.1-0.9)，控制掩膜的严格程度
        border_trim_px: 边界收缩像素数
    
    返回:
        mask: 二值掩膜，True表示投影区域
    """
    if len(images) < 3:
        raise ValueError(f"至少需要3张图像，但只提供了{len(images)}张")
    
    # 计算基础特征
    imgs = np.stack([img.astype(np.float32) for img in images], axis=2)
    I_max = np.max(imgs, axis=2)
    I_min = np.min(imgs, axis=2)
    I_mean = np.mean(imgs, axis=2)
    I_std = np.std(imgs, axis=2)

    # 基本特征计算
    A = (I_max - I_min) / 2.0  # 振幅
    M = (I_max - I_min) / (I_max + I_min + 1e-9)  # 调制度
    
    # 新增特征：相位稳定性（基于标准差）
    S = I_std / (I_mean + 1e-9)  # 归一化标准差，反映相位稳定性
    
    # 噪声检测和抑制
    A, M, S = apply_noise_reduction_calibration(A, M, S, I_mean)
    
    # 归一化特征
    A_norm = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    M_norm = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    S_norm = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    I_norm = cv2.normalize(I_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 选择阈值化方法
    if method == 'adaptive':
        mask = adaptive_thresholding_calibration(A_norm, M_norm, S_norm, I_norm, confidence)
    elif method == 'otsu':
        mask = otsu_thresholding_calibration(A_norm, M_norm, I_norm)
    elif method == 'relative':
        mask = relative_thresholding_calibration(A_norm, M_norm, I_norm, thresh_rel, confidence)
    else:
        raise ValueError(f"不支持的阈值化方法: {method}")
    
    print(f"掩膜生成参数: 方法={method}, 置信度={confidence:.2f}")
    
    # 渐进式形态学处理
    mask = progressive_morphological_processing_calibration(mask, min_area)
    
    # 智能边界处理
    mask = smart_border_trimming_calibration(mask, border_trim_px)
    
    # 连通性分析和优化
    mask = connectivity_optimization_calibration(mask)
    
    return mask.astype(bool)


def apply_noise_reduction_calibration(A, M, S, I_mean):
    """应用噪声抑制 - 投影仪标定版本"""
    # 使用高斯滤波减少噪声
    A_filtered = cv2.GaussianBlur(A, (5, 5), 1.0)
    M_filtered = cv2.GaussianBlur(M, (5, 5), 1.0)
    S_filtered = cv2.GaussianBlur(S, (5, 5), 1.0)
    
    # 检测异常高的噪声区域
    noise_mask = (S > np.percentile(S, 95)) & (I_mean < np.percentile(I_mean, 20))
    
    # 在噪声区域使用滤波后的值
    A = np.where(noise_mask, A_filtered, A)
    M = np.where(noise_mask, M_filtered, M)
    S = np.where(noise_mask, S_filtered, S)
    
    return A, M, S


def adaptive_thresholding_calibration(A_norm, M_norm, S_norm, I_norm, confidence):
    """自适应阈值化方法 - 投影仪标定版本"""
    # 计算每个特征的多级阈值
    th_A_low = np.percentile(A_norm[A_norm > 0], 30)
    th_A_high = np.percentile(A_norm[A_norm > 0], 70)
    
    th_M_low = np.percentile(M_norm[M_norm > 0], 30)
    th_M_high = np.percentile(M_norm[M_norm > 0], 70)
    
    th_I = np.percentile(I_norm, 50)
    
    # 稳定性阈值（高稳定性意味着低标准差）
    th_S = np.percentile(S_norm, 60)
    
    # 多级判断
    # 高置信度区域：振幅和调制度都高
    high_confidence = (A_norm >= th_A_high) & (M_norm >= th_M_high) & (S_norm <= th_S)
    
    # 中等置信度区域：振幅或调制度高
    medium_confidence = ((A_norm >= th_A_low) | (M_norm >= th_M_low)) & (S_norm <= th_S)
    
    # 强度约束
    intensity_valid = I_norm > th_I
    
    # 根据置信度参数选择区域
    if confidence >= 0.7:
        mask = high_confidence & intensity_valid
    elif confidence >= 0.4:
        mask = (high_confidence | medium_confidence) & intensity_valid
    else:
        mask = medium_confidence & intensity_valid
    
    return mask


def otsu_thresholding_calibration(A_norm, M_norm, I_norm):
    """改进的Otsu方法 - 投影仪标定版本"""
    _, mask_A = cv2.threshold(A_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_M = cv2.threshold(M_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    th_I = np.percentile(I_norm, 50)
    mask_I = (I_norm > th_I).astype(np.uint8) * 255
    
    mask = np.logical_or(mask_A > 0, mask_M > 0)
    mask = np.logical_and(mask, mask_I > 0)
    
    return mask


def relative_thresholding_calibration(A_norm, M_norm, I_norm, thresh_rel, confidence):
    """改进的相对阈值方法 - 投影仪标定版本"""
    if thresh_rel is None:
        thresh_rel = 0.2
    
    adjusted_thresh_rel = float(thresh_rel) * (2 - float(confidence))
    th_A = np.percentile(A_norm, 100 * (1 - adjusted_thresh_rel))
    th_M = np.percentile(M_norm, 100 * (1 - adjusted_thresh_rel))
    th_I = np.percentile(I_norm, 50)
    
    mask = np.logical_or(A_norm >= th_A, M_norm >= th_M)
    mask = np.logical_and(mask, I_norm > th_I)
    
    return mask


def progressive_morphological_processing_calibration(mask, min_area):
    """渐进式形态学处理 - 投影仪标定版本"""
    mask = mask.astype(np.uint8)
    
    # 小尺度闭运算
    mask = morphology.binary_closing(mask, morphology.disk(3))
    
    # 移除小对象
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area//4)
    
    # 中等尺度闭运算
    mask = morphology.binary_closing(mask, morphology.disk(5))
    
    # 填充小孔洞
    mask = morphology.remove_small_holes(mask, area_threshold=min_area//2)
    
    # 最终移除小对象
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    
    return mask.astype(np.uint8)


def smart_border_trimming_calibration(mask, border_trim_px):
    """智能边界处理 - 投影仪标定版本"""
    if not border_trim_px or border_trim_px <= 0:
        return mask
    
    h, w = mask.shape
    bt = min(border_trim_px, min(h // 15, w // 15))  # 更保守的边界收缩
    
    if bt > 0:
        # 只在边界区域确实存在噪声时才进行收缩
        border_noise_ratio = 0.1  # 边界噪声比例阈值
        
        # 检查上下边界
        top_ratio = np.mean(mask[:bt, :])
        bottom_ratio = np.mean(mask[-bt:, :])
        left_ratio = np.mean(mask[:, :bt])
        right_ratio = np.mean(mask[:, -bt:])
        
        if top_ratio < border_noise_ratio:
            mask[:bt, :] = 0
        if bottom_ratio < border_noise_ratio:
            mask[-bt:, :] = 0
        if left_ratio < border_noise_ratio:
            mask[:, :bt] = 0
        if right_ratio < border_noise_ratio:
            mask[:, -bt:] = 0
    
    return mask


def connectivity_optimization_calibration(mask):
    """连通性优化 - 投影仪标定版本"""
    # 保留最大的几个连通分量
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    if num_labels <= 2:  # 背景 + 1个前景
        return mask
    
    # 计算每个连通分量的面积
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    
    if len(areas) == 0:
        return mask
    
    # 保留最大的连通分量，以及面积超过最大面积30%的其他分量
    max_area = max(areas)
    area_threshold = max_area * 0.3
    
    new_mask = np.zeros_like(mask)
    for i, area in enumerate(areas):
        if area >= area_threshold:
            new_mask[labels == (i + 1)] = 1
    
    return new_mask.astype(np.uint8)


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


def quality_guided_unwrap(wrapped_phase: np.ndarray, quality_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
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
    
    # 掩膜准备
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.shape != (height, width):
            raise ValueError(f"掩膜尺寸 {mask.shape} 与相位图尺寸 {(height, width)} 不匹配")

    # 创建访问标记数组
    visited = np.zeros((height, width), dtype=bool)
    
    # 创建输出的解包裹相位图
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # 掩膜外质量置零
    qm = quality_map.copy()
    qm[~mask] = 0

    # 创建质量排序索引
    quality_flat = qm.flatten()
    indices = np.argsort(-quality_flat)  # 按质量降序排序的索引
    
    # 找到质量最高的点作为种子点
    # 选择掩膜内质量最高的点作为种子
    seed_y, seed_x = None, None
    for idx in indices:
        y, x = np.unravel_index(idx, (height, width))
        if mask[y, x]:
            seed_y, seed_x = y, x
            break
    if seed_y is None:
        # 掩膜为空
        return unwrapped_phase
    
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
            
            # 必须在掩膜内
            if not mask[ny, nx]:
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
    
    # 掩膜外清零
    unwrapped_phase[~mask] = 0
    return unwrapped_phase


def visualize_phase(phase_data: np.ndarray, title: str, save_path: str, show_plots: bool, 
                    is_wrapped: bool, quality_map: Optional[np.ndarray] = None):
    """通用相位可视化函数"""
    plt.figure(figsize=(12, 9 if is_wrapped and quality_map is not None else 8))
    
    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 1)

    img = plt.imshow(phase_data, cmap='jet')
    plt.colorbar(img, label='Phase (rad)')
    plt.title(title)

    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 2)
        quality_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(quality_img, label='Quality')
        plt.title("Phase Quality Map")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()


def process_four_step_images_integrated(image_paths: List[str], output_dir: str, 
                                       show_plots: bool = True, return_quality: bool = False,
                                       mask_method: str = 'otsu', mask_confidence: float = 0.5,
                                       min_area: int = 1000):
    """
    处理四步相移图像并生成解包裹相位 (集成版本)
    
    参数:
        image_paths: 四张相移图像的路径列表
        output_dir: 输出目录
        show_plots: 是否显示图形
        return_quality: 是否返回相位质量图
    
    返回:
        unwrapped_phase: 解包裹相位图
        quality_map: (可选) 相位质量图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if any(img is None for img in images):
        raise ValueError("无法加载一张或多张图像")
    
    if len(images) != 4:
        raise ValueError(f"需要4张相移图像，但收到了{len(images)}张")

    # 先生成掩膜（可调方法与置信度）
    mask = generate_projection_mask(images, PhaseShiftingAlgorithm.four_step, method=mask_method, min_area=min_area, confidence=mask_confidence)
    cv2.imwrite(os.path.join(output_dir, "projection_mask.png"), (mask.astype(np.uint8) * 255))

    # 使用掩膜约束计算包裹相位与质量图
    wrapped_phase, _ = compute_phasor_and_phase_masked(images, mask, PhaseShiftingAlgorithm.four_step)
    quality_map = compute_phase_quality_masked(images, mask)
    
    visualize_phase(wrapped_phase, "Wrapped Phase", os.path.join(output_dir, "wrapped_phase.png"), 
                    show_plots, True, quality_map)
    
    unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map, mask=mask)
    
    visualize_phase(unwrapped_phase, "Unwrapped Phase", os.path.join(output_dir, "unwrapped_phase.png"), 
                    show_plots, False)
    
    np.save(os.path.join(output_dir, "wrapped_phase.npy"), wrapped_phase)
    np.save(os.path.join(output_dir, "unwrapped_phase.npy"), unwrapped_phase)
    np.save(os.path.join(output_dir, "quality_map.npy"), quality_map)
    
    print(f"相位处理完成。结果已保存到 {output_dir} 目录。")
    
    if return_quality:
        return unwrapped_phase, quality_map
    return unwrapped_phase

# --- 集成功能结束 ---


# 定义OpenCV显示中文文本的函数
def put_chinese_text(img, text, position, font_size=30, color=(0, 0, 255)):
    """在OpenCV图像上显示中文文本

    参数:
        img: 输入图像
        text: 要显示的文本
        position: 位置元组 (x, y)
        font_size: 字体大小
        color: 字体颜色 (B, G, R)

    返回:
        添加文本后的图像
    """
    try:
        # 深拷贝图像，以免修改原图
        img_copy = img.copy()

        # 在matplotlib中绘制文本再转换为OpenCV图像格式
        fig = Figure(figsize=(1, 1), dpi=font_size*2)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center',
                fontsize=font_size, color=(color[2]/255.0, color[1]/255.0, color[0]/255.0))
        ax.axis('off')
        fig.tight_layout(pad=0)
        canvas.draw()

        # 将matplotlib图像转换为OpenCV图像
        buf = canvas.buffer_rgba()
        text_img = np.asarray(buf)
        text_img = cv2.cvtColor(text_img, cv2.COLOR_RGBA2BGRA)

        # 创建透明蒙版
        alpha = text_img[:,:,3]/255.0
        alpha = np.repeat(alpha[:,:,np.newaxis], 3, axis=2)

        # 提取文本颜色
        text_img = text_img[:,:,:3]

        # 计算放置文本的区域
        x, y = position
        h, w = text_img.shape[:2]

        # 确保坐标不越界
        if y + h > img_copy.shape[0]:
            h = img_copy.shape[0] - y
        if x + w > img_copy.shape[1]:
            w = img_copy.shape[1] - x

        if h <= 0 or w <= 0:
            return img_copy

        # 融合文本和图像
        roi = img_copy[y:y+h, x:x+w]
        roi_result = (1 - alpha[:h,:w]) * roi + alpha[:h,:w] * text_img[:h,:w]
        img_copy[y:y+h, x:x+w] = roi_result.astype(np.uint8)

        return img_copy
    except:
        # 如果失败，则使用OpenCV的英文文本作为备选
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2)
        return img

def select_calibration_board_type():
    """
    交互式选择标定板类型

    返回:
        board_type: 选择的标定板类型字符串 ('chessboard', 'circles' 或 'ring_circles')
    """
    print("\n请选择标定板类型：")
    print("1. 棋盘格标定板 (适合精确角点检测)")
    print("2. 圆形标定板-白底黑圆 (适合光照变化较大的场景)")
    print("3. 空心圆环标定板-白底空心圆 (适合高反光或特殊光照条件)")

    while True:
        choice = input("请输入选择 [1-3，默认：1]: ").strip()
        if choice == '' or choice == '1':
            return 'chessboard'
        elif choice == '2':
            return 'circles'
        elif choice == '3':
            return 'ring_circles'
        else:
            print("无效选择，请重新输入！")

def configure_detection_parameters(board_type):
    """
    根据标定板类型配置特定的检测参数

    参数:
        board_type: 标定板类型

    返回:
        params: 包含检测参数的字典
    """
    params = {}

    if board_type == 'chessboard':
        params['criteria'] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        params['flags'] = None
    elif board_type == 'circles':
        # 白底黑圆参数
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = 50
        blob_params.maxArea = 5000
        params['detector'] = cv2.SimpleBlobDetector_create(blob_params)
        params['flags'] = cv2.CALIB_CB_SYMMETRIC_GRID
    elif board_type == 'ring_circles':
        # 白底空心圆参数
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = 50
        blob_params.maxArea = 5000
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.7
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.8
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.7
        params['detector'] = cv2.SimpleBlobDetector_create(blob_params)
        params['flags'] = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

    return params

def preprocess_image(img, board_type):
    """
    根据标定板类型优化图像预处理

    参数:
        img: 输入图像
        board_type: 标定板类型

    返回:
        处理后的灰度图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if board_type == 'chessboard':
        # 基本处理，提高对比度
        gray = cv2.equalizeHist(gray)
    elif board_type == 'circles':
        # 增强圆形检测
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif board_type == 'ring_circles':
        # 空心圆环特殊处理
        gray = cv2.bitwise_not(gray)  # 反转图像使圆环区域为暗色
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray

def assess_calibration_quality(reprojection_error, board_type):
    """
    评估标定质量并提供改进建议

    参数:
        reprojection_error: 重投影误差
        board_type: 标定板类型
    """
    print("\n【标定质量评估】")
    print(f"平均重投影误差: {reprojection_error:.4f} 像素")

    if reprojection_error < 0.5:
        quality = "极佳"
    elif reprojection_error < 1.0:
        quality = "良好"
    elif reprojection_error < 2.0:
        quality = "一般"
    else:
        quality = "较差"

    print(f"标定质量: {quality}")

    if quality == "较差":
        print("\n【改进建议】")
        if board_type == 'chessboard':
            print("- 检查棋盘格是否平整无变形")
            print("- 尝试在更均匀的光照条件下拍摄")
        elif board_type == 'circles':
            print("- 检查圆点是否清晰可见")
            print("- 尝试调整照明减少反光")
        elif board_type == 'ring_circles':
            print("- 确保圆环闭合且形状规则")
            print("- 考虑增加图像对比度")

        print("- 尝试增加标定图像数量，覆盖更多角度和位置") 

# 添加自适应质量阈值计算函数
def compute_adaptive_quality_threshold(phase_quality):
    """
    基于相位质量图自适应计算质量阈值
    
    参数:
        phase_quality: 相位质量图
        
    返回:
        quality_threshold: 自适应质量阈值
    """
    if phase_quality is None:
        return 0.3  # 默认阈值
    
    # 计算相位质量的直方图
    valid_quality = phase_quality[phase_quality > 0]
    if len(valid_quality) == 0:
        return 0.3
    
    # 使用Otsu方法自动确定阈值
    try:
        # 将质量值缩放到0-255范围
        quality_8bit = ((valid_quality - np.min(valid_quality)) / 
                       (np.max(valid_quality) - np.min(valid_quality)) * 255).astype(np.uint8)
        
        # 使用Otsu方法找到最佳阈值
        _, threshold = cv2.threshold(quality_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将阈值转换回原始范围
        adaptive_threshold = threshold / 255 * (np.max(valid_quality) - np.min(valid_quality)) + np.min(valid_quality)
        
        print(f"自适应相位质量阈值: {adaptive_threshold:.4f}")
        return float(adaptive_threshold)
    except:
        # 如果Otsu方法失败，使用固定百分比
        sorted_quality = np.sort(valid_quality)
        threshold_idx = int(len(sorted_quality) * 0.2)  # 使用20%作为阈值点
        adaptive_threshold = sorted_quality[threshold_idx]
        
        print(f"使用百分比方法确定相位质量阈值: {adaptive_threshold:.4f}")
        return float(adaptive_threshold)

# 添加错误处理类
class CalibrationError(Exception):
    """标定过程中的异常基类"""
    def __init__(self, message="投影仪标定过程中发生错误"):
        self.message = message
        super().__init__(self.message)

class PhaseUnwrappingError(CalibrationError):
    """相位解包裹过程中的异常"""
    def __init__(self, message="相位解包裹失败"):
        super().__init__(message)

class BoardDetectionError(CalibrationError):
    """标定板检测异常"""
    def __init__(self, message="标定板检测失败"):
        super().__init__(message)

class CorrespondenceError(CalibrationError):
    """对应关系建立异常"""
    def __init__(self, message="无法建立足够的点对应关系"):
        super().__init__(message)

# 改进ProjectorCalibration类，添加错误处理和日志记录
class ProjectorCalibration:
    """投影仪标定类，实现投影仪内参外参标定"""

    def __init__(self, projector_width=1280, projector_height=800, log_file=None):
        """
        初始化投影仪标定对象

        参数:
            projector_width: 投影仪分辨率宽度
            projector_height: 投影仪分辨率高度
            log_file: 日志文件路径，如果为None则不记录日志
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.projector_matrix = None  # 投影仪内参矩阵
        self.projector_dist = None    # 投影仪畸变系数
        self.R = None                 # 从投影仪到相机的旋转矩阵
        self.T = None                 # 从投影仪到相机的平移向量
        self.log_file = log_file
        
        # 初始化日志文件
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"投影仪标定日志 - 开始时间: {timestamp}\n")
                f.write(f"投影仪分辨率: {projector_width}x{projector_height}\n")
                f.write("="*50 + "\n\n")
    
    def log(self, message):
        """记录日志消息"""
        print(message)  # 始终打印到控制台
        
        # 如果提供了日志文件，也写入文件
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
    
    def load_camera_params(self, camera_params_file=None, camera_matrix=None, camera_dist=None):
        """
        加载相机标定参数

        参数:
            camera_params_file: 相机标定参数文件路径(.npy或.json)，如果为None，则尝试使用提供的矩阵和畸变系数
            camera_matrix: 手动提供的相机内参矩阵
            camera_dist: 手动提供的相机畸变系数

        返回:
            camera_matrix: 相机内参矩阵
            camera_dist: 相机畸变系数
        """
        try:
            # 如果直接提供了相机参数
            if camera_params_file is None and camera_matrix is not None and camera_dist is not None:
                self.log("使用手动提供的相机标定参数")
                return camera_matrix, camera_dist

            # 如果提供了文件路径
            if camera_params_file and os.path.exists(camera_params_file):
                if camera_params_file.endswith('.npy'):
                    # 从npy文件加载
                    data = np.load(camera_params_file, allow_pickle=True).item()
                    camera_matrix = data['camera_matrix']
                    camera_dist = data['dist_coeffs']
                elif camera_params_file.endswith('.json'):
                    # 从json文件加载
                    with open(camera_params_file, 'r') as f:
                        data = json.load(f)
                    camera_matrix = np.array(data['camera_matrix'])
                    camera_dist = np.array(data['dist_coeffs'])
                else:
                    raise ValueError("不支持的相机参数文件格式，请提供.npy或.json文件")

                self.log("成功加载相机标定参数:")
                self.log("相机内参矩阵:")
                self.log(str(camera_matrix))
                self.log("\n相机畸变系数:")
                self.log(str(camera_dist))

                return camera_matrix, camera_dist

            # 如果没有提供文件路径也没有提供直接参数，则尝试查找默认位置的文件
            default_paths = [
                "./calibration_results/camera_calibration_latest.json",  # 当前目录的结果文件夹
                "./camera_calibration_latest.json",  # 当前目录
            ]

            for path in default_paths:
                if os.path.exists(path):
                    self.log(f"找到默认相机标定文件: {path}")
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        camera_matrix = np.array(data['camera_matrix'])
                        camera_dist = np.array(data['dist_coeffs'])

                        self.log("成功加载相机标定参数:")
                        self.log("相机内参矩阵:")
                        self.log(str(camera_matrix))
                        self.log("\n相机畸变系数:")
                        self.log(str(camera_dist))

                        return camera_matrix, camera_dist
                    except Exception as e:
                        self.log(f"警告：无法从文件 {path} 加载标定参数: {e}")

            # 如果都找不到，则抛出错误
            raise ValueError("未提供相机标定参数文件或直接参数，且未找到默认位置的标定文件")
            
        except Exception as e:
            self.log(f"错误：加载相机参数失败 - {str(e)}")
            raise

    def calibrate_projector_with_camera(self, camera_matrix, camera_distortion,
                                     proj_cam_correspondences, board_points):
        """
        使用相机和投影仪对应点标定投影仪内参外参

        参数:
            camera_matrix: 相机内参矩阵
            camera_distortion: 相机畸变系数
            proj_cam_correspondences: 投影点和相机点的对应关系列表
            board_points: 棋盘格角点的世界坐标

        返回:
            projector_matrix: 投影仪内参矩阵
            projector_dist: 投影仪畸变系数
            R: 从投影仪到相机的旋转矩阵
            T: 从投影仪到相机的平移向量
            ret: 投影仪标定的重投影误差
        """
        # 提取投影仪点和相机点
        projector_points = []
        camera_points = []
        object_points = []
        
        for corr in proj_cam_correspondences:
            projector_points.append(corr['projector_point'])
            camera_points.append(corr['camera_point'])
            object_points.append(board_points[corr['board_index']])
        
        # 转换为numpy数组
        projector_points = np.array(projector_points, dtype=np.float32)
        camera_points = np.array(camera_points, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)

        # 首先从相机角度求解棋盘格的外参
        _, rvec, tvec = cv2.solvePnP(
            object_points, camera_points, camera_matrix, camera_distortion
        )
        
        # 旋转向量转换为旋转矩阵
        R_cam, _ = cv2.Rodrigues(rvec)
        T_cam = tvec

        # 构建优化问题的初始估计
        fx_proj = self.projector_width
        fy_proj = self.projector_width
        cx_proj = self.projector_width / 2
        cy_proj = self.projector_height / 2

        initial_projector_matrix = np.array([
            [fx_proj, 0, cx_proj],
            [0, fy_proj, cy_proj],
            [0, 0, 1]
        ], dtype=np.float32)

        initial_projector_dist = np.zeros(5, dtype=np.float32)

        # 使用OpenCV的标定函数进行标定
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
        ret, self.projector_matrix, self.projector_dist, rvecs_proj, tvecs_proj = cv2.calibrateCamera(
            [object_points], [projector_points],
            (self.projector_width, self.projector_height),
            initial_projector_matrix, initial_projector_dist,
            flags=flags
        )
        
        # 计算投影仪相对于相机的外参
        R_proj, _ = cv2.Rodrigues(rvecs_proj[0])
        T_proj = tvecs_proj[0]
        
        # 相对变换: R和T从投影仪到相机的变换
        self.R = R_cam @ R_proj.T
        self.T = T_cam - self.R @ T_proj

        print("\n投影仪标定结果:")
        print(f"投影仪重投影误差 (RMS): {ret:.4f} 像素")
        print("投影仪内参矩阵:")
        print(self.projector_matrix)
        print("\n投影仪畸变系数:")
        print(self.projector_dist)
        print("\n从投影仪到相机的旋转矩阵:")
        print(self.R)
        print("\n从投影仪到相机的平移向量 (mm):")
        print(self.T)

        return self.projector_matrix, self.projector_dist, self.R, self.T, ret

    def save_calibration(self, filename, include_metadata=True):
        """
        保存标定结果到文件

        参数:
            filename: 保存文件名
            include_metadata: 是否包含额外的元数据
        """
        # 生成当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保文件名有正确的扩展名
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        # 基本标定数据
        calibration_data = {
            'projector_width': self.projector_width,
            'projector_height': self.projector_height,
            'projector_matrix': self.projector_matrix.tolist() if self.projector_matrix is not None else None,
            'projector_dist': self.projector_dist.tolist() if self.projector_dist is not None else None,
            'rotation_matrix': self.R.tolist() if self.R is not None else None,
            'translation_vector': self.T.tolist() if self.T is not None else None
        }

        # 添加元数据
        if include_metadata:
            calibration_data['calibration_time'] = timestamp
            calibration_data['description'] = "投影仪标定数据 (基于相位解包裹)"
            calibration_data['method'] = "phase_unwrapping"

        # 保存为JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)

        print(f"投影仪标定数据已保存至 {filename}")

        # 同时保存一个固定名称的文件，方便其他程序直接引用
        default_json_file = os.path.join(os.path.dirname(filename), "projector_calibration_latest.json")
        with open(default_json_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)

        print(f"最新投影仪标定结果: {default_json_file} (供其他程序引用)")

        return filename, default_json_file

    def load_calibration(self, filename):
        """
        从文件加载标定结果

        参数:
            filename: 标定文件名
        """
        with open(filename, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)

        self.projector_width = calibration_data['projector_width']
        self.projector_height = calibration_data['projector_height']

        if calibration_data.get('projector_matrix'):
            self.projector_matrix = np.array(calibration_data['projector_matrix'])
        if calibration_data.get('projector_dist'):
            self.projector_dist = np.array(calibration_data['projector_dist'])
        if calibration_data.get('rotation_matrix'):
            self.R = np.array(calibration_data['rotation_matrix'])
        if calibration_data.get('translation_vector'):
            self.T = np.array(calibration_data['translation_vector'])

        print(f"从 {filename} 加载投影仪标定数据")
        return calibration_data 

def detect_calibration_board(image, board_type, chessboard_size, square_size):
    """
    在图像中检测标定板，获取角点在世界坐标系和图像坐标系中的位置
    
    参数:
        image: 输入图像
        board_type: 标定板类型
        chessboard_size: 标定板内角点数量 (宽, 高)
        square_size: 标定板方格尺寸(mm)
        
    返回:
        board_points: 标定板角点的3D坐标
        corners: 标定板角点的图像坐标
    """
    # 创建标定板对象点 (3D点)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 应用实际方格尺寸
    
    # 检测标定板
    gray = preprocess_image(image, board_type)
    detection_params = configure_detection_parameters(board_type)
    
    if board_type == 'chessboard':
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # 使用亚像素精度优化角点位置
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), detection_params['criteria'])
            return objp, corners
    
    elif board_type in ['circles', 'ring_circles']:
        ret, corners = cv2.findCirclesGrid(
            image=gray, 
            patternSize=chessboard_size, 
            flags=detection_params['flags'],
            blobDetector=detection_params.get('detector')
        )
        
        if ret:
            return objp, corners
            
    return None, None

# 使用向量化操作提取相位对应关系 - 改进版本，支持掩码约束
def extract_phase_correspondence(horizontal_unwrapped, vertical_unwrapped, 
                              projector_width, projector_height, sampling_step=5, 
                              mask=None):
    """
    从水平和垂直方向的解包裹相位中提取投影仪-相机像素对应关系，使用向量化操作提高性能
    改进版本：支持掩码约束，只在投影区域内建立对应关系
    
    参数:
        horizontal_unwrapped: 水平方向解包裹相位图
        vertical_unwrapped: 垂直方向解包裹相位图
        projector_width: 投影仪宽度
        projector_height: 投影仪高度
        sampling_step: 采样步长，减少点数提高性能
        mask: 投影区域掩膜，True表示有效投影区域（可选）
        
    返回:
        correspondences: 投影仪点和相机点的对应关系字典
    """
    if horizontal_unwrapped is None and vertical_unwrapped is None:
        raise ValueError("需要至少一个方向的解包裹相位")
    
    # 获取相位图尺寸
    height, width = horizontal_unwrapped.shape if horizontal_unwrapped is not None else vertical_unwrapped.shape
    
    # 如果没有提供掩膜，创建一个基于相位值的简单掩膜
    if mask is None:
        if horizontal_unwrapped is not None and vertical_unwrapped is not None:
            mask = (horizontal_unwrapped > 0) & (vertical_unwrapped > 0)
        elif horizontal_unwrapped is not None:
            mask = horizontal_unwrapped > 0
        else:
            mask = vertical_unwrapped > 0
        print("警告：未提供掩膜，使用基于相位值的简单掩膜")
    else:
        mask = mask.astype(bool)
        print(f"使用提供的掩膜，有效像素数: {np.sum(mask)}")
    
    # 创建结果字典
    correspondences = {}
    
    # 使用进度条来显示处理进度
    if TQDM_AVAILABLE:
        print("提取相位对应关系（掩膜约束）...")
    
    if horizontal_unwrapped is not None and vertical_unwrapped is not None:
        # 两个方向都有相位信息，使用向量化操作
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:height:sampling_step, 0:width:sampling_step]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        # 获取相应位置的相位值
        h_phases = horizontal_unwrapped[y_coords, x_coords]
        v_phases = vertical_unwrapped[y_coords, x_coords]
        
        # 获取掩膜状态
        mask_values = mask[y_coords, x_coords]
        
        # 获取相位图的最大值（只在掩膜区域内计算）
        h_max_phase = np.max(horizontal_unwrapped[mask]) if np.any(mask) else 1.0
        v_max_phase = np.max(vertical_unwrapped[mask]) if np.any(mask) else 1.0
        
        # 查找有效点（必须在掩膜内且两个方向都有有效相位值）
        valid_mask = mask_values & (h_phases > 0) & (v_phases > 0)
        
        # 计算有效点的投影仪坐标
        if np.any(valid_mask):
            valid_y_coords = y_coords[valid_mask]
            valid_x_coords = x_coords[valid_mask]
            valid_h_phases = h_phases[valid_mask]
            valid_v_phases = v_phases[valid_mask]
            
            # 计算投影仪坐标
            proj_x_coords = (valid_h_phases / h_max_phase) * projector_width
            proj_y_coords = (valid_v_phases / v_max_phase) * projector_height
            
            # 存储对应关系，使用tqdm显示进度
            items = zip(valid_y_coords, valid_x_coords, proj_x_coords, proj_y_coords)
            if TQDM_AVAILABLE:
                items = tqdm(items, total=len(valid_y_coords), desc="构建对应关系")
            
            for y, x, proj_x, proj_y in items:
                correspondences[(y, x)] = (proj_x, proj_y)
    
    elif horizontal_unwrapped is not None:
        # 只有水平方向相位信息
        y_coords, x_coords = np.mgrid[0:height:sampling_step, 0:width:sampling_step]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        h_phases = horizontal_unwrapped[y_coords, x_coords]
        mask_values = mask[y_coords, x_coords]
        h_max_phase = np.max(horizontal_unwrapped[mask]) if np.any(mask) else 1.0
        
        # 必须在掩膜内且有有效相位值
        valid_mask = mask_values & (h_phases > 0)
        if np.any(valid_mask):
            valid_y_coords = y_coords[valid_mask]
            valid_x_coords = x_coords[valid_mask]
            valid_h_phases = h_phases[valid_mask]
            
            # 计算投影仪坐标
            proj_x_coords = (valid_h_phases / h_max_phase) * projector_width
            proj_y_coords = valid_y_coords * projector_height / height  # 简单的线性映射
            
            # 存储对应关系，使用tqdm显示进度
            items = zip(valid_y_coords, valid_x_coords, proj_x_coords, proj_y_coords)
            if TQDM_AVAILABLE:
                items = tqdm(items, total=len(valid_y_coords), desc="构建对应关系")
                
            for y, x, proj_x, proj_y in items:
                correspondences[(y, x)] = (proj_x, proj_y)
    
    elif vertical_unwrapped is not None:
        # 只有垂直方向相位信息
        y_coords, x_coords = np.mgrid[0:height:sampling_step, 0:width:sampling_step]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        v_phases = vertical_unwrapped[y_coords, x_coords]
        mask_values = mask[y_coords, x_coords]
        v_max_phase = np.max(vertical_unwrapped[mask]) if np.any(mask) else 1.0
        
        # 必须在掩膜内且有有效相位值
        valid_mask = mask_values & (v_phases > 0)
        if np.any(valid_mask):
            valid_y_coords = y_coords[valid_mask]
            valid_x_coords = x_coords[valid_mask]
            valid_v_phases = v_phases[valid_mask]
            
            # 计算投影仪坐标
            proj_x_coords = valid_x_coords * projector_width / width  # 简单的线性映射
            proj_y_coords = (valid_v_phases / v_max_phase) * projector_height
            
            # 存储对应关系，使用tqdm显示进度
            items = zip(valid_y_coords, valid_x_coords, proj_x_coords, proj_y_coords)
            if TQDM_AVAILABLE:
                items = tqdm(items, total=len(valid_y_coords), desc="构建对应关系")
                
            for y, x, proj_x, proj_y in items:
                correspondences[(y, x)] = (proj_x, proj_y)
    
    print(f"从相位图中提取了 {len(correspondences)} 个像素对应关系（掩膜约束）")
    if mask is not None:
        total_mask_pixels = np.sum(mask)
        sampled_mask_pixels = total_mask_pixels // (sampling_step * sampling_step)
        coverage_ratio = len(correspondences) / max(sampled_mask_pixels, 1) * 100
        print(f"掩膜覆盖率: {coverage_ratio:.1f}% ({len(correspondences)}/{sampled_mask_pixels})")
    return correspondences

# 在process_phase_images函数中添加相位质量评估和优化
def process_phase_images(phase_images_folder, output_dir=None, visualize=True, quality_threshold=None, adaptive_threshold=True):
    """
    处理相移图案图像，进行相位解包裹
    
    参数:
        phase_images_folder: 包含相移图案图像的文件夹
        output_dir: 输出目录
        visualize: 是否可视化结果
        quality_threshold: 相位质量阈值，低于此值的相位将被视为无效。如果为None且adaptive_threshold=True，则自动计算
        adaptive_threshold: 是否使用自适应方法计算质量阈值
        
    返回:
        unwrapped_results: 包含水平和垂直方向解包裹相位和相位质量的字典
    """
    if output_dir is None:
        output_dir = os.path.join(phase_images_folder, "phase_unwrapping_results")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找相移图像
    horizontal_images = sorted(glob.glob(os.path.join(phase_images_folder, "h_phase_shift_*.png")))
    if not horizontal_images:
        horizontal_images = sorted(glob.glob(os.path.join(phase_images_folder, "horizontal", "*.png")))
    if not horizontal_images:
        horizontal_images = sorted(glob.glob(os.path.join(phase_images_folder, "h_*.jpg")))
    
    vertical_images = sorted(glob.glob(os.path.join(phase_images_folder, "v_phase_shift_*.png")))
    if not vertical_images:
        vertical_images = sorted(glob.glob(os.path.join(phase_images_folder, "vertical", "*.png")))
    if not vertical_images:
        vertical_images = sorted(glob.glob(os.path.join(phase_images_folder, "v_*.jpg")))
    
    if not horizontal_images and not vertical_images:
        raise ValueError(f"在文件夹 '{phase_images_folder}' 中未找到相移图像")
    
    print(f"找到 {len(horizontal_images)} 张水平相移图像和 {len(vertical_images)} 张垂直相移图像")
    
    # 相位解包裹处理
    unwrapped_results = {}
    
    if horizontal_images:
        print("\n处理水平方向相移图像...")
        horizontal_unwrapped, h_quality = process_four_step_images_integrated(
            image_paths=horizontal_images,
            output_dir=os.path.join(output_dir, "horizontal"),
            show_plots=visualize,
            return_quality=True
        )
        
        # 使用自适应阈值
        h_threshold = compute_adaptive_quality_threshold(h_quality) if adaptive_threshold else (quality_threshold or 0.3)
        
        # 过滤低质量相位点
        if h_quality is not None:
            mask = h_quality < h_threshold
            horizontal_unwrapped[mask] = 0
            print(f"过滤了 {np.sum(mask)} 个低质量水平相位点 (阈值: {h_threshold:.4f})")
            
        # 计算有效相位的最大值和最小值
        valid_mask = horizontal_unwrapped > 0
        if np.any(valid_mask):
            h_min_phase = np.min(horizontal_unwrapped[valid_mask])
            h_max_phase = np.max(horizontal_unwrapped[valid_mask])
            print(f"水平相位范围: {h_min_phase:.4f} - {h_max_phase:.4f}")
        else:
            h_max_phase = 2 * np.pi  # 默认值
            print("警告: 未找到有效的水平相位值")
            
        unwrapped_results["horizontal"] = horizontal_unwrapped
        unwrapped_results["horizontal_quality"] = h_quality
        unwrapped_results["horizontal_max_phase"] = h_max_phase
    
    if vertical_images:
        print("\n处理垂直方向相移图像...")
        vertical_unwrapped, v_quality = process_four_step_images_integrated(
            image_paths=vertical_images,
            output_dir=os.path.join(output_dir, "vertical"),
            show_plots=visualize,
            return_quality=True
        )
        
        # 使用自适应阈值
        v_threshold = compute_adaptive_quality_threshold(v_quality) if adaptive_threshold else (quality_threshold or 0.3)
        
        # 过滤低质量相位点
        if v_quality is not None:
            mask = v_quality < v_threshold
            vertical_unwrapped[mask] = 0
            print(f"过滤了 {np.sum(mask)} 个低质量垂直相位点 (阈值: {v_threshold:.4f})")
            
        # 计算有效相位的最大值和最小值
        valid_mask = vertical_unwrapped > 0
        if np.any(valid_mask):
            v_min_phase = np.min(vertical_unwrapped[valid_mask])
            v_max_phase = np.max(vertical_unwrapped[valid_mask])
            print(f"垂直相位范围: {v_min_phase:.4f} - {v_max_phase:.4f}")
        else:
            v_max_phase = 2 * np.pi  # 默认值
            print("警告: 未找到有效的垂直相位值")
            
        unwrapped_results["vertical"] = vertical_unwrapped
        unwrapped_results["vertical_quality"] = v_quality
        unwrapped_results["vertical_max_phase"] = v_max_phase
    
    # 如果有两个方向的相位图，生成组合视图
    if "horizontal" in unwrapped_results and "vertical" in unwrapped_results:
        print("\n生成水平和垂直方向相位组合图...")
        
        # 获取相位图
        h_unwrapped = unwrapped_results["horizontal"]
        v_unwrapped = unwrapped_results["vertical"]
        
        # 归一化
        h_norm = np.zeros_like(h_unwrapped, dtype=np.float32)
        v_norm = np.zeros_like(v_unwrapped, dtype=np.float32)
        
        h_valid = h_unwrapped > 0
        v_valid = v_unwrapped > 0
        
        if np.any(h_valid):
            h_min, h_max = np.min(h_unwrapped[h_valid]), np.max(h_unwrapped[h_valid])
            h_norm[h_valid] = (h_unwrapped[h_valid] - h_min) / (h_max - h_min)
            
        if np.any(v_valid):
            v_min, v_max = np.min(v_unwrapped[v_valid]), np.max(v_unwrapped[v_valid])
            v_norm[v_valid] = (v_unwrapped[v_valid] - v_min) / (v_max - v_min)
        
        # 创建组合图
        height, width = h_unwrapped.shape
        combined_rgb = np.zeros((height, width, 3), dtype=np.float32)
        combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
        combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
        
        # 只在两个方向都有有效相位的区域显示蓝色
        both_valid = h_valid & v_valid
        combined_rgb[:,:,2][both_valid] = (h_norm[both_valid] + v_norm[both_valid]) / 2
        
        # 保存组合图
        plt.figure(figsize=(10, 8))
        plt.imshow(combined_rgb)
        plt.title('水平和垂直方向相位组合图')
        plt.colorbar(label='归一化相位值')
        plt.savefig(os.path.join(output_dir, "combined_phase.png"), dpi=300, bbox_inches='tight')
        
        if visualize:
            plt.show()
        else:
            plt.close()
            
    return unwrapped_results

# 改进双线性插值函数，增强边界处理和鲁棒性
def bilinear_interpolate(data, y, x, default_value=0):
    """
    在数据图像上对亚像素坐标进行双线性插值，增强边界处理和鲁棒性
    
    参数:
        data: 二维数组，例如相位图
        y, x: 需要插值的亚像素坐标
        default_value: 当坐标超出图像范围或周围点无效时的默认返回值
        
    返回:
        插值后的值
    """
    height, width = data.shape
    
    # 确保坐标在图像范围内
    if x < 0 or y < 0 or x >= width-1 or y >= height-1:
        return default_value
        
    # 获取坐标的整数和小数部分
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    
    # 计算插值权重
    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1
    
    # 获取周围四个点的值
    f00 = data[y0, x0]
    f01 = data[y0, x1]
    f10 = data[y1, x0]
    f11 = data[y1, x1]
    
    # 检查周围点是否有效(对于相位图，通常0或负值表示无效)
    valid_points = 0
    valid_sum = 0
    
    if f00 > 0:
        valid_points += 1
        valid_sum += f00 * wx0 * wy0
    
    if f01 > 0:
        valid_points += 1
        valid_sum += f01 * wx1 * wy0
        
    if f10 > 0:
        valid_points += 1
        valid_sum += f10 * wx0 * wy1
        
    if f11 > 0:
        valid_points += 1
        valid_sum += f11 * wx1 * wy1
    
    # 如果周围没有有效点，返回默认值
    if valid_points == 0:
        return default_value
    
    # 如果所有点都有效，进行标准双线性插值
    if valid_points == 4:
        value = wx0 * wy0 * f00 + wx1 * wy0 * f01 + wx0 * wy1 * f10 + wx1 * wy1 * f11
    else:
        # 只使用有效点，并根据权重重新归一化
        total_weight = 0
        if f00 > 0: total_weight += wx0 * wy0
        if f01 > 0: total_weight += wx1 * wy0
        if f10 > 0: total_weight += wx0 * wy1
        if f11 > 0: total_weight += wx1 * wy1
        
        if total_weight > 0:
            value = valid_sum / total_weight
        else:
            value = default_value
    
    return value

# 添加全局优化标定方法
def calibrate_projector_with_camera_global(self, camera_matrix, camera_distortion,
                                       proj_cam_correspondences, board_points):
    """
    使用全局优化方法进行投影仪标定，同时优化相机和投影仪参数
    
    参数:
        camera_matrix: 相机内参矩阵
        camera_distortion: 相机畸变系数
        proj_cam_correspondences: 投影点和相机点的对应关系列表
        board_points: 标定板角点的世界坐标
        
    返回:
        projector_matrix: 投影仪内参矩阵
        projector_dist: 投影仪畸变系数
        R: 从投影仪到相机的旋转矩阵
        T: 从投影仪到相机的平移向量
        ret: 投影仪标定的重投影误差
    """
    # 提取投影仪点和相机点
    projector_points = []
    camera_points = []
    object_points = []
    
    for corr in proj_cam_correspondences:
        projector_points.append(corr['projector_point'])
        camera_points.append(corr['camera_point'])
        object_points.append(board_points[corr['board_index']])
    
    # 转换为numpy数组
    projector_points = np.array(projector_points, dtype=np.float32)
    camera_points = np.array(camera_points, dtype=np.float32)
    object_points = np.array(object_points, dtype=np.float32)
    
    # 初始投影仪内参估计
    fx_proj = self.projector_width
    fy_proj = self.projector_width
    cx_proj = self.projector_width / 2
    cy_proj = self.projector_height / 2
    
    initial_projector_matrix = np.array([
        [fx_proj, 0, cx_proj],
        [0, fy_proj, cy_proj],
        [0, 0, 1]
    ], dtype=np.float32)
    
    initial_projector_dist = np.zeros(5, dtype=np.float32)
    
    # 首先使用PnP求解相机的位姿
    _, rvec_cam, tvec_cam = cv2.solvePnP(
        object_points, camera_points, camera_matrix, camera_distortion
    )
    
    # 然后使用另一个PnP求解投影仪的位姿
    _, rvec_proj, tvec_proj = cv2.solvePnP(
        object_points, projector_points, initial_projector_matrix, initial_projector_dist
    )
    
    # 将初始估计组合为一个参数向量
    initial_params = np.zeros(12 + 5 + 5, dtype=np.float32)
    
    # 投影仪内参 (5个): fx, fy, cx, cy, skew
    initial_params[0:5] = [
        initial_projector_matrix[0, 0],  # fx_proj
        initial_projector_matrix[1, 1],  # fy_proj
        initial_projector_matrix[0, 2],  # cx_proj
        initial_projector_matrix[1, 2],  # cy_proj
        0.0                             # skew
    ]
    
    # 投影仪畸变系数 (5个)
    initial_params[5:10] = initial_projector_dist.flatten()
    
    # 投影仪位姿 (6个): 旋转向量(3) + 平移向量(3)
    initial_params[10:13] = rvec_proj.flatten()
    initial_params[13:16] = tvec_proj.flatten()
    
    # 相机位姿 (6个): 旋转向量(3) + 平移向量(3)
    initial_params[16:19] = rvec_cam.flatten()
    initial_params[19:22] = tvec_cam.flatten()
    
    # 定义全局优化的目标函数
    def objective_function(params):
        # 解包参数
        fx_proj, fy_proj, cx_proj, cy_proj, skew = params[0:5]
        proj_dist = params[5:10]
        rvec_proj = params[10:13].reshape(3, 1)
        tvec_proj = params[13:16].reshape(3, 1)
        rvec_cam = params[16:19].reshape(3, 1)
        tvec_cam = params[19:22].reshape(3, 1)
        
        # 重建投影仪内参矩阵
        proj_matrix = np.array([
            [fx_proj, skew, cx_proj],
            [0, fy_proj, cy_proj],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 计算重投影误差
        total_error = 0.0
        
        # 计算相机重投影误差
        camera_proj_pts, _ = cv2.projectPoints(
            object_points, rvec_cam, tvec_cam, camera_matrix, camera_distortion
        )
        camera_error = np.sum((camera_proj_pts.reshape(-1, 2) - camera_points) ** 2)
        
        # 计算投影仪重投影误差
        proj_proj_pts, _ = cv2.projectPoints(
            object_points, rvec_proj, tvec_proj, proj_matrix, proj_dist
        )
        proj_error = np.sum((proj_proj_pts.reshape(-1, 2) - projector_points) ** 2)
        
        # 总误差为相机和投影仪误差之和
        total_error = camera_error + proj_error
        return total_error
    
    # 使用scipy优化
    try:
        from scipy import optimize
        result = optimize.minimize(
            objective_function, 
            initial_params, 
            method='Powell',
            options={'disp': True, 'maxiter': 1000}
        )
        optimized_params = result.x
        print(f"全局优化完成，状态: {result.message}")
    except ImportError:
        print("警告: 未找到scipy模块，无法执行全局优化。使用初始估计代替。")
        optimized_params = initial_params
    
    # 解包优化后的参数
    fx_proj, fy_proj, cx_proj, cy_proj, skew = optimized_params[0:5]
    proj_dist = optimized_params[5:10]
    rvec_proj = optimized_params[10:13].reshape(3, 1)
    tvec_proj = optimized_params[13:16].reshape(3, 1)
    rvec_cam = optimized_params[16:19].reshape(3, 1)
    tvec_cam = optimized_params[19:22].reshape(3, 1)
    
    # 重建投影仪内参矩阵
    self.projector_matrix = np.array([
        [fx_proj, skew, cx_proj],
        [0, fy_proj, cy_proj],
        [0, 0, 1]
    ], dtype=np.float32)
    
    self.projector_dist = proj_dist.astype(np.float32)
    
    # 转换旋转向量为旋转矩阵
    R_proj, _ = cv2.Rodrigues(rvec_proj)
    T_proj = tvec_proj
    R_cam, _ = cv2.Rodrigues(rvec_cam)
    T_cam = tvec_cam
    
    # 计算投影仪相对于相机的外参
    self.R = R_cam @ R_proj.T
    self.T = T_cam - self.R @ T_proj
    
    # 计算最终的重投影误差
    camera_proj_pts, _ = cv2.projectPoints(
        object_points, rvec_cam, tvec_cam, camera_matrix, camera_distortion
    )
    camera_error = np.sqrt(np.mean(np.sum((camera_proj_pts.reshape(-1, 2) - camera_points) ** 2, axis=1)))
    
    proj_proj_pts, _ = cv2.projectPoints(
        object_points, rvec_proj, tvec_proj, self.projector_matrix, self.projector_dist
    )
    proj_error = np.sqrt(np.mean(np.sum((proj_proj_pts.reshape(-1, 2) - projector_points) ** 2, axis=1)))
    
    avg_error = (camera_error + proj_error) / 2
    
    print("\n全局优化标定结果:")
    print(f"相机重投影误差 (RMSE): {camera_error:.4f} 像素")
    print(f"投影仪重投影误差 (RMSE): {proj_error:.4f} 像素")
    print(f"平均重投影误差 (RMSE): {avg_error:.4f} 像素")
    print("投影仪内参矩阵:")
    print(self.projector_matrix)
    print("\n投影仪畸变系数:")
    print(self.projector_dist)
    print("\n从投影仪到相机的旋转矩阵:")
    print(self.R)
    print("\n从投影仪到相机的平移向量 (mm):")
    print(self.T)
    
    return self.projector_matrix, self.projector_dist, self.R, self.T, proj_error

def process_n_step_images_integrated(image_paths: List[str], output_dir: str, 
                                     show_plots: bool = True, return_quality: bool = False,
                                     mask_method: str = 'otsu', mask_confidence: float = 0.5,
                                     min_area: int = 500):
    """
    处理N步相移图像并生成解包裹相位 (通用版本)
    
    参数:
        image_paths: N张相移图像的路径列表
        output_dir: 输出目录
        show_plots: 是否显示图形
        return_quality: 是否返回相位质量图
    
    返回:
        unwrapped_phase: 解包裹相位图
        quality_map: 相位质量图
        wrapped_phase: 包裹相位图
        base_image: 用于角点检测的基准图像 (平均强度图)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if any(img is None for img in images):
        img_path_str = "\n".join(image_paths)
        raise ValueError(f"无法加载一张或多张图像，请检查路径:\n{img_path_str}")
    
    n_steps = len(images)
    if n_steps < 3:
        raise ValueError(f"N步相移至少需要3张图像，但收到了{n_steps}张")

    # 计算平均强度图作为基准图
    base_image = np.mean(np.stack([img.astype(np.float32) for img in images], axis=0), axis=0).astype(np.uint8)

    # 先生成掩膜并保存
    mask = generate_projection_mask(images, PhaseShiftingAlgorithm.n_step, method=mask_method, min_area=min_area, confidence=mask_confidence)
    cv2.imwrite(os.path.join(output_dir, "projection_mask.png"), (mask.astype(np.uint8) * 255))

    # 保存掩膜特征图，便于诊断
    imgs = np.stack([img.astype(np.float32) for img in images], axis=2)
    I_max = np.max(imgs, axis=2)
    I_min = np.min(imgs, axis=2)
    I_mean = np.mean(imgs, axis=2)
    A = (I_max - I_min) / 2.0
    M = (I_max - I_min) / (I_max + I_min + 1e-9)
    cv2.imwrite(os.path.join(output_dir, "Amplitude.png"), cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "Modulation.png"), cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "MeanIntensity.png"), cv2.normalize(I_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # 使用掩膜约束计算包裹相位与质量图
    wrapped_phase, _ = compute_phasor_and_phase_masked(images, mask, PhaseShiftingAlgorithm.n_step)
    quality_map = compute_phase_quality_masked(images, mask)
    
    visualize_phase(wrapped_phase, "Wrapped Phase", os.path.join(output_dir, "wrapped_phase.png"), 
                    show_plots, True, quality_map)
    
    # 在掩膜内进行质量引导解包裹
    unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map, mask=mask)
    
    visualize_phase(unwrapped_phase, "Unwrapped Phase", os.path.join(output_dir, "unwrapped_phase.png"), 
                    show_plots, False)
    
    np.save(os.path.join(output_dir, "wrapped_phase.npy"), wrapped_phase)
    np.save(os.path.join(output_dir, "unwrapped_phase.npy"), unwrapped_phase)
    np.save(os.path.join(output_dir, "quality_map.npy"), quality_map)
    
    print(f"{n_steps}步相移处理完成。结果已保存到 {output_dir} 目录。")
    
    if return_quality:
        return unwrapped_phase, quality_map, wrapped_phase, base_image
    return unwrapped_phase, None, wrapped_phase, base_image

def phase_based_projector_calibration(projector_width, projector_height, camera_params_file,
                                   phase_images_folder, board_type="chessboard", chessboard_size=(9, 6),
                                   square_size=20.0, output_folder=None, visualize=True,
                                   global_optimization=True, sampling_step=5, adaptive_threshold=True,
                                   n_steps=4, print_func=print,
                                   mask_method: str = 'otsu', mask_confidence: float = 0.5):
    """
    基于相位解包裹的投影仪标定主函数 (新版，支持N步相移和子文件夹)

    通过处理包含多个子文件夹的图像目录来进行投影仪标定，每个子文件夹代表一个标定姿态。
    每个子文件夹内应包含 2*N 张图像 (N为相移步数):
    - I1 ... IN: 水平方向的N步相移图像
    - I(N+1) ... I(2N): 垂直方向的N步相移图像
    (支持 .png, .jpg, .jpeg, .bmp, .tif, .tiff 格式)

    参数:
        projector_width: 投影仪分辨率宽度
        projector_height: 投影仪分辨率高度
        camera_params_file: 相机标定参数文件路径
        phase_images_folder: 包含相移图案图像的文件夹
        board_type: 标定板类型
        chessboard_size: 标定板内角点数量 (宽, 高)
        square_size: 标定板方格尺寸或圆心间距(mm)
        output_folder: 输出文件夹
        visualize: 是否可视化结果
        global_optimization: 是否使用全局优化方法提高精度
        sampling_step: 从相位图中提取对应关系的采样步长
        adaptive_threshold: 是否使用自适应方法计算质量阈值
        n_steps: 相移步数 (N), 默认为4.
        print_func: 用于打印消息的函数，默认为print。
    """
    print_func(f"投影仪标定程序 (基于相位解包裹 - v2.1 支持N步相移)")
    print_func("=" * 60)

    # 检查输入文件夹是否存在
    if not os.path.isdir(phase_images_folder):
        raise FileNotFoundError(f"指定的相位图像文件夹不存在: {phase_images_folder}")

    # 创建输出文件夹
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(phase_images_folder, f"projector_calibration_{timestamp}")
    
    os.makedirs(output_folder, exist_ok=True)
    phase_unwrap_dir = os.path.join(output_folder, "phase_unwrapping")
    os.makedirs(phase_unwrap_dir, exist_ok=True)
    
    print_func(f"标定结果将保存至: {output_folder}")

    # 初始化标定对象
    calibration = ProjectorCalibration(projector_width, projector_height, log_file=os.path.join(output_folder, "calibration_log.txt"))

    # 加载相机参数
    print_func("\n步骤 1: 加载相机参数...")
    try:
        calib_data = calibration.load_camera_params(camera_params_file=camera_params_file)
        camera_matrix = calib_data['camera_matrix']
        camera_distortion = calib_data['dist_coeffs']
        print_func(f"成功从 {camera_params_file} 加载相机参数。")
    except Exception as e:
        raise CalibrationError(f"加载相机参数失败: {e}")

    # 准备存储所有姿态的数据
    all_obj_points = []     # 所有姿态的三维物体点
    all_cam_points = []     # 所有姿态的相机图像点
    all_proj_points = []    # 所有姿态的投影仪"图像"点 (相位坐标)

    # 定义图像文件扩展名
    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    # 步骤 2 & 3: 遍历子文件夹，处理每个姿态
    print_func("\n步骤 2 & 3: 循环处理每个姿态的图像，进行相位解包裹和角点检测...")
    
    pose_dirs = [d for d in os.listdir(phase_images_folder) if os.path.isdir(os.path.join(phase_images_folder, d))]
    
    if not pose_dirs:
        raise FileNotFoundError(f"在 '{phase_images_folder}' 中未找到任何姿态子文件夹。")

    valid_poses_count = 0
    iterator = tqdm(pose_dirs, desc="处理姿态子文件夹", unit="pose") if TQDM_AVAILABLE else pose_dirs

    for pose_name in iterator:
        pose_dir = os.path.join(phase_images_folder, pose_name)
        print_func(f"\n--- 正在处理姿态: {pose_name} ({n_steps}步相移) ---")

        try:
            # 查找 2*N 张图像
            image_files = {}
            total_images = 2 * n_steps
            for i in range(1, total_images + 1):
                found = False
                for ext in IMG_EXTENSIONS:
                    path = os.path.join(pose_dir, f"I{i}{ext}")
                    if os.path.exists(path):
                        image_files[f'I{i}'] = path
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"在 {pose_dir} 中找不到图像 I{i} (需要1到{total_images}的图像)")

            h_paths = [image_files[f'I{i}'] for i in range(1, n_steps + 1)]
            v_paths = [image_files[f'I{i}'] for i in range(n_steps + 1, total_images + 1)]

            # 为每个姿态创建单独的输出子目录
            pose_output_dir = os.path.join(phase_unwrap_dir, pose_name)
            os.makedirs(pose_output_dir, exist_ok=True)
            h_output_dir = os.path.join(pose_output_dir, "horizontal")
            v_output_dir = os.path.join(pose_output_dir, "vertical")

            # 处理水平条纹
            print_func("  - 解包裹水平相位...")
            h_unwrapped, h_quality, _, _ = process_n_step_images_integrated(
                h_paths, h_output_dir, show_plots=visualize, return_quality=True,
                mask_method=mask_method, mask_confidence=mask_confidence
            )
            
            # 处理垂直条纹
            print_func("  - 解包裹垂直相位...")
            v_unwrapped, v_quality, _, base_image_v = process_n_step_images_integrated(
                v_paths, v_output_dir, show_plots=visualize, return_quality=True,
                mask_method=mask_method, mask_confidence=mask_confidence
            )
            
            # 生成组合掩膜（水平和垂直相位的交集）
            print_func("  - 生成投影区域掩膜...")
            # 读取水平和垂直方向的掩膜
            h_mask_path = os.path.join(h_output_dir, "projection_mask.png")
            v_mask_path = os.path.join(v_output_dir, "projection_mask.png")
            
            combined_mask = None
            if os.path.exists(h_mask_path) and os.path.exists(v_mask_path):
                h_mask = cv2.imread(h_mask_path, cv2.IMREAD_GRAYSCALE) > 0
                v_mask = cv2.imread(v_mask_path, cv2.IMREAD_GRAYSCALE) > 0
                combined_mask = h_mask & v_mask  # 取交集
                print_func(f"  - 组合掩膜有效像素: {np.sum(combined_mask)}")
            else:
                print_func("  - 警告: 无法读取掩膜文件，将使用相位值判断有效区域")

            # 过滤低质量相位点
            if adaptive_threshold:
                print_func("  - 应用自适应阈值过滤低质量相位点...")
                h_threshold = compute_adaptive_quality_threshold(h_quality)
                v_threshold = compute_adaptive_quality_threshold(v_quality)
                h_unwrapped[h_quality < h_threshold] = 0
                v_unwrapped[v_quality < v_threshold] = 0

            # 获取有效相位的动态范围
            h_valid_mask = h_unwrapped != 0
            v_valid_mask = v_unwrapped != 0
            
            if not np.any(h_valid_mask) or not np.any(v_valid_mask):
                raise PhaseUnwrappingError("解包裹后的水平或垂直相位图为空，无法继续。请检查图像质量。")

            h_min, h_max = np.min(h_unwrapped[h_valid_mask]), np.max(h_unwrapped[h_valid_mask])
            v_min, v_max = np.min(v_unwrapped[v_valid_mask]), np.max(v_unwrapped[v_valid_mask])
            
            # 使用垂直相移的平均强度图进行角点检测
            cam_img = base_image_v
            if cam_img is None: # 如果处理函数没返回，就自己读一张
                cam_img = cv2.imread(v_paths[0])
            
            # 检测标定板角点
            print_func("  - 检测标定板角点...")
            obj_points_pose, cam_points_pose = detect_calibration_board(cam_img, board_type, chessboard_size, square_size)
            
            if cam_points_pose is None or len(cam_points_pose) == 0:
                raise BoardDetectionError(f"在姿态 {pose_name} 的图像中未能检测到标定板。")

            print_func(f"  - 成功检测到 {len(cam_points_pose)} 个角点。")

            # 提取投影仪中的对应点（掩膜约束）
            proj_points_pose = []
            for point in cam_points_pose:
                x, y = point[0]
                
                # 首先检查角点是否在掩膜有效区域内
                is_in_mask = True
                if combined_mask is not None:
                    # 检查角点位置是否在掩膜内
                    if 0 <= int(y) < combined_mask.shape[0] and 0 <= int(x) < combined_mask.shape[1]:
                        is_in_mask = combined_mask[int(y), int(x)]
                    else:
                        is_in_mask = False
                
                if is_in_mask:
                    phi_h = bilinear_interpolate(h_unwrapped, y, x)
                    phi_v = bilinear_interpolate(v_unwrapped, y, x)
                    
                    # 检查插值后的相位是否有效（掩膜约束下的双重检查）
                    if phi_h != 0 and phi_v != 0:
                        # 相位值到投影仪像素坐标的映射 (使用动态范围)
                        px = (phi_h - h_min) / (h_max - h_min) * (projector_width - 1)
                        py = (phi_v - v_min) / (v_max - v_min) * (projector_height - 1)
                        proj_points_pose.append([px, py])
            
            if len(proj_points_pose) < len(cam_points_pose) * 0.8:
                print_func(f"  - 警告: {len(cam_points_pose) - len(proj_points_pose)} 个角点位于无效相位区域，已被丢弃。")

            if len(proj_points_pose) < 10: # 至少需要10个点
                raise CorrespondenceError(f"在姿态 {pose_name} 中未能找到足够多的有效角点对应关系 (仅找到 {len(proj_points_pose)} 个)。")

            proj_points_pose = np.array(proj_points_pose, dtype=np.float32).reshape(-1, 1, 2)
            
            # 由于部分角点可能被丢弃，需要筛选对应的相机点和物体点（掩膜约束）
            valid_indices = []
            for i, point in enumerate(cam_points_pose):
                x, y = point[0]
                
                # 检查是否在掩膜内
                is_in_mask = True
                if combined_mask is not None:
                    if 0 <= int(y) < combined_mask.shape[0] and 0 <= int(x) < combined_mask.shape[1]:
                        is_in_mask = combined_mask[int(y), int(x)]
                    else:
                        is_in_mask = False
                
                # 检查相位值是否有效
                if is_in_mask:
                    phi_h = bilinear_interpolate(h_unwrapped, y, x)
                    phi_v = bilinear_interpolate(v_unwrapped, y, x)
                    if phi_h != 0 and phi_v != 0:
                        valid_indices.append(i)

            if len(valid_indices) != len(proj_points_pose):
                 # 这是一个内部逻辑检查，正常不应触发
                 raise Exception("内部错误：有效角点和投影仪点数量不匹配。")
            
            cam_points_pose_valid = cam_points_pose[valid_indices]
            obj_points_pose_valid = obj_points_pose[valid_indices]

            all_obj_points.append(obj_points_pose_valid)
            all_cam_points.append(cam_points_pose_valid)
            all_proj_points.append(proj_points_pose)
            
            valid_poses_count += 1
            print_func(f"  - 姿态 {pose_name} 处理成功。")

        except (FileNotFoundError, BoardDetectionError, PhaseUnwrappingError) as e:
            print_func(f"警告: 跳过姿态 '{pose_name}'，原因: {e}")
            continue
        except Exception as e:
            print_func(f"警告: 处理姿态 '{pose_name}' 时发生未知错误，已跳过。错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    if valid_poses_count < 3:
        raise CorrespondenceError(f"未能处理足够数量的有效姿态。至少需要3个有效姿态，但只处理了 {valid_poses_count} 个。")

    print_func(f"\n成功处理了 {valid_poses_count} / {len(pose_dirs)} 个姿态。")

    # 步骤 4: 执行立体标定
    # =========================================================================
    print_func("\n步骤 4: 执行立体标定以计算投影仪参数...")
    
    if global_optimization:
        print_func("使用全局优化方法 (Levenberg-Marquardt)...")
        reprojection_error, calibration_data = calibration.calibrate_projector_with_camera_global(
            camera_matrix=camera_matrix,
            camera_distortion=camera_distortion,
            proj_cam_correspondences=list(zip(all_proj_points, all_cam_points)),
            board_points=all_obj_points
        )
    else:
        print_func("使用OpenCV标准立体标定方法...")
        reprojection_error, calibration_data = calibration.calibrate_projector_with_camera(
            camera_matrix=camera_matrix,
            camera_distortion=camera_distortion,
            proj_cam_correspondences=list(zip(all_proj_points, all_cam_points)),
            board_points=all_obj_points
        )

    print_func(f"标定完成！最终重投影误差: {reprojection_error:.4f} 像素")

    # 保存标定结果
    calibration_file_path = os.path.join(output_folder, "projector_calibration.json")
    calibration.save_calibration(calibration_file_path, include_metadata=True)
    print_func(f"标定结果已保存至: {calibration_file_path}")
    
    print_func("\n标定成功！")
    print_func("=" * 60)
    
    return calibration, calibration_file_path

def show_calibration_tips(board_type):
    """显示一些有用的标定提示"""
    print("\n标定提示:")
    if board_type == 'chessboard':
        print("请确保棋盘格标定板平整，无明显变形。")
        print("建议在均匀光照条件下拍摄，以获得最佳效果。")
    elif board_type == 'circles':
        print("请确保圆形标定板清晰可见，无明显反光。")
        print("建议在光照变化较小的环境中拍摄。")
    elif board_type == 'ring_circles':
        print("请确保空心圆环标定板闭合且形状规则，无明显变形。")
        print("建议在特殊光照条件下拍摄，如高反光或特殊背景。")
    print("标定过程中请勿移动标定板。")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='投影仪标定工具 (基于相位解包裹)')
    parser.add_argument('--camera-params', type=str, help='相机标定参数文件路径(.json或.npy)')
    parser.add_argument('--phase-images', type=str, help='包含相移图案图像的文件夹')
    parser.add_argument('--projector-width', type=int, default=1280, help='投影仪宽度分辨率')
    parser.add_argument('--projector-height', type=int, default=720, help='投影仪高度分辨率')
    parser.add_argument('--board-type', type=str, default=None,
                       choices=['chessboard', 'circles', 'ring_circles'], help='标定板类型')
    parser.add_argument('--chessboard-width', type=int, default=9, help='标定板宽度点数量')
    parser.add_argument('--chessboard-height', type=int, default=6, help='标定板高度点数量')
    parser.add_argument('--square-size', type=float, default=20.0, help='标定板方格尺寸或圆心间距(mm)')
    parser.add_argument('--output-folder', type=str, help='输出结果文件夹')
    parser.add_argument('--no-visualize', action='store_true', help='不显示可视化结果')
    parser.add_argument('--no-global-optimization', action='store_true', help='不使用全局优化')
    parser.add_argument('--sampling-step', type=int, default=4, help='相位图采样步长')
    parser.add_argument('--no-adaptive-threshold', action='store_true', help='不使用自适应质量阈值')
    parser.add_argument('--quality-threshold', type=float, default=None, help='手动设置相位质量阈值(0-1)')
    
    args = parser.parse_args()
    
    # 如果没有提供任何参数，显示交互式菜单
    if len(sys.argv) == 1:
        print("\n欢迎使用基于相位解包裹的投影仪标定工具！")
        print("=" * 60)
            
        camera_params_file = input("请输入相机标定参数文件路径 (留空则自动查找): ").strip()
        phase_images_folder = input("请输入包含相移图案图像的文件夹: ").strip()
        
        if not phase_images_folder or not os.path.exists(phase_images_folder):
            print("未提供有效的图像文件夹，无法继续标定。")
            return
            
        args.camera_params = camera_params_file
        args.phase_images = phase_images_folder
        
        try:
            res_input = input(f"请输入投影仪分辨率 (宽 高) [默认: {args.projector_width} {args.projector_height}]: ").strip()
            if res_input:
                args.projector_width, args.projector_height = map(int, res_input.split())
        except:
            print(f"使用默认投影仪分辨率: {args.projector_width}x{args.projector_height}")
        
        args.board_type = select_calibration_board_type()
        
        try:
            dims_input = input(f"请输入标定板点数量 (宽 高) [默认: {args.chessboard_width} {args.chessboard_height}]: ").strip()
            if dims_input:
                args.chessboard_width, args.chessboard_height = map(int, dims_input.split())
        except:
            print(f"使用默认标定板尺寸: {args.chessboard_width}x{args.chessboard_height}")
            
        try:
            size_input = input(f"请输入标定板方格尺寸或圆心间距(mm) [默认: {args.square_size}]: ").strip()
            if size_input:
                args.square_size = float(size_input)
        except:
            print(f"使用默认方格尺寸或圆心间距: {args.square_size}mm")
            
        try:
            global_opt_input = input(f"是否使用全局优化? (y/n) [默认: y]: ").strip().lower()
            if global_opt_input == 'n':
                args.no_global_optimization = True
        except:
            print("使用全局优化")
            
        try:
            sampling_input = input(f"请输入采样步长 [默认: {args.sampling_step}]: ").strip()
            if sampling_input:
                args.sampling_step = int(sampling_input)
        except:
            print(f"使用默认采样步长: {args.sampling_step}")
    
    # 检查必要参数
    if not args.phase_images:
        print("错误: 必须提供包含相移图案图像的文件夹路径 --phase-images")
        parser.print_help()
        return
        
    # 获取标定板类型
    board_type = args.board_type if args.board_type else select_calibration_board_type()
    show_calibration_tips(board_type)
    
    # 执行基于相位解包裹的投影仪标定
    try:
        print("\n开始基于相位解包裹的投影仪标定...")
        calibration, calibration_file = phase_based_projector_calibration(
            projector_width=args.projector_width, 
            projector_height=args.projector_height,
            camera_params_file=args.camera_params,
            phase_images_folder=args.phase_images,
            board_type=board_type,
            chessboard_size=(args.chessboard_width, args.chessboard_height),
            square_size=args.square_size,
            output_folder=args.output_folder,
            visualize=not args.no_visualize,
            global_optimization=not args.no_global_optimization,
            sampling_step=args.sampling_step,
            adaptive_threshold=not args.no_adaptive_threshold
        )

        print(f"\n基于相位解包裹的投影仪标定完成，结果已保存至: {calibration_file}")

        # 显示投影仪与相机之间的位姿关系
        print("\n投影仪与相机的位姿关系:")
        print("旋转矩阵 (从投影仪到相机):")
        print(calibration.R)
        print("\n平移向量 (从投影仪到相机，单位:mm):")
        print(calibration.T)

    except PhaseUnwrappingError as e:
        print(f"\n相位解包裹失败: {e}")
        print("\n建议: 检查相位图像质量，确保照明条件适合，并尝试调整相位图像获取过程。")
        
    except BoardDetectionError as e:
        print(f"\n标定板检测失败: {e}")
        print("\n建议: 确保标定板清晰可见且无变形，尝试改进照明条件，或检查标定板尺寸参数是否正确。")
        
    except CorrespondenceError as e:
        print(f"\n点对应关系建立失败: {e}")
        print("\n建议: 确保标定板上有足够多的角点，并且相位图质量良好，角点位于相位图的有效区域内。")
        
    except CalibrationError as e:
        print(f"\n投影仪标定失败: {e}")
        print("\n建议: 检查标定过程中的参数设置，尝试调整优化方法，或收集更多更高质量的数据。")
        
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n建议: 这可能是一个程序bug，请检查错误信息并考虑报告此问题。")

    print("\n所有操作完成！")

if __name__ == "__main__":
    main() 