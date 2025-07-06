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
                                        show_plots: bool = True, return_quality: bool = False):
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

    wrapped_phase = compute_wrapped_phase(images, PhaseShiftingAlgorithm.four_step)
    quality_map = compute_phase_quality(images)
    
    visualize_phase(wrapped_phase, "Wrapped Phase", os.path.join(output_dir, "wrapped_phase.png"), 
                    show_plots, True, quality_map)
    
    unwrapped_phase = quality_guided_unwrap(wrapped_phase, quality_map)
    
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

# 使用向量化操作提取相位对应关系
def extract_phase_correspondence(horizontal_unwrapped, vertical_unwrapped, 
                              projector_width, projector_height, sampling_step=5):
    """
    从水平和垂直方向的解包裹相位中提取投影仪-相机像素对应关系，使用向量化操作提高性能
    
    参数:
        horizontal_unwrapped: 水平方向解包裹相位图
        vertical_unwrapped: 垂直方向解包裹相位图
        projector_width: 投影仪宽度
        projector_height: 投影仪高度
        sampling_step: 采样步长，减少点数提高性能
        
    返回:
        correspondences: 投影仪点和相机点的对应关系字典
    """
    if horizontal_unwrapped is None and vertical_unwrapped is None:
        raise ValueError("需要至少一个方向的解包裹相位")
    
    # 获取相位图尺寸
    height, width = horizontal_unwrapped.shape if horizontal_unwrapped is not None else vertical_unwrapped.shape
    
    # 创建结果字典
    correspondences = {}
    
    # 使用进度条来显示处理进度
    if TQDM_AVAILABLE:
        print("提取相位对应关系...")
    
    if horizontal_unwrapped is not None and vertical_unwrapped is not None:
        # 两个方向都有相位信息，使用向量化操作
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:height:sampling_step, 0:width:sampling_step]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        # 获取相应位置的相位值
        h_phases = horizontal_unwrapped[y_coords, x_coords]
        v_phases = vertical_unwrapped[y_coords, x_coords]
        
        # 获取相位图的最大值
        h_max_phase = np.max(horizontal_unwrapped[horizontal_unwrapped > 0])
        v_max_phase = np.max(vertical_unwrapped[vertical_unwrapped > 0])
        
        # 查找有效点（两个方向都有有效相位值）
        valid_mask = (h_phases > 0) & (v_phases > 0)
        
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
        h_max_phase = np.max(horizontal_unwrapped[horizontal_unwrapped > 0])
        
        valid_mask = h_phases > 0
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
        v_max_phase = np.max(vertical_unwrapped[vertical_unwrapped > 0])
        
        valid_mask = v_phases > 0
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
    
    print(f"从相位图中提取了 {len(correspondences)} 个像素对应关系")
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

def phase_based_projector_calibration(projector_width, projector_height, camera_params_file,
                                   phase_images_folder, board_type="chessboard", chessboard_size=(9, 6), 
                                   square_size=20.0, output_folder=None, visualize=True, 
                                   global_optimization=True, sampling_step=5, adaptive_threshold=True):
    """
    使用相位解包裹方法进行投影仪标定
    
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
        
    返回:
        calibration: 投影仪标定对象
        calibration_file: 保存的标定文件路径
        
    异常:
        PhaseUnwrappingError: 相位解包裹失败
        BoardDetectionError: 标定板检测失败
        CorrespondenceError: 无法建立足够的点对应关系
        CalibrationError: 其他标定过程中的错误
    """
    try:
        # 创建输出文件夹
        if output_folder is None:
            output_folder = os.path.join(phase_images_folder, "projector_calibration_results")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_folder, f"calibration_log_{timestamp}.txt")

        # 创建投影仪标定对象
        calibration = ProjectorCalibration(
            projector_width, projector_height, log_file=log_file
        )
        
        calibration.log(f"开始投影仪标定 - 时间: {timestamp}")
        calibration.log(f"投影仪分辨率: {projector_width}x{projector_height}")
        calibration.log(f"使用标定板类型: {board_type}, 尺寸: {chessboard_size}, 方格大小: {square_size}mm")
        calibration.log(f"相位图文件夹: {phase_images_folder}")
        
        # 加载相机参数
        camera_matrix, camera_dist = calibration.load_camera_params(camera_params_file)
        
        try:
            # 处理相移图案图像，获取解包裹相位
            calibration.log("\n处理相移图案图像，进行相位解包裹...")
            unwrapped_results = process_phase_images(
                phase_images_folder, 
                output_dir=os.path.join(output_folder, "phase_unwrapping"),
                visualize=visualize,
                adaptive_threshold=adaptive_threshold
            )
        except Exception as e:
            raise PhaseUnwrappingError(f"相位解包裹失败: {str(e)}")
        
        try:
            # 从相位图中提取投影仪-相机像素对应关系
            calibration.log("\n从相位图提取投影仪-相机像素对应关系...")
            correspondences = extract_phase_correspondence(
                unwrapped_results.get("horizontal"), 
                unwrapped_results.get("vertical"),
                projector_width,
                projector_height,
                sampling_step=sampling_step
            )
            
            if not correspondences:
                raise CorrespondenceError("未能从相位图中提取有效的像素对应关系")
        except Exception as e:
            if isinstance(e, CorrespondenceError):
                raise
            else:
                raise CorrespondenceError(f"提取像素对应关系失败: {str(e)}")
        
        try:
            # 找到标定板位置
            calibration.log("\n检测标定板位置...")
            # 使用第一张相移图像来检测标定板
            if "horizontal" in unwrapped_results:
                reference_image_path = sorted(glob.glob(os.path.join(phase_images_folder, "h_*.png")))[0]
            else:
                reference_image_path = sorted(glob.glob(os.path.join(phase_images_folder, "v_*.png")))[0]
                
            reference_image = cv2.imread(reference_image_path)
            if reference_image is None:
                raise BoardDetectionError(f"无法读取参考图像: {reference_image_path}")
            
            board_points, board_corners = detect_calibration_board(
                reference_image, board_type, chessboard_size, square_size
            )
            
            if board_points is None or board_corners is None:
                raise BoardDetectionError("无法在参考图像中检测到标定板")
        except Exception as e:
            if isinstance(e, BoardDetectionError):
                raise
            else:
                raise BoardDetectionError(f"标定板检测失败: {str(e)}")
        
        # 将标定板角点与相位对应关系结合
        calibration.log("\n将标定板角点与相位对应关系结合...")
        
        # 准备标定数据
        calibration_points = []
        
        # 对于检测到的每个标定板角点
        corner_iter = enumerate(board_corners)
        if TQDM_AVAILABLE:
            corner_iter = tqdm(corner_iter, total=len(board_corners), desc="处理标定板角点")
            
        for i, corner in corner_iter:
            # 获取角点在图像中的位置（亚像素精度）
            camera_x, camera_y = corner.ravel()
            
            # 使用双线性插值获取该亚像素位置的相位值
            if "horizontal" in unwrapped_results and "vertical" in unwrapped_results:
                h_unwrapped = unwrapped_results["horizontal"]
                v_unwrapped = unwrapped_results["vertical"]
                
                # 使用双线性插值获取水平和垂直相位值
                h_phase = bilinear_interpolate(h_unwrapped, camera_y, camera_x)
                v_phase = bilinear_interpolate(v_unwrapped, camera_y, camera_x)
                
                # 获取相位最大值
                h_max_phase = unwrapped_results["horizontal_max_phase"]
                v_max_phase = unwrapped_results["vertical_max_phase"]
                
                # 计算对应的投影仪坐标
                if h_phase > 0 and v_phase > 0:  # 确保相位值有效
                    proj_x = (h_phase / h_max_phase) * projector_width
                    proj_y = (v_phase / v_max_phase) * projector_height
                    
                    # 添加到标定点列表，保持亚像素精度
                    calibration_points.append({
                        'projector_point': np.array([proj_x, proj_y]),
                        'camera_point': corner.ravel(),
                        'board_index': i
                    })
                    
                    if not TQDM_AVAILABLE:  # 如果没有进度条，则打印每个点的信息
                        calibration.log(f"角点 {i}: 相机坐标 ({camera_x:.2f}, {camera_y:.2f}), "
                              f"投影仪坐标 ({proj_x:.2f}, {proj_y:.2f})")
            else:
                calibration.log(f"警告: 角点 {i} 处无法进行插值，缺少完整的相位数据")
                
        if not calibration_points:
            raise CorrespondenceError("未能建立有效的标定点对应关系")
            
        calibration.log(f"成功建立了 {len(calibration_points)} 个标定点对应关系")
        
        # 检查是否有足够的点用于标定
        min_points_needed = 10  # 至少需要10个点进行可靠的标定
        if len(calibration_points) < min_points_needed:
            raise CorrespondenceError(f"标定点数量不足，需要至少{min_points_needed}个点，但只找到了{len(calibration_points)}个")
        
        try:
            # 执行投影仪标定
            calibration.log("\n执行投影仪标定...")
            
            if global_optimization:
                calibration.log("使用全局优化方法进行标定...")
                projector_matrix, projector_dist, R, T, reproj_error = calibration.calibrate_projector_with_camera_global(
                    camera_matrix, camera_dist, calibration_points, board_points
                )
            else:
                projector_matrix, projector_dist, R, T, reproj_error = calibration.calibrate_projector_with_camera(
                    camera_matrix, camera_dist, calibration_points, board_points
                )
            
            # 评估标定质量
            assess_calibration_quality(reproj_error, board_type)
            
            # 保存标定结果
            calibration_file = os.path.join(output_folder, f"projector_calibration_phase_{timestamp}.json")
            calibration_file, default_file = calibration.save_calibration(calibration_file)
            
            # 记录完成信息
            calibration.log("\n标定完成！")
            calibration.log(f"标定结果已保存至: {calibration_file}")
            calibration.log(f"标定精度 (重投影误差): {reproj_error:.4f} 像素")
            
            return calibration, calibration_file
            
        except Exception as e:
            raise CalibrationError(f"执行标定失败: {str(e)}")
    
    except Exception as e:
        # 捕获并重新抛出所有异常，确保它们得到适当处理
        if isinstance(e, (PhaseUnwrappingError, BoardDetectionError, CorrespondenceError, CalibrationError)):
            raise
        else:
            raise CalibrationError(f"投影仪标定过程中发生未预期的错误: {str(e)}")

def show_calibration_tips(board_type):
    """
    根据标定板类型显示标定提示
    """
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