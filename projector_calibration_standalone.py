#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影仪标定独立程序

该程序允许用户使用相机拍摄的投影图案图像对投影仪进行标定，获取投影仪的内参矩阵、畸变系数以及投影仪与相机的相对位置关系。
需要先完成相机标定，并提供相机的内参矩阵和畸变系数。

使用方法:
1. 确保已经完成相机标定，并有相机标定结果文件
2. 准备投影仪投射相移条纹到标定板上，相机拍摄的图像
3. 运行程序，按照提示输入相关参数
4. 程序将自动进行标定，并保存结果

作者: [Your Name]
日期: [Date]
"""

import os
import sys
import numpy as np
import cv2
import glob
import json
import matplotlib.pyplot as plt
from scipy import optimize
import argparse
from datetime import datetime

class ProjectorCalibration:
    """投影仪标定类，实现伽马校正和投影仪内参外参标定"""
    
    def __init__(self, projector_width=1280, projector_height=800):
        """
        初始化投影仪标定对象
        
        参数:
            projector_width: 投影仪分辨率宽度
            projector_height: 投影仪分辨率高度
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.projector_matrix = None  # 投影仪内参矩阵
        self.projector_dist = None    # 投影仪畸变系数
        self.R = None                 # 从投影仪到相机的旋转矩阵
        self.T = None                 # 从投影仪到相机的平移向量
        self.gamma_a = 1.0            # 伽马校正参数 a
        self.gamma_b = 1.0            # 伽马校正参数 b
        self.gamma_c = 0.0            # 伽马校正参数 c
        
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
        # 如果直接提供了相机参数
        if camera_params_file is None and camera_matrix is not None and camera_dist is not None:
            print("使用手动提供的相机标定参数")
            return camera_matrix, camera_dist
        
        # 如果提供了文件路径
        if camera_params_file:
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
            
            print("成功加载相机标定参数:")
            print("相机内参矩阵:")
            print(camera_matrix)
            print("\n相机畸变系数:")
            print(camera_dist)
            
            return camera_matrix, camera_dist
        
        # 如果没有提供文件路径也没有提供直接参数，则尝试查找默认位置的文件
        default_paths = [
            "./calibration_results/camera_calibration_latest.json",  # 当前目录的结果文件夹
            "./camera_calibration_latest.json",  # 当前目录
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                print(f"找到默认相机标定文件: {path}")
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    camera_matrix = np.array(data['camera_matrix'])
                    camera_dist = np.array(data['dist_coeffs'])
                    
                    print("成功加载相机标定参数:")
                    print("相机内参矩阵:")
                    print(camera_matrix)
                    print("\n相机畸变系数:")
                    print(camera_dist)
                    
                    return camera_matrix, camera_dist
                except:
                    print(f"无法从文件 {path} 加载标定参数")
        
        # 如果都找不到，则抛出错误
        raise ValueError("未提供相机标定参数文件或直接参数，且未找到默认位置的标定文件")
        
    def calibrate_gamma(self, brightness_data, intensity_data, visualize=True):
        """
        根据亮度-强度数据校正投影仪伽马曲线
        
        参数:
            brightness_data: 相机捕获的亮度值列表
            intensity_data: 对应的投影强度值列表
            visualize: 是否可视化伽马校正结果
        
        返回:
            gamma_params: 伽马校正参数 (a, b, c)
        """
        # 转换为numpy数组
        brightness = np.array(brightness_data)
        intensity = np.array(intensity_data)
        
        # 查找饱和水平
        saturation_level = 0.95
        saturation = len(intensity) - 1  # 默认使用全部数据
        
        k = 0
        for i in range(len(intensity)):
            if brightness[i] > np.max(brightness) * saturation_level:
                k = k + 1
                if k > 3:
                    saturation = i - 2
                    break
        
        # 减少序列到饱和水平
        int_reduced = intensity[:saturation]
        brt_reduced = brightness[:saturation]
        
        # 定义伽马函数拟合
        gamma_func = lambda x, a, b, c: a * (x + c) ** b
        
        # 对减少后的亮度与强度序列拟合伽马函数参数
        try:
            popt, pcov = optimize.curve_fit(gamma_func, int_reduced, brt_reduced, p0=(1, 1, 0))
            
            print(f"拟合的伽马函数 - Iout = {popt[0]:.3f} * (Iin + {popt[2]:.3f}) ^ {popt[1]:.3f}")
            
            # 保存伽马校正参数
            self.gamma_a = popt[0]
            self.gamma_b = popt[1]
            self.gamma_c = popt[2]
            
            # 绘制拟合的伽马函数
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.plot(intensity, brightness, "b+", label="测量数据")
                plt.plot(intensity, gamma_func(intensity, *popt), "r-", label="拟合曲线")
                plt.xlabel("输入强度")
                plt.ylabel("输出亮度")
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.grid(True)
                plt.legend()
                plt.title("投影仪伽马校正曲线")
                plt.show()
            
            return popt
            
        except Exception as e:
            print(f"伽马校正拟合失败: {str(e)}")
            print("使用默认伽马校正参数: a=1.0, b=2.2, c=0.0")
            self.gamma_a = 1.0
            self.gamma_b = 2.2
            self.gamma_c = 0.0
            return (self.gamma_a, self.gamma_b, self.gamma_c)
    
    def apply_gamma_correction(self, image):
        """
        对图像应用伽马校正
        
        参数:
            image: 输入图像 (0-1范围)
        
        返回:
            corrected_image: 伽马校正后的图像
        """
        # 应用伽马校正公式: Iout = a * (Iin + c) ^ b
        corrected_image = self.gamma_a * np.power(image + self.gamma_c, self.gamma_b)
        
        # 裁剪到0-1范围
        corrected_image = np.clip(corrected_image, 0, 1)
        
        return corrected_image
        
    def collect_gamma_calibration_data(self, gray_images_folder):
        """
        从一组灰度图像中收集伽马校正数据
        
        参数:
            gray_images_folder: 包含投影不同强度灰度图像的文件夹
            
        返回:
            brightness: 亮度值数组
            intensity: 强度值数组
        """
        # 查找灰度图像
        gray_images = sorted(glob.glob(os.path.join(gray_images_folder, '*.jpg')))
        gray_images.extend(sorted(glob.glob(os.path.join(gray_images_folder, '*.png'))))
        
        if len(gray_images) == 0:
            raise ValueError(f"在文件夹 '{gray_images_folder}' 中未找到图像文件")
        
        print(f"找到 {len(gray_images)} 张灰度图像")
        
        # 收集数据
        brightness = []
        intensity = []
        
        for i, img_path in enumerate(gray_images):
            # 获取文件名中的强度值，假设文件名格式为 "gray_X.XX.jpg"，其中X.XX是0-1之间的强度值
            try:
                intensity_value = float(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
            except:
                intensity_value = i / (len(gray_images) - 1)  # 如果无法从文件名获取，则使用线性值
            
            # 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
                
            # 计算平均亮度
            avg_brightness = np.mean(img) / 255.0  # 归一化到0-1范围
            
            brightness.append(avg_brightness)
            intensity.append(intensity_value)
            
            print(f"图像 {os.path.basename(img_path)}: 强度 = {intensity_value:.3f}, 亮度 = {avg_brightness:.3f}")
        
        # 按强度值排序
        sorted_indices = np.argsort(intensity)
        intensity = np.array(intensity)[sorted_indices]
        brightness = np.array(brightness)[sorted_indices]
        
        return brightness, intensity
    
    def create_phase_shifting_patterns(self, frequencies=[1, 4, 12], shifts=4, vertical=True):
        """
        创建相移图案用于投影仪标定
        
        参数:
            frequencies: 使用的频率列表
            shifts: 每个频率的相移数量
            vertical: 是否创建垂直条纹(True)或水平条纹(False)
        
        返回:
            patterns: 相移图案列表的列表，按频率和相移组织
            phase_shifts: 相移值列表
        """
        patterns = []
        phase_shifts = np.linspace(0, 2*np.pi, shifts, endpoint=False)
        
        for freq in frequencies:
            freq_patterns = []
            for phase in phase_shifts:
                # 创建对应尺寸的图案
                pattern = np.zeros((self.projector_height, self.projector_width), dtype=np.float32)
                
                # 根据方向生成条纹
                if vertical:
                    # 垂直条纹 - x坐标变化
                    x = np.arange(self.projector_width)
                    for y in range(self.projector_height):
                        pattern[y, :] = 0.5 + 0.5 * np.cos(2 * np.pi * freq * x / self.projector_width + phase)
                else:
                    # 水平条纹 - y坐标变化
                    y = np.arange(self.projector_height)
                    for x in range(self.projector_width):
                        pattern[:, x] = 0.5 + 0.5 * np.cos(2 * np.pi * freq * y / self.projector_height + phase)
                
                # 应用伽马校正
                if hasattr(self, 'gamma_a') and self.gamma_a != 1.0:
                    pattern = self.apply_gamma_correction(pattern)
                
                freq_patterns.append(pattern)
            
            patterns.append(freq_patterns)
        
        return patterns, phase_shifts
    
    def save_patterns(self, patterns, output_folder, file_prefix="pattern", file_format="png"):
        """
        保存相移图案到文件
        
        参数:
            patterns: 相移图案列表的列表 (按频率和相移组织)
            output_folder: 输出文件夹
            file_prefix: 文件名前缀
            file_format: 文件格式 ('png' 或 'jpg')
        
        返回:
            pattern_files: 保存的图案文件路径列表
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        pattern_files = []
        
        for freq_idx, freq_patterns in enumerate(patterns):
            for phase_idx, pattern in enumerate(freq_patterns):
                # 转换为8位图像
                pattern_8bit = (pattern * 255).astype(np.uint8)
                
                # 构建文件名: pattern_freq<freq_idx>_phase<phase_idx>.<format>
                file_name = f"{file_prefix}_freq{freq_idx}_phase{phase_idx}.{file_format}"
                file_path = os.path.join(output_folder, file_name)
                
                # 保存图像
                cv2.imwrite(file_path, pattern_8bit)
                pattern_files.append(file_path)
                
                print(f"已保存图案: {file_name}")
        
        return pattern_files
    
    def visualize_patterns(self, patterns, phase_shifts, show=True, save_folder=None):
        """
        可视化相移图案
        
        参数:
            patterns: 相移图案列表的列表 (按频率和相移组织)
            phase_shifts: 相移值列表
            show: 是否显示图像
            save_folder: 保存可视化结果的文件夹 (如果需要保存)
        """
        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        for freq_idx, freq_patterns in enumerate(patterns):
            # 创建一个大图，显示该频率下的所有相移图案
            n_phases = len(freq_patterns)
            fig, axes = plt.subplots(1, n_phases, figsize=(n_phases * 4, 4))
            
            if n_phases == 1:
                axes = [axes]
                
            fig.suptitle(f"频率 {freq_idx+1} 的相移图案")
            
            for phase_idx, pattern in enumerate(freq_patterns):
                ax = axes[phase_idx]
                im = ax.imshow(pattern, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"相移 = {phase_shifts[phase_idx]:.2f}")
                ax.axis('off')
            
            plt.tight_layout()
            
            if save_folder:
                fig_path = os.path.join(save_folder, f"patterns_freq{freq_idx}.png")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"已保存可视化结果: {fig_path}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
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
        # 初始投影仪内参 (基于典型投影仪参数)
        fx_proj = self.projector_width
        fy_proj = self.projector_width
        cx_proj = self.projector_width / 2
        cy_proj = self.projector_height / 2
        
        initial_projector_matrix = np.array([
            [fx_proj, 0, cx_proj],
            [0, fy_proj, cy_proj],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 初始投影仪畸变系数
        initial_projector_dist = np.zeros(5, dtype=np.float32)
        
        # 使用OpenCV的标定函数进行标定
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        ret, self.projector_matrix, self.projector_dist, rvec_proj, tvec_proj = cv2.calibrateCamera(
            [object_points], [projector_points], 
            (self.projector_width, self.projector_height),
            initial_projector_matrix, initial_projector_dist,
            flags=flags
        )
        
        # 计算投影仪相对于相机的外参
        R_proj, _ = cv2.Rodrigues(rvec_proj[0])
        T_proj = tvec_proj[0]
        
        # 相对变换: R和T从投影仪到相机的变换
        self.R = R_cam @ R_proj.T
        self.T = T_cam - self.R @ T_proj
        
        # 计算重投影误差
        reprojerr = 0
        for i in range(len(object_points)):
            # 将世界点投影到相机上
            imgpt, _ = cv2.projectPoints(object_points[i].reshape(1, 3), rvec, tvec, camera_matrix, camera_distortion)
            # 计算与实际相机点的距离
            err = cv2.norm(camera_points[i], imgpt, cv2.NORM_L2)
            reprojerr += err
            
        reprojerr /= len(object_points)
        print(f"相机重投影误差: {reprojerr} 像素")
        
        print("\n投影仪标定结果:")
        print("投影仪内参矩阵:")
        print(self.projector_matrix)
        print("\n投影仪畸变系数:")
        print(self.projector_dist)
        print("\n从投影仪到相机的旋转矩阵:")
        print(self.R)
        print("\n从投影仪到相机的平移向量:")
        print(self.T)
        
        return self.projector_matrix, self.projector_dist, self.R, self.T
        
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
            'translation_vector': self.T.tolist() if self.T is not None else None,
            'gamma_a': float(self.gamma_a),
            'gamma_b': float(self.gamma_b),
            'gamma_c': float(self.gamma_c)
        }
        
        # 添加元数据
        if include_metadata:
            calibration_data['calibration_time'] = timestamp
            calibration_data['description'] = "投影仪标定数据"
        
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
        
        if calibration_data['projector_matrix']:
            self.projector_matrix = np.array(calibration_data['projector_matrix'])
        else:
            self.projector_matrix = None
            
        if calibration_data['projector_dist']:
            self.projector_dist = np.array(calibration_data['projector_dist'])
        else:
            self.projector_dist = None
            
        if 'rotation_matrix' in calibration_data and calibration_data['rotation_matrix']:
            self.R = np.array(calibration_data['rotation_matrix'])
        else:
            self.R = None
            
        if 'translation_vector' in calibration_data and calibration_data['translation_vector']:
            self.T = np.array(calibration_data['translation_vector'])
        else:
            self.T = None
            
        self.gamma_a = calibration_data['gamma_a']
        self.gamma_b = calibration_data['gamma_b']
        self.gamma_c = calibration_data['gamma_c']
        
        print(f"从 {filename} 加载投影仪标定数据")
        
        return calibration_data

def detect_chessboard_and_calibrate(projector_width, projector_height, camera_params_file, 
                               pattern_images_folder, chessboard_size=(9, 6), square_size=20.0,
                               output_folder=None, visualize=True, camera_matrix=None, camera_dist=None):
    """
    使用投影图案到棋盘格上的图像进行投影仪标定
    
    参数:
        projector_width: 投影仪分辨率宽度
        projector_height: 投影仪分辨率高度
        camera_params_file: 相机标定参数文件，如果为None，则使用提供的矩阵和畸变系数
        pattern_images_folder: 包含投影图案到棋盘格上的图像文件夹
        chessboard_size: 棋盘格内角点数量 (宽, 高)
        square_size: 棋盘格方格尺寸(mm)
        output_folder: 输出文件夹
        visualize: 是否可视化结果
        camera_matrix: 手动提供的相机内参矩阵
        camera_dist: 手动提供的相机畸变系数
    
    返回:
        calibration: 投影仪标定对象
        calibration_file: 保存的标定文件路径
    """
    # 创建输出文件夹
    if output_folder is None:
        output_folder = os.path.join(pattern_images_folder, "projector_calibration_results")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建投影仪标定对象
    calibration = ProjectorCalibration(projector_width, projector_height)
    
    # 加载相机参数
    camera_matrix, camera_dist = calibration.load_camera_params(camera_params_file, camera_matrix, camera_dist)
    
    # 查找图像
    pattern_images = sorted(glob.glob(os.path.join(pattern_images_folder, '*.jpg')))
    pattern_images.extend(sorted(glob.glob(os.path.join(pattern_images_folder, '*.png'))))
    
    if len(pattern_images) == 0:
        raise ValueError(f"在文件夹 '{pattern_images_folder}' 中未找到图像文件")
    
    print(f"找到 {len(pattern_images)} 张图像")
    
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 应用实际方格尺寸
    
    # 检测棋盘格角点
    board_points = []  # 世界坐标系中的点
    correspondences = []  # 投影仪点和相机点的对应关系
    
    for i, img_path in enumerate(pattern_images):
        print(f"\n处理图像 {i+1}/{len(pattern_images)}: {os.path.basename(img_path)}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if not ret:
            print(f"未能在图像 {img_path} 中找到所有棋盘格角点")
            continue
        
        # 使用亚像素精度优化角点位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 绘制并显示角点
        if visualize:
            img_display = img.copy()
            cv2.drawChessboardCorners(img_display, chessboard_size, corners2, ret)
            # 保存标记了角点的图像
            marked_img_path = os.path.join(output_folder, f"corners_{os.path.basename(img_path)}")
            cv2.imwrite(marked_img_path, img_display)
            # 显示图像
            cv2.imshow('Chessboard Corners', cv2.resize(img_display, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(500)
        
        # 获取图像中的投影仪点和相机点
        # 这里简化处理，假设每个图像对应一个不同的棋盘格位置
        # 实际应用中需要根据具体的编码方案提取投影仪点
        
        # 这里我们假设投影仪投影的是一个结构光图案（如相移条纹）
        # 我们需要从图像中解码出投影仪坐标
        # 由于这是一个复杂的过程，这里我们简化为直接使用棋盘格角点的相对位置
        # 实际应用中需要实现真正的解码过程
        
        # 简化：假设投影仪投影的是棋盘格图案，投影仪坐标就是棋盘格角点的相对位置
        proj_points = []
        for j in range(chessboard_size[1]):
            for i in range(chessboard_size[0]):
                # 生成投影仪坐标 (归一化到投影仪分辨率)
                proj_x = int(i * projector_width / (chessboard_size[0] - 1))
                proj_y = int(j * projector_height / (chessboard_size[1] - 1))
                proj_points.append([proj_x, proj_y])
        
        # 转换为numpy数组
        proj_points = np.array(proj_points, dtype=np.float32)
        
        # 添加到对应关系列表
        for j in range(len(corners2)):
            correspondences.append({
                'projector_point': proj_points[j],
                'camera_point': corners2[j][0],
                'board_index': j
            })
        
        board_points.append(objp)
        
    if visualize:
        cv2.destroyAllWindows()
    
    if len(correspondences) == 0:
        raise ValueError("未能建立足够的投影仪-相机对应关系，无法进行标定")
    
    print(f"\n成功建立了 {len(correspondences)} 个投影仪-相机对应点")
    
    # 执行投影仪标定
    calibration.calibrate_projector_with_camera(
        camera_matrix, camera_dist, correspondences, board_points[0]
    )
    
    # 保存标定结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_file = os.path.join(output_folder, f"projector_calibration_{timestamp}.json")
    calibration_file, default_file = calibration.save_calibration(calibration_file)
    
    return calibration, calibration_file

def simulate_brightness_intensity_data():
    """生成模拟的亮度-强度数据用于测试"""
    # 模拟的强度值 (0-1范围)
    intensity = np.linspace(0, 1, 100)
    
    # 模拟的伽马响应 (非线性)
    gamma = 2.2
    brightness = np.power(intensity, 1/gamma)
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.02, len(brightness))
    brightness = brightness + noise
    brightness = np.clip(brightness, 0, 1)
    
    return brightness, intensity

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='投影仪标定工具')
    parser.add_argument('--camera-params', type=str, help='相机标定参数文件路径(.npy或.json)')
    parser.add_argument('--pattern-images', type=str, help='包含投影图案到棋盘格上的图像文件夹')
    parser.add_argument('--projector-width', type=int, default=1280, help='投影仪宽度分辨率')
    parser.add_argument('--projector-height', type=int, default=720, help='投影仪高度分辨率')
    parser.add_argument('--chessboard-width', type=int, default=9, help='棋盘格宽度内角点数量')
    parser.add_argument('--chessboard-height', type=int, default=6, help='棋盘格高度内角点数量')
    parser.add_argument('--square-size', type=float, default=20.0, help='棋盘格方格尺寸(mm)')
    parser.add_argument('--gamma-images', type=str, help='包含伽马校正灰度图像的文件夹')
    parser.add_argument('--output-folder', type=str, help='输出结果文件夹')
    parser.add_argument('--no-visualize', action='store_true', help='不显示可视化结果')
    parser.add_argument('--generate-patterns', action='store_true', help='生成相移图案并保存')
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据进行测试')
    
    args = parser.parse_args()
    
    # 如果使用模拟模式
    if args.simulate:
        print("\n=== 使用模拟数据进行投影仪标定演示 ===")
        
        # 创建投影仪标定对象
        calibration = ProjectorCalibration(projector_width=1280, projector_height=720)
        
        # 执行伽马校正
        print("\n1. 执行投影仪伽马校正...")
        brightness, intensity = simulate_brightness_intensity_data()
        gamma_params = calibration.calibrate_gamma(brightness, intensity)
        
        # 创建相移图案
        print("\n2. 创建相移图案...")
        patterns, phase_shifts = calibration.create_phase_shifting_patterns(frequencies=[1, 4, 12], shifts=4)
        
        # 显示第一个频率的第一个相移图案
        calibration.visualize_patterns(patterns, phase_shifts)
        
        # 保存相移图案
        if args.output_folder:
            pattern_folder = os.path.join(args.output_folder, "patterns")
        else:
            pattern_folder = "patterns"
            
        if not os.path.exists(pattern_folder):
            os.makedirs(pattern_folder)
            
        calibration.save_patterns(patterns, pattern_folder)
        
        # 模拟标定结果
        print("\n3. 模拟投影仪标定结果...")
        calibration.projector_matrix = np.array([
            [1280.0, 0.0, 640.0],
            [0.0, 1280.0, 360.0],
            [0.0, 0.0, 1.0]
        ])
        calibration.projector_dist = np.array([0.1, -0.05, 0.001, 0.001, 0.0])
        calibration.R = np.eye(3)
        calibration.T = np.array([100.0, 0.0, 0.0])
        
        # 保存标定结果
        if args.output_folder:
            calibration_file = os.path.join(args.output_folder, "projector_calibration_simulated.json")
        else:
            calibration_file = "projector_calibration_simulated.json"
            
        calibration.save_calibration(calibration_file)
        
        print("\n=== 模拟投影仪标定完成 ===")
        return
    
    # 初始化相机参数
    camera_matrix = None
    camera_dist = None
    camera_params_file = args.camera_params
    
    # 如果未提供相机参数文件，尝试查找默认位置或手动输入
    if camera_params_file is None:
        # 尝试查找默认相机标定文件
        default_paths = [
            "./calibration_results/camera_calibration_latest.json",
            "./camera_calibration_latest.json"
        ]
        
        found = False
        for path in default_paths:
            if os.path.exists(path):
                camera_params_file = path
                print(f"找到默认相机标定文件: {path}")
                found = True
                break
                
        if not found:
            # 如果没有找到默认文件，询问用户
            response = input("未找到默认相机标定文件。是否手动输入相机标定文件路径? (y/n) [默认:y]: ").strip().lower()
            
            if not response or response == 'y':
                camera_params_file = input("请输入相机标定参数文件路径(.npy或.json): ").strip()
                if not camera_params_file or not os.path.exists(camera_params_file):
                    print("未提供有效的相机标定参数文件。")
                    
                    # 询问是否手动输入相机内参
                    response = input("是否手动输入相机内参矩阵和畸变系数? (y/n) [默认:n]: ").strip().lower()
                    if response == 'y':
                        try:
                            print("请输入3x3相机内参矩阵 (每行用空格分隔值):")
                            matrix = []
                            for i in range(3):
                                row = list(map(float, input(f"行 {i+1}: ").strip().split()))
                                if len(row) != 3:
                                    raise ValueError("每行应该有3个值")
                                matrix.append(row)
                            camera_matrix = np.array(matrix)
                            
                            print("请输入畸变系数 (k1 k2 p1 p2 k3，用空格分隔):")
                            dist = list(map(float, input().strip().split()))
                            if len(dist) != 5:
                                raise ValueError("应该有5个畸变系数")
                            camera_dist = np.array(dist)
                            
                            print("成功输入相机参数")
                        except Exception as e:
                            print(f"输入相机参数时出错: {str(e)}")
                            return
                    else:
                        print("没有相机参数，无法进行标定。请先运行相机标定程序。")
                        return
    
    # 设置投影仪分辨率
    projector_width = args.projector_width
    projector_height = args.projector_height
    
    try:
        user_input = input(f"请输入投影仪分辨率 (宽 高) [默认: {projector_width} {projector_height}]: ").strip()
        if user_input:
            projector_width, projector_height = map(int, user_input.split())
    except:
        print(f"使用默认投影仪分辨率: {projector_width}x{projector_height}")
    
    # 创建投影仪标定对象
    calibration = ProjectorCalibration(projector_width, projector_height)
    
    # 检查是否需要执行伽马校正
    if args.gamma_images:
        gamma_folder = args.gamma_images
    else:
        gamma_calibration = input("\n是否需要执行伽马校正? (y/n) [默认:y]: ").strip().lower()
        if not gamma_calibration or gamma_calibration == 'y':
            gamma_folder = input("请输入包含伽马校正灰度图像的文件夹: ").strip()
            if not gamma_folder or not os.path.exists(gamma_folder):
                print("未提供有效的伽马校正图像文件夹，跳过伽马校正。")
                gamma_folder = None
        else:
            gamma_folder = None
    
    if gamma_folder and os.path.exists(gamma_folder):
        print("\n执行伽马校正...")
        try:
            brightness, intensity = calibration.collect_gamma_calibration_data(gamma_folder)
            gamma_params = calibration.calibrate_gamma(brightness, intensity, visualize=not args.no_visualize)
            print(f"伽马校正完成: a={gamma_params[0]:.3f}, b={gamma_params[1]:.3f}, c={gamma_params[2]:.3f}")
        except Exception as e:
            print(f"伽马校正失败: {str(e)}")
    
    # 检查是否需要生成相移图案
    if args.generate_patterns:
        print("\n生成相移图案...")
        patterns, phase_shifts = calibration.create_phase_shifting_patterns(frequencies=[1, 4, 12], shifts=4)
        
        if not args.no_visualize:
            calibration.visualize_patterns(patterns, phase_shifts)
        
        # 保存相移图案
        if args.output_folder:
            pattern_folder = os.path.join(args.output_folder, "patterns")
        else:
            pattern_folder = "patterns"
            
        if not os.path.exists(pattern_folder):
            os.makedirs(pattern_folder)
            
        calibration.save_patterns(patterns, pattern_folder)
        print(f"相移图案已保存至: {pattern_folder}")
    
    # 执行投影仪标定
    if args.pattern_images:
        pattern_images_folder = args.pattern_images
    else:
        calibrate_projector = input("\n是否执行投影仪标定? (y/n) [默认:y]: ").strip().lower()
        if not calibrate_projector or calibrate_projector == 'y':
            pattern_images_folder = input("请输入包含投影图案到棋盘格上的图像文件夹: ").strip()
            if not pattern_images_folder or not os.path.exists(pattern_images_folder):
                print("未提供有效的图像文件夹，退出程序。")
                return
        else:
            pattern_images_folder = None
    
    if pattern_images_folder and os.path.exists(pattern_images_folder):
        try:
            # 设置棋盘格参数
            chessboard_width = args.chessboard_width
            chessboard_height = args.chessboard_height
            square_size = args.square_size
            
            # 如果未通过命令行提供，则询问用户
            try:
                user_input = input(f"请输入棋盘格内角点数量 (宽 高) [默认: {chessboard_width} {chessboard_height}]: ").strip()
                if user_input:
                    chessboard_width, chessboard_height = map(int, user_input.split())
            except:
                print(f"使用默认棋盘格尺寸: {chessboard_width}x{chessboard_height}")
            
            try:
                user_input = input(f"请输入棋盘格方格尺寸(mm) [默认: {square_size}]: ").strip()
                if user_input:
                    square_size = float(user_input)
            except:
                print(f"使用默认方格尺寸: {square_size}mm")
            
            # 执行标定
            print("\n开始投影仪标定...")
            calibration, calibration_file = detect_chessboard_and_calibrate(
                projector_width, projector_height,
                camera_params_file,
                pattern_images_folder,
                chessboard_size=(chessboard_width, chessboard_height),
                square_size=square_size,
                output_folder=args.output_folder,
                visualize=not args.no_visualize,
                camera_matrix=camera_matrix,
                camera_dist=camera_dist
            )
            
            print(f"\n投影仪标定完成，结果已保存至: {calibration_file}")
            
            # 显示投影仪与相机之间的位姿关系
            print("\n投影仪与相机的位姿关系:")
            print("旋转矩阵 (从投影仪到相机):")
            print(calibration.R)
            print("\n平移向量 (从投影仪到相机，单位:mm):")
            print(calibration.T)
            
        except Exception as e:
            print(f"\n投影仪标定失败: {str(e)}")
    
    print("\n所有操作完成！")

if __name__ == "__main__":
    main() 