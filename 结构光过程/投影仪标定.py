import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json

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
        self.gamma_a = 1.0            # 伽马校正参数 a
        self.gamma_b = 1.0            # 伽马校正参数 b
        self.gamma_c = 0.0            # 伽马校正参数 c
    
    def calibrate_gamma(self, brightness_data, intensity_data):
        """
        根据亮度-强度数据校正投影仪伽马曲线
        
        参数:
            brightness_data: 相机捕获的亮度值列表
            intensity_data: 对应的投影强度值列表
        
        返回:
            gamma_params: 伽马校正参数 (a, b, c)
        """
        # 转换为numpy数组
        brightness = np.array(brightness_data)
        intensity = np.array(intensity_data)
        
        # 查找饱和水平
        saturation_level = 0.95
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
        popt, pcov = optimize.curve_fit(gamma_func, int_reduced, brt_reduced, p0=(1, 1, 0))
        
        print(f"拟合的伽马函数 - Iout = {popt[0]:.3f} * (Iin + {popt[2]:.3f}) ^ {popt[1]:.3f}")
        
        # 保存伽马校正参数
        self.gamma_a = popt[0]
        self.gamma_b = popt[1]
        self.gamma_c = popt[2]
        
        # 绘制拟合的伽马函数
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
        plt.savefig("projector_gamma_curve.png")
        plt.show()
        
        return popt
    
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
    
    def create_phase_shifting_patterns(self, frequencies=[1, 4, 12], shifts=4, vertical=True):
        """
        创建相移图案用于投影仪标定
        
        参数:
            frequencies: 使用的频率列表
            shifts: 每个频率的相移数量
            vertical: 是否创建垂直条纹(True)或水平条纹(False)
        
        返回:
            patterns: 相移图案列表的列表，按频率和相移组织
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
        
        # 初始投影仪位姿 (假设与相机位置相近但有一定偏移)
        initial_R_proj = np.eye(3, dtype=np.float32)  # 单位旋转矩阵
        initial_T_proj = np.array([100, 0, 0], dtype=np.float32)  # 假设投影仪在相机右侧100mm处
        
        # 使用OpenCV的标定函数进行标定
        # 注意：这里的实现是简化的，实际中可能需要更复杂的优化过程
        ret, self.projector_matrix, self.projector_dist, rvec_proj, tvec_proj = cv2.calibrateCamera(
            [object_points], [projector_points], 
            (self.projector_width, self.projector_height),
            initial_projector_matrix, initial_projector_dist,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        
        # 计算投影仪相对于相机的外参
        R_proj, _ = cv2.Rodrigues(rvec_proj[0])
        T_proj = tvec_proj[0]
        
        # 相对变换: R和T从投影仪到相机的变换
        R = R_cam @ R_proj.T
        T = T_cam - R @ T_proj
        
        print("\n投影仪标定结果:")
        print("投影仪内参矩阵:")
        print(self.projector_matrix)
        print("\n投影仪畸变系数:")
        print(self.projector_dist)
        
        return self.projector_matrix, self.projector_dist, R, T
    
    def save_calibration(self, filename):
        """
        保存标定结果到文件
        
        参数:
            filename: 保存文件名
        """
        calibration_data = {
            'projector_width': self.projector_width,
            'projector_height': self.projector_height,
            'projector_matrix': self.projector_matrix.tolist() if self.projector_matrix is not None else None,
            'projector_dist': self.projector_dist.tolist() if self.projector_dist is not None else None,
            'gamma_a': float(self.gamma_a),
            'gamma_b': float(self.gamma_b),
            'gamma_c': float(self.gamma_c)
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"投影仪标定数据已保存至 {filename}")
    
    def load_calibration(self, filename):
        """
        从文件加载标定结果
        
        参数:
            filename: 标定文件名
        """
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        self.projector_width = calibration_data['projector_width']
        self.projector_height = calibration_data['projector_height']
        self.projector_matrix = np.array(calibration_data['projector_matrix']) if calibration_data['projector_matrix'] else None
        self.projector_dist = np.array(calibration_data['projector_dist']) if calibration_data['projector_dist'] else None
        self.gamma_a = calibration_data['gamma_a']
        self.gamma_b = calibration_data['gamma_b']
        self.gamma_c = calibration_data['gamma_c']
        
        print(f"从 {filename} 加载投影仪标定数据")


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


if __name__ == "__main__":
    # 创建投影仪标定对象
    calibration = ProjectorCalibration(projector_width=1280, projector_height=800)
    
    # 示例1: 伽马校正
    print("执行投影仪伽马校正...")
    brightness, intensity = simulate_brightness_intensity_data()
    gamma_params = calibration.calibrate_gamma(brightness, intensity)
    
    # 示例2: 创建相移图案
    print("\n创建相移图案...")
    patterns, phase_shifts = calibration.create_phase_shifting_patterns(frequencies=[1, 4, 12], shifts=4)
    
    # 显示第一个频率的第一个相移图案
    plt.figure(figsize=(8, 6))
    plt.imshow(patterns[0][0], cmap='gray')
    plt.title(f"频率1的第一个相移图案 (相移={phase_shifts[0]:.2f})")
    plt.colorbar()
    plt.savefig("phase_pattern_example.png")
    plt.show()
    
    # 保存标定结果
    calibration.save_calibration("projector_calibration.json")
    
    print("\n完成投影仪标定示例。在实际应用中，需要相机捕获图像进行完整标定。") 