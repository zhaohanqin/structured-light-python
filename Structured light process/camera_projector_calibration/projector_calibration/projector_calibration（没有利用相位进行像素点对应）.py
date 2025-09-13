#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影仪标定独立程序 (简化版)

该程序允许用户使用相机拍摄的投影图案图像对投影仪进行标定，获取投影仪的内参矩阵、畸变系数以及投影仪与相机的相对位置关系。
需要先完成相机标定，并提供相机的内参矩阵和畸变系数。

【像素点匹配方法】
本程序使用传统的基于标定板的方法进行投影仪标定，不使用相位解包裹技术：
1. 投影仪投射特定图案（如棋盘格）到标定板上
2. 相机拍摄标定板图像
3. 检测标定板上的特征点（棋盘格角点或圆心）
4. 假设投影仪图案与标定板点一一对应，直接建立对应关系
5. 使用这些对应关系计算投影仪参数

此方法实现简单，但精度相对较低，适用于一般应用场景。

使用方法:
1. 确保已经完成相机标定，并有相机标定结果文件
2. 准备好相机拍摄的，投影仪投射了图案（例如棋盘格）到标定板上的图像
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
import argparse
from datetime import datetime

# 配置matplotlib支持中文显示
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    print("已配置matplotlib支持中文显示")
except ImportError:
    print("警告: 未安装matplotlib，无法在图像上显示中文。")
    FigureCanvasAgg = None


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
    if FigureCanvasAgg is None:
        # 如果matplotlib未导入，则使用OpenCV的英文文本作为备选
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2)
        return img

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

def show_calibration_tips(board_type):
    """
    为用户提供针对所选标定板类型的最佳实践建议

    参数:
        board_type: 标定板类型
    """
    print("\n===== 标定板使用建议 =====")

    if board_type == 'chessboard':
        print("【棋盘格标定板最佳实践】")
        print("- 确保棋盘格打印精确，无变形")
        print("- 标定板应固定在坚硬平面上，避免弯曲")
        print("- 拍摄时覆盖相机视场的不同区域（特别是边角）")
        print("- 以不同角度（约15-45度）拍摄至少10张图像")
        print("- 避免强反光和不均匀光照")
        print("- 确保整个标定板清晰可见")
    elif board_type == 'circles':
        print("【圆形标定板(白底黑圆)最佳实践】")
        print("- 适用于光照条件不理想的场景")
        print("- 圆点大小应适中，过大或过小都会影响检测")
        print("- 拍摄时避免极端视角，防止圆变成椭圆影响检测")
        print("- 确保光照均匀，减少阴影")
        print("- 对于光滑表面，考虑使用哑光材料打印以减少反光")
    elif board_type == 'ring_circles':
        print("【空心圆环标定板最佳实践】")
        print("- 最适合强光或反光条件下使用")
        print("- 圆环厚度应适中，太细可能检测不到")
        print("- 可以在白纸上用黑色记号笔手绘，确保圆环闭合")
        print("- 特别适合玻璃表面或有反光的场景")
        print("- 保持圆环形状规则，避免变形")

    print("\n【通用技巧】")
    print("- 标定板与相机距离应适中，不要太近或太远")
    print("- 确保图像中的标定点至少占图像的1/3区域")
    print("- 保持标定板完全位于相机视场内")
    print("- 图像应该清晰无模糊")
    print("============================")

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

class ProjectorCalibration:
    """投影仪标定类，实现投影仪内参外参标定"""

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
                except Exception as e:
                    print(f"无法从文件 {path} 加载标定参数: {e}")

        # 如果都找不到，则抛出错误
        raise ValueError("未提供相机标定参数文件或直接参数，且未找到默认位置的标定文件")

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
        projector_points_list = []
        camera_points_list = []
        object_points_list = []

        # OpenCV标定函数需要每个视图一个点列表
        num_views = len(proj_cam_correspondences)
        for i in range(num_views):
            projector_points_list.append(np.array([corr['projector_point'] for corr in proj_cam_correspondences[i]], dtype=np.float32))
            camera_points_list.append(np.array([corr['camera_point'] for corr in proj_cam_correspondences[i]], dtype=np.float32))
            object_points_list.append(board_points) # 假设所有视图的标定板都一样

        # 首先从相机角度求解每个视图的外参
        rvecs_cam = []
        tvecs_cam = []
        for i in range(num_views):
            _, rvec, tvec = cv2.solvePnP(object_points_list[i], camera_points_list[i], camera_matrix, camera_distortion)
            rvecs_cam.append(rvec)
            tvecs_cam.append(tvec)

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
            object_points_list, projector_points_list,
            (self.projector_width, self.projector_height),
            initial_projector_matrix, initial_projector_dist,
            flags=flags
        )
        
        # 为了得到唯一的R,T，我们只使用第一个视图的相机和投影仪外参来计算相对位姿
        # 假设相机与投影仪的位置关系在所有拍摄中是固定的
        R_cam, _ = cv2.Rodrigues(rvecs_cam[0])
        T_cam = tvecs_cam[0]
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
            calibration_data['description'] = "投影仪标定数据 (简化版)"

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

def detect_chessboard_and_calibrate(projector_width, projector_height, camera_params_file,
                               pattern_images_folder, board_type="chessboard", chessboard_size=(9, 6), square_size=20.0,
                               output_folder=None, visualize=True, camera_matrix=None, camera_dist=None):
    """
    使用投影图案到标定板上的图像进行投影仪标定

    参数:
        projector_width: 投影仪分辨率宽度
        projector_height: 投影仪分辨率高度
        camera_params_file: 相机标定参数文件，如果为None，则使用提供的矩阵和畸变系数
        pattern_images_folder: 包含投影图案到标定板上的图像文件夹
        board_type: 标定板类型 ('chessboard', 'circles' 或 'ring_circles')
        chessboard_size: 标定板内角点数量 (宽, 高)
        square_size: 标定板方格尺寸或圆心间距(mm)
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
    image_paths = sorted(glob.glob(os.path.join(pattern_images_folder, '*.jpg')))
    image_paths.extend(sorted(glob.glob(os.path.join(pattern_images_folder, '*.png'))))
    if not image_paths:
        raise ValueError(f"在文件夹 '{pattern_images_folder}' 中未找到图像文件")
    print(f"找到 {len(image_paths)} 张图像")

    # 准备对象点 (世界坐标系)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 获取标定板检测参数
    detection_params = configure_detection_parameters(board_type)

    # 检测标定板角点或圆心
    all_correspondences = []  # 包含所有视图的对应关系

    for i, img_path in enumerate(image_paths):
        print(f"\n处理图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        gray = preprocess_image(img, board_type)
        found, corners = False, None

        if board_type == 'chessboard':
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                found = True
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), detection_params['criteria'])
        elif board_type in ['circles', 'ring_circles']:
            ret, corners = cv2.findCirclesGrid(
                image=gray, patternSize=chessboard_size, flags=detection_params['flags'],
                blobDetector=detection_params.get('detector')
            )
            found = ret

        if found:
            print(f"在图像 {os.path.basename(img_path)} 中找到标定点")
            if visualize:
                img_display = img.copy()
                cv2.drawChessboardCorners(img_display, chessboard_size, corners, True)
                marked_img_path = os.path.join(output_folder, f"corners_marked_{os.path.basename(img_path)}")
                cv2.imwrite(marked_img_path, img_display)
                cv2.imshow('Calibration Points', cv2.resize(img_display, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(500)

            # 假设投影仪图案与标定板点一一对应
            proj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 2), np.float32)
            proj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

            # 归一化投影仪坐标到像素坐标
            proj_points[:, 0] = proj_points[:, 0] * (projector_width - 1) / (chessboard_size[0] - 1)
            proj_points[:, 1] = proj_points[:, 1] * (projector_height - 1) / (chessboard_size[1] - 1)
            
            view_correspondences = []
            for j in range(len(corners)):
                view_correspondences.append({
                    'projector_point': proj_points[j],
                    'camera_point': corners[j][0]
                })
            all_correspondences.append(view_correspondences)
        else:
            print(f"未能在图像 {os.path.basename(img_path)} 中找到所有标定点")

    if visualize:
        cv2.destroyAllWindows()

    if not all_correspondences:
        raise ValueError("未能建立任何投影仪-相机对应关系，无法进行标定")

    print(f"\n成功处理了 {len(all_correspondences)} 张图像，建立了对应点关系")

    # 执行投影仪标定
    _, _, _, _, reproj_error = calibration.calibrate_projector_with_camera(
        camera_matrix, camera_dist, all_correspondences, objp
    )

    # 评估标定质量
    assess_calibration_quality(reproj_error, board_type)

    # 保存标定结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_file = os.path.join(output_folder, f"projector_calibration_{timestamp}.json")
    calibration_file, _ = calibration.save_calibration(calibration_file)

    return calibration, calibration_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='投影仪标定工具 (简化版)')
    parser.add_argument('--camera-params', type=str, help='相机标定参数文件路径(.json或.npy)')
    parser.add_argument('--pattern-images', type=str, help='包含投影图案到标定板上的图像文件夹')
    parser.add_argument('--projector-width', type=int, default=1280, help='投影仪宽度分辨率')
    parser.add_argument('--projector-height', type=int, default=720, help='投影仪高度分辨率')
    parser.add_argument('--board-type', type=str, default=None,
                       choices=['chessboard', 'circles', 'ring_circles'], help='标定板类型')
    parser.add_argument('--chessboard-width', type=int, default=9, help='标定板宽度点数量')
    parser.add_argument('--chessboard-height', type=int, default=6, help='标定板高度点数量')
    parser.add_argument('--square-size', type=float, default=20.0, help='标定板方格尺寸或圆心间距(mm)')
    parser.add_argument('--output-folder', type=str, help='输出结果文件夹')
    parser.add_argument('--no-visualize', action='store_true', help='不显示可视化结果')

    args = parser.parse_args()

    # 如果没有提供任何参数，显示交互式菜单
    if len(sys.argv) == 1:
        print("\n欢迎使用投影仪标定工具！")
        print("=" * 50)
        camera_params_file = input("请输入相机标定参数文件路径 (留空则自动查找): ").strip()
        pattern_images_folder = input("请输入包含标定图像的文件夹: ").strip()
        
        if not pattern_images_folder or not os.path.exists(pattern_images_folder):
            print("未提供有效的图像文件夹，无法继续标定。")
            return
            
        args.camera_params = camera_params_file
        args.pattern_images = pattern_images_folder
        
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
    
    # 检查必要参数
    if not args.pattern_images:
        print("错误: 必须提供包含标定图像的文件夹路径 --pattern-images")
        parser.print_help()
        return

    # 获取标定板类型
    board_type = args.board_type if args.board_type else select_calibration_board_type()
    show_calibration_tips(board_type)

    # 执行标定
    try:
        print("\n开始投影仪标定...")
        calibration, calibration_file = detect_chessboard_and_calibrate(
            projector_width=args.projector_width, 
            projector_height=args.projector_height,
            camera_params_file=args.camera_params,
            pattern_images_folder=args.pattern_images,
            board_type=board_type,
            chessboard_size=(args.chessboard_width, args.chessboard_height),
            square_size=args.square_size,
            output_folder=args.output_folder,
            visualize=not args.no_visualize
        )

        print(f"\n投影仪标定完成，结果已保存至: {calibration_file}")

        # 显示投影仪与相机之间的位姿关系
        print("\n投影仪与相机的位姿关系:")
        print("旋转矩阵 (从投影仪到相机):")
        print(calibration.R)
        print("\n平移向量 (从投影仪到相机，单位:mm):")
        print(calibration.T)

    except Exception as e:
        print(f"\n投影仪标定失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n所有操作完成！")

if __name__ == "__main__":
    main() 