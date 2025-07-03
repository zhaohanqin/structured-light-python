#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定独立程序

该程序允许用户使用一组棋盘格标定板图像对相机进行标定，获取相机的内参矩阵和畸变系数。
可以直接运行此程序，按照提示输入相关参数。

使用方法:
1. 准备多张不同角度拍摄的棋盘格标定板图像，保存在一个文件夹中
2. 运行程序，按照提示输入图像文件夹路径、棋盘格角点数量和实际尺寸
3. 程序将自动进行标定，并保存结果

作者: [Your Name]
日期: [Date]
"""

import os
import sys
import numpy as np
import cv2
import glob
import json
from matplotlib import pyplot as plt
import argparse
from datetime import datetime

def calibrate_camera(images_folder, chessboard_size=(9, 6), square_size=20.0, 
                     visualize=True, delay=500):
    """
    使用棋盘格图像进行相机标定
    
    参数:
        images_folder: 包含棋盘格图像的文件夹路径
        chessboard_size: 棋盘格内角点数量 (宽, 高)
        square_size: 棋盘格方格尺寸(mm)
        visualize: 是否可视化角点检测结果
        delay: 可视化图像显示的延迟时间(ms)
    
    返回:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        rvecs: 旋转向量
        tvecs: 平移向量
        reprojection_error: 重投影误差
        image_size: 图像尺寸
    """
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 应用实际方格尺寸
    
    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D空间中的点
    imgpoints = []  # 图像平面上的点
    
    # 获取图像文件列表
    images = glob.glob(os.path.join(images_folder, '*.jpg'))
    images.extend(glob.glob(os.path.join(images_folder, '*.png')))
    
    if len(images) == 0:
        raise ValueError(f"在文件夹 '{images_folder}' 中未找到jpg或png图像文件")
    
    # 图像尺寸
    img_shape = None
    
    print(f"找到 {len(images)} 张校准图像")
    
    # 创建结果文件夹
    results_folder = os.path.join(images_folder, "calibration_results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # 处理每张图像
    successful_images = []
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 如果找到角点，添加对象点和图像点
        if ret:
            objpoints.append(objp)
            
            # 使用亚像素精度优化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images.append(os.path.basename(fname))
            
            # 绘制并显示角点
            if visualize:
                img_display = img.copy()
                cv2.drawChessboardCorners(img_display, chessboard_size, corners2, ret)
                
                # 保存标记了角点的图像
                marked_img_path = os.path.join(results_folder, f"corners_{os.path.basename(fname)}")
                cv2.imwrite(marked_img_path, img_display)
                
                # 显示图像
                cv2.imshow('Chessboard Corners', cv2.resize(img_display, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(delay)
                
                # 显示进度
                sys.stdout.write(f"\r处理图像 {i+1}/{len(images)}: {os.path.basename(fname)}")
                sys.stdout.flush()
        else:
            print(f"\n未能在图像 {fname} 中找到所有棋盘格角点")
    
    print(f"\n成功处理了 {len(objpoints)}/{len(images)} 张图像")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if len(objpoints) == 0:
        raise ValueError("未找到任何有效的棋盘格图像，无法进行标定")
    
    # 执行相机标定
    flags = 0
    # 可以添加额外的标志，如 cv2.CALIB_FIX_K3
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None, flags=flags
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    print(f"平均重投影误差: {reprojection_error} 像素")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error, img_shape, successful_images

def save_calibration_results(output_folder, output_basename, camera_matrix, dist_coeffs, 
                           image_size, reprojection_error, chessboard_info, successful_images):
    """
    保存标定结果到文件
    
    参数:
        output_folder: 输出文件夹
        output_basename: 输出文件基本名称
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        image_size: 图像尺寸
        reprojection_error: 重投影误差
        chessboard_info: 棋盘格信息 (尺寸和方格大小)
        successful_images: 成功处理的图像列表
    """
    # 创建结果目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 生成当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为NumPy格式 (.npy)
    npy_file = os.path.join(output_folder, f"{output_basename}_{timestamp}.npy")
    calibration_data_npy = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }
    np.save(npy_file, calibration_data_npy)
    
    # 保存为JSON格式 (.json)
    json_file = os.path.join(output_folder, f"{output_basename}_{timestamp}.json")
    calibration_data_json = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'image_size': image_size,
        'reprojection_error': float(reprojection_error),
        'chessboard_size': chessboard_info['size'],
        'square_size': chessboard_info['square_size'],
        'calibration_time': timestamp,
        'successful_images': successful_images
    }
    
    with open(json_file, 'w') as f:
        json.dump(calibration_data_json, f, indent=4)
    
    # 同时保存一个固定名称的文件，方便其他程序直接引用
    default_json_file = os.path.join(output_folder, "camera_calibration_latest.json")
    with open(default_json_file, 'w') as f:
        json.dump(calibration_data_json, f, indent=4)
    
    print(f"标定结果已保存至:")
    print(f"  - NumPy格式: {npy_file}")
    print(f"  - JSON格式: {json_file}")
    print(f"  - 最新标定结果: {default_json_file} (供其他程序引用)")
    
    return json_file, default_json_file

def test_undistortion(image_path, camera_matrix, dist_coeffs, output_folder=None, alpha=1.0):
    """
    测试畸变校正效果
    
    参数:
        image_path: 输入图像路径
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        output_folder: 输出文件夹路径(如果需要保存结果)
        alpha: 缩放参数(0-1), 0表示最小裁剪, 1表示无裁剪
    
    返回:
        dst: 校正后的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # 获取最佳相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
    
    # 校正图像
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # 裁剪结果
    if roi[2] > 0 and roi[3] > 0:  # 确保ROI有效
        x, y, w, h = roi
        dst_cropped = dst[y:y+h, x:x+w]
    else:
        dst_cropped = dst
    
    # 创建比较图像
    comparison = np.hstack((img, dst))
    
    # 显示原始图像和校正后的图像
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 校正后未裁剪的图像
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('校正后图像 (未裁剪)')
    plt.axis('off')
    
    # 校正后裁剪的图像
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(dst_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后图像 (裁剪)')
    plt.axis('off')
    
    # 原始与校正后的对比
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('原始图像 vs 校正后图像')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存校正后的图像
        undistorted_path = os.path.join(output_folder, f"{base_name}_undistorted.jpg")
        cv2.imwrite(undistorted_path, dst)
        
        # 保存裁剪后的图像
        cropped_path = os.path.join(output_folder, f"{base_name}_undistorted_cropped.jpg")
        cv2.imwrite(cropped_path, dst_cropped)
        
        # 保存对比图
        comparison_path = os.path.join(output_folder, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        # 保存matplotlib图
        plt_path = os.path.join(output_folder, f"{base_name}_undistortion_results.png")
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        
        print(f"校正结果已保存至: {output_folder}")
    
    plt.show()
    
    return dst_cropped

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机标定工具')
    parser.add_argument('--images', type=str, help='包含标定图像的文件夹路径')
    parser.add_argument('--width', type=int, default=9, help='棋盘格宽度内角点数量')
    parser.add_argument('--height', type=int, default=6, help='棋盘格高度内角点数量')
    parser.add_argument('--size', type=float, default=20.0, help='棋盘格方格尺寸(mm)')
    parser.add_argument('--no-visualize', action='store_true', help='不显示角点检测过程')
    parser.add_argument('--delay', type=int, default=500, help='角点检测可视化时的延迟(ms)')
    parser.add_argument('--test-image', type=str, help='用于测试畸变校正的图像路径')
    parser.add_argument('--output', type=str, help='自定义输出文件夹路径')
    
    args = parser.parse_args()
    
    # 如果没有通过命令行提供图像文件夹，则请用户输入
    images_folder = args.images
    if images_folder is None:
        images_folder = input("请输入包含棋盘格图像的文件夹路径: ").strip()
        if not images_folder:
            print("未提供有效的图像文件夹路径，退出程序。")
            return
    
    # 确保文件夹存在
    if not os.path.exists(images_folder):
        print(f"错误: 文件夹 {images_folder} 不存在")
        return
    
    # 获取棋盘格参数
    board_width = args.width
    board_height = args.height
    square_size = args.size
    
    # 如果未通过命令行提供，则询问用户
    if args.width == 9 and args.height == 6:  # 检查是否使用了默认值
        try:
            user_input = input(f"请输入棋盘格内角点数量 (宽 高) [默认: {board_width} {board_height}]: ").strip()
            if user_input:
                board_width, board_height = map(int, user_input.split())
        except:
            print(f"使用默认棋盘格尺寸: {board_width}x{board_height}")
    
    if args.size == 20.0:  # 检查是否使用了默认值
        try:
            user_input = input(f"请输入棋盘格方格尺寸(mm) [默认: {square_size}]: ").strip()
            if user_input:
                square_size = float(user_input)
        except:
            print(f"使用默认方格尺寸: {square_size}mm")
    
    print(f"\n使用以下参数进行标定:")
    print(f"  - 图像文件夹: {images_folder}")
    print(f"  - 棋盘格内角点: {board_width}x{board_height}")
    print(f"  - 方格尺寸: {square_size}mm")
    print(f"  - 可视化: {'否' if args.no_visualize else '是'}")
    print()
    
    try:
        # 执行标定
        camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error, image_size, successful_images = calibrate_camera(
            images_folder,
            chessboard_size=(board_width, board_height),
            square_size=square_size,
            visualize=not args.no_visualize,
            delay=args.delay
        )
        
        # 打印标定结果
        print("\n相机标定结果:")
        print("相机内参矩阵:")
        print(camera_matrix)
        print("\n畸变系数 (k1, k2, p1, p2, k3):")
        print(dist_coeffs)
        print(f"\n图像尺寸: {image_size[0]}x{image_size[1]}")
        
        # 保存标定结果
        chessboard_info = {
            'size': (board_width, board_height),
            'square_size': square_size
        }
        
        # 确定输出文件夹
        if args.output:
            results_folder = args.output
        else:
            results_folder = os.path.join(images_folder, "calibration_results")
        
        # 保存标定结果
        calibration_file, default_file = save_calibration_results(
            results_folder, 
            "camera_calibration", 
            camera_matrix, 
            dist_coeffs, 
            image_size, 
            reprojection_error, 
            chessboard_info,
            successful_images
        )
        
        print(f"\n默认标定文件路径: {default_file}")
        print("可以使用此文件路径作为投影仪标定程序的输入")
        
        # 测试畸变校正
        test_image = args.test_image
        if test_image is None:
            # 如果没有提供测试图像，询问用户是否要使用一张标定图像作为测试
            response = input("\n是否要测试畸变校正? (y/n) [默认:y]: ").strip().lower()
            if not response or response == 'y':
                images = glob.glob(os.path.join(images_folder, '*.jpg'))
                images.extend(glob.glob(os.path.join(images_folder, '*.png')))
                if images:
                    test_image = images[0]  # 使用第一张图像作为测试
        
        if test_image and os.path.exists(test_image):
            print(f"\n使用图像 {test_image} 测试畸变校正...")
            test_undistortion(
                test_image, 
                camera_matrix, 
                dist_coeffs, 
                output_folder=results_folder
            )
        
        print("\n相机标定完成！")
        
    except Exception as e:
        print(f"\n标定过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 