'''工具函数模块'''

from __future__ import annotations
from typing import Optional

import os
import json
import time
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cv2
import open3d as o3d

from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm, CameraMeasurement


def create_measurement_folder(folder_path: str, cameras_folders: list[str]) -> str:
    '''
    创建测量数据存储文件夹结构
    
    参数:
        folder_path (str): 基本文件夹路径
        cameras_folders (list[str]): 相机文件夹名称列表
    
    返回:
        measurement_folder_path (str): 创建的测量文件夹路径
    '''
    # 获取当前时间戳
    timestamp = time.time()
    date_time = datetime.datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建测量文件夹
    measurement_folder_path = os.path.join(folder_path, str_date_time)
    Path(measurement_folder_path).mkdir(parents=True, exist_ok=True)
    
    # 创建相机子文件夹
    for cam_folder in cameras_folders:
        cam_folder_path = os.path.join(measurement_folder_path, cam_folder)
        Path(cam_folder_path).mkdir(parents=True, exist_ok=True)
    
    return measurement_folder_path


def save_measurement_data(measurement, measurement_folder_path: str, measurement_file_name: str) -> None:
    '''
    保存测量数据到JSON文件
    
    参数:
        measurement: 要保存的测量对象
        measurement_folder_path (str): 保存文件夹路径
        measurement_file_name (str): 保存文件名
    '''
    # 创建要保存的数据字典
    data = {}
    
    # 保存测量基本信息
    data['phase_shifting_type'] = measurement.phase_shifting_type
    data['frequencies'] = measurement.frequencies
    data['shifts'] = measurement.shifts
    
    # 保存相机测量数据
    data['cameras_data'] = []
    for i, camera_data in enumerate(measurement.camera_results):
        cam_data = {}
        cam_data['fringe_orientation'] = camera_data.fringe_orientation
        
        # 保存图像文件名
        cam_data['imgs_file_names'] = []
        for freq_imgs in camera_data.imgs_file_names:
            freq_imgs_files = []
            for img_file in freq_imgs:
                freq_imgs_files.append(img_file)
            cam_data['imgs_file_names'].append(freq_imgs_files)
        
        data['cameras_data'].append(cam_data)
    
    # 将数据保存到JSON文件
    file_path = os.path.join(measurement_folder_path, measurement_file_name)
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_measurement_data(measurement_folder_path: str, measurement_file_name: str):
    '''
    从JSON文件加载测量数据
    
    参数:
        measurement_folder_path (str): 测量文件夹路径
        measurement_file_name (str): 测量文件名
    
    返回:
        measurement: 加载的测量对象
    '''
    from fpp_structures import FPPMeasurement, CameraMeasurement, PhaseShiftingAlgorithm
    
    # 读取JSON文件
    file_path = os.path.join(measurement_folder_path, measurement_file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 创建测量对象
    measurement = FPPMeasurement(
        phase_shifting_type=data['phase_shifting_type'],
        frequencies=data['frequencies'],
        shifts=data['shifts']
    )
    
    # 加载相机数据
    for cam_data in data['cameras_data']:
        camera_measurement = CameraMeasurement()
        camera_measurement.fringe_orientation = cam_data['fringe_orientation']
        
        # 加载图像文件名
        camera_measurement.imgs_file_names = []
        for freq_imgs_files in cam_data['imgs_file_names']:
            freq_imgs = []
            for img_file in freq_imgs_files:
                freq_imgs.append(img_file)
            camera_measurement.imgs_file_names.append(freq_imgs)
        
        # 加载图像
        camera_measurement.imgs_list = []
        for freq_imgs_files in camera_measurement.imgs_file_names:
            freq_imgs = []
            for img_file in freq_imgs_files:
                img_path = os.path.join(measurement_folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                freq_imgs.append(img)
            camera_measurement.imgs_list.append(freq_imgs)
        
        measurement.camera_results.append(camera_measurement)
    
    return measurement


def visualize_phases(measurement, camera_num: int = 0):
    '''
    可视化相位图
    
    参数:
        measurement: 测量对象
        camera_num (int): 要可视化的相机索引
    '''
    # 获取相机测量数据
    camera_data = measurement.camera_results[camera_num]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 绘制包裹相位图
    for i, phase in enumerate(camera_data.phases):
        ax = fig.add_subplot(2, len(camera_data.phases), i + 1)
        im = ax.imshow(phase, cmap='jet')
        ax.set_title(f'包裹相位 {i+1}')
        plt.colorbar(im, ax=ax)
    
    # 绘制解包相位图
    for i, unwrapped_phase in enumerate(camera_data.unwrapped_phases):
        ax = fig.add_subplot(2, len(camera_data.unwrapped_phases), i + 1 + len(camera_data.phases))
        im = ax.imshow(unwrapped_phase, cmap='jet')
        ax.set_title(f'解包相位 {i+1}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def visualize_3d_phase(measurement, camera_num: int = 0, frequency_num: int = 0):
    '''
    3D可视化相位图
    
    参数:
        measurement: 测量对象
        camera_num (int): 要可视化的相机索引
        frequency_num (int): 要可视化的频率索引
    '''
    # 获取相机测量数据
    camera_data = measurement.camera_results[camera_num]
    
    # 获取解包相位
    unwrapped_phase = camera_data.unwrapped_phases[frequency_num]
    
    # 创建图形
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # 创建网格
    X = np.arange(0, unwrapped_phase.shape[1], 1)
    Y = np.arange(0, unwrapped_phase.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    
    # 绘制3D表面
    surf = ax.plot_surface(X, Y, unwrapped_phase, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # 自定义Z轴
    ax.set_zlim(np.min(unwrapped_phase), np.max(unwrapped_phase))
    ax.zaxis.set_major_locator(LinearLocator(10))
    
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


def visualize_point_cloud(points, colors=None):
    '''
    使用Open3D可视化点云
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3), 可选
    '''
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置颜色（如果提供）
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
