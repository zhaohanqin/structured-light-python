#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相位解包裹示例程序2 - 水平和垂直方向

该脚本展示如何使用 unwrap_phase_modified.py 同时处理水平和垂直方向的
四步相移图像，并生成解包裹相位结果。示例使用模拟生成的条纹投影图像。
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import unwrap_phase as unwrap_phase


def generate_test_images(width=640, height=480, h_frequency=10, v_frequency=10, 
                         noise_level=0.05, direction="both"):
    """
    生成用于测试的四步相移图像，支持水平和垂直两个方向
    
    参数:
        width: 图像宽度
        height: 图像高度
        h_frequency: 水平条纹频率
        v_frequency: 垂直条纹频率
        noise_level: 噪声水平
        direction: 生成方向，可选 "horizontal", "vertical", "both"
    
    返回:
        images_dict: 包含水平和垂直方向图像的字典
        paths_dict: 包含图像路径的字典
    """
    # 创建保存目录
    os.makedirs("test_images", exist_ok=True)
    
    # 相移角度 (0°, 90°, 180°, 270°)
    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    # 创建坐标网格
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # 添加3D对象 (球形突起)
    cx, cy = width // 2, height // 2
    r = min(width, height) // 4
    zz = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            d = np.sqrt((j - cx)**2 + (i - cy)**2)
            if d < r:
                # 创建球面高度
                zz[i, j] = np.sqrt(r**2 - d**2) * 0.5
    
    images_dict = {"horizontal": [], "vertical": []}
    paths_dict = {"horizontal": [], "vertical": []}
    
    # 生成水平方向的条纹图像 (垂直条纹)
    if direction in ["horizontal", "both"]:
        for i, shift in enumerate(phase_shifts):
            # 垂直条纹 (水平方向变化)
            pattern = np.cos(2 * np.pi * h_frequency * xx / width + shift)
            
            # 添加物体变形 (相位调制)
            deformed_pattern = np.cos(2 * np.pi * h_frequency * xx / width + shift + zz * 0.3)
            
            # 转换到0-255范围
            image = ((deformed_pattern + 1) * 127.5).astype(np.uint8)
            
            # 添加噪声
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 保存图像
            image_path = f"test_images/h_phase_shift_{i}.png"
            cv2.imwrite(image_path, image)
            
            images_dict["horizontal"].append(image)
            paths_dict["horizontal"].append(image_path)
            
            print(f"已生成水平方向相移图像 {i+1}/4: {image_path}")
    
    # 生成垂直方向的条纹图像 (水平条纹)
    if direction in ["vertical", "both"]:
        for i, shift in enumerate(phase_shifts):
            # 水平条纹 (垂直方向变化)
            pattern = np.cos(2 * np.pi * v_frequency * yy / height + shift)
            
            # 添加物体变形 (相位调制)
            deformed_pattern = np.cos(2 * np.pi * v_frequency * yy / height + shift + zz * 0.3)
            
            # 转换到0-255范围
            image = ((deformed_pattern + 1) * 127.5).astype(np.uint8)
            
            # 添加噪声
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 保存图像
            image_path = f"test_images/v_phase_shift_{i}.png"
            cv2.imwrite(image_path, image)
            
            images_dict["vertical"].append(image)
            paths_dict["vertical"].append(image_path)
            
            print(f"已生成垂直方向相移图像 {i+1}/4: {image_path}")
    
    return images_dict, paths_dict


def visualize_3d_results(h_unwrapped=None, v_unwrapped=None, output_dir="test_results"):
    """
    可视化解包裹相位的3D结果
    
    参数:
        h_unwrapped: 水平方向解包裹相位
        v_unwrapped: 垂直方向解包裹相位
        output_dir: 输出目录
    """
    if h_unwrapped is None and v_unwrapped is None:
        print("没有可显示的解包裹相位数据")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算坐标网格
    if h_unwrapped is not None:
        height, width = h_unwrapped.shape
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # 绘制3D表面
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        
        # 使用解包裹相位作为高度
        ax.plot_surface(xx, yy, h_unwrapped, cmap='viridis', edgecolor='none')
        ax.set_title('水平方向解包裹相位 3D 表面')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('相位值')
        
        plt.savefig(os.path.join(output_dir, "horizontal_3d.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    if v_unwrapped is not None:
        height, width = v_unwrapped.shape
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        
        ax.plot_surface(xx, yy, v_unwrapped, cmap='plasma', edgecolor='none')
        ax.set_title('垂直方向解包裹相位 3D 表面')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('相位值')
        
        plt.savefig(os.path.join(output_dir, "vertical_3d.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 如果两个方向都有数据，生成组合视图
    if h_unwrapped is not None and v_unwrapped is not None:
        # 归一化两个相位图
        h_norm = (h_unwrapped - np.min(h_unwrapped)) / (np.max(h_unwrapped) - np.min(h_unwrapped))
        v_norm = (v_unwrapped - np.min(v_unwrapped)) / (np.max(v_unwrapped) - np.min(v_unwrapped))
        
        # 组合两个方向的相位图得到伪彩色图像
        combined_rgb = np.zeros((height, width, 3), dtype=np.float32)
        combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
        combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
        combined_rgb[:,:,2] = (h_norm + v_norm) / 2  # 蓝色通道为两者平均
        
        plt.figure(figsize=(10, 8))
        plt.imshow(combined_rgb)
        plt.title('水平和垂直方向相位组合图')
        plt.colorbar(label='归一化相位值')
        plt.savefig(os.path.join(output_dir, "combined_phase.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return combined_rgb


def generate_combined_phase_image(h_unwrapped, v_unwrapped, output_path=None):
    """
    生成水平和垂直方向相位组合图
    
    参数:
        h_unwrapped: 水平方向解包裹相位
        v_unwrapped: 垂直方向解包裹相位
        output_path: 输出路径 (可选)
    
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
    
    # 如果指定了输出路径，保存图像
    if output_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(combined_rgb)
        plt.title('水平和垂直方向相位组合图')
        plt.colorbar(label='归一化相位值')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return combined_rgb


def main():
    """主函数"""
    print("生成测试用的水平和垂直方向四步相移图像...")
    _, image_paths = generate_test_images(
        width=640, 
        height=480, 
        h_frequency=15,  # 水平方向频率 
        v_frequency=12,  # 垂直方向频率
        noise_level=0.03,
        direction="both"  # 同时生成水平和垂直方向
    )
    
    output_dir = "test_results_both"
    unwrapped_results = {}
    
    print("\n处理水平方向相移图像...")
    horizontal_unwrapped = unwrap_phase.process_four_step_images(
        image_paths=image_paths["horizontal"],
        output_dir=os.path.join(output_dir, "horizontal"),
        method="quality_guided",
        show_plots=True  # 显示图形
    )
    unwrapped_results["horizontal"] = horizontal_unwrapped
    
    print("\n处理垂直方向相移图像...")
    vertical_unwrapped = unwrap_phase.process_four_step_images(
        image_paths=image_paths["vertical"],
        output_dir=os.path.join(output_dir, "vertical"),
        method="quality_guided",
        show_plots=True  # 显示图形
    )
    unwrapped_results["vertical"] = vertical_unwrapped
    
    # 可视化3D结果
    print("\n生成3D可视化结果...")
    visualize_3d_results(
        h_unwrapped=unwrapped_results["horizontal"],
        v_unwrapped=unwrapped_results["vertical"],
        output_dir=output_dir
    )
    
    # 生成并显示组合相位图
    print("\n生成水平和垂直方向相位组合图...")
    combined_phase = generate_combined_phase_image(
        h_unwrapped=unwrapped_results["horizontal"],
        v_unwrapped=unwrapped_results["vertical"],
        output_path=os.path.join(output_dir, "combined_phase_2.png")
    )
    
    # 显示组合相位图
    plt.figure(figsize=(12, 10))
    plt.imshow(combined_phase)
    plt.title('水平和垂直方向相位组合图')
    plt.colorbar(label='归一化相位值')
    plt.show()
    
    print("\n示例运行完成！请查看 test_results_both 目录中的结果。")


if __name__ == "__main__":
    main() 