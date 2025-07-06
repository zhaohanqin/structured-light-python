#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相移条纹图案生成器

该程序用于生成结构光3D重建和投影仪标定所需的相移条纹图案。
生成的图案将保存为图像文件，可直接通过投影仪投射到标定板或待重建物体上。

使用方法:
1. 设置投影仪分辨率
2. 可选：设置伽马校正参数
3. 生成并保存相移图案
4. 可选：可视化预览生成的图案

使用示例:
$ python phase_pattern_generator.py --width 1280 --height 720 --freqs 1,4,12 --shifts 4 --output patterns
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class PhasePatternGenerator:
    """相移条纹图案生成类"""
    
    def __init__(self, projector_width=1280, projector_height=800):
        """
        初始化图案生成器
        
        参数:
            projector_width: 投影仪分辨率宽度
            projector_height: 投影仪分辨率高度
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.gamma_a = 1.0            # 伽马校正参数 a
        self.gamma_b = 1.0            # 伽马校正参数 b
        self.gamma_c = 0.0            # 伽马校正参数 c
    
    def set_gamma_correction(self, a=1.0, b=1.0, c=0.0):
        """
        设置伽马校正参数
        
        参数:
            a, b, c: 伽马校正公式参数 - Iout = a * (Iin + c) ^ b
        """
        self.gamma_a = a
        self.gamma_b = b
        self.gamma_c = c
        print(f"设置伽马校正参数: a={a}, b={b}, c={c}")
        
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
                if self.gamma_a != 1.0 or self.gamma_b != 1.0 or self.gamma_c != 0.0:
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
    
    def create_gray_patterns(self, levels=10):
        """
        创建灰度级别图案，用于伽马校正
        
        参数:
            levels: 灰度级别数量
            
        返回:
            patterns: 灰度图案列表
            intensities: 对应的强度值列表
        """
        patterns = []
        intensities = np.linspace(0, 1, levels)
        
        for intensity in intensities:
            # 创建均匀灰度图案
            pattern = np.ones((self.projector_height, self.projector_width), dtype=np.float32) * intensity
            patterns.append(pattern)
            
        return patterns, intensities

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相移条纹图案生成器')
    parser.add_argument('--width', type=int, default=1280, help='投影仪宽度分辨率')
    parser.add_argument('--height', type=int, default=720, help='投影仪高度分辨率')
    parser.add_argument('--freqs', type=str, default="1,4,12", help='频率列表，用逗号分隔')
    parser.add_argument('--shifts', type=int, default=4, help='每个频率的相移数量')
    parser.add_argument('--horizontal', action='store_true', help='创建水平条纹，默认为垂直条纹')
    parser.add_argument('--gamma-a', type=float, default=1.0, help='伽马校正参数a')
    parser.add_argument('--gamma-b', type=float, default=1.0, help='伽马校正参数b')
    parser.add_argument('--gamma-c', type=float, default=0.0, help='伽马校正参数c')
    parser.add_argument('--output', type=str, default='patterns', help='输出文件夹')
    parser.add_argument('--prefix', type=str, default='pattern', help='文件名前缀')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg'], help='输出图像格式')
    parser.add_argument('--no-visualize', action='store_true', help='不显示可视化结果')
    parser.add_argument('--gray', action='store_true', help='同时生成灰度图案用于伽马校正')
    parser.add_argument('--gray-levels', type=int, default=10, help='灰度级别数量')
    
    args = parser.parse_args()
    
    # 创建图案生成器
    generator = PhasePatternGenerator(args.width, args.height)
    
    # 设置伽马校正参数
    generator.set_gamma_correction(args.gamma_a, args.gamma_b, args.gamma_c)
    
    # 解析频率列表
    frequencies = [int(f) for f in args.freqs.split(',')]
    
    # 创建相移图案
    print(f"\n生成{'水平' if args.horizontal else '垂直'}相移条纹图案...")
    patterns, phase_shifts = generator.create_phase_shifting_patterns(
        frequencies=frequencies,
        shifts=args.shifts,
        vertical=not args.horizontal
    )
    
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{args.output}_{timestamp}" if args.output == 'patterns' else args.output
    
    # 保存相移图案
    pattern_files = generator.save_patterns(
        patterns, 
        output_folder, 
        file_prefix=args.prefix, 
        file_format=args.format
    )
    
    # 可视化预览
    if not args.no_visualize:
        generator.visualize_patterns(
            patterns, 
            phase_shifts, 
            show=True, 
            save_folder=output_folder
        )
    
    # 如果需要，生成灰度图案
    if args.gray:
        print(f"\n生成灰度图案用于伽马校正...")
        gray_patterns, intensities = generator.create_gray_patterns(levels=args.gray_levels)
        
        # 保存灰度图案
        gray_folder = os.path.join(output_folder, 'gray')
        if not os.path.exists(gray_folder):
            os.makedirs(gray_folder)
            
        for i, pattern in enumerate(gray_patterns):
            intensity = intensities[i]
            file_name = f"gray_{intensity:.2f}.{args.format}"
            file_path = os.path.join(gray_folder, file_name)
            
            # 转换为8位图像
            pattern_8bit = (pattern * 255).astype(np.uint8)
            cv2.imwrite(file_path, pattern_8bit)
            print(f"已保存灰度图案: {file_name}")
    
    print(f"\n所有图案已保存至: {output_folder}")
    print(f"您可以使用这些图案通过投影仪投射到标定板上，然后用相机拍摄用于投影仪标定。")

if __name__ == "__main__":
    main() 