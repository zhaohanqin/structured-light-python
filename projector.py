'''投影仪控制模块'''
from __future__ import annotations

import cv2
import numpy as np

import config


class Projector():
    '''
    投影仪控制类
    
    用于控制投影仪投影图案，包括窗口创建、图像校正和投影
    '''
    def __init__(self, min_brightness=0.0, max_brightness=1.0, gamma_a=1.0, gamma_b=2.2, gamma_c=0.0):
        '''
        初始化投影仪控制对象
        
        参数:
            min_brightness (float): 投影图像的最小亮度 (0.0-1.0)
            max_brightness (float): 投影图像的最大亮度 (0.0-1.0)
            gamma_a (float): 伽马校正参数a
            gamma_b (float): 伽马校正参数b
            gamma_c (float): 伽马校正参数c
        '''
        # 保存参数
        self.min_image_brightness = min_brightness
        self.max_image_brightness = max_brightness
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.gamma_c = gamma_c
        
        # 窗口名称
        self.window_name = "Projector"
        
        # 窗口是否已创建
        self.window_created = False
    
    def set_up_window(self):
        '''创建投影窗口'''
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        # 将窗口移动到投影仪显示器
        cv2.moveWindow(self.window_name, config.PROJECTOR_WINDOW_SHIFT, 0)
        # 设置窗口为全屏
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 更新窗口状态
        self.window_created = True
    
    def close_window(self):
        '''关闭投影窗口'''
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
    
    def project_pattern(self, pattern):
        '''
        投影图案
        
        参数:
            pattern: 要投影的图案，可以是灰度图或彩色图
        '''
        # 如果窗口未创建，先创建窗口
        if not self.window_created:
            self.set_up_window()
        
        # 应用亮度校正
        pattern_corrected = self._correct_image_brightness(pattern)
        
        # 显示图案
        cv2.imshow(self.window_name, pattern_corrected)
        cv2.waitKey(1)
    
    def _correct_image_brightness(self, image):
        '''
        校正图像亮度
        
        参数:
            image: 输入图像
        
        返回:
            corrected_image: 校正后的图像
        '''
        # 将图像转换为浮点型
        image_float = image.astype(np.float32)
        
        # 如果是彩色图像
        if len(image.shape) == 3:
            # 归一化到0-1范围
            image_float = image_float / 255.0
            
            # 应用伽马校正: a * (x + c) ^ b
            corrected = self.gamma_a * np.power(image_float + self.gamma_c, self.gamma_b)
            
            # 应用亮度范围校正
            corrected = self.min_image_brightness + corrected * (self.max_image_brightness - self.min_image_brightness)
            
            # 裁剪到0-1范围
            corrected = np.clip(corrected, 0.0, 1.0)
            
            # 转换回8位无符号整数
            return (corrected * 255).astype(np.uint8)
        # 如果是灰度图像
        else:
            # 归一化到0-1范围
            image_float = image_float / 255.0
            
            # 应用伽马校正: a * (x + c) ^ b
            corrected = self.gamma_a * np.power(image_float + self.gamma_c, self.gamma_b)
            
            # 应用亮度范围校正
            corrected = self.min_image_brightness + corrected * (self.max_image_brightness - self.min_image_brightness)
            
            # 裁剪到0-1范围
            corrected = np.clip(corrected, 0.0, 1.0)
            
            # 转换回8位无符号整数
            return (corrected * 255).astype(np.uint8)

    def project_black_background(self) -> None:
        '''
        通过OpenCV GUI窗口投影黑色图案，用于模拟投影仪关闭状态
        '''
        # 如果窗口尚未打开，则打开OpenCV GUI窗口
        if not self.window_created:
            self.set_up_window()
        
        # 创建黑色背景图像(全零矩阵)
        background = np.zeros((config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH))
        
        # 在OpenCV GUI窗口显示图像
        cv2.imshow(self.window_name, background)
        cv2.waitKey(200)  # 等待200毫秒，确保显示

    def project_white_background(self) -> None:
        '''
        通过OpenCV GUI窗口投影白色图案，用于将投影仪用作光源
        '''
        # 如果窗口尚未打开，则打开OpenCV GUI窗口
        if not self.window_created:
            self.set_up_window()
        
        # 创建白色背景图像(全255矩阵)
        background = np.ones((config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH)) * 255
        
        # 在OpenCV GUI窗口显示图像
        cv2.imshow(self.window_name, background)
        cv2.waitKey(200)  # 等待200毫秒，确保显示

    @property
    def corrected_pattern(self) -> np.ndarray:
        '''
        返回最后一次投影的校正后图案作为numpy数组
        '''
        return self._correct_image_brightness(np.zeros((config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH)))

    @property
    def resolution(self) -> tuple[int, int]:
        '''
        返回投影仪分辨率(宽度和高度)
        '''
        return config.PROJECTOR_WIDTH, config.PROJECTOR_HEIGHT

    @property
    def min_image_brightness(self) -> float:
        '''
        获取最小图像亮度值
        '''
        return self.min_image_brightness

    @min_image_brightness.setter
    def min_image_brightness(self, value: float):
        '''
        设置最小图像亮度值
        '''
        self.min_image_brightness = value

    @property
    def max_image_brightness(self) -> float:
        '''
        获取最大图像亮度值
        '''
        return self.max_image_brightness

    @max_image_brightness.setter
    def max_image_brightness(self, value: float):
        '''
        设置最大图像亮度值
        '''
        self.max_image_brightness = value

    @property
    def image_brightness_rescale_factor(self) -> float:
        '''
        计算图像亮度重缩放因子(最大亮度与最小亮度之差)
        用于亮度校正过程
        '''
        return (self.max_image_brightness - self.min_image_brightness)
