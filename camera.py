from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class Camera(ABC):

    @staticmethod
    @abstractmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        '''
        获取可用相机列表
        
        参数:
            cameras_num_to_find (int) = 2 : 尝试查找的相机数量

        返回:
            cameras (list of Camera): 找到的相机列表
        '''

    @abstractmethod
    def get_image(self) -> np.array:
        '''
        从相机获取图像
        
        返回:
            image (numpy array): 图像作为numpy数组(取决于颜色模式为2D或3D)
        '''
    
    @abstractproperty
    def exposure(self):
        '''曝光值'''
    
    @exposure.setter
    @abstractmethod
    def exposure(self):
        '''设置曝光值'''

    @abstractproperty
    def gain(self):
        '''增益'''

    @gain.setter
    @abstractmethod
    def gain(self):
        '''设置增益'''
    
    @abstractproperty
    def gamma(self):
        '''伽马值'''
    
    @gamma.setter
    @abstractmethod
    def gamma(self):
        '''设置伽马值'''
