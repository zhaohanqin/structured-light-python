'''FPP数据结构模块'''

from __future__ import annotations
from typing import Optional

import enum
from dataclasses import dataclass, field

import numpy as np


class PhaseShiftingAlgorithm(enum.IntEnum):
    '''相移算法类型枚举'''
    n_step = 1
    double_three_step = 2


@dataclass
class CameraMeasurement:
    '''
    存储单个相机测量结果的类
    '''
    fringe_orientation: Optional[str] = 'vertical'    
    imgs_list: Optional[list[list[np.ndarray]]] = field(default_factory=lambda:list())
    imgs_file_names: Optional[list[list[str]]] = field(default_factory=lambda:list())

    # 计算属性
    phases: Optional[list[np.ndarray]] = field(init=False)
    unwrapped_phases: Optional[list[np.ndarray]] = field(init=False)
    average_intensities: Optional[list[np.ndarray]] = field(init=False)
    modulated_intensities: Optional[list[np.ndarray]] = field(init=False)
    signal_to_noise_mask: Optional[np.ndarray] = field(init=False)
    ROI: Optional[np.array[list]] = field(init=False)
    ROI_mask: Optional[np.ndarray] = field(init=False)
    use_ROI_mask: bool = field(init=False, default=True)


@dataclass
class FPPMeasurement:
    '''
    存储FPP测量数据的类    
    '''
    phase_shifting_type: PhaseShiftingAlgorithm
    frequencies: list[float]
    shifts: list[float]

    camera_results: list[CameraMeasurement] = field(default_factory=lambda:list())

    @property
    def frequency_counts(self) -> int:
        return len(self.frequencies)

    @property
    def shifts_count(self) -> int:
        return len(self.shifts)