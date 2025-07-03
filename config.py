'''
配置管理模块

校准数据存储在单独的config.json文件中，在导入模块时加载
'''

import json

# 投影仪配置
# 投影仪分辨率宽度和高度(像素)
PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 720
# OpenCV GUI窗口相对于系统第一屏幕的偏移
PROJECTOR_WINDOW_SHIFT = 1920

# 投影图像校正的最大和最小亮度
# 给定的值为默认值，当前值从校准文件加载
PROJECTOR_MIN_BRIGHTNESS = 0.0
PROJECTOR_MAX_BRIGHTNESS = 1.0

# 伽马校正系数，用于公式 Iout = a * (Iin + c) ^ b
# 给定的值为默认值，当前值从校准文件加载
PROJECTOR_GAMMA_A = 1.0
PROJECTOR_GAMMA_B = 2.2
PROJECTOR_GAMMA_C = 0

# 相机配置
# 测量中使用的相机数量
CAMERAS_COUNT = 2

# 测量中使用的相机类型
CAMERA_TYPE = 'web'

# 相机参数默认值，当前值从校准文件加载
CAMERA_EXPOSURE = [20000, 20000]
CAMERA_GAIN = [1, 1]
CAMERA_GAMMA = [1, 1]

# 测量配置
# 保存测量数据的路径
DATA_PATH = './data'

# 图像文件名掩码
IMAGES_FILENAME_MASK = 'frame_{2}_{0}_{1}.png'

# 测量文件名掩码
MEASUREMENT_FILENAME_MASK = 'fpp_measurement.json'

# 测量文件夹中的相机文件夹
CAMERAS_FOLDER_NAMES = ['cam1', 'cam2']

# 是否保存测量图像文件
SAVE_MEASUREMENT_IMAGE_FILES = False

# 图案投影和相机图像捕获之间的延迟(毫秒)
MEASUREMENT_CAPTURE_DELAY = 300  # ms

# 校准数据的文件名
CONFIG_FILENAME = r"./config.json"

# 是否使用多进程提高处理速度
USE_MULTIPROCESSING = False

# 并行处理中使用的进程池数量
POOLS_NUMBER = 5

# 最后一次测量结果的路径
LAST_MEASUREMENT_PATH = None


# 从json文件加载校准数据
try:
    with open('config.json') as f:
        calibration_data = json.load(f)

        try:
            PROJECTOR_MIN_BRIGHTNESS = float(calibration_data['projector']['min_brightness'])
            PROJECTOR_MAX_BRIGHTNESS = float(calibration_data['projector']['max_brightness'])

            PROJECTOR_GAMMA_A = float(calibration_data['projector']['gamma_a'])
            PROJECTOR_GAMMA_B = float(calibration_data['projector']['gamma_b'])
            PROJECTOR_GAMMA_C = float(calibration_data['projector']['gamma_c'])
        except:
            pass

        try:
            CAMERA_EXPOSURE = [int(calibration_data['cameras']['baumer'][0]['exposure']),
                               int(calibration_data['cameras']['baumer'][1]['exposure'])]
            CAMERA_GAIN = [float(calibration_data['cameras']['baumer'][0]['gain']),
                           float(calibration_data['cameras']['baumer'][1]['gain'])]
            CAMERA_GAMMA = [float(calibration_data['cameras']['baumer'][0]['gamma']),
                            float(calibration_data['cameras']['baumer'][1]['gamma'])]
        except:
            pass

        try:
            LAST_MEASUREMENT_PATH = calibration_data['measurements']['last_measurement_path']
        except:
            pass
except:
    pass


def save_calibration_data() -> None:
    '''
    将校准数据保存到config.json文件
    '''
    try:
        with open("config.json") as f:
            calibration_data = json.load(f)

            calibration_data['projector']['gamma_a'] = PROJECTOR_GAMMA_A
            calibration_data['projector']['gamma_b'] = PROJECTOR_GAMMA_B
            calibration_data['projector']['gamma_c'] = PROJECTOR_GAMMA_C

        for i in range(CAMERAS_COUNT):
            calibration_data["cameras"]["baumer"][i]["exposure"] = CAMERA_EXPOSURE[i]
            calibration_data["cameras"]["baumer"][i]["gain"] = CAMERA_GAIN[i]
            calibration_data["cameras"]["baumer"][i]["gamma"] = CAMERA_GAIN[i]

        calibration_data['measurements']['last_measurement_path'] = LAST_MEASUREMENT_PATH
    except:
        pass
    else:
        with open("config.json", "w") as f:
            json.dump(calibration_data, f, ensure_ascii=False, indent=4)
