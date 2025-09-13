from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

import config
from camera import Camera
from projector import Projector
from camera_web import CameraWeb
from camera_baumer import CameraBaumer, NEOAPI_AVAILABLE
from camera_simulated import CameraSimulated
from create_patterns import create_psp_templates
from hand_set_up_camera import camera_adjust, camera_baumer_adjust
from min_max_projector_calibration import MinMaxProjectorCalibration
from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm, CameraMeasurement

from examples.test_plate_phasogrammetry import process_with_phasogrammetry

def initialize_cameras(
    camera_type: str, 
    projector: Projector=None, 
    cam_to_found_number: int = 2, 
    cameras_serial_numbers: List[str] = []
    ) -> list[Camera]:
    '''
    搜索并初始化指定类型的相机，并与投影仪实例关联
    
    参数:
        camera_type (str): 相机类型('web'、'baumer'或'simulated')
        projector (Projector): 关联的投影仪实例
        cam_to_found_number (int): 要查找的相机数量
        cameras_serial_numbers (List[str]): 相机序列号列表(用于Baumer相机)
    
    返回:
        cameras (list[Camera]): 已检测到的相机列表
    '''
    if camera_type == 'web':
        # 初始化网络摄像头
        cameras = CameraWeb.get_available_cameras(cam_to_found_number)
    elif camera_type == 'baumer':
        # 初始化Baumer工业相机
        if not NEOAPI_AVAILABLE:
            print("警告: 选择了Baumer相机但neoapi不可用。")
            cameras = []
        else:
            cameras = CameraBaumer.get_available_cameras(cam_to_found_number, cameras_serial_numbers)
    elif camera_type == 'simulated':
        # 初始化模拟相机(用于测试)
        cameras = CameraSimulated.get_available_cameras(cam_to_found_number)
        # 为模拟相机设置投影仪
        if projector is not None:
            for camera in cameras:
                camera.projector = projector
    return cameras


def adjust_cameras(cameras: list[Camera]) -> None:
    '''
    调整相机捕获参数(焦距、曝光时间等)，提供视觉反馈
    
    参数:
        cameras (list[Camera]): 要调整的相机列表
    '''
    for i, camera in enumerate(cameras):
        if camera.type == "web":
            # 调整网络摄像头
            camera_adjust(camera)
        elif camera.type == "baumer":
            # 调整Baumer工业相机
            exposure, gamma, gain = camera_baumer_adjust(camera)
            # 保存调整后的参数到配置
            config.CAMERA_EXPOSURE[i] = exposure
            config.CAMERA_GAIN[i] = gain
            config.CAMERA_GAMMA[i] = gamma
    # 将校准数据保存到文件
    config.save_calibration_data()


def calibrate_projector(cameras: list[Camera], projector: Projector) -> None:
    '''
    校准投影仪图像，应用伽马校正
    
    参数:
        cameras (list[Camera]): 用于捕获测量图像的相机列表
        projector (Projector): 需要校准的投影仪
    '''
    # 获取亮度与强度关系数据(无校正)
    brightness, _ = get_brightness_vs_intensity(cameras, projector, use_correction=False)

    # 计算伽马系数
    # 创建强度线性空间
    intensity = np.linspace(0, np.max(brightness), len(brightness))

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
    lam = lambda x, a, b, c: a * (x + c) ** b

    # 对减少后的亮度与强度序列拟合伽马函数参数
    popt, pcov = optimize.curve_fit(lam, int_reduced, brt_reduced, p0=(1, 1, 1))
    print(
        f"拟合的伽马函数 - Iout = {popt[0]:.3f} * (Iin + {popt[2]:.3f}) ^ {popt[1]:.3f}"
    )

    # 绘制拟合的伽马函数
    gg = lam(intensity, *popt)

    plt.plot(intensity, brightness, "b+")
    plt.plot(intensity, gg, "r-")
    plt.xlabel("强度，相对单位")
    plt.ylabel("亮度，相对单位")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()

    # 保存新的伽马校正系数
    config.PROJECTOR_GAMMA_A = popt[0]
    config.PROJECTOR_GAMMA_B = popt[1]
    config.PROJECTOR_GAMMA_C = popt[2]
    config.save_calibration_data()

    # 检查伽马校正效果
    brt_corrected, _ = get_brightness_vs_intensity(
        cameras, projector, use_correction=True
    )

    # 绘制校正后的亮度与强度关系
    plt.plot(intensity, brt_corrected, "b+")
    plt.xlabel("强度，相对单位")
    plt.ylabel("亮度，相对单位")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def get_brightness_vs_intensity(cameras : List[Camera], projector: Projector, use_correction: bool) -> Tuple[List[float], List[float]]:
    '''
    通过在屏幕上投影恒定强度并用相机捕获图像，获取亮度与强度的关系。
    亮度在多张捕获图像的小区域内取平均值。
    
    参数:
        cameras (list[Camera]): 用于捕获测量图像的相机列表
        projector (Projector): 用于投影图案的投影仪
        use_correction (bool): 是否对投影图案使用校正
        
    返回:
        Tuple[List[float], List[float]]: 两个相机的亮度值列表
    '''
    # 创建OpenCV窗口显示图像
    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    # 设置参数 (TODO: 添加到配置)
    win_size_x = 50  # 测量窗口X尺寸
    win_size_y = 50  # 测量窗口Y尺寸
    max_intensity = 1024  # 最大强度值
    average_num = 5  # 平均帧数
    border_width = 20  # 边框宽度

    projector.set_up_window()  # 设置投影窗口

    # 亮度存储列表 (TODO: 支持任意数量相机)
    brightness1 = []
    brightness2 = []

    # 创建带有细黑白边框的图像
    image = np.zeros((projector.height, projector.width))
    image[border_width:-border_width, border_width:-border_width] = max_intensity

    temp_img = cameras[0].get_image()  # 获取图像尺寸参考

    for intensity in range(max_intensity):
        # 设置中心区域强度
        image[2 * border_width: -2 * border_width, 2 * border_width: -2 * border_width] = intensity / max_intensity
        projector.project_pattern(image, use_correction)

        # 初始化累积图像
        img1 = np.zeros(temp_img.shape, dtype=np.float64)
        img2 = np.zeros(temp_img.shape, dtype=np.float64)

        # 捕获多帧并平均
        for _ in range(average_num):
            cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)  # 等待捕获延迟

            img1 = img1 + cameras[0].get_image()
            img2 = img2 + cameras[1].get_image()
        
        # 计算平均值
        img1 = img1 / average_num
        img2 = img2 / average_num
        
        # 定义感兴趣区域(图像中心)
        roi_x = slice(int(img1.shape[1] / 2 - win_size_x), int(img1.shape[1] / 2 + win_size_x))
        roi_y = slice(int(img1.shape[0] / 2 - win_size_y), int(img1.shape[0] / 2 + win_size_y))
        
        # 计算ROI区域平均亮度
        brt1 = np.mean(img1[roi_y, roi_x]) / max_intensity
        brt2 = np.mean(img2[roi_y, roi_x]) / max_intensity

        # 存储亮度值
        brightness1.append(brt1)
        brightness2.append(brt2)

        # 显示第一个相机图像
        img_to_display1 = img1.astype(np.uint16)
        # 在图像上标记ROI区域
        cv2.rectangle(
            img_to_display1,
            (roi_x.start, roi_y.start),
            (roi_x.stop, roi_y.stop),
            (255, 0, 0), 3,
        )
        # 添加强度和亮度文本信息
        cv2.putText(
            img_to_display1,
            f"强度 = {intensity}",
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.putText(
            img_to_display1,
            f"亮度 = {brt1:.3f}",
            (50, 100),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.imshow('cam1', img_to_display1)

        # 显示第二个相机图像
        img_to_display2 = img2.astype(np.uint16)
        cv2.rectangle(
            img_to_display2,
            (roi_x.start, roi_y.start),
            (roi_x.stop, roi_y.stop),
            (255, 0, 0), 3,
        )
        cv2.putText(
            img_to_display2,
            f"强度 = {intensity}",
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.putText(
            img_to_display2,
            f"亮度 = {brt2:.3f}",
            (50, 100),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.imshow('cam2', img_to_display2)

    # 关闭投影窗口
    projector.close_window()
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    return brightness1, brightness2


def capture_measurement_images(
    cameras: List[Camera],
    projector: Projector, 
    phase_shift_type: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.n_step
    ) -> FPPMeasurement:
    '''
    执行条纹投影测量。生成图案，通过投影仪投影并使用相机捕获图像。
    
    参数:
        cameras (list[Camera]): 用于捕获测量图像的相机列表
        projector (Projector): 用于投影图案的投影仪
        phase_shift_type (PhaseShiftingAlgorithm): 相移算法类型
    
    返回:
        meas (FPPMeasurement): 包含第一和第二相机测量数据的FPP测量对象
    '''
    # 创建OpenCV GUI窗口显示捕获的图像
    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    # 设置相移和频率参数
    shift_num = 4  # 相移数量
    frequencies = [1, 4, 12, 48, 90]  # 使用的频率列表

    # 创建相移轮廓测量图案
    patterns_v, _ = create_psp_templates(  # 垂直条纹图案
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        frequencies,
        phase_shift_type,
        shifts_number=shift_num,
        vertical=True,
    )
    patterns_h, phase_shifts = create_psp_templates(  # 水平条纹图案
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        frequencies,
        phase_shift_type,
        shifts_number=shift_num,
        vertical=False,
    )

    # 组织垂直和水平图案
    patterns_vh = {'vertical': patterns_v, 'horizontal': patterns_h}

    # 初始化相机结果
    cam_results = [
        CameraMeasurement(fringe_orientation='vertical'),    # 相机1垂直条纹
        CameraMeasurement(fringe_orientation='vertical'),    # 相机2垂直条纹
        CameraMeasurement(fringe_orientation='horizontal'),  # 相机1水平条纹
        CameraMeasurement(fringe_orientation='horizontal'),  # 相机2水平条纹
    ]

    # 创建FPP测量实例
    meas = FPPMeasurement(phase_shift_type, frequencies, phase_shifts, cam_results)

    # 如果配置中启用了保存测量图像，创建保存文件夹
    if config.SAVE_MEASUREMENT_IMAGE_FILES:
        # 使用当前时间创建唯一文件夹名
        measure_name = f'{datetime.now():%d-%m-%Y_%H-%M-%S}'
        last_measurement_path = f'{config.DATA_PATH}/{measure_name}'
        os.makedirs(f'{last_measurement_path}/')
        os.makedirs(f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[0]}/')
        os.makedirs(f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[1]}/')

    # 设置投影仪
    projector.set_up_window()

    # 处理垂直和水平条纹图案
    for res1, res2 in ((cam_results[0], cam_results[1]), (cam_results[2], cam_results[3])):
        
        orientation = res1.fringe_orientation  # 当前条纹方向
        patterns = patterns_vh[orientation]    # 当前方向的图案集

        # 遍历生成的图案
        for i in range(len(patterns)):  # 遍历频率

            # 初始化图像文件名或图像列表
            if config.SAVE_MEASUREMENT_IMAGE_FILES:
                res1.imgs_file_names.append([])
                res2.imgs_file_names.append([])
            else:
                res1.imgs_list.append([])
                res2.imgs_list.append([])

            for j in range(len(patterns[i])):  # 遍历相移
                # 投影当前图案
                projector.project_pattern(patterns[i][j])

                # 对于网络摄像头，在测量前捕获一帧(清除缓冲)
                if cameras[0].type == 'web':
                    cameras[0].get_image()
                if cameras[1].type == 'web':
                    cameras[1].get_image()

                # 在图案投影和图像捕获之间等待延迟时间
                cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

                # 捕获图像
                frames_1 = []
                frames_2 = []
                for _ in range(1):  # 可设置多帧平均
                    frames_1.append(cameras[0].get_image())
                    frames_2.append(cameras[1].get_image())

                # 计算平均帧
                frame_1 = np.mean(frames_1, axis=0).astype(np.uint8)
                frame_2 = np.mean(frames_2, axis=0).astype(np.uint8)

                # 显示捕获的图像
                cv2.imshow('cam1', frame_1)
                cv2.imshow('cam2', frame_2)

                # 如果配置中启用了保存，则保存图像
                if config.SAVE_MEASUREMENT_IMAGE_FILES:
                    # 构建文件名
                    filename1 = f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[0]}/' + config.IMAGES_FILENAME_MASK.format(i, j, orientation)
                    filename2 = f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[1]}/' + config.IMAGES_FILENAME_MASK.format(i, j, orientation)
                    saved1 = cv2.imwrite(filename1, frame_1)
                    saved2 = cv2.imwrite(filename2, frame_2)

                    # 存储已保存的图像文件名
                    if saved1 and saved2:
                        res1.imgs_file_names[-1].append(filename1)
                        res2.imgs_file_names[-1].append(filename2)
                    else:
                        raise Exception('图像保存过程中出错!')
                else:
                    # 直接存储图像数据
                    res1.imgs_list[-1].append(frame_1)
                    res2.imgs_list[-1].append(frame_2)

    # 停止投影仪
    projector.close_window()

    # 关闭OpenCV GUI窗口
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    # 如果配置中启用了保存，将测量结果保存为JSON文件
    if config.SAVE_MEASUREMENT_IMAGE_FILES:        
        with open(f'{last_measurement_path}/' + config.MEASUREMENT_FILENAME_MASK.format(measure_name), 'x') as f:
            json.dump(meas, f, ensure_ascii=False, indent=4, default=vars)
        config.LAST_MEASUREMENT_PATH = last_measurement_path
        config.save_calibration_data()

    return meas


if __name__ == '__main__':
    # 主程序入口

    # 初始化投影仪
    projector = Projector(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        config.PROJECTOR_MIN_BRIGHTNESS,
        config.PROJECTOR_MAX_BRIGHTNESS,
    )

    # 初始化相机
    cameras = initialize_cameras(config.CAMERA_TYPE, projector, cam_to_found_number=2)

    # 可选功能菜单
    choices = {i for i in range(6)}

    # 主交互循环
    while True:
        print(f"已连接 {len(cameras)} 个相机")
        print("==========================================================")
        print("1 - 调整相机参数")
        print("2 - 投影仪伽马校正校准")
        print("3 - 检查亮度分布")
        print("4 - 执行测量")
        print("==========================================================")
        print("0 - 退出脚本")
        answer = input("请从上面列表中选择一项功能: ")

        # 验证用户输入
        try:
            if int(answer) not in choices:
                raise Exception()
        except:
            continue
        else:
            choice = int(answer)

        # 处理用户选择
        if choice == 0:
            # 退出程序
            break

        elif choice == 1:
            # 调整相机参数
            adjust_cameras(cameras)

        elif choice == 2:
            # 投影仪伽马校正校准
            calibrate_projector(cameras, projector)

        elif choice == 3:
            # 检查亮度分布
            frequencies = [1, 4, 16, 64, 100, 120]
            test_pattern, _ = create_psp_templates(
                config.PROJECTOR_WIDTH,
                config.PROJECTOR_HEIGHT,
                frequencies,
                PhaseShiftingAlgorithm.n_step,
                1,
                vertical=False,
            )
            MinMaxProjectorCalibration(test_pattern, cameras, projector)

        elif choice == 4:
            # 执行测量并处理
            measurement = capture_measurement_images(
                cameras, projector, phase_shift_type=PhaseShiftingAlgorithm.n_step
            )
            process_with_phasogrammetry(measurement)
