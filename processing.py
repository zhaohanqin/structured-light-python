'''图像处理算法模块'''

from __future__ import annotations
import multiprocessing
from multiprocessing import Pool
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import signal
from scipy.optimize import fsolve

import config
from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm


def calculate_phase_generic(images: list[np.ndarray], phase_shifts: Optional[list[float]]=None, frequency: Optional[float]=None, phase_shifting_type: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.n_step, direct_formula: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
    '''
    通过通用公式计算多张相移轮廓测量图像的包裹相位
    参考公式(8)：https://doi.org/10.1016/j.optlaseng.2018.04.019

    参数:
        images (list of numpy arrays): 相移轮廓测量图像列表
        phase_shifts = None (list): 每张图像的相移值列表，
                                   如果未定义，则自动计算均匀步长的相移
        frequency = None (float): 测量频率，用于为单位频率图像添加PI
        phase_shifting_type = n_step (enum(int)): 用于相位计算的相移算法类型
        direct_formula = False (bool): 是否使用3步和4步相移的直接公式计算相位

    返回:
        result_phase (2D numpy array): 从图像计算的包裹相位
        average_intensity (2D numpy array): 图像上的平均强度
        modulated_intensity (2D numpy array): 图像上的调制强度
    '''
    def calculate_n_step_phase(imgs: list[np.ndarray], phase_shifts: list[float]):
        # 根据相移长度使用特定情况
        if direct_formula and len(phase_shifts) == 3:
            # 计算公式 (14-16) https://doi.org/10.1016/j.optlaseng.2018.04.019
            sum12 = imgs[1] - imgs[2]
            sum012 = 2 * imgs[0] - imgs[1] - imgs[2]
            result_phase = np.arctan2(np.sqrt(3) * (sum12), sum012)
            average_intensity = (imgs[0] + imgs[1] + imgs[2]) / 3
            modulated_intensity = 1 / 3 * np.sqrt(3 * (sum12) ** 2 + (sum012) ** 2)
        elif direct_formula and len(phase_shifts) == 4:
            # 计算公式 (21-23) https://doi.org/10.1016/j.optlaseng.2018.04.019
            sum13 = imgs[1] - imgs[3]
            sum02 = imgs[0] - imgs[2]
            result_phase = np.arctan2(sum13, sum02)
            average_intensity = (imgs[0] + imgs[1] + imgs[2] + imgs[3]) / 4
            modulated_intensity = 0.5 * np.sqrt(sum13**2 + sum02**2)
        else:
            # 重塑相移数组以便广播乘法
            phase_shifts = np.array(phase_shifts).reshape((-1,) + (1, 1))

            # 添加补充相位以获取单位频率测量的相位
            phase_sup = 0
            if frequency is not None and frequency == 1:
                phase_sup = np.pi

            # 计算公式 (8) https://doi.org/10.1016/j.optlaseng.2018.04.019
            # 计算图像与正弦相移的乘积
            temp1 = np.multiply(imgs, np.sin(phase_shifts + phase_sup))
            # 计算图像与余弦相移的乘积
            temp2 = np.multiply(imgs, np.cos(phase_shifts + phase_sup))

            # 对所有图像求和
            sum1 = np.sum(temp1, 0)
            sum2 = np.sum(temp2, 0)

            # 计算相位(反正切)
            result_phase = np.arctan2(sum1, sum2)

            # 计算公式 (9-10) https://doi.org/10.1016/j.optlaseng.2018.04.019
            # 计算平均强度
            average_intensity = np.mean(imgs, 0)
            # 计算调制强度
            modulated_intensity = 2 * np.sqrt(np.power(sum1, 2) + np.power(sum2, 2)) / len(images)
        return result_phase, average_intensity, modulated_intensity
    
    # 如果未定义相移，则计算均匀步长的相移
    if phase_shifts is None:
        phase_shifts = [2 * np.pi / len(images) * n for n in range(len(images))]

    # 形成numpy数组以便广播
    imgs = np.zeros((len(images), images[0].shape[0], images[0].shape[1]))

    # 将图像添加到形成的numpy数组中
    for i in range(len(images)):
        imgs[i] = images[i]

    # 根据相移算法类型计算包裹相位场
    if phase_shifting_type == PhaseShiftingAlgorithm.n_step:
        # 经典N步方法
        result_phase, average_intensity, modulated_intensity = calculate_n_step_phase(images, phase_shifts)
    elif phase_shifting_type == PhaseShiftingAlgorithm.double_three_step:
        # 双三步方法 - 两个3步相位的平均值(第二个相移PI/3)
        # 计算公式 (26-31) 参考文献3.2节 https://doi.org/10.1016/j.optlaseng.2018.04.019
        result_phase1, average_intensity1, modulated_intensity1 = calculate_n_step_phase(imgs[:3,:,:], phase_shifts[:3])
        result_phase2, average_intensity2, modulated_intensity2 = calculate_n_step_phase(imgs[3:,:,:], phase_shifts[3:])
        
        # 计算两组结果的平均值
        result_phase = (result_phase1 + result_phase2) / 2
        average_intensity = (average_intensity1 + average_intensity2) / 2
        modulated_intensity = (modulated_intensity1 + modulated_intensity2) / 2

    return result_phase, average_intensity, modulated_intensity


def calculate_unwraped_phase(phase_l: np.ndarray, phase_h: np.ndarray, lamb_l:float , lamb_h: float) -> np.ndarray:
    '''
    通过两组相移轮廓测量图像计算解包裹相位
    使用公式(94-95)：https://doi.org/10.1016/j.optlaseng.2018.04.019
    采用标准时间相位解包裹(TPU)算法

    参数:
        phase_l (2D numpy array): 低频(lamb_l)PSP图像集的计算相位
        phase_h (2D numpy array): 高频(lamb_h)PSP图像集的计算相位
        lamb_l (float): 第一个相位数组(phase_l)的低空间频率
        lamb_h (float): 第二个相位数组(phase_h)的高空间频率

    返回:
        unwrapped_phase (2D numpy array): 解包裹的相位
    '''
    assert phase_h.shape == phase_l.shape, \
    'phase_l和phase_h的形状必须相等'

    # 公式(95) https://doi.org/10.1016/j.optlaseng.2018.04.019
    # 计算相位解包裹的整数倍数k
    k = np.round(((lamb_l / lamb_h) * phase_l - phase_h) / (2 * np.pi)).astype(int)

    # 公式(94) https://doi.org/10.1016/j.optlaseng.2018.04.019
    # 计算最终的解包裹相位
    unwrapped_phase = phase_h + 2 * np.pi * k

    return unwrapped_phase


def calculate_phase_for_fppmeasurement(measurement: FPPMeasurement):
    '''
    使用calculate_phase_generic和calculate_unwraped_phase函数
    为FPP测量实例计算解包裹相位。
    计算的相位场将存储在输入的measurement参数中。

    参数:
        measurement (FPPMeasurement): 包含图像的FPP测量实例
    '''
    # 加载测量数据
    frequencies = measurement.frequencies  # 频率列表
    shifts = measurement.shifts  # 相移列表
    frequency_counts = len(measurement.frequencies)  # 频率数量

    # 处理每个相机的结果
    for cam_result in measurement.camera_results:
        # 为相位、解包裹相位、平均强度和调制强度创建空列表
        phases = []
        unwrapped_phases = []
        avg_ints = []
        mod_ints = []

        # 获取一个相机的图像
        images = cam_result.imgs_list

        # 计算每个频率的场
        for i in range(frequency_counts):
            # 获取一个频率的图像
            images_for_one_frequency = images[i]

            # 计算相位、平均强度和调制强度
            phase, avg_int, mod_int = calculate_phase_generic(images_for_one_frequency, shifts, frequencies[i], phase_shifting_type=measurement.phase_shifting_type)

            # 使用阈值过滤相位场
            # 创建掩码，调制强度大于5的区域为1，其他为0
            mask = np.where(mod_int > 5, 1, 0)
            # 应用掩码到相位场
            phase = phase * mask

            # 存储计算结果
            phases.append(phase)
            avg_ints.append(avg_int)
            mod_ints.append(mod_int)

            if i == 0:
                # 第一个相位场应该是单位频率，没有歧义
                unwrapped_phases.append(phase)
            else:
                # 后续相位场需要解包裹
                # 使用前一个频率的解包裹相位和当前频率的包裹相位计算
                unwrapped_phase = calculate_unwraped_phase(unwrapped_phases[i-1], phases[i], 1 / frequencies[i-1], 1 / frequencies[i])
                unwrapped_phases.append(unwrapped_phase)

        # 将计算的场设置到当前相机结果实例
        cam_result.phases = phases  # 包裹相位
        cam_result.unwrapped_phases = unwrapped_phases  # 解包裹相位
        cam_result.average_intensities = avg_ints  # 平均强度
        cam_result.modulated_intensities = mod_ints  # 调制强度


def create_polygon(shape: Tuple[int], vertices: np.ndarray) -> np.ndarray:
    '''
    创建一个由顶点定义的多边形的2D numpy数组。
    多边形内的点填充为1，其他点填充为0。

    来源: https://stackoverflow.com/a/37123933

    参数:
        shape (tuple of int): 要生成的2D numpy数组的形状
        vertices (2D numpy array): 形状为(N, 2)的2D numpy数组，包含多边形顶点的坐标 ([[x0, y0], ..., [xn, yn]])

    返回:
        base_array (2D numpy array): 多边形区域填充为1的2D numpy数组
    '''

    def check(p1, p2, arr):
        """
        使用由p1和p2定义的直线检查输入索引数组与插值之间的关系

        返回布尔数组，形状内为True，形状外为False
        """
        # 创建3D索引数组
        idxs = np.indices(arr.shape) 

        p1 = p1[::-1].astype(float)
        p2 = p2[::-1].astype(float)

        # 基于两点之间的插值线计算每行索引的最大列索引
        if p1[0] == p2[0]:
            max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
            sign = np.sign(p2[1] - p1[1])
        else:
            max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
            sign = np.sign(p2[0] - p1[0])

        return idxs[1] * sign <= max_col_idx * sign

    # 初始化全零数组
    base_array = np.zeros(shape, dtype=float)

    # 初始化定义形状填充的布尔数组
    fill = np.ones(base_array.shape) * True

    # 为每个边缘段创建检查数组，组合到fill数组中
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # 将多边形内的所有值设置为1
    base_array[fill] = 1

    return base_array


def point_inside_polygon(x: int, y: int, poly: list[tuple[int, int]], include_edges: bool = True) -> bool:
    '''
    测试点(x,y)是否在多边形poly内部

    如果从点向右发射的水平光线与多边形相交的次数为奇数，则点在多边形内部。
    对于非凸多边形也能正常工作。
    来源: https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon

    参数:
        x (int): 点的水平坐标
        y (int): 点的垂直坐标
        poly (list[tuple(int, int)]): N个顶点的多边形，定义为[(x1,y1),...,(xN,yN)]或[(x1,y1),...,(xN,yN),(x1,y1)]
        include_edges (bool): 是否将边界上的点视为内部点，默认为True

    返回:
        inside (bool): 如果点在多边形内部则为True
    '''
    n = len(poly)

    inside = False

    p1x, p1y = poly[0]

    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:  # 水平边
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # 点在水平边上
                    inside = include_edges
                    break
                # 点在当前边的左侧
                elif x < min(p1x, p2x):
                    inside = not inside
        else:  # p1y!= p2y，非水平边
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                # 计算边与水平线的交点的x坐标
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                # 点正好在边上
                if x == xinters:
                    inside = include_edges
                    break

                # 点在当前边的左侧
                if x < xinters:
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def triangulate_points(calibration_data: dict, image1_points: np.ndarray, image2_points: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    '''
    将两组2D点三角测量为一组3D点

    参数:
        calibration_data (dictionary): 用于三角测量的标定数据
        image1_points (numpy arrray [N, 2]): 第一组2D点
        image2_points (numpy arrray [N, 2]): 第二组2D点
    返回:
        points_3d (numpy arrray [N, 3]): 三角测量的3D点
        rms1 (float): 第一个相机的总体重投影误差
        rms2 (float): 第二个相机的总体重投影误差
        reproj_err1, reproj_err2 (numpy arrray [N]): 第一个和第二个相机每个三角测量点的重投影误差
    '''
    # 根据立体标定数据计算投影矩阵
    cam1_mtx = np.array(calibration_data['camera_0']['mtx'])  # 第一个相机的内参矩阵
    cam2_mtx = np.array(calibration_data['camera_1']['mtx'])  # 第二个相机的内参矩阵
    dist1_mtx = np.array(calibration_data['camera_0']['dist'])  # 第一个相机的畸变系数
    dist2_mtx = np.array(calibration_data['camera_1']['dist'])  # 第二个相机的畸变系数

    # 计算相机的投影矩阵
    # 第一个相机的投影矩阵(假设为世界坐标系原点)
    proj_mtx_1 = np.dot(cam1_mtx, np.hstack((np.identity(3), np.zeros((3,1)))))
    # 第二个相机的投影矩阵(使用旋转和平移)
    proj_mtx_2 = np.dot(cam2_mtx, np.hstack((calibration_data['R'], calibration_data['T'])))

    # 对2D点进行去畸变处理
    points_2d_1 = np.array(image1_points, dtype=np.float32)
    points_2d_2 = np.array(image2_points, dtype=np.float32)
    # 使用相机内参和畸变系数去畸变
    undist_points_2d_1 = cv2.undistortPoints(points_2d_1, cam1_mtx, dist1_mtx, P=cam1_mtx)
    undist_points_2d_2 = cv2.undistortPoints(points_2d_2, cam2_mtx, dist2_mtx, P=cam2_mtx)

    # 计算3D点的三角测量
    # 使用OpenCV的triangulatePoints函数
    points_hom = cv2.triangulatePoints(proj_mtx_1, proj_mtx_2, undist_points_2d_1, undist_points_2d_2)
    # 将齐次坐标转换为3D坐标
    points_3d = cv2.convertPointsFromHomogeneous(points_hom.T)

    # 重塑3D点数组
    points_3d = np.reshape(points_3d, (points_3d.shape[0], points_3d.shape[2]))

    # 重投影三角测量点
    # 将3D点投影回第一个相机
    reproj_points, _ = cv2.projectPoints(points_3d, np.identity(3), np.zeros((3,1)), cam1_mtx, dist1_mtx)
    # 将3D点投影回第二个相机
    reproj_points2, _ = cv2.projectPoints(points_3d, np.array(calibration_data['R']), np.array(calibration_data['T']), cam2_mtx, dist2_mtx)

    # 计算重投影误差
    # 第一个相机的重投影误差
    reproj_err1 = np.sum(np.square(points_2d_1[:,np.newaxis,:] - reproj_points), axis=2)
    rms1 = np.sqrt(np.sum(reproj_err1)/reproj_points.shape[0])

    # 第二个相机的重投影误差
    reproj_err2 = np.sum(np.square(points_2d_2[:,np.newaxis,:] - reproj_points2), axis=2)
    rms2 = np.sqrt(np.sum(reproj_err2)/reproj_points.shape[0])

    # 重塑误差数组
    reproj_err1 = np.reshape(reproj_err1, (reproj_err1.shape[0]))
    reproj_err2 = np.reshape(reproj_err2, (reproj_err2.shape[0]))

    return points_3d, rms1, rms2, reproj_err1, reproj_err2


def calculate_bilinear_interpolation_coeficients(points: tuple[tuple]) -> np.ndarray:
    '''
    计算2D数据的双线性插值系数。双线性插值定义为
    多项式拟合 f(x0, y0) = a0 + a1 * x0 + a2 * y0 + a3 * x0 * y0。
    使用维基百科的方程:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    参数:
        points (tuple[tuple]): 四个元素，格式为 (x, y, f(x, y))

    返回:
        bilinear_coeficients (numpy array): 输入点的双线性插值的四个系数
    '''
    # 排序点
    points = sorted(points)

    # 获取x、y坐标和这些点的值
    (x1, y1, q11), (_, y2, q12), (x2, _, q21), (_, _, q22) = points

    # 获取矩阵A
    A = np.array(
        [
            [x2 * y2, -x2 * y1, -x1 * y2, x1 * y1],
            [-y2, y1, y2, -y1],
            [-x2, x2, x1, -x1],
            [1, -1, -1, 1],
        ]
    )

    # 获取向量B
    B = np.array([q11, q12, q21, q22])

    # 计算双线性插值的系数
    bilinear_coeficients = (1 / ((x2 - x1) * (y2 - y1))) * A.dot(B)
    return bilinear_coeficients


def bilinear_phase_fields_approximation(p: tuple[float, float], *data: tuple) -> tuple[float, float]:
    '''
    计算水平和垂直相位场双线性插值的残差。
    该函数用于find_phasogrammetry_corresponding_point中的fsolve函数。

    参数:
        p (tuple[float, float]): 计算残差的点的x和y坐标
        data (tuple): 计算残差的数据
            - a (numpy array): 定义水平相位场线性插值的四个系数
            - b (numpy array): 定义垂直相位场线性插值的四个系数
            - p_h (float): 在插值场中匹配的水平相位
            - p_v (float): 在插值场中匹配的垂直相位

    返回:
        res_h, res_v (tuple[float, float]): 点(x, y)处水平和垂直场的残差
    '''
    x, y = p

    a, b, p_h, p_v = data

    return (
        a[0] + a[1] * x + a[2] * y + a[3] * x * y - p_h,
        b[0] + b[1] * x + b[2] * y + b[3] * x * y - p_v,
    )


def find_phasogrammetry_corresponding_point(p1_h: np.ndarray, p1_v: np.ndarray, p2_h: np.ndarray, p2_v: np.ndarray, x: int, y: int, LUT:list[list[list[int]]]=None) -> tuple[float, float]:
    '''
    使用相位测量法找到第二幅图像的对应点坐标

    对于给定的x和y坐标，确定第一个相机的垂直和水平条纹的相位场上的相位值。
    然后在第二个相机的相应场上找到具有定义相位值的两条等相线。
    等相线的交点给出了第二个相机图像上对应点的坐标。

    参数:
        p1_h (numpy array): 第一个相机的水平条纹相位场
        p1_v (numpy array): 第一个相机的垂直条纹相位场
        p2_h (numpy array): 第二个相机的水平条纹相位场
        p2_v (numpy array): 第二个相机的垂直条纹相位场
        x (int): 第一个相机的点的水平坐标
        y (int): 第一个相机的点的垂直坐标
        LUT (list[list[list[int]]]): 查找表，用于加速相位测量计算

    返回:
        x2, y2 (tuple[float, float]): 第二个相机对应点的水平和垂直坐标
    '''
    # 获取垂直和水平相位场上的相位值
    phase_h = p1_h[y, x]
    phase_v = p1_v[y, x]

    retval = [np.inf, np.inf]

    # 如果LUT可用，使用它计算对应点
    if LUT is not None:
        # 从LUT获取x, y坐标的值作为第一次近似
        try:
            # 找到最接近phase_h和phase_v的索引
            phase_h_index = np.argmin(np.abs(LUT[-2] - phase_h))
            phase_v_index = np.argmin(np.abs(LUT[-1] - phase_v))
        except:
            # LUT中找不到相位值
            return -1, -1, retval

        # 获取对应点
        cor_points = LUT[phase_v_index][phase_h_index]
        cor_points = np.array(cor_points)

        if len(cor_points) > 0 and len(cor_points) < 20:
            # 获取LUT中点的x, y坐标的平均值作为第二次近似
            x0, y0 = np.mean(cor_points, axis=0)
            # 获取相位差最小的点的x, y坐标作为第二次近似
            p2_h_d = np.abs(p2_h[cor_points[:,1], cor_points[:,0]] - phase_h)
            p2_v_d = np.abs(p2_v[cor_points[:,1], cor_points[:,0]] - phase_v)
            x0, y0 = cor_points[np.argmin(p2_h_d + p2_v_d)]

            iter_num = 0

            # 迭代x和y的变体，其中场接近phase_v和phase_h
            while iter_num < 5:
                # 获取当前x和y值的最近坐标
                if int(np.round(x0)) - x0 == 0:
                    x1 = int(x0 - 1)
                    x2 = int(x0 + 1)
                else:
                    x1 = int(np.floor(x0))
                    x2 = int(np.ceil(x0))

                if int(np.round(y0)) - y0 == 0:
                    y1 = int(y0 - 1)
                    y2 = int(y0 + 1)
                else:
                    y1 = int(np.floor(y0))
                    y2 = int(np.ceil(y0))

                # 检查坐标是否在场上(是否为正且小于场的形状)
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and x1 < p1_h.shape[1] and x2 < p1_h.shape[1] and y1 < p1_h.shape[0] and y2 < p1_h.shape[0]:

                    # 获取水平相位的双线性插值系数
                    aa = calculate_bilinear_interpolation_coeficients(((x1, y1, p2_h[y1, x1]), (x1, y2, p2_h[y2, x1]),
                                                                       (x2, y2, p2_h[y2, x2]), (x2, y1, p2_h[y2, x1])))
                    # 获取垂直相位的双线性插值系数
                    bb = calculate_bilinear_interpolation_coeficients(((x1, y1, p2_v[y1, x1]), (x1, y2, p2_v[y2, x1]),
                                                                       (x2, y2, p2_v[y2, x2]), (x2, y1, p2_v[y2, x1])))

                    # 找到双线性插值等于phase_h和phase_v的位置
                    x0, y0 =  fsolve(bilinear_phase_fields_approximation, (x1, y1), args=(aa, bb, phase_h, phase_v))
                    # x0, y0 = minimize(
                    #     bilinear_phase_fields_approximation,
                    #     (x0, y0),
                    #     args=(aa, bb, phase_h, phase_v),
                    #     bounds=Bounds([x1, y1], [x2, y2]),
                    #     method="Powell",
                    # ).x

                    # 计算残差
                    h_res, v_res = bilinear_phase_fields_approximation((x0, y0), aa, bb, phase_h, phase_v) 

                    # 检查x和y是否在x1, x2, y1和y2之间
                    if x2 >= x0 >= x1 and y2 >= y0 >= y1:
                        return x0, y0, [h_res, v_res]
                    else:
                        iter_num = iter_num + 1
                else:
                    return -1, -1, [np.inf, np.inf]

            return -1, -1, [np.inf, np.inf]
        else:
            return -1, -1, [np.inf, np.inf]

    # 找到等相位曲线的坐标
    y_h, x_h = np.where(np.isclose(p2_h, phase_h, atol=10**-1))
    y_v, x_v = np.where(np.isclose(p2_v, phase_v, atol=10**-1))

    # 如果找不到等相线，则中断
    if y_h.size == 0 or y_v.size == 0:
        return -1, -1

    # 使用展平数组计算的更快方法
    # _, yx_h = np.unravel_index(np.where(np.isclose(p2_h, p1_h[y, x], atol=10**-1)), p2_h.shape)
    # _, yx_v = np.unravel_index(np.where(np.isclose(p2_v, p1_v[y, x], atol=10**-1)), p2_v.shape)

    # 找到坐标交集的ROI
    y_h_min = np.min(y_h)
    y_h_max = np.max(y_h)
    x_v_min = np.min(x_v)
    x_v_max = np.max(x_v)

    # 为等相位曲线的坐标应用ROI
    y_h = y_h[(x_h >= x_v_min) & (x_h <= x_v_max)]
    x_h = x_h[(x_h >= x_v_min) & (x_h <= x_v_max)]
    x_v = x_v[(y_v >= y_h_min) & (y_v <= y_h_max)]
    y_v = y_v[(y_v >= y_h_min) & (y_v <= y_h_max)]

    # 如果等相位线中的点太多，则中断
    if len(y_h) > 500 or len(y_v) > 500:
        return -1, -1

    # 如果没有找到点，则中断
    if x_h.size == 0 or x_v.size == 0:
        return -1, -1

    # 重塑坐标以使用广播
    x_h = x_h[:, np.newaxis]
    y_h = y_h[:, np.newaxis]
    y_v = y_v[np.newaxis, :]
    x_v = x_v[np.newaxis, :]

    # 计算坐标中点之间的距离
    distance = np.sqrt((x_h - x_v) ** 2 + (y_h - y_v) ** 2)

    # 找到最小距离的索引
    i_h_min, i_v_min = np.where(distance == distance.min())
    i_v_min = i_v_min[0]
    i_h_min = i_h_min[0]

    # 使用展平数组计算的更快方法
    # i_h_min, i_v_min = np.unravel_index(np.where(distance.ravel()==distance.min()), distance.shape)
    # i_v_min = i_v_min[0][0]
    # i_h_min = i_h_min[0][0]

    # 计算交点坐标(取两条线上最近点的平均值)
    x2, y2 = ((x_v[0, i_v_min] + x_h[i_h_min, 0]) / 2, (y_v[0, i_v_min] + y_h[i_h_min, 0]) / 2)

    return x2, y2


def get_phasogrammetry_correlation(p1_h: np.ndarray, p1_v: np.ndarray, p2_h: np.ndarray, p2_v: np.ndarray, x1: int, y1: int, x2: int, y2: int, window_size: int) -> np.ndarray:
    '''
    计算水平和垂直相位场的相关函数

    参数:
        p1_h (numpy array): 第一个相机的水平条纹相位场
        p1_v (numpy array): 第一个相机的垂直条纹相位场
        p2_h (numpy array): 第二个相机的水平条纹相位场
        p2_v (numpy array): 第二个相机的垂直条纹相位场
        x1 (int): 第一个相机点的水平坐标
        y1 (int): 第一个相机点的垂直坐标
        x2 (int): 第二个相机点的水平坐标
        y2 (int): 第二个相机点的垂直坐标
        window_size (int): 计算相关函数的窗口大小

    返回:
        (x, y) (tuple): 相关场中最大值的坐标，表示最佳匹配位置
    '''
    # 提取第一个相机的窗口区域
    p1_h_ij = p1_h[int(y1 - window_size//2):int(y1 + window_size//2), int(x1 - window_size//2):int(x1 + window_size//2)]
    p1_v_ij = p1_v[int(y1 - window_size//2):int(y1 + window_size//2), int(x1 - window_size//2):int(x1 + window_size//2)]
    # 计算窗口区域的平均值
    p1_h_m = np.mean(p1_h_ij)
    p1_v_m = np.mean(p1_v_ij)
    # 计算第一个相机的模板
    t1_h = (p1_h_ij - p1_h_m) ** 2
    t1_v = (p1_v_ij - p1_v_m) ** 2

    # 初始化相关场
    corelation_field = np.zeros((window_size, window_size))

    # 创建第二个相机的搜索区域网格
    xx = np.linspace(x2 - window_size // 2, x2 + window_size // 2, window_size)
    yy = np.linspace(y2 - window_size // 2, y2 + window_size // 2, window_size)

    # 在搜索区域中移动窗口并计算相关性
    for j in range(yy.shape[0]):
        for i in range(xx.shape[0]):
            x0 = xx[i]
            y0 = yy[j]
            # 提取第二个相机的窗口区域
            p2_h_ij = p2_h[int(y0 - window_size //2):int(y0 + window_size //2), int(x0 - window_size//2):int(x0 + window_size//2)]
            p2_v_ij = p2_v[int(y0 - window_size //2):int(y0 + window_size //2), int(x0 - window_size//2):int(x0 + window_size//2)]
            # 计算窗口区域的平均值
            p2_h_m = np.mean(p2_h_ij)
            p2_v_m = np.mean(p2_v_ij)
            # 计算第二个相机的模板
            t2_h = (p2_h_ij - p2_h_m) ** 2
            t2_v = (p2_v_ij - p2_v_m) ** 2

            # 确保窗口大小匹配，然后计算相关系数
            if p2_h_ij.size == p1_h_ij.size and p2_v_ij.size == p1_v_ij.size:
                # 计算归一化相关系数
                t = np.sum(t1_h * t1_v * t2_h * t2_v) / np.sqrt(np.sum(t1_h * t1_v) * np.sum(t2_h * t2_v))
                # 将相关系数存储到相关场中
                corelation_field[j, i] = t

    # 找到相关场中的最大值索引
    maximum = np.unravel_index(corelation_field.argmax(), corelation_field.shape)

    # 获取最大值X轴附近的像素
    cx0 = np.fabs(corelation_field[maximum[0], maximum[1] - 1])
    cx1 = np.fabs(corelation_field[maximum[0], maximum[1]    ])

    if maximum[1] == corelation_field.shape[1]:
        cx2 = np.fabs(corelation_field[maximum[0], maximum[1] + 1])
    else:
        cx2 = np.fabs(corelation_field[maximum[0], 0])

    # 获取最大值Y轴附近的像素
    cy0 = np.fabs(corelation_field[maximum[0] - 1, maximum[1]])
    cy1 = np.fabs(corelation_field[maximum[0]    , maximum[1]])

    if maximum[0] == corelation_field.shape[0]:
        cy2 = np.fabs(corelation_field[maximum[0] + 1, maximum[1]])
    else:
        cy2 = np.fabs(corelation_field[0, maximum[1]])

    # 使用3点高斯拟合进行亚像素精度定位
    try:
        # 计算X方向的亚像素位置
        x_max = maximum[1] + (np.log(np.abs(cx0))  - np.log(np.abs(cx2)))/(2 * np.log(np.abs(cx0)) - 4 * np.log(np.abs(cx1)) + 2 * np.log(np.abs(cx2)))
    except (ZeroDivisionError, ValueError):
        x_max = 0
    try:
        # 计算Y方向的亚像素位置
        y_max = maximum[0] + (np.log(np.abs(cy0))  - np.log(np.abs(cy2)))/(2 * np.log(np.abs(cy0)) - 4 * np.log(np.abs(cy1)) + 2 * np.log(np.abs(cy2)))
    except (ZeroDivisionError, ValueError):
        y_max = 0

    # 由于相关函数的周期性，调整最大值位置
    if x_max > corelation_field.shape[0] / 2:
        x_max = x_max - corelation_field.shape[0]
    elif np.fabs(x_max) < 0.01:
        x_max = 0

    # 由于相关函数的周期性，调整最大值位置
    if y_max > corelation_field.shape[1] / 2:
        y_max = y_max - corelation_field.shape[1]
    elif np.fabs(y_max) < 0.01:
        y_max = 0

    # 返回亚像素精度的最佳匹配位置
    if not np.isnan(x_max) and not np.isnan(y_max):
        return x2 + x_max, y2 + y_max
    else:
        return -1, -1


def get_phase_field_ROI(fpp_measurement: FPPMeasurement, signal_to_nose_threshold: float = 0.25):
    '''
    通过信噪比阈值处理获取FPP测量的感兴趣区域(ROI)。
    ROI存储为掩码(fpp_measurement.signal_to_noise_mask)，信噪比低于阈值的点为0，高于阈值的点为1。
    此外，ROI还存储为由四个点定义的四边形，这些点由信噪比高于阈值的点的最小和最大x和y坐标组成。
    计算的ROI将存储在输入的fpp_measurement参数中。

    参数:
        fpp_measurement (FPPMeasurement): 用于计算ROI的FPP测量
        signal_to_nose_threshold (float) = 0.25: 计算ROI的信噪比阈值
    '''
    # 处理每个相机结果
    for cam_result in fpp_measurement.camera_results:
        # 计算信噪比(调制强度与平均强度的比值)
        signal_to_nose = cam_result.modulated_intensities[-1] / cam_result.average_intensities[-1]
        # 使用定义的阈值对信噪比进行阈值处理，找出高于阈值的点的坐标
        thresholded_coords = np.argwhere(signal_to_nose > signal_to_nose_threshold)

        # 存储ROI掩码(信噪比高于阈值的区域为1，其他为0)
        cam_result.signal_to_noise_mask = np.zeros(signal_to_nose.shape, dtype=int)
        cam_result.signal_to_noise_mask[signal_to_nose > signal_to_nose_threshold] = 1

        # 确定阈值区域周围的四个点
        x_min = np.min(thresholded_coords[:, 1])  # 最小x坐标
        x_max = np.max(thresholded_coords[:, 1])  # 最大x坐标
        y_min = np.min(thresholded_coords[:, 0])  # 最小y坐标
        y_max = np.max(thresholded_coords[:, 0])  # 最大y坐标

        # 存储确定的ROI(四边形的四个角点)
        cam_result.ROI = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


def get_phase_field_LUT(measurement: FPPMeasurement) -> list[list[list]]:
    '''
    获取水平和垂直相位场的查找表(LUT)，以提高相位测量计算速度。
    LUT是一个二维数组，其索引对应于这些坐标中的水平和垂直相位值。
    通过知道水平和垂直相位的值，可以快速找到具有这些值的点的坐标。
    LUT是包含二维坐标的三层嵌套列表。

    参数:
        measurement (FPPMeasurement): 具有水平和垂直条纹的FPP测量
    返回:
        LUT (list[list[list]]): 包含水平和垂直相位值对应点坐标的LUT结构
    '''
    assert len(measurement.camera_results) == 4, 'FPP测量的camera_results中应该有四(4)个相机结果'
    
    # 获取第二个相机的水平和垂直解包裹相位
    cam1_meas_h = measurement.camera_results[2]
    cam1_meas_v = measurement.camera_results[0]
    cam2_meas_h = measurement.camera_results[3]
    cam2_meas_v = measurement.camera_results[1]

    p_h = cam2_meas_h.unwrapped_phases[-1]
    p_v = cam2_meas_v.unwrapped_phases[-1]

    # 查找水平和垂直相位的范围
    if cam1_meas_h.ROI_mask is not None:
        # 如果定义了ROI掩码 - 从第一个相机的ROI区域获取最小最大相位
        p_h1 = cam1_meas_h.unwrapped_phases[-1][cam1_meas_h.ROI_mask == 1]
        p_v1 = cam1_meas_v.unwrapped_phases[-1][cam1_meas_v.ROI_mask == 1]
        ph_max = np.max(p_h1)
        ph_min = np.min(p_h1)
        pv_max = np.max(p_v1)
        pv_min = np.min(p_v1)
    else:
        ph_max = np.max(p_h)
        ph_min = np.min(p_h)
        pv_max = np.max(p_v)
        pv_min = np.min(p_v)

    # 设置相位步长
    step = 1.0

    # 创建相位范围数组
    h_range = np.arange(ph_min, ph_max, step)
    v_range = np.arange(pv_min, pv_max, step)

    # 确定LUT结构的大小
    w, h = h_range.shape[0] + 1, v_range.shape[0] + 1

    # 创建LUT结构(三层嵌套列表)
    LUT = [[[] for x in range(w)] for y in range(h)]

    # 获取相位场的宽度和高度
    w = p_h.shape[1]
    h = p_h.shape[0]

    # 相位四舍五入，并偏移使其从零开始
    p_h_r = np.round((p_h - ph_min) / step).astype(int).tolist()
    p_v_r = np.round((p_v - pv_min) / step).astype(int).tolist()

    # 用水平和垂直值作为索引的点的坐标填充LUT
    for y in range(h):
        for x in range(w):
            # 检查点是否在信噪比掩码的有效区域内
            if cam2_meas_h.signal_to_noise_mask[y, x] == 1 and cam2_meas_v.signal_to_noise_mask[y, x] == 1:
                # 检查相位值是否在有效范围内
                if (pv_max - pv_min) / step >= p_v_r[y][x] >= 0 and (ph_max - ph_min) / step >= p_h_r[y][x] >= 0:
                    # 将坐标添加到对应的LUT位置
                    LUT[p_v_r[y][x]][p_h_r[y][x]].append([x, y])

    # 在LUT末尾添加水平和垂直相位的范围
    LUT.append(h_range)
    LUT.append(v_range)

    return LUT


def process_fppmeasurement_with_phasogrammetry(measurement: FPPMeasurement, step_x: float, step_y: float, LUT:list[list[list[int]]]=None) -> tuple[np.ndarray, np.ndarray]:
    '''
    使用相位测量法为两组相位场找到2D对应点

    参数:
        measurement (FPPMeasurement): 包含两个相机的水平和垂直相位场的FPPMeasurement实例
        step_x, step_y (float): 计算对应点的水平和垂直步长
        LUT (list[list[list]]): 包含水平和垂直相位值对应点坐标的LUT结构
    返回:
        points_1 (numpy array [N, 2]): 第一个相机的对应2D点
        points_2 (numpy array [N, 2]): 第二个相机的对应2D点
    '''
    # 获取最高频率的相位
    p1_h = measurement.camera_results[2].unwrapped_phases[-1]
    p2_h = measurement.camera_results[3].unwrapped_phases[-1]

    p1_v = measurement.camera_results[0].unwrapped_phases[-1]
    p2_v = measurement.camera_results[1].unwrapped_phases[-1]

    # 从测量对象获取ROI
    ROI1 = measurement.camera_results[0].ROI

    # 从第二个相机的相位场中剪切ROI
    ROIx = slice(0, measurement.camera_results[1].unwrapped_phases[-1].shape[1])
    ROIy = slice(0, measurement.camera_results[1].unwrapped_phases[-1].shape[0])

    p2_h = p2_h[ROIy, ROIx]
    p2_v = p2_v[ROIy, ROIx]

    # 计算第一幅图像上的坐标网格
    xx = np.arange(0, p1_h.shape[1], step_x, dtype=np.int32)
    yy = np.arange(0, p1_h.shape[0], step_y, dtype=np.int32)

    # 创建坐标列表
    coords1 = []

    # 只保留ROI掩码内的坐标
    for y in yy:
        for x in xx:
            if measurement.camera_results[0].ROI_mask[y, x] == 1:
                coords1.append((x, y))

    coords2 = []
    errors = []

    coords_to_delete = []

    if config.USE_MULTIPROCESSING:
        # 使用并行计算提高处理速度
        with multiprocessing.Pool(config.POOLS_NUMBER) as p:
            coords2 = p.starmap(find_phasogrammetry_corresponding_point, [(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1], LUT) for i in range(len(coords1))])

        # 搜索未找到的对应点
        for i in range(len(coords2)):
            if coords2[i][0] < 0 and coords2[i][1] < 0:
                coords_to_delete.append(i)
            else:
                # 添加ROI左上角坐标
                coords2[i] = (coords2[i][0] + ROIx.start, coords2[i][1] + ROIy.start)

        # 如果未找到点，则从网格中删除坐标
        for index in reversed(coords_to_delete):
            coords1.pop(index)
            coords2.pop(index)
    else:
        # 非并行处理方式
        for i in range(len(coords1)):
            # 在第二幅图像上找到对应点坐标
            x, y, err = find_phasogrammetry_corresponding_point(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1], LUT)
            # 如果未找到点，则从网格中删除坐标
            if (x == -1 and y == -1) or (abs(err[0]) > 0.1 or abs(err[1]) > 0.1):
                coords_to_delete.append(i)
            else:
                coords2.append((x + ROIx.start, y + ROIy.start))
                errors.append(err)

        # 删除在第二幅图像上没有对应点的网格点
        for index in reversed(coords_to_delete):
            coords1.pop(index)

    # 形成第一和第二幅图像上对应点的坐标集
    image1_points = []
    image2_points = []
    distance = []

    # 计算对应点之间的距离
    for point1, point2 in zip(coords1, coords2):
        image1_points.append([point1[0], point1[1]])
        image2_points.append([point2[0], point2[1]])
        distance.append(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5)

    # 移除异常值(距离超过标准差10倍的点)
    std_d = np.std(distance)
    indicies_to_delete = [i for i in range(len(distance)) if distance[i] > std_d * 10]
    for index in reversed(indicies_to_delete):
        image1_points.pop(index)
        image2_points.pop(index)
        errors.pop(index)

    # 在从函数返回结果之前将列表转换为数组
    image1_points = np.array(image1_points, dtype=np.float32)
    image2_points = np.array(image2_points, dtype=np.float32)
    errors = np.array(errors, dtype=np.float32)

    return image1_points, image2_points, errors


def calculate_displacement_field(field1: np.ndarray, field2: np.ndarray, win_size_x: int, win_size_y: int, step_x: int, step_y: int) -> np. ndarray:
    '''
    通过相关计算两个标量场之间的位移场。

    参数:
        field1 (2D numpy array): 第一个标量场
        field2 (2D numpy array): 第二个标量场
        win_size_x (int): 询问窗口水平尺寸
        win_size_y (int): 询问窗口垂直尺寸
        step_x (int): 询问窗口划分的水平步长
        step_y (int): 询问窗口划分的垂直步长
    返回:
        vector_field (): 位移向量场
    '''
    assert field1.shape == field2.shape, 'field1和field2的形状必须相等'
    assert win_size_x > 4 and win_size_y > 4, '询问窗口的尺寸应大于4像素'
    assert step_x > 0 and step_y > 0, '水平和垂直步长应大于零'

    # 获取询问窗口
    list_of_windows = [[], []]
    list_of_coords = []

    # 获取场的尺寸
    width = field1.shape[1]
    height = field1.shape[0]
    # 计算窗口数量
    num_win_x = range(int(np.floor((width - win_size_x) / step_x + 1)))
    num_win_y = range(int(np.floor((height - win_size_y) / step_y + 1)))

    # 遍历所有窗口位置
    for i in num_win_x:
        start_x = step_x * i
        end_x = step_x * i + win_size_x
        center_x = np.round(end_x - win_size_x / 2)

        for j in num_win_y:
            start_y = step_y * j
            end_y = step_y * j + win_size_y
            center_y = np.round(end_y - win_size_y / 2)

            # 提取窗口区域
            window1 = field1[start_y:end_y, start_x:end_x]
            window2 = field2[start_y:end_y, start_x:end_x]
            # 存储窗口和中心坐标
            list_of_windows[0].append(window1)
            list_of_windows[1].append(window2)
            list_of_coords.append([center_x, center_y])

    # 计算相关函数
    correlation_list = []

    # 创建2D高斯核
    gauss = np.outer(signal.windows.gaussian(win_size_x, win_size_x / 2),
                     signal.windows.gaussian(win_size_y, win_size_y / 2))

    # 对每个窗口计算相关性
    for i in range(len(list_of_windows[0])):
        # 对询问窗口应用窗函数
        list_of_windows[0][i] = list_of_windows[0][i] * gauss
        list_of_windows[1][i] = list_of_windows[1][i] * gauss
        
        # 计算窗口的均值和标准差
        mean1 = np.mean(list_of_windows[0][i])
        std1 = np.std(list_of_windows[0][i])
        mean2 = np.mean(list_of_windows[1][i])
        std2 = np.std(list_of_windows[1][i])
        
        # 使用FFT计算归一化互相关
        a = np.fft.rfft2(list_of_windows[0][i] - mean1, norm='ortho')
        b = np.fft.rfft2(list_of_windows[1][i] - mean2, norm='ortho')
        c = np.multiply(a, b.conjugate())
        d = np.fft.irfft2(c)
        
        # 防止除零错误
        if std1 == 0:
            std1 = 1
        if std2 == 0:
            std2 = 1
            
        # 归一化相关结果
        e = d / (std1 * std2)
        correlation_list.append(e)

    # 找到最大值
    maximums_list = []

    # 处理每个相关结果
    for i in range(len(correlation_list)):

        # 找到x和y的最大索引
        maximum = np.unravel_index(correlation_list[i].argmax(), correlation_list[i].shape)                        

        # 获取X轴最大值附近的像素
        cx0 = np.fabs(correlation_list[i][maximum[0], maximum[1] - 1])
        cx1 = np.fabs(correlation_list[i][maximum[0], maximum[1]    ])

        if maximum[1] == correlation_list[i].shape[1]:
            cx2 = np.fabs(correlation_list[i][maximum[0], maximum[1] + 1])
        else:
            cx2 = np.fabs(correlation_list[i][maximum[0], 0])

        # 获取Y轴最大值附近的像素
        cy0 = np.fabs(correlation_list[i][maximum[0] - 1, maximum[1]])
        cy1 = np.fabs(correlation_list[i][maximum[0]    , maximum[1]])

        if maximum[0] == correlation_list[i].shape[0]:
            cy2 = np.fabs(correlation_list[i][maximum[0] + 1, maximum[1]])
        else:
            cy2 = np.fabs(correlation_list[i][0, maximum[1]])

        # 3点高斯拟合
        try:
            x_max = maximum[1] + (np.log(np.abs(cx0))  - np.log(np.abs(cx2)))/(2 * np.log(np.abs(cx0)) - 4 * np.log(np.abs(cx1)) + 2 * np.log(np.abs(cx2)))
        except (ZeroDivisionError, ValueError):
            x_max = 0
        try:
            y_max = maximum[0] + (np.log(np.abs(cy0))  - np.log(np.abs(cy2)))/(2 * np.log(np.abs(cy0)) - 4 * np.log(np.abs(cy1)) + 2 * np.log(np.abs(cy2)))
        except (ZeroDivisionError, ValueError):
            y_max = 0

        # 由于相关函数的周期性，调整最大值
        if x_max > correlation_list[i].shape[0] / 2:
            x_max = x_max - correlation_list[i].shape[0]
        elif np.fabs(x_max) < 0.01:
            x_max = 0

        # 由于相关函数的周期性，调整最大值
        if y_max > correlation_list[i].shape[1] / 2:
            y_max = y_max - correlation_list[i].shape[1]
        elif np.fabs(y_max) < 0.01:
            y_max = 0

        # 存储最大值信息
        maximums_list.append([x_max, y_max, np.max(correlation_list[i])])

    # 创建向量场
    vector_field = []

    return np.array(list_of_coords), np.array(maximums_list)
