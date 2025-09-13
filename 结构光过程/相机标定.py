import os
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

def calibrate_camera(images_folder, chessboard_size=(9, 6), square_size=20.0):
    """
    使用棋盘格图像进行相机标定
    
    参数:
        images_folder: 包含棋盘格图像的文件夹路径
        chessboard_size: 棋盘格内角点数量 (宽, 高)
        square_size: 棋盘格方格尺寸(mm)
    
    返回:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        rvecs: 旋转向量
        tvecs: 平移向量
        reprojection_error: 重投影误差
    """
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 应用实际方格尺寸
    
    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D空间中的点
    imgpoints = []  # 图像平面上的点
    
    # 获取图像文件列表
    images = glob.glob(f'{images_folder}/*.jpg')
    
    # 图像尺寸
    img_shape = None
    
    print(f"找到 {len(images)} 张校准图像")
    
    # 处理每张图像
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 如果找到角点，添加对象点和图像点
        if ret:
            objpoints.append(objp)
            
            # 使用亚像素精度优化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # 绘制并显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)
        else:
            print(f"未能在图像 {fname} 中找到所有棋盘格角点")
    
    cv2.destroyAllWindows()
    
    print(f"成功处理了 {len(objpoints)} 张图像")
    
    if len(objpoints) == 0:
        raise Exception("未找到任何有效的棋盘格图像，无法进行标定")
    
    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    print(f"平均重投影误差: {reprojection_error}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error

def save_calibration_results(output_file, camera_matrix, dist_coeffs):
    """保存标定结果到文件"""
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    np.save(output_file, calibration_data)
    print(f"标定结果已保存至 {output_file}")

def test_undistortion(image_path, camera_matrix, dist_coeffs):
    """测试畸变校正效果"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 获取最佳相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # 校正图像
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # 裁剪结果
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # 显示原始图像和校正后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
    plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title('校正后图像')
    plt.show()
    
    return dst

if __name__ == "__main__":
    # 标定相机
    calibration_images_folder = "../calibration_images"  # 包含棋盘格图像的文件夹
    
    # 确保文件夹存在
    if not os.path.exists(calibration_images_folder):
        print(f"警告: 文件夹 {calibration_images_folder} 不存在，创建示例文件夹")
        os.makedirs(calibration_images_folder)
        print(f"请在 {calibration_images_folder} 中放置棋盘格图像后再运行")
        exit(0)
    
    # 执行标定
    camera_matrix, dist_coeffs, rvecs, tvecs, error = calibrate_camera(
        calibration_images_folder,
        chessboard_size=(9, 6),  # 根据实际棋盘格尺寸调整
        square_size=20.0         # 根据实际方格尺寸调整(mm)
    )
    
    # 打印标定结果
    print("\n相机标定结果:")
    print("相机内参矩阵:")
    print(camera_matrix)
    print("\n畸变系数:")
    print(dist_coeffs)
    
    # 保存标定结果
    save_calibration_results("camera_calibration.npy", camera_matrix, dist_coeffs)
    
    # 测试校正效果 (如果有图像可用)
    test_images = glob.glob(f'{calibration_images_folder}/*.jpg')
    if test_images:
        test_undistortion(test_images[0], camera_matrix, dist_coeffs) 