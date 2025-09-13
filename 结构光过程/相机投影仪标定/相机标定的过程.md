# 相机标定过程详解

在结构光三维扫描系统中，相机标定是第一个也是最基础的步骤。相机标定的目的是求解相机的内部参数（内参矩阵和畸变系数）以及相机与世界坐标系之间的关系（外参矩阵）。本文档详细介绍了相机标定的原理、方法和实现细节。

## 1. 相机标定的基本原理

相机标定是确定相机成像几何模型参数的过程。相机成像过程可以简化为针孔相机模型，该模型将三维世界中的点通过透视投影映射到二维图像平面上。

### 1.1 针孔相机模型

针孔相机模型描述了三维世界坐标系中的点 $(X, Y, Z)$ 如何投影到二维图像平面上的点 $(u, v)$：

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

其中：

- $(u, v)$ 是图像平面上的像素坐标
- $(X, Y, Z)$ 是世界坐标系中的三维点坐标
- $f_x, f_y$ 是相机的焦距（以像素为单位）
- $(c_x, c_y)$ 是主点坐标（光轴与图像平面的交点）
- $R$ 是旋转矩阵（3x3）
- $T$ 是平移向量（3x1）
- $s$ 是比例因子

### 1.2 相机内参和外参

根据针孔相机模型，相机参数可以分为内参和外参：

- **内参矩阵**：描述相机的内部光学特性
  
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- **外参矩阵**：描述相机在世界坐标系中的位置和姿态
  
  $$[R|T] = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix}$$

### 1.3 镜头畸变

实际相机的镜头存在畸变，主要包括径向畸变和切向畸变：

- **径向畸变**：由镜头的形状引起，使得图像点向内或向外偏移
  
  $$x_{distorted} = x(1 + k_1r^2 + k_2r^4 + k_3r^6)$$
  $$y_{distorted} = y(1 + k_1r^2 + k_2r^4 + k_3r^6)$$

- **切向畸变**：由镜头与图像平面不平行引起
  
  $$x_{distorted} = x + [2p_1xy + p_2(r^2 + 2x^2)]$$
  $$y_{distorted} = y + [p_1(r^2 + 2y^2) + 2p_2xy]$$

其中 $(x, y)$ 是归一化的图像坐标，$r^2 = x^2 + y^2$，$k_1, k_2, k_3$ 是径向畸变系数，$p_1, p_2$ 是切向畸变系数。

## 2. 相机标定的方法

目前最常用的相机标定方法是基于平面标定板（如棋盘格）的张氏标定法（Zhang's method）。该方法的基本思路是：

1. 准备具有已知几何特征的标定板（如棋盘格）
2. 从不同角度拍摄标定板的多张图像
3. 在图像中检测标定板的特征点（如棋盘格的内角点）
4. 建立三维世界坐标系中的特征点与其在图像中投影点之间的对应关系
5. 通过求解最小化重投影误差的优化问题，估计相机的内参和畸变系数

## 3. 相机标定的实现

在代码实现中，我们使用 OpenCV 库中的函数进行相机标定。主要步骤如下：

### 3.1 数据准备

首先，我们需要定义标定板的参数并准备世界坐标系中的点：

```python
def calibrate_camera(images_folder, chessboard_size=(9, 6), square_size=20.0):
    """
    使用棋盘格图像进行相机标定
    
    参数:
        images_folder: 包含棋盘格图像的文件夹路径
        chessboard_size: 棋盘格内角点数量 (宽, 高)
        square_size: 棋盘格方格尺寸(mm)
    """
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 应用实际方格尺寸
    
    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D空间中的点
    imgpoints = []  # 图像平面上的点
```

### 3.2 角点检测

接下来，我们需要加载图像并检测棋盘格角点：

```python
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
```

### 3.3 相机标定

然后，使用 OpenCV 的 `calibrateCamera` 函数进行标定：

```python
    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
```

### 3.4 计算重投影误差

标定完成后，我们需要计算重投影误差来评估标定的准确性：

```python
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    print(f"平均重投影误差: {reprojection_error}")
```

### 3.5 保存标定结果

最后，我们保存标定结果以供后续使用：

```python
def save_calibration_results(output_file, camera_matrix, dist_coeffs):
    """保存标定结果到文件"""
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    np.save(output_file, calibration_data)
    print(f"标定结果已保存至 {output_file}")
```

## 4. 畸变校正

使用标定得到的参数，我们可以对相机捕获的图像进行畸变校正：

```python
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
```

## 5. 相机标定的完整流程

结合以上步骤，相机标定的完整流程如下：

```python
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
```

## 6. 相机标定结果分析

### 6.1 内参矩阵

相机内参矩阵通常具有以下形式：

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

内参矩阵中的参数含义如下：

- $f_x, f_y$：相机的焦距（以像素为单位）
- $c_x, c_y$：主点坐标（通常接近图像中心）

### 6.2 畸变系数

畸变系数通常包括：

- 径向畸变系数：$k_1, k_2, k_3$
- 切向畸变系数：$p_1, p_2$

### 6.3 重投影误差

重投影误差是评估标定质量的重要指标，它表示三维世界点通过标定参数重新投影到图像平面后与原始检测点之间的距离。较小的重投影误差表示标定结果更准确。

## 7. 提高标定精度的方法

为了提高相机标定的精度，可以采取以下措施：

1. **使用高质量的标定板**：确保标定板的制作精度高，角点清晰可见
2. **采集足够多的图像**：从不同角度和距离拍摄至少10-20张标定板图像
3. **保证标定板的覆盖范围**：标定板图像应尽可能覆盖整个相机视场
4. **避免极端视角**：过于倾斜的视角可能导致角点检测不准确
5. **稳定的光照条件**：避免过暗或过亮的环境，减少反光和阴影
6. **亚像素角点检测**：使用亚像素精度的角点检测方法提高准确性
7. **迭代优化**：可以多次运行标定过程，筛选出重投影误差较小的结果

## 8. 在结构光系统中的应用

在结构光三维扫描系统中，相机标定是第一步，也是最重要的步骤之一。准确的相机标定为后续的投影仪标定和三维重建奠定基础。相机标定的结果将直接用于：

1. **投影仪标定**：与相机配合完成投影仪的内参和外参标定
2. **相位解包裹**：将相位值转换为三维坐标需要相机的内参和外参
3. **点云生成**：基于相机模型将二维图像点转换为三维世界点
4. **多相机系统的配准**：在多相机系统中，相机之间的相对位置关系依赖于各自的标定结果

通过高精度的相机标定，可以显著提高结构光三维扫描系统的测量精度和重建质量。
