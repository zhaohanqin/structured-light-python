# 基于相位解包裹的投影仪标定系统

## 项目概述

基于相位解包裹的投影仪标定系统是一个高精度的投影仪标定工具，通过相移条纹图案和相位解包裹方法建立投影仪与相机之间的像素精确对应关系，进而计算投影仪的内参矩阵、畸变系数以及与相机的相对位置关系。该方法提供亚像素级别的精度，适用于高精度结构光3D扫描系统。

## 输入图像格式要求

### 图像命名规则

系统支持以下图像命名格式：

#### 水平方向相移图像

- `h_phase_shift_*.png`（推荐）
- `horizontal/*.png`
- `h_*.jpg`

#### 垂直方向相移图像

- `v_phase_shift_*.png`（推荐）
- `vertical/*.png`
- `v_*.jpg`

### 图像内容要求

- 使用四步相移法（或N步相移法）拍摄的图像
- 每张图像中应包含清晰可见的标定板
- 标定板上应投影有相移条纹图案
- 图像应按照相位偏移顺序命名（如0°、90°、180°、270°）

## 标定步骤详解

### 1. 准备工作

- 完成相机标定，并获取相机标定参数文件
- 准备标定板（支持棋盘格、圆形标定板或环形标定板）
- 准备投影仪和相机，固定其相对位置

### 2. 数据采集

- 投影水平方向的相移条纹图案到标定板上
- 用相机拍摄投影结果
- 投影垂直方向的相移条纹图案到标定板上
- 用相机拍摄投影结果
- 将所有图像按命名规则保存在同一文件夹中

### 3. 相位解包裹处理

标定程序会执行以下步骤：

- 加载相移图像
- 分别处理水平和垂直方向的相移图像
- 使用quality_guided方法进行相位解包裹
- 生成解包裹相位图和可视化结果

### 4. 建立投影仪-相机像素对应关系

核心代码在`extract_phase_correspondence`函数中：

```python
for y in range(height):
    for x in range(width):
        if horizontal_unwrapped is not None and vertical_unwrapped is not None:
            # 获取该像素的水平和垂直相位值
            h_phase = horizontal_unwrapped[y, x]
            v_phase = vertical_unwrapped[y, x]
            
            # 计算投影仪坐标
            proj_x = (h_phase / h_max_phase) * projector_width
            proj_y = (v_phase / v_max_phase) * projector_height
            
            # 存储对应关系
            correspondences[(y, x)] = (proj_x, proj_y)
```

### 5. 标定板检测与匹配

- 在参考图像中检测标定板
- 提取标定板角点的3D坐标和图像坐标
- 将标定板角点与相位对应关系结合
- **使用双线性插值获取亚像素精度的相位值**

标定板角点通常以亚像素精度检测，为了充分利用这一精度，系统使用双线性插值计算角点位置处的精确相位值：

```python
# 使用双线性插值获取水平和垂直相位值
h_phase = bilinear_interpolate(h_unwrapped, camera_y, camera_x)
v_phase = bilinear_interpolate(v_unwrapped, camera_y, camera_x)

# 计算对应的投影仪坐标
proj_x = (h_phase / h_max_phase) * projector_width
proj_y = (v_phase / v_max_phase) * projector_height
```

双线性插值通过角点周围的四个像素相位值计算精确的亚像素相位，相比简单的四舍五入方法，可以显著提高标定精度。

### 6. 投影仪标定

- 使用OpenCV的标定函数计算投影仪内参
- 计算投影仪相对于相机的外参（旋转矩阵和平移向量）
- 评估标定质量（重投影误差）
- 保存标定结果

## 使用方法

### 命令行方式

```bash
python projector_calibration_phase.py --camera-params camera_calibration.json --phase-images ./phase_images_folder --projector-width 1280 --projector-height 720 --board-type chessboard --chessboard-width 9 --chessboard-height 6 --square-size 20.0
```

### 交互式方式

直接运行程序，按提示输入各项参数：

```bash
python projector_calibration_phase.py
```

### 主要参数说明

- `--camera-params`: 相机标定参数文件路径
- `--phase-images`: 包含相移图案图像的文件夹
- `--projector-width`: 投影仪宽度分辨率
- `--projector-height`: 投影仪高度分辨率
- `--board-type`: 标定板类型（棋盘格、圆点或环形）
- `--chessboard-width`: 标定板宽度点数量
- `--chessboard-height`: 标定板高度点数量
- `--square-size`: 标定板方格尺寸或圆心间距(mm)
- `--output-folder`: 输出结果文件夹
- `--no-visualize`: 不显示可视化结果
- `--no-global-optimization`: 不使用全局优化方法
- `--sampling-step`: 相位图采样步长（默认为4）
- `--no-adaptive-threshold`: 不使用自适应质量阈值
- `--quality-threshold`: 手动设置相位质量阈值(0-1)

## 输出结果

- 相位解包裹可视化图像
- 水平和垂直相位组合图
- 投影仪内参矩阵和畸变系数
- 投影仪与相机的相对位置关系（旋转矩阵和平移向量）
- 标定质量评估报告
- JSON格式的标定结果文件

## 技术细节

### 相位解包裹方法

系统使用四步相移法和质量引导的相位解包裹算法，通过解析相位值获得亚像素级别的对应关系。相位解包裹过程由集成的`compute_wrapped_phase`、`compute_phase_quality`和`quality_guided_unwrap`函数实现。

### 像素对应关系原理

- 水平条纹的解包裹相位值映射到投影仪的X坐标
- 垂直条纹的解包裹相位值映射到投影仪的Y坐标
- 通过相位的线性映射建立亚像素级精度的对应关系

### 双线性插值技术

系统使用双线性插值技术提高标定精度。对于每个亚像素精度的标定板角点，通过以下步骤获取精确相位：

1. 找到角点周围的四个整数像素点
2. 获取这四个点的相位值
3. 根据角点到四个点的距离计算权重
4. 按权重计算出角点位置的精确相位值

双线性插值公式：

```bash
f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
```

这种方法避免了简单四舍五入带来的精度损失，显著提升了标定结果的准确性。

### 全局优化方法

系统支持全局优化标定方法，可以同时优化相机和投影仪参数，提高标定精度：

```python
# 使用全局优化方法进行标定
if global_optimization:
    calibration.log("使用全局优化方法进行标定...")
    projector_matrix, projector_dist, R, T, reproj_error = calibration.calibrate_projector_with_camera_global(
        camera_matrix, camera_dist, calibration_points, board_points
    )
```

全局优化通过最小化相机和投影仪的总重投影误差，获得更准确的标定结果。

### 支持的标定板类型

- 棋盘格标定板：适合精确角点检测
- 圆形标定板：适合光照变化较大的场景
- 环形标定板：适合高反光或特殊光照条件

## 系统依赖

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- scipy (用于全局优化，可选)
- tqdm (用于显示进度条，可选)

## 在其他程序中使用标定结果

### 加载标定结果

```python
import json
import numpy as np

def load_projector_calibration(calibration_file):
    """加载投影仪标定结果"""
    with open(calibration_file, 'r') as f:
        data = json.load(f)
    
    # 提取参数
    projector_width = data['projector_width']
    projector_height = data['projector_height']
    projector_matrix = np.array(data['projector_matrix'])
    projector_dist = np.array(data['projector_dist'])
    rotation_matrix = np.array(data['rotation_matrix'])
    translation_vector = np.array(data['translation_vector'])
    
    return {
        'projector_width': projector_width,
        'projector_height': projector_height,
        'projector_matrix': projector_matrix,
        'projector_dist': projector_dist,
        'R': rotation_matrix,
        'T': translation_vector
    }

# 使用示例
calibration_params = load_projector_calibration('projector_calibration_latest.json')
```

### 将3D点投影到投影仪平面

```python
import cv2
import numpy as np

def project_3d_points_to_projector(points_3d, camera_matrix, camera_dist, 
                                 projector_matrix, projector_dist, R, T):
    """将3D点投影到投影仪平面"""
    # 先将3D点投影到相机平面
    camera_points, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), 
                                       camera_matrix, camera_dist)
    camera_points = camera_points.reshape(-1, 2)
    
    # 计算从相机到投影仪的变换
    R_inv = R.T
    T_inv = -R_inv @ T
    
    # 将点从相机坐标系转换到投影仪坐标系
    points_projector = []
    for point in points_3d:
        # 转换到投影仪坐标系
        point_proj = R_inv @ point.reshape(3, 1) + T_inv
        points_projector.append(point_proj.ravel())
    
    # 将3D点投影到投影仪平面
    projector_points, _ = cv2.projectPoints(np.array(points_projector), 
                                          np.zeros(3), np.zeros(3),
                                          projector_matrix, projector_dist)
    
    return projector_points.reshape(-1, 2)
```

### 创建投影图案

```python
import cv2
import numpy as np

def create_projection_pattern(projector_width, projector_height, points_2d, radius=5):
    """创建包含指定点的投影图案"""
    # 创建空白图像
    pattern = np.zeros((projector_height, projector_width), dtype=np.uint8)
    
    # 在图像上绘制点
    for point in points_2d:
        x, y = int(round(point[0])), int(round(point[1]))
        # 检查点是否在图像范围内
        if 0 <= x < projector_width and 0 <= y < projector_height:
            cv2.circle(pattern, (x, y), radius, 255, -1)
    
    return pattern
```

### 完整使用示例

```python
import cv2
import numpy as np
import json

# 1. 加载标定结果
with open('projector_calibration_latest.json', 'r') as f:
    calib_data = json.load(f)

projector_matrix = np.array(calib_data['projector_matrix'])
projector_dist = np.array(calib_data['projector_dist'])
R = np.array(calib_data['rotation_matrix'])
T = np.array(calib_data['translation_vector'])
projector_width = calib_data['projector_width']
projector_height = calib_data['projector_height']

# 2. 加载相机参数
with open('camera_calibration_latest.json', 'r') as f:
    camera_data = json.load(f)

camera_matrix = np.array(camera_data['camera_matrix'])
camera_dist = np.array(camera_data['dist_coeffs'])

# 3. 创建3D点
points_3d = np.array([
    [0, 0, 100],
    [50, 0, 100],
    [0, 50, 100],
    [50, 50, 100]
], dtype=np.float32)

# 4. 将3D点投影到投影仪平面
projector_points = project_3d_points_to_projector(
    points_3d, camera_matrix, camera_dist, 
    projector_matrix, projector_dist, R, T
)

# 5. 创建投影图案
pattern = create_projection_pattern(projector_width, projector_height, projector_points)

# 6. 保存或显示投影图案
cv2.imwrite('projection_pattern.png', pattern)
cv2.imshow('Projection Pattern', pattern)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 注意事项

1. 确保相机已经过良好标定
2. 标定过程中，保持投影仪和相机的相对位置不变
3. 标定板应平整，无明显变形
4. 拍摄环境光线适中，避免过曝或欠曝
5. 相移图案应覆盖整个标定板
6. 使用自适应质量阈值可以自动过滤低质量相位点
7. 采样步长参数可以根据需要调整，较小的值提供更多的对应点但会增加计算量
8. 全局优化方法通常能提供更高的标定精度，但需要更长的计算时间

## 错误处理

程序提供了详细的错误处理机制，包括以下异常类型：

- `PhaseUnwrappingError`: 相位解包裹失败
- `BoardDetectionError`: 标定板检测失败
- `CorrespondenceError`: 无法建立足够的点对应关系
- `CalibrationError`: 其他标定过程中的错误

每种错误都会提供具体的错误信息和改进建议。

## 标定质量评估

系统会根据重投影误差评估标定质量：

- 小于0.5像素：极佳
- 0.5-1.0像素：良好
- 1.0-2.0像素：一般
- 大于2.0像素：较差

如果标定质量较差，系统会提供针对性的改进建议。
