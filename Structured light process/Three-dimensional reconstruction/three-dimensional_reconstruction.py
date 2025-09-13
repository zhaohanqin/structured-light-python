import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import os
import json
import argparse
from pathlib import Path


def phase_to_pointcloud(
    unwrapped_phase_x: np.ndarray,
    unwrapped_phase_y: np.ndarray,
    mask: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    projector_matrix: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    projector_width: int = 1280,
    projector_height: int = 800
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据解包裹的X和Y方向相位生成3D点云
    
    参数:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
        mask: 有效区域掩码 (可选)
        camera_matrix: 相机内参矩阵 (可选)
        projector_matrix: 投影仪内参矩阵 (可选)
        R: 从投影仪到相机的旋转矩阵 (可选)
        T: 从投影仪到相机的平移向量 (可选)
        projector_width: 投影仪宽度 (像素)
        projector_height: 投影仪高度 (像素)
    
    返回:
        points: 点云坐标数组 (N, 3)
        colors: 点云颜色数组 (N, 3)，如果没有颜色信息则为None
    """
    # 获取图像尺寸
    height, width = unwrapped_phase_x.shape
    
    # 如果没有提供掩码，创建全部为1的掩码
    if mask is None:
        mask = np.ones_like(unwrapped_phase_x, dtype=bool)
    
    # 计算投影仪坐标 (归一化到投影仪分辨率)
    proj_x = unwrapped_phase_x * projector_width / (2 * np.pi)
    proj_y = unwrapped_phase_y * projector_height / (2 * np.pi)
    
    # 如果提供了相机和投影仪参数，使用三角测量重建3D点
    if camera_matrix is not None and projector_matrix is not None and R is not None and T is not None:
        # 构建投影矩阵
        P_cam = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P_proj = projector_matrix @ np.hstack((R, T))
        
        # 准备存储3D点的数组
        points_3d = []
        
        # 为每个有效像素计算3D点
        for v in range(height):
            for u in range(width):
                if not mask[v, u]:
                    continue
                
                # 相机和投影仪的像素坐标
                cam_pixel = np.array([u, v, 1])  # 齐次坐标
                proj_pixel = np.array([proj_x[v, u], proj_y[v, u], 1])  # 齐次坐标
                
                # 构建线性方程组
                A = np.zeros((4, 4))
                
                # 相机投影方程
                A[0, :] = cam_pixel[0] * P_cam[2, :] - P_cam[0, :]
                A[1, :] = cam_pixel[1] * P_cam[2, :] - P_cam[1, :]
                
                # 投影仪投影方程
                A[2, :] = proj_pixel[0] * P_proj[2, :] - P_proj[0, :]
                A[3, :] = proj_pixel[1] * P_proj[2, :] - P_proj[1, :]
                
                # 使用SVD求解齐次线性方程组 AX = 0
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1, :]
                
                # 齐次坐标转为3D点
                X = X / X[3]
                points_3d.append(X[:3])
        
        # 转换为numpy数组
        points = np.array(points_3d)
        
        # 创建简单的颜色 (基于深度值的伪彩色)
        if len(points) > 0:
            # 归一化深度值
            z_values = points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_normalized = (z_values - z_min) / (z_max - z_min)
            
            # 创建伪彩色
            colors = plt.cm.jet(z_normalized)[:, :3]
        else:
            colors = None
    
    else:
        # 如果没有提供相机和投影仪参数，使用简化的方法生成点云
        # 此方法不能得到准确的3D重建，仅用于可视化
        
        # 创建网格坐标
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # 提取有效区域的坐标
        valid_x = xx[mask]
        valid_y = yy[mask]
        valid_phase_x = unwrapped_phase_x[mask]
        valid_phase_y = unwrapped_phase_y[mask]
        
        # 创建简化的3D点
        # 使用相位值作为深度信息，x和y坐标则使用图像坐标
        points = np.column_stack([
            valid_x,
            valid_y,
            (valid_phase_x + valid_phase_y) / 2  # 简化：使用相位和作为深度值
        ])
        
        # 创建伪彩色
        z_values = points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        z_normalized = (z_values - z_min) / (z_max - z_min)
        colors = plt.cm.jet(z_normalized)[:, :3]
    
    return points, colors


def create_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    创建Open3D点云对象
    
    参数:
        points: 点云坐标数组 (N, 3)
        colors: 点云颜色数组 (N, 3)，范围为[0,1]
    
    返回:
        pcd: Open3D点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def visualize_pointcloud(pcd: o3d.geometry.PointCloud, window_name: str = "3D点云") -> None:
    """
    可视化点云
    
    参数:
        pcd: Open3D点云对象
        window_name: 可视化窗口名称
    """
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
    
    # 显示点云
    o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=window_name)


def create_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, 
                              voxel_size: float = 0.05, 
                              depth: int = 9,
                              method: str = 'poisson') -> o3d.geometry.TriangleMesh:
    """
    从点云创建网格
    
    参数:
        pcd: Open3D点云对象
        voxel_size: 体素大小，用于降采样
        depth: 泊松重建深度
        method: 重建方法 ('poisson' 或 'alpha_shape')
    
    返回:
        mesh: 三角网格
    """
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # 确保法线方向一致
    pcd.orient_normals_consistent_tangent_plane(k=20)
    
    if method == 'poisson':
        # 泊松表面重建
        print("使用泊松重建生成网格...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        
        # 根据密度裁剪网格
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    elif method == 'alpha_shape':
        # Alpha Shape重建
        print("使用Alpha Shape重建生成网格...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=voxel_size * 5
        )
    
    else:
        raise ValueError(f"不支持的网格重建方法: {method}")
    
    # 简化网格
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    
    # 平滑网格
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    
    # 计算法线
    mesh.compute_vertex_normals()
    
    return mesh


def load_camera_params(file_path: str) -> np.ndarray:
    """
    加载相机参数
    
    参数:
        file_path: 相机参数文件路径
    
    返回:
        camera_matrix: 相机内参矩阵
    """
    try:
        if file_path.endswith('.npy'):
            camera_data = np.load(file_path, allow_pickle=True).item()
            camera_matrix = np.array(camera_data['camera_matrix'])
        else:
            with open(file_path, 'r') as f:
                camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'])
        print(f"已加载相机内参矩阵:\n{camera_matrix}")
        return camera_matrix
    except Exception as e:
        print(f"加载相机参数失败: {e}")
        return None


def load_projector_params(file_path: str) -> Tuple[np.ndarray, int, int]:
    """
    加载投影仪参数
    
    参数:
        file_path: 投影仪参数文件路径
    
    返回:
        projector_matrix: 投影仪内参矩阵
        projector_width: 投影仪宽度
        projector_height: 投影仪高度
    """
    try:
        if file_path.endswith('.npy'):
            projector_data = np.load(file_path, allow_pickle=True).item()
        else:
            with open(file_path, 'r') as f:
                projector_data = json.load(f)

        projector_matrix = np.array(projector_data['projector_matrix'])
        projector_width = projector_data.get('projector_width', 1280)
        projector_height = projector_data.get('projector_height', 800)
        
        print(f"已加载投影仪内参矩阵:\n{projector_matrix}")
        print(f"投影仪分辨率: {projector_width}x{projector_height}")
        
        return projector_matrix, projector_width, projector_height
    except Exception as e:
        print(f"加载投影仪参数失败: {e}")
        return None, 1280, 800


def load_extrinsics(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载外参数据
    
    参数:
        file_path: 外参文件路径
    
    返回:
        R: 旋转矩阵
        T: 平移向量
    """
    try:
        if file_path.endswith('.npy'):
            extrinsics_data = np.load(file_path, allow_pickle=True).item()
        else:
            with open(file_path, 'r') as f:
                extrinsics_data = json.load(f)
        
        R = np.array(extrinsics_data['R'])
        T = np.array(extrinsics_data['T'])
        
        print(f"已加载外参数据:")
        print(f"旋转矩阵R:\n{R}")
        print(f"平移向量T:\n{T}")
        
        return R, T
    except Exception as e:
        print(f"加载外参数据失败: {e}")
        return None, None


def load_unwrapped_phases(phase_x_path: str, phase_y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载解包裹相位数据
    
    参数:
        phase_x_path: X方向解包裹相位文件路径
        phase_y_path: Y方向解包裹相位文件路径
    
    返回:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
    """
    def read_phase_file(file_path: str) -> np.ndarray:
        """读取单个相位文件，支持.npy和图像格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.npy':
            return np.load(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            # 使用IMREAD_UNCHANGED来读取原始数据，支持更高位深
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"无法读取图像文件: {file_path}")
            
            # 如果图像是多通道的（如RGB），转换为灰度图
            if len(img.shape) == 3 and img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            return img.astype(np.float32) # 转换为浮点数以进行计算
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。请使用.npy或图像文件。")

    try:
        unwrapped_phase_x = read_phase_file(phase_x_path)
        unwrapped_phase_y = read_phase_file(phase_y_path)
        
        print(f"已加载解包裹相位数据:")
        print(f"X方向相位形状: {unwrapped_phase_x.shape}")
        print(f"Y方向相位形状: {unwrapped_phase_y.shape}")
        
        return unwrapped_phase_x, unwrapped_phase_y
    except Exception as e:
        print(f"加载解包裹相位数据失败: {e}")
        return None, None


def create_mask(unwrapped_phase_x: np.ndarray, unwrapped_phase_y: np.ndarray, percentile_threshold: float = 98.0) -> np.ndarray:
    """
    根据解包裹相位创建有效区域掩码
    
    参数:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
        percentile_threshold: 相位梯度阈值的百分位数(默认: 98.0)，值越大，保留的区域越多
    
    返回:
        mask: 有效区域掩码
    """
    # 计算相位梯度
    phase_gradient_x = np.gradient(unwrapped_phase_x)
    phase_gradient_y = np.gradient(unwrapped_phase_y)
    gradient_magnitude = np.sqrt(phase_gradient_x[0]**2 + phase_gradient_x[1]**2 +
                               phase_gradient_y[0]**2 + phase_gradient_y[1]**2)
    
    # 设置阈值，排除梯度过大的区域
    mask = gradient_magnitude < np.percentile(gradient_magnitude, percentile_threshold)
    
    # 可选：添加更多的掩码条件，例如排除相位值异常的区域
    # mask = np.logical_and(mask, unwrapped_phase_x > min_phase_x)
    # mask = np.logical_and(mask, unwrapped_phase_x < max_phase_x)
    
    return mask


def reconstruct_3d_scene(
    unwrapped_phase_x: np.ndarray,
    unwrapped_phase_y: np.ndarray,
    camera_matrix: np.ndarray,
    projector_matrix: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    projector_width: int,
    projector_height: int,
    output_dir: str = "output",
    create_mesh: bool = True,
    mask_percentile: float = 98.0
) -> None:
    """
    从解包裹相位重建3D场景并保存结果
    
    参数:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
        camera_matrix: 相机内参矩阵
        projector_matrix: 投影仪内参矩阵
        R: 从投影仪到相机的旋转矩阵
        T: 从投影仪到相机的平移向量
        projector_width: 投影仪宽度 (像素)
        projector_height: 投影仪高度 (像素)
        output_dir: 输出目录
        create_mesh: 是否创建网格
        mask_percentile: 掩码阈值百分位数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化解包裹相位
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(unwrapped_phase_x, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("X方向解包裹相位")
    
    plt.subplot(122)
    plt.imshow(unwrapped_phase_y, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("Y方向解包裹相位")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unwrapped_phases.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建有效区域掩码
    print(f"创建有效区域掩码 (梯度阈值百分位数: {mask_percentile})...")
    mask = create_mask(unwrapped_phase_x, unwrapped_phase_y, mask_percentile)
    
    # 可视化掩码
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f"有效区域掩码 (阈值百分位数: {mask_percentile}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mask.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 从相位生成点云
    print("根据解包裹相位生成点云...")
    points, colors = phase_to_pointcloud(
        unwrapped_phase_x, unwrapped_phase_y, mask,
        camera_matrix, projector_matrix, R, T,
        projector_width, projector_height
    )
    
    if len(points) == 0:
        print("警告: 生成的点云为空")
        return
    
    # 创建Open3D点云
    pcd = create_open3d_pointcloud(points, colors)
    
    # 移除噪声点
    print("移除噪声点...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 保存点云
    output_ply_file = os.path.join(output_dir, 'reconstructed_pointcloud.ply')
    o3d.io.write_point_cloud(output_ply_file, pcd)
    print(f"点云已保存至 {output_ply_file}")
    
    # 可视化点云
    print("可视化点云...")
    visualize_pointcloud(pcd, "重建的3D点云")
    
    # 如果需要，创建并保存网格
    if create_mesh:
        try:
            print("从点云创建网格...")
            mesh = create_mesh_from_pointcloud(pcd, voxel_size=0.01, depth=9)
            
            # 保存网格
            output_mesh_file = os.path.join(output_dir, 'reconstructed_mesh.ply')
            o3d.io.write_triangle_mesh(output_mesh_file, mesh)
            print(f"网格已保存至 {output_mesh_file}")
            
            # 可视化网格
            print("可视化网格...")
            o3d.visualization.draw_geometries([mesh], window_name="重建的3D网格")
        except Exception as e:
            print(f"网格创建失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从相位图和标定参数重建3D场景')
    
    parser.add_argument('--camera-params', type=str, required=True,
                        help='相机参数文件路径 (.npy 或 .json)')
    
    parser.add_argument('--projector-params', type=str, required=True,
                        help='投影仪参数文件路径 (.json 或 .npy)')
    
    parser.add_argument('--extrinsics', type=str, required=True,
                        help='相机和投影仪之间的外参文件路径 (.npy 或 .json)')
    
    parser.add_argument('--phase-x', type=str, required=True,
                        help='X方向解包裹相位文件路径 (.npy 或 图像文件)')
    
    parser.add_argument('--phase-y', type=str, required=True,
                        help='Y方向解包裹相位文件路径 (.npy 或 图像文件)')
    
    parser.add_argument('--output-dir', type=str, default='reconstruction_output',
                        help='输出目录路径')
    
    parser.add_argument('--create-mesh', action='store_true',
                        help='是否从点云创建网格')
    
    parser.add_argument('--mask-percentile', type=float, default=98.0,
                        help='掩码阈值的百分位数 (默认: 98.0)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 确认Open3D是否可用
    try:
        import open3d as o3d
    except ImportError:
        print("警告: Open3D库未安装，将无法进行3D可视化和网格重建")
        print("可以使用 'pip install open3d' 安装此库")
        return
    
    # 解析命令行参数
    try:
        args = parse_arguments()
    except SystemExit:
        # 如果没有提供命令行参数，则使用交互模式
        print("\n未提供足够的命令行参数，切换到交互模式...\n")
        args = None
    
    # 交互模式
    if args is None:
        print("==== 三维重建系统 ====")
        print("请提供以下参数以进行三维重建:\n")
        
        camera_params_path = input("1. 输入相机内参文件路径 (.npy 或 .json): ")
        projector_params_path = input("2. 输入投影仪内参文件路径 (.json 或 .npy): ")
        extrinsics_path = input("3. 输入相机和投影仪之间的外参文件路径 (.npy 或 .json): ")
        phase_x_path = input("4. 输入X方向解包裹相位文件路径 (.npy 或 图像文件): ")
        phase_y_path = input("5. 输入Y方向解包裹相位文件路径 (.npy 或 图像文件): ")
        output_dir = input("6. 输入输出目录路径 (默认: reconstruction_output): ") or "reconstruction_output"
        mask_percentile_input = input("7. 输入掩码阈值的百分位数 (默认: 98.0): ")
        mask_percentile = float(mask_percentile_input) if mask_percentile_input else 98.0
        create_mesh = input("8. 是否从点云创建网格? (y/n, 默认: y): ").lower() != 'n'
    else:
        camera_params_path = args.camera_params
        projector_params_path = args.projector_params
        extrinsics_path = args.extrinsics
        phase_x_path = args.phase_x
        phase_y_path = args.phase_y
        output_dir = args.output_dir
        mask_percentile = args.mask_percentile
        create_mesh = args.create_mesh
    
    # 检查文件是否存在
    for file_path in [camera_params_path, projector_params_path, extrinsics_path, phase_x_path, phase_y_path]:
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在!")
            return
    
    # 加载参数
    print("\n加载参数文件...")
    camera_matrix = load_camera_params(camera_params_path)
    projector_matrix, projector_width, projector_height = load_projector_params(projector_params_path)
    R, T = load_extrinsics(extrinsics_path)
    unwrapped_phase_x, unwrapped_phase_y = load_unwrapped_phases(phase_x_path, phase_y_path)
    
    # 检查是否所有参数都成功加载
    if None in [camera_matrix, projector_matrix, R, T, unwrapped_phase_x, unwrapped_phase_y]:
        print("错误: 未能成功加载所有必要参数!")
        return
    
    # 执行三维重建
    print("\n开始三维重建过程...")
    reconstruct_3d_scene(
        unwrapped_phase_x, unwrapped_phase_y,
        camera_matrix, projector_matrix, R, T,
        projector_width, projector_height,
        output_dir, create_mesh, mask_percentile
    )
    
    print("\n三维重建完成!")


if __name__ == "__main__":
    main() 