import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import os
import json

# 导入自定义模块
from 相位计算 import PhaseShiftingAlgorithm
from 解包裹相位 import unwrap_x_y_phases, temporal_phase_unwrap


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


def reconstruct_3d_from_unwrapped_phases(
    unwrapped_phase_x: np.ndarray,
    unwrapped_phase_y: np.ndarray,
    camera_params_file: str = None,
    projector_params_file: str = None,
    output_ply_file: str = "reconstructed_pointcloud.ply",
    output_mesh_file: str = "reconstructed_mesh.ply"
) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    """
    从解包裹相位重建3D场景并保存结果
    
    参数:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
        camera_params_file: 相机参数文件路径 (可选)
        projector_params_file: 投影仪参数文件路径 (可选)
        output_ply_file: 输出点云文件路径
        output_mesh_file: 输出网格文件路径
    
    返回:
        pcd: Open3D点云对象
        mesh: Open3D网格对象 (如果创建网格)
    """
    # 加载相机和投影仪参数
    camera_matrix = None
    projector_matrix = None
    R = None
    T = None
    
    if camera_params_file and os.path.exists(camera_params_file):
        try:
            camera_data = np.load(camera_params_file, allow_pickle=True).item()
            camera_matrix = np.array(camera_data['camera_matrix'])
            print("已加载相机参数")
        except Exception as e:
            print(f"加载相机参数失败: {e}")
    
    if projector_params_file and os.path.exists(projector_params_file):
        try:
            with open(projector_params_file, 'r') as f:
                projector_data = json.load(f)
            
            projector_matrix = np.array(projector_data['projector_matrix'])
            projector_width = projector_data['projector_width']
            projector_height = projector_data['projector_height']
            
            # 这里假设外参也存储在投影仪参数文件中
            if 'R' in projector_data and 'T' in projector_data:
                R = np.array(projector_data['R'])
                T = np.array(projector_data['T'])
            
            print("已加载投影仪参数")
        except Exception as e:
            print(f"加载投影仪参数失败: {e}")
    
    # 创建有效区域掩码 (排除噪声和无效区域)
    # 这里简单地排除相位变化过大的区域
    phase_gradient_x = np.gradient(unwrapped_phase_x)
    phase_gradient_y = np.gradient(unwrapped_phase_y)
    gradient_magnitude = np.sqrt(phase_gradient_x[0]**2 + phase_gradient_x[1]**2 +
                                phase_gradient_y[0]**2 + phase_gradient_y[1]**2)
    
    # 设置阈值，排除梯度过大的区域
    mask = gradient_magnitude < np.percentile(gradient_magnitude, 98)
    
    # 从相位生成点云
    points, colors = phase_to_pointcloud(
        unwrapped_phase_x, unwrapped_phase_y, mask,
        camera_matrix, projector_matrix, R, T,
        projector_width if 'projector_width' in locals() else 1280,
        projector_height if 'projector_height' in locals() else 800
    )
    
    if len(points) == 0:
        print("警告: 生成的点云为空")
        return None, None
    
    # 创建Open3D点云
    pcd = create_open3d_pointcloud(points, colors)
    
    # 移除噪声点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 保存点云
    o3d.io.write_point_cloud(output_ply_file, pcd)
    print(f"点云已保存至 {output_ply_file}")
    
    # 创建网格
    try:
        mesh = create_mesh_from_pointcloud(pcd, voxel_size=0.01, depth=9)
        
        # 保存网格
        o3d.io.write_triangle_mesh(output_mesh_file, mesh)
        print(f"网格已保存至 {output_mesh_file}")
    except Exception as e:
        print(f"网格创建失败: {e}")
        mesh = None
    
    return pcd, mesh


def process_measurement_data(
    data_folder: str,
    frequencies: List[int],
    algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
    camera_params_file: str = None,
    projector_params_file: str = None,
    output_folder: str = "reconstruction_results"
) -> None:
    """
    处理测量数据并重建3D模型
    
    参数:
        data_folder: 包含相移图像的文件夹
        frequencies: 使用的频率列表
        algorithm: 相移算法类型
        camera_params_file: 相机参数文件路径
        projector_params_file: 投影仪参数文件路径
        output_folder: 输出结果文件夹
    """
    from 相位计算 import process_phase_shifting_images
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 处理垂直条纹图像，计算X方向的包裹相位
    print("\n处理垂直条纹图像计算X方向包裹相位...")
    x_wrapped_phases = process_phase_shifting_images(
        data_folder,
        pattern_type='vertical',
        algorithm=algorithm,
        frequencies=frequencies
    )
    
    # 处理水平条纹图像，计算Y方向的包裹相位
    print("\n处理水平条纹图像计算Y方向包裹相位...")
    y_wrapped_phases = process_phase_shifting_images(
        data_folder,
        pattern_type='horizontal',
        algorithm=algorithm,
        frequencies=frequencies
    )
    
    # 解包裹相位
    print("\n解包裹X和Y方向相位...")
    x_unwrapped, y_unwrapped = unwrap_x_y_phases(
        x_wrapped_phases, y_wrapped_phases, 
        frequencies,
        method='temporal'
    )
    
    # 保存解包裹相位图
    np.save(os.path.join(output_folder, 'unwrapped_phase_x.npy'), x_unwrapped)
    np.save(os.path.join(output_folder, 'unwrapped_phase_y.npy'), y_unwrapped)
    
    # 可视化解包裹相位
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(x_unwrapped, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("X方向解包裹相位")
    
    plt.subplot(122)
    plt.imshow(y_unwrapped, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("Y方向解包裹相位")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'unwrapped_phases.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 重建3D模型
    print("\n从解包裹相位重建3D模型...")
    pcd, mesh = reconstruct_3d_from_unwrapped_phases(
        x_unwrapped, y_unwrapped,
        camera_params_file, projector_params_file,
        os.path.join(output_folder, 'pointcloud.ply'),
        os.path.join(output_folder, 'mesh.ply')
    )
    
    # 可视化结果
    if pcd is not None:
        print("\n可视化3D点云...")
        visualize_pointcloud(pcd, "重建的3D点云")
    
    if mesh is not None:
        print("\n可视化3D网格...")
        o3d.visualization.draw_geometries([mesh], window_name="重建的3D网格")


def simulate_3d_object():
    """生成模拟的3D对象数据用于测试"""
    # 创建一个3D高斯面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.exp(-(xx**2 + yy**2) / 2)
    
    # 添加一些噪声
    zz += np.random.normal(0, 0.01, zz.shape)
    
    # 创建相位图 (与深度相关)
    # 这里简化处理，将深度直接映射到相位
    phase_scale = 20 * np.pi
    unwrapped_phase_x = phase_scale * (xx + zz)
    unwrapped_phase_y = phase_scale * (yy + zz)
    
    return unwrapped_phase_x, unwrapped_phase_y


if __name__ == "__main__":
    # 检查Open3D是否可用
    try:
        import open3d as o3d
    except ImportError:
        print("警告: Open3D库未安装，将无法进行3D可视化和网格重建")
        print("可以使用 'pip install open3d' 安装此库")
    
    # 示例1: 使用模拟数据测试3D重建
    print("生成模拟3D对象数据...")
    unwrapped_phase_x, unwrapped_phase_y = simulate_3d_object()
    
    # 可视化解包裹相位
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(unwrapped_phase_x, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("X方向解包裹相位 (模拟)")
    
    plt.subplot(122)
    plt.imshow(unwrapped_phase_y, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("Y方向解包裹相位 (模拟)")
    
    plt.tight_layout()
    plt.savefig('simulated_unwrapped_phases.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 重建3D点云
    print("\n从模拟相位重建3D点云...")
    points, colors = phase_to_pointcloud(unwrapped_phase_x, unwrapped_phase_y)
    
    if len(points) > 0:
        # 创建Open3D点云
        pcd = create_open3d_pointcloud(points, colors)
        
        # 保存点云
        o3d.io.write_point_cloud("simulated_pointcloud.ply", pcd)
        print("模拟点云已保存至 simulated_pointcloud.ply")
        
        # 可视化点云
        visualize_pointcloud(pcd, "模拟3D对象点云")
        
        # 尝试创建网格
        try:
            mesh = create_mesh_from_pointcloud(pcd, voxel_size=0.05, depth=8)
            
            # 保存网格
            o3d.io.write_triangle_mesh("simulated_mesh.ply", mesh)
            print("模拟网格已保存至 simulated_mesh.ply")
            
            # 可视化网格
            o3d.visualization.draw_geometries([mesh], window_name="模拟3D对象网格")
        except Exception as e:
            print(f"网格创建失败: {e}")
    
    # 示例2: 处理实际测量数据 (需要提供数据文件夹)
    # 如果有实际数据，取消下面的注释并修改路径
    '''
    data_folder = "../measurement_data/camera1"
    camera_params_file = "camera_calibration.npy"
    projector_params_file = "projector_calibration.json"
    
    if os.path.exists(data_folder):
        print(f"\n处理实际测量数据文件夹: {data_folder}")
        process_measurement_data(
            data_folder,
            frequencies=[1, 4, 12, 48],
            algorithm=PhaseShiftingAlgorithm.four_step,
            camera_params_file=camera_params_file if os.path.exists(camera_params_file) else None,
            projector_params_file=projector_params_file if os.path.exists(projector_params_file) else None,
            output_folder="reconstruction_results"
        )
    ''' 