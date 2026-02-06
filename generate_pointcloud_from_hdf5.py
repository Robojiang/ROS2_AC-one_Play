#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
从 HDF5 数据集文件中读取 RGB、深度和机器人位姿，生成拼接点云
"""

import os
import sys
import cv2
import h5py
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
HDF5_FILE = "act/datasets/episode_0.hdf5"
INTRINSICS_FILE = "calibration_data_ark/intrinsics.json"
CALIBRATION_DIR = "calibration_results"
MAX_DEPTH = 1  # 最大深度 (米)
TARGET_POINTS = 4096  # 目标点云采样数
FRAME_IDX = 0  # 处理第一帧

# ================= 辅助函数 =================

def load_intrinsics(camera_name):
    """加载相机内参"""
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    fx, fy = d['fx'], d['fy']
    cx, cy = d['cx'], d['cy']
    return fx, fy, cx, cy


def load_calibration_matrix(filename):
    """加载标定矩阵"""
    path = os.path.join(CALIBRATION_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)


def decompress_image(compressed_data, camera_name):
    """解压缩 JPEG 图像"""
    img = cv2.imdecode(np.frombuffer(compressed_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 解压缩失败: {camera_name}")
        return None
    return img


def qpos_to_transform_matrix(qpos_7):
    """
    将 7D 关节位置转换为 4x4 变换矩阵 (用于末端执行器位姿)
    假设 qpos_7 = [j0, j1, j2, j3, j4, j5, gripper]
    
    注意: 这里需要正运动学计算,暂时使用占位符
    实际应该调用机器人的正运动学函数
    """
    # 这是简化版本,实际需要根据你的机器人模型进行正运动学计算
    # 这里我们假设 eef 数据中包含了末端位姿
    return np.eye(4)


def eef_to_transform_matrix(eef_data):
    """
    将末端执行器数据转换为 4x4 变换矩阵
    eef_data: [x, y, z, rx, ry, rz, gripper] 或者 14D (双臂)
    """
    if len(eef_data) >= 7:
        x, y, z = eef_data[0], eef_data[1], eef_data[2]
        rx, ry, rz = eef_data[3], eef_data[4], eef_data[5]
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        T[:3, :3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        return T
    return np.eye(4)


def depth_to_point_cloud(depth_img, color_img, fx, fy, cx, cy, max_depth=1.5):
    """
    深度图转点云
    
    参数:
      depth_img: (H, W) uint16, 单位 mm
      color_img: (H, W, 3) BGR uint8
      max_depth: 最大深度 (米)
    
    返回:
      points: (N, 6) [x, y, z, r, g, b] in meters
    """
    h, w = depth_img.shape
    
    # 创建网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤无效深度
    max_depth_mm = max_depth * 1000
    valid = (depth_img > 0) & (depth_img < max_depth_mm)
    
    z = depth_img[valid].astype(np.float32) / 1000.0  # mm -> m
    u = u[valid]
    v = v[valid]
    
    # 反投影到3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 颜色 (BGR -> RGB, 0-1)
    b = color_img[valid, 0].astype(np.float32) / 255.0
    g = color_img[valid, 1].astype(np.float32) / 255.0
    r = color_img[valid, 2].astype(np.float32) / 255.0
    
    # 拼接 (N, 6)
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    
    return np.hstack((xyz, rgb))


def transform_point_cloud(cloud, T):
    """点云变换"""
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:]
    
    # 齐次变换
    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.hstack((xyz, ones))
    
    xyz_trans = (T @ xyz_homo.T).T
    xyz_new = xyz_trans[:, :3]
    
    return np.hstack((xyz_new, rgb))


def numpy_to_o3d(cloud_np):
    """转换 numpy 数组到 open3d 点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
    return pcd


def visualize_point_cloud(cloud_np, title="Point Cloud", camera_frames=None):
    """可视化点云，可选显示相机位置
    
    参数:
        cloud_np: 点云数组 (N, 6)
        title: 窗口标题
        camera_frames: [(T_matrix, name, size), ...] 相机变换矩阵列表
    """
    print(f"👀 显示: {title} (点数: {len(cloud_np)})")
    
    geometries = []
    
    # 点云
    pcd = numpy_to_o3d(cloud_np)
    geometries.append(pcd)
    
    # 世界坐标系（原点）
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    geometries.append(world_axes)
    
    # 相机坐标系
    if camera_frames is not None:
        for T, name, size in camera_frames:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
            cam_frame.transform(T)
            geometries.append(cam_frame)
            print(f"  📷 {name} 相机位置: [{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]")
    
    o3d.visualization.draw_geometries(
        geometries, 
        window_name=title, 
        width=1280, 
        height=720
    )


def downsample_point_cloud(cloud_np, target_points=4096):
    """点云降采样"""
    print(f"\n📊 原始点数: {len(cloud_np)}")
    
    pcd = numpy_to_o3d(cloud_np)
    
    # 1. 统计滤波去噪
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"🧹 去噪后: {len(pcd.points)} 点")
    
    # 2. 体素下采样
    voxel_size = 0.01  # 1cm
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"📦 体素采样: {len(pcd_down.points)} 点")
    
    # 3. FPS 最远点采样
    if len(pcd_down.points) > target_points:
        pcd_fps = pcd_down.farthest_point_down_sample(target_points)
        print(f"🎯 FPS采样: {len(pcd_fps.points)} 点")
        pcd_final = pcd_fps
    else:
        pcd_final = pcd_down
    
    # 转回 numpy
    pts = np.asarray(pcd_final.points)
    clrs = np.asarray(pcd_final.colors)
    
    return np.hstack((pts, clrs))


# ================= 主程序 =================

def main():
    print("="*60)
    print("🚀 从 HDF5 数据集生成拼接点云")
    print("="*60)
    
    # 1. 加载标定结果
    print("\n📂 加载标定矩阵...")
    
    # Eye-in-Hand: T_End_Cam (相机在末端执行器坐标系)
    T_LE_LC = load_calibration_matrix("left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix("right_eye_in_hand.npy")
    print(f"  Left Eye-in-Hand:\n{T_LE_LC}")
    print(f"  Right Eye-in-Hand:\n{T_RE_RC}")
    
    # Eye-to-Base: T_Base_HeadCam (Head相机在机器人基座坐标系)
    # 优先使用 ICP 修正后的结果
    if os.path.exists(os.path.join(CALIBRATION_DIR, "head_base_to_left_refined_icp.npy")):
        T_LB_H = load_calibration_matrix("head_base_to_left_refined_icp.npy")
        print("  ✅ 使用 ICP 修正后的 head_base_to_left")
    else:
        T_LB_H = load_calibration_matrix("head_base_to_left.npy")
        print("  ⚠️  使用原始 head_base_to_left")
    print(f"  T_LB_H (Base->Head):\n{T_LB_H}")
    
    if os.path.exists(os.path.join(CALIBRATION_DIR, "head_base_to_right_refined_icp.npy")):
        T_RB_H = load_calibration_matrix("head_base_to_right_refined_icp.npy")
        print("  ✅ 使用 ICP 修正后的 head_base_to_right")
    else:
        T_RB_H = load_calibration_matrix("head_base_to_right.npy")
        print("  ⚠️  使用原始 head_base_to_right")
    print(f"  T_RB_H (Base->Head):\n{T_RB_H}")
    
    # 2. 加载相机内参
    print("\n📷 加载相机内参...")
    intrinsics = {}
    for name in ['head', 'left', 'right']:
        intrinsics[name] = load_intrinsics(name)
        fx, fy, cx, cy = intrinsics[name]
        print(f"  {name}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 3. 打开 HDF5 文件
    print(f"\n📖 读取 HDF5 文件: {HDF5_FILE}")
    with h5py.File(HDF5_FILE, 'r') as f:
        total_frames = len(f['action'])
        print(f"  总帧数: {total_frames}")
        print(f"  处理帧: {FRAME_IDX}")
        
        if FRAME_IDX >= total_frames:
            print(f"❌ 帧索引超出范围 (最大: {total_frames-1})")
            return
        
        # 读取第 FRAME_IDX 帧数据
        print("\n📥 读取数据...")
        
        # RGB 图像 (压缩格式)
        img_head_compressed = f['observations/images/head'][FRAME_IDX]
        img_left_compressed = f['observations/images/left_wrist'][FRAME_IDX]
        img_right_compressed = f['observations/images/right_wrist'][FRAME_IDX]
        
        # 深度图像
        depth_head = f['observations/images_depth/head'][FRAME_IDX]
        depth_left = f['observations/images_depth/left_wrist'][FRAME_IDX]
        depth_right = f['observations/images_depth/right_wrist'][FRAME_IDX]
        
        # 机器人状态
        eef = f['observations/eef'][FRAME_IDX]  # [14] 双臂末端位姿
        qpos = f['observations/qpos'][FRAME_IDX]  # [14] 关节位置
        
        print(f"  ✅ RGB 数据大小: Head={len(img_head_compressed)}, Left={len(img_left_compressed)}, Right={len(img_right_compressed)} bytes")
        print(f"  ✅ 深度形状: {depth_head.shape}")
        print(f"  ✅ EEF: {eef.shape}, QPose: {qpos.shape}")
    
    # 4. 解压缩图像
    print("\n🖼️  解压缩图像...")
    img_head = decompress_image(img_head_compressed, "head")
    img_left = decompress_image(img_left_compressed, "left")
    img_right = decompress_image(img_right_compressed, "right")
    
    if img_head is None or img_left is None or img_right is None:
        print("❌ 图像解压缩失败")
        return
    
    print(f"  ✅ Head: {img_head.shape}, Left: {img_left.shape}, Right: {img_right.shape}")
    
    # 5. 生成局部点云
    print("\n☁️  生成局部点云...")
    
    fx, fy, cx, cy = intrinsics['head']
    cloud_head_local = depth_to_point_cloud(depth_head, img_head, fx, fy, cx, cy, MAX_DEPTH)
    print(f"  Head: {len(cloud_head_local)} 点")
    if len(cloud_head_local) > 0:
        print(f"    范围: X[{cloud_head_local[:, 0].min():.3f}, {cloud_head_local[:, 0].max():.3f}] "
              f"Y[{cloud_head_local[:, 1].min():.3f}, {cloud_head_local[:, 1].max():.3f}] "
              f"Z[{cloud_head_local[:, 2].min():.3f}, {cloud_head_local[:, 2].max():.3f}]")
    
    fx, fy, cx, cy = intrinsics['left']
    cloud_left_local = depth_to_point_cloud(depth_left, img_left, fx, fy, cx, cy, MAX_DEPTH)
    print(f"  Left: {len(cloud_left_local)} 点")
    if len(cloud_left_local) > 0:
        print(f"    范围: X[{cloud_left_local[:, 0].min():.3f}, {cloud_left_local[:, 0].max():.3f}] "
              f"Y[{cloud_left_local[:, 1].min():.3f}, {cloud_left_local[:, 1].max():.3f}] "
              f"Z[{cloud_left_local[:, 2].min():.3f}, {cloud_left_local[:, 2].max():.3f}]")
    
    fx, fy, cx, cy = intrinsics['right']
    cloud_right_local = depth_to_point_cloud(depth_right, img_right, fx, fy, cx, cy, MAX_DEPTH)
    print(f"  Right: {len(cloud_right_local)} 点")
    if len(cloud_right_local) > 0:
        print(f"    范围: X[{cloud_right_local[:, 0].min():.3f}, {cloud_right_local[:, 0].max():.3f}] "
              f"Y[{cloud_right_local[:, 1].min():.3f}, {cloud_right_local[:, 1].max():.3f}] "
              f"Z[{cloud_right_local[:, 2].min():.3f}, {cloud_right_local[:, 2].max():.3f}]")
    
    # 6. 构建变换矩阵并转换到全局坐标系
    print("\n🌍 转换到全局坐标系 (Head Camera Frame)...")
    
    clouds_global = []
    camera_frames = []  # 存储相机位置用于可视化
    
    # Head 相机 (作为全局坐标系原点)
    clouds_global.append(cloud_head_local)
    camera_frames.append((np.eye(4), "Head", 0.2))  # 位置、名称、坐标轴大小
    print(f"  ✅ Head: 作为世界原点 ({len(cloud_head_local)} 点)")
    
    # Left 手腕相机
    # 路径: Cam -> End (Eye-in-Hand) -> Base (Forward Kinematics) -> Head (Eye-to-Base inverse)
    # P_Head = inv(T_Base_Head) @ T_Base_LeftEnd @ T_LeftEnd_LeftCam @ P_LeftCam
    
    left_eef = eef[:7]  # 前7个是左臂 [x, y, z, rx, ry, rz, gripper]
    print(f"\n  📍 Left EEF: pos=[{left_eef[0]:.3f}, {left_eef[1]:.3f}, {left_eef[2]:.3f}] "
          f"rot=[{left_eef[3]:.3f}, {left_eef[4]:.3f}, {left_eef[5]:.3f}] gripper={left_eef[6]:.3f}")
    
    T_Base_LeftEnd = eef_to_transform_matrix(left_eef)
    print(f"  T_Base_LeftEnd:\n{T_Base_LeftEnd}")
    
    T_HeadCam_Base_L = np.linalg.inv(T_LB_H)
    T_LeftEnd_LeftCam = T_LE_LC
    
    T_total_left = T_HeadCam_Base_L @ T_Base_LeftEnd @ T_LeftEnd_LeftCam
    print(f"  T_total_left (最终变换):\n{T_total_left}")
    
    cloud_left_global = transform_point_cloud(cloud_left_local, T_total_left)
    clouds_global.append(cloud_left_global)
    camera_frames.append((T_total_left, "Left", 0.15))  # 记录Left相机位置
    print(f"  ✅ Left: {len(cloud_left_global)} 点")
    if len(cloud_left_global) > 0:
        print(f"    全局范围: X[{cloud_left_global[:, 0].min():.3f}, {cloud_left_global[:, 0].max():.3f}] "
              f"Y[{cloud_left_global[:, 1].min():.3f}, {cloud_left_global[:, 1].max():.3f}] "
              f"Z[{cloud_left_global[:, 2].min():.3f}, {cloud_left_global[:, 2].max():.3f}]")
    
    # Right 手腕相机
    right_eef = eef[7:]  # 后7个是右臂
    print(f"\n  📍 Right EEF: pos=[{right_eef[0]:.3f}, {right_eef[1]:.3f}, {right_eef[2]:.3f}] "
          f"rot=[{right_eef[3]:.3f}, {right_eef[4]:.3f}, {right_eef[5]:.3f}] gripper={right_eef[6]:.3f}")
    
    T_Base_RightEnd = eef_to_transform_matrix(right_eef)
    print(f"  T_Base_RightEnd:\n{T_Base_RightEnd}")
    
    T_HeadCam_Base_R = np.linalg.inv(T_RB_H)
    T_RightEnd_RightCam = T_RE_RC
    
    T_total_right = T_HeadCam_Base_R @ T_Base_RightEnd @ T_RightEnd_RightCam
    print(f"  T_total_right (最终变换):\n{T_total_right}")
    
    cloud_right_global = transform_point_cloud(cloud_right_local, T_total_right)
    clouds_global.append(cloud_right_global)
    camera_frames.append((T_total_right, "Right", 0.15))  # 记录Right相机位置
    print(f"  ✅ Right: {len(cloud_right_global)} 点")
    if len(cloud_right_global) > 0:
        print(f"    全局范围: X[{cloud_right_global[:, 0].min():.3f}, {cloud_right_global[:, 0].max():.3f}] "
              f"Y[{cloud_right_global[:, 1].min():.3f}, {cloud_right_global[:, 1].max():.3f}] "
              f"Z[{cloud_right_global[:, 2].min():.3f}, {cloud_right_global[:, 2].max():.3f}]")
    
    # 7. 合并点云
    print("\n🔗 合并点云...")
    merged_cloud = np.vstack(clouds_global)
    print(f"  合并后总点数: {len(merged_cloud)}")
    
    # 8. 降采样
    print("\n⬇️  降采样处理...")
    final_cloud = downsample_point_cloud(merged_cloud, TARGET_POINTS)
    
    # 9. 保存点云
    output_path = "point_cloud_merged_frame0.ply"
    pcd_save = numpy_to_o3d(final_cloud)
    o3d.io.write_point_cloud(output_path, pcd_save)
    print(f"\n💾 点云已保存: {output_path}")
    
    # 10. 可视化
    print("\n" + "="*60)
    print("🎨 可视化步骤:")
    print("  1. 显示原始合并点云 + 三个相机位置")
    print("  2. 显示降采样后点云 + 三个相机位置")
    print("="*60)
    
    # 第一次：显示原始合并点云 + 相机位置
    visualize_point_cloud(
        merged_cloud, 
        f"原始合并点云 (Frame {FRAME_IDX}) - {len(merged_cloud)} 点",
        camera_frames=camera_frames
    )
    
    # 第二次：显示降采样后点云 + 相机位置
    visualize_point_cloud(
        final_cloud, 
        f"降采样点云 (Frame {FRAME_IDX}) - {len(final_cloud)} 点",
        camera_frames=camera_frames
    )
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
