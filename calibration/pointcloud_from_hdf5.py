#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import h5py
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# ================= 配置 =================
HDF5_PATH = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/datasets/episode_4.hdf5"
CALIBRATION_DIR = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/calibration_results"
INTRINSICS_FILE = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/calibration_results/intrinsics.json"

FRAME_INDEX = 0  # 读取第一帧
MAX_DEPTH_Head = 1 # 最大深度限制 (米)
MAX_DEPTH_Hand = 0.6  # 最大深度限制 (米)
FPS_SAMPLE_POINTS = 2048  # FPS采样点数

# 工作空间裁剪 (相对于左臂基座坐标系)
USE_WORKSPACE_CROP = True  # 是否启用工作空间裁剪
WORKSPACE_X_RANGE = [-0.4, 0.5]  # x轴范围 (米)
WORKSPACE_Y_RANGE = [-0.5, 3]  # y轴范围 (米)
WORKSPACE_Z_RANGE = [-0.2, 1]  # z轴范围 (米)

# 输出坐标系选择
OUTPUT_FRAME = 'left_base'  # 'head' 或 'left_base' - 最终点云相对的坐标系

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
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.txt'):
            return np.loadtxt(path)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)

def eef_to_matrix(eef_pose):
    """
    将 end-effector pose 转换为 4x4 变换矩阵
    eef_pose: [x, y, z, rx, ry, rz, gripper] 长度为7
    """
    if eef_pose is None or len(eef_pose) < 6:
        return np.eye(4)
    
    t = np.array(eef_pose[:3])
    r = R.from_euler('xyz', eef_pose[3:6]).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def depth_to_point_cloud(depth_img, color_img, fx, fy, cx, cy, max_depth=None):
    """
    将深度图和彩色图转换为点云
    
    输入:
      depth_img: (H, W) uint16, 单位 mm
      color_img: (H, W, 3) BGR uint8
      max_depth: 最大深度限制 (米), 超过此深度的点将被过滤
    输出:
      points: (N, 6) [x, y, z, r, g, b] in meters
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤无效深度
    valid = depth_img > 0
    
    # 过滤距离过远的点
    if max_depth is not None:
        valid = valid & (depth_img < max_depth * 1000)  # 转换为mm
    
    z = depth_img[valid].astype(np.float32) / 1000.0  # mm -> m
    u = u[valid]
    v = v[valid]
    
    # 反投影
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 颜色 (BGR -> RGB, 0-1)
    b = color_img[valid, 0].astype(np.float32) / 255.0
    g = color_img[valid, 1].astype(np.float32) / 255.0
    r = color_img[valid, 2].astype(np.float32) / 255.0
    
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    
    return np.hstack((xyz, rgb))

def transform_point_cloud(cloud, T):
    """变换点云, cloud: (N, 6), T: (4, 4)"""
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:]
    
    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.hstack((xyz, ones))
    
    xyz_trans = (T @ xyz_homo.T).T
    xyz_new = xyz_trans[:, :3]
    
    return np.hstack((xyz_new, rgb))

def numpy_to_o3d(cloud_np):
    """转换 (N, 6) numpy 数组到 open3d.geometry.PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
    return pcd

def crop_point_cloud(cloud_np, x_range, y_range, z_range):
    """
    裁剪点云到指定的xyz范围
    cloud_np: (N, 6) [x, y, z, r, g, b]
    x_range, y_range, z_range: [min, max]
    """
    xyz = cloud_np[:, :3]
    
    mask = (
        (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
        (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1]) &
        (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    )
    
    return cloud_np[mask]

def visualize_merged(clouds_list, title="Point Cloud", coordinate_frames=[]):
    """
    可视化合并的点云
    clouds_list: list of numpy arrays (N, 6) or o3d.PointCloud
    coordinate_frames: list of (T_matrix, label, size)
    """
    print(f"👀 显示: {title}")
    geometries = []
    
    # World Origin
    axes_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(axes_world)
    
    # Point Clouds
    for cloud_obj in clouds_list:
        if isinstance(cloud_obj, np.ndarray):
            if len(cloud_obj) > 0:
                pcd = numpy_to_o3d(cloud_obj)
                geometries.append(pcd)
        else:
            geometries.append(cloud_obj)
    
    # Coordinate Frames
    for item in coordinate_frames:
        if len(item) == 2:
            T, label = item
            size = 0.15
        else:
            T, label, size = item
            
        if T is not None:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            frame.transform(T)
            geometries.append(frame)
            print(f"  -> Frame: {label} at {T[:3, 3]}")
    
    o3d.visualization.draw_geometries(geometries, window_name=title, width=1280, height=720)

def decode_jpeg(data):
    """解码JPEG数据"""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

# ================= 主程序 =================

def main():
    print("="*80)
    print("从 HDF5 文件生成拼接点云")
    print("="*80)
    
    # 1. 加载标定结果 (使用修正后的 ICP 结果)
    print("\n📁 加载标定文件...")
    
    # Eye-in-Hand: T_End_Cam
    T_LE_LC = load_calibration_matrix("left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix("right_eye_in_hand.npy")
    
    # Eye-to-Hand: T_Base_HeadCam (使用 ICP 修正版本)
    T_LB_H = load_calibration_matrix("head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix("head_base_to_right_refined_icp.txt")
    
    # 如果没有修正版本,尝试加载原始版本
    if np.array_equal(T_LB_H, np.eye(4)):
        print("  ⚠️ 未找到 left ICP 修正版本,使用原始标定")
        T_LB_H = load_calibration_matrix("head_base_to_left.npy")
    
    if np.array_equal(T_RB_H, np.eye(4)):
        print("  ⚠️ 未找到 right ICP 修正版本,使用原始标定")
        T_RB_H = load_calibration_matrix("head_base_to_right.npy")
    
    # 转换为 T_HeadCam_Base
    T_H_LB = np.linalg.inv(T_LB_H)
    T_H_RB = np.linalg.inv(T_RB_H)
    
    print("✅ 标定文件加载完成")
    
    # 2. 加载内参
    print("\n📁 加载相机内参...")
    intrinsics = {
        'head': load_intrinsics('head'),
        'left': load_intrinsics('left'),
        'right': load_intrinsics('right')
    }
    print("✅ 内参加载完成")
    
    # 3. 读取 HDF5 数据
    print(f"\n📁 读取 HDF5 文件: {os.path.basename(HDF5_PATH)}")
    print(f"   帧索引: {FRAME_INDEX}")
    
    with h5py.File(HDF5_PATH, 'r') as f:
        # 读取机器人状态
        eef_data = f['observations/eef'][FRAME_INDEX]  # shape: (14,) = left(7) + right(7)
        
        # 分离左右臂
        left_eef = eef_data[:7]  # [x, y, z, rx, ry, rz, gripper]
        right_eef = eef_data[7:14]
        
        print(f"\n🤖 机器人状态:")
        print(f"   Left EEF:  {left_eef[:3]} (位置)")
        print(f"   Right EEF: {right_eef[:3]} (位置)")
        
        # 转换为变换矩阵
        T_LB_LE = eef_to_matrix(left_eef)  # Left Base -> Left End
        T_RB_RE = eef_to_matrix(right_eef)  # Right Base -> Right End
        
        # 读取图像数据
        head_color_data = f['observations/images/head'][FRAME_INDEX]
        left_color_data = f['observations/images/left_wrist'][FRAME_INDEX]
        right_color_data = f['observations/images/right_wrist'][FRAME_INDEX]
        
        head_depth = f['observations/images_depth/head'][FRAME_INDEX]
        left_depth = f['observations/images_depth/left_wrist'][FRAME_INDEX]
        right_depth = f['observations/images_depth/right_wrist'][FRAME_INDEX]
        
        # 解码 JPEG 图像
        head_color = decode_jpeg(head_color_data)
        left_color = decode_jpeg(left_color_data)
        right_color = decode_jpeg(right_color_data)
    
    print(f"✅ 数据读取完成")
    print(f"   图像尺寸: {head_color.shape}")
    print(f"   深度尺寸: {head_depth.shape}")
    
    # 4. 生成点云
    print(f"\n🌐 生成点云 (最大深度: Head={MAX_DEPTH_Head}m, Hand={MAX_DEPTH_Hand}m)...")
    
    clouds_local = {}
    clouds_global = []
    coordinate_frames = []
    
    # --- Head Camera (作为世界坐标系原点) ---
    print("  处理 Head Camera...")
    fx, fy, cx, cy = intrinsics['head']
    pc_head = depth_to_point_cloud(head_depth, head_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Head)
    print(f"    点数: {len(pc_head)}")
    
    clouds_local['head'] = pc_head
    clouds_global.append(pc_head)
    coordinate_frames.append((np.eye(4), "Head_Camera", 0.2))
    
    # --- Left Wrist Camera ---
    print("  处理 Left Wrist Camera...")
    fx, fy, cx, cy = intrinsics['left']
    pc_left = depth_to_point_cloud(left_depth, left_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    print(f"    点数: {len(pc_left)}")
    
    if len(pc_left) > 0:
        # 变换路径: Cam -> End -> Base -> HeadCam
        # P_Head = T_H_LB @ T_LB_LE @ T_LE_LC @ P_Cam
        T_total_left = T_H_LB @ T_LB_LE @ T_LE_LC
        pc_left_global = transform_point_cloud(pc_left, T_total_left)
        
        clouds_local['left'] = pc_left
        clouds_global.append(pc_left_global)
        coordinate_frames.append((T_total_left, "Left_Camera", 0.15))
        
        print(f"    变换后位置: {T_total_left[:3, 3]}")
    
    # --- Right Wrist Camera ---
    print("  处理 Right Wrist Camera...")
    fx, fy, cx, cy = intrinsics['right']
    pc_right = depth_to_point_cloud(right_depth, right_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    print(f"    点数: {len(pc_right)}")
    
    if len(pc_right) > 0:
        # 变换路径: Cam -> End -> Base -> HeadCam
        # P_Head = T_H_RB @ T_RB_RE @ T_RE_RC @ P_Cam
        T_total_right = T_H_RB @ T_RB_RE @ T_RE_RC
        pc_right_global = transform_point_cloud(pc_right, T_total_right)
        
        clouds_local['right'] = pc_right
        clouds_global.append(pc_right_global)
        coordinate_frames.append((T_total_right, "Right_Camera", 0.15))
        
        print(f"    变换后位置: {T_total_right[:3, 3]}")
    
    # 5. 坐标系转换 (可选: 转换到左臂基座坐标系)
    print("\n" + "="*80)
    if OUTPUT_FRAME == 'left_base':
        print("🔄 转换坐标系: Head -> Left Base")
        # T_LB_H 是 Left Base -> Head 的变换
        # 我们需要 Head -> Left Base, 所以取逆
        T_LB_H_inv = T_LB_H  # 注意: T_LB_H 本身就是 Base->Head, 所以 inv 是 Head->Base
        
        clouds_in_left_base = []
        for cloud in clouds_global:
            if len(cloud) > 0:
                cloud_transformed = transform_point_cloud(cloud, T_LB_H_inv)
                clouds_in_left_base.append(cloud_transformed)
        
        clouds_global = clouds_in_left_base
        
        # 更新坐标系标记
        coordinate_frames_left_base = []
        for T, label, size in coordinate_frames:
            T_new = T_LB_H_inv @ T
            coordinate_frames_left_base.append((T_new, label, size))
        coordinate_frames = coordinate_frames_left_base
        
        print(f"   ✅ 已转换到左臂基座坐标系")
    
    # 6. 工作空间裁剪
    if USE_WORKSPACE_CROP:
        print("\n✂️  工作空间裁剪:")
        print(f"   X: [{WORKSPACE_X_RANGE[0]}, {WORKSPACE_X_RANGE[1]}] m")
        print(f"   Y: [{WORKSPACE_Y_RANGE[0]}, {WORKSPACE_Y_RANGE[1]}] m")
        print(f"   Z: [{WORKSPACE_Z_RANGE[0]}, {WORKSPACE_Z_RANGE[1]}] m")
        
        clouds_cropped = []
        for cloud in clouds_global:
            if len(cloud) > 0:
                cloud_cropped = crop_point_cloud(cloud, 
                                                 WORKSPACE_X_RANGE, 
                                                 WORKSPACE_Y_RANGE, 
                                                 WORKSPACE_Z_RANGE)
                if len(cloud_cropped) > 0:
                    clouds_cropped.append(cloud_cropped)
                    print(f"   保留点数: {len(cloud)} -> {len(cloud_cropped)}")
        
        clouds_global = clouds_cropped
        print(f"   ✅ 裁剪完成")
    
    # 7. 显示原始拼接结果
    print("\n" + "="*80)
    print("📊 拼接统计:")
    total_points = sum(len(c) for c in clouds_global)
    print(f"   总点数: {total_points}")
    print(f"   相机数: {len(clouds_global)}")
    print(f"   坐标系: {OUTPUT_FRAME}")
    
    if len(clouds_global) > 0:
        print("\n👀 显示原始拼接结果...")
        frame_name = "Left_Base" if OUTPUT_FRAME == 'left_base' else "Head"
        visualize_merged(clouds_global, 
                        title=f"Original Merged (Frame: {frame_name})", 
                        coordinate_frames=coordinate_frames)
    
    # 8. FPS 下采样
    print("\n" + "="*80)
    print(f"🎯 FPS 下采样到 {FPS_SAMPLE_POINTS} 点...")
    
    if len(clouds_global) > 0:
        # 合并所有点云
        merged_cloud = np.vstack(clouds_global)
        print(f"   合并前总点数: {len(merged_cloud)}")
        
        # 转换为 Open3D
        pcd_merged = numpy_to_o3d(merged_cloud)
        
        # 去除离群点
        pcd_clean, ind = pcd_merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"   去噪后点数: {len(pcd_clean.points)}")
        
        # 体素下采样 (预处理)
        voxel_size = 0.005  # 5mm
        pcd_voxel = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
        print(f"   体素下采样({voxel_size}m)后: {len(pcd_voxel.points)}")
        
        # FPS 下采样
        if len(pcd_voxel.points) > FPS_SAMPLE_POINTS:
            pcd_fps = pcd_voxel.farthest_point_down_sample(FPS_SAMPLE_POINTS)
            print(f"   FPS 采样后: {len(pcd_fps.points)}")
        else:
            pcd_fps = pcd_voxel
            print(f"   ⚠️ 点数不足 {FPS_SAMPLE_POINTS}, 保留所有点: {len(pcd_fps.points)}")
        
        # 转换回 numpy
        fps_points = np.asarray(pcd_fps.points)
        fps_colors = np.asarray(pcd_fps.colors)
        fps_cloud = np.hstack((fps_points, fps_colors))
        
        # 显示 FPS 结果
        print(f"\n👀 显示 FPS 下采样结果 ({len(fps_cloud)} 点)...")
        frame_name = "Left_Base" if OUTPUT_FRAME == 'left_base' else "Head"
        visualize_merged([fps_cloud], 
                        title=f"FPS Sampled ({FPS_SAMPLE_POINTS} points, Frame: {frame_name})", 
                        coordinate_frames=coordinate_frames)
        
        # 保存结果
        frame_suffix = "leftbase" if OUTPUT_FRAME == 'left_base' else "head"
        output_path = f"pointcloud_frame{FRAME_INDEX}_fps{FPS_SAMPLE_POINTS}_{frame_suffix}.npy"
        np.save(output_path, fps_cloud)
        print(f"\n💾 已保存点云到: {output_path}")
        print(f"   形状: {fps_cloud.shape}")
        print(f"   坐标系: {OUTPUT_FRAME}")
    
    print("\n" + "="*80)
    print("✅ 完成!")

if __name__ == "__main__":
    main()
