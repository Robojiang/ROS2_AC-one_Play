#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
可视化某个 HDF5 文件中指定帧生成的合成点云。
"""

import os
import sys
import h5py
import numpy as np
import cv2
import json
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # 回到上级目录，即项目根目录
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_utils.pointcloud_generator import PointCloudGenerator

CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_results")
INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, "D435_intrinsics.json")

# ================= 辅助函数 =================
def load_intrinsics(camera_name):
    """加载相机内参"""
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    return d['fx'], d['fy'], d['cx'], d['cy']

def load_calibration_matrix(filename):
    """加载外参矩阵"""
    path = os.path.join(CALIBRATION_DIR, filename)
    if os.path.exists(path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.txt'):
            return np.loadtxt(path)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)

def decode_jpeg(data):
    """解码JPEG数据"""
    img_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # RGB 格式用于点云色彩和保存

def load_hdf5_frame(hdf5_path, frame_idx):
    """只读取并返回指定帧的数据"""
    with h5py.File(hdf5_path, 'r') as f:
        T = f['observations/eef'].shape[0]
        if frame_idx >= T or frame_idx < -T:
            raise ValueError(f"帧索引 {frame_idx} 超出范围，总帧数为 {T}")
            
        eef = f['observations/eef'][frame_idx]
        
        # JPEG -> RGB
        head_img = decode_jpeg(f['observations/images/head'][frame_idx])
        left_img = decode_jpeg(f['observations/images/left_wrist'][frame_idx])
        right_img = decode_jpeg(f['observations/images/right_wrist'][frame_idx])
        
        # Depths
        head_depth = f['observations/images_depth/head'][frame_idx]
        left_depth = f['observations/images_depth/left_wrist'][frame_idx]
        right_depth = f['observations/images_depth/right_wrist'][frame_idx]
        
        return {
            'eef': eef,
            'head_img': head_img,
            'left_img': left_img,
            'right_img': right_img,
            'head_depth': head_depth,
            'left_depth': left_depth,
            'right_depth': right_depth,
            'total_frames': T
        }

def eef_to_matrix(eef_pose):
    if eef_pose is None or len(eef_pose) < 6:
        return np.eye(4)
    t = np.array(eef_pose[:3])
    r = R.from_euler('xyz', eef_pose[3:6]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

# ================= 主程序 =================
def main(args):
    print(f"📂 读取文件: {args.hdf5_path}")
    
    # 1. 读取指定帧
    data = load_hdf5_frame(args.hdf5_path, args.frame)
    print(f"✅ 成功读取第 {args.frame} 帧 (总帧数: {data['total_frames']})")
    
    left_eef = data['eef'][:7]
    right_eef = data['eef'][7:14]

    # 2. 加载标定参数
    print("📁 加载外参和内参文件...")
    T_LE_LC = load_calibration_matrix("left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix("right_eye_in_hand.npy")
    T_LB_H = load_calibration_matrix("head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix("head_base_to_right_refined_icp.txt")
    
    if np.array_equal(T_LB_H, np.eye(4)):
        T_LB_H = load_calibration_matrix("head_base_to_left.npy")
    if np.array_equal(T_RB_H, np.eye(4)):
        T_RB_H = load_calibration_matrix("head_base_to_right.npy")

    T_H_LB = T_LB_H
    T_H_RB = np.linalg.inv(T_RB_H)

    intrinsics = {
        'head': load_intrinsics('head'),
        'left': load_intrinsics('left'),
        'right': load_intrinsics('right')
    }
    print("✅ 参数加载完毕")

    # 3. 初始化点云生成器
    # 注意：此处使用的参数需与 convert_hdf5_to_zarr 保持一致
    pc_generator = PointCloudGenerator(
        intrinsics=intrinsics,
        fps_sample_points=4000, # 为了清晰，可视化可以使用更多的点数 (你可以改回1024)
        use_workspace_crop=True,
        workspace_z_range=(-3,3),
        downsample_size=(160, 120),
          # 与生成器中保持一致
    )
    
    print("🌟 正在生成组合点云...")
    # 4. 生成点云
    merged_pc_np, eef_trans = pc_generator.generate(
        head_depth=data['head_depth'],
        head_color=data['head_img'],
        left_depth=data['left_depth'],
        left_color=data['left_img'],
        right_depth=data['right_depth'],
        right_color=data['right_img'],
        left_eef=left_eef,
        right_eef=right_eef,
        intrinsics=intrinsics,
        T_H_LB=T_H_LB,
        T_H_RB=T_H_RB,
        T_LE_LC=T_LE_LC,
        T_RE_RC=T_RE_RC,
        T_LB_H=T_LB_H
    )
    
    num_pts = merged_pc_np.shape[0]
    print(f"☁️ 点云生成成功，包含采样点数: {num_pts} 个")

    if num_pts == 0:
        print("❌ 生成的点云为空，可能是由于深度图无效或者被裁切掉（Workingspace bounds）。")
        return

    # 5. Open3D 显示
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pc_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(merged_pc_np[:, 3:]) # PointCloudGenerator 输出的颜色已经位于 [0, 1] 区间

    # 添加一个坐标系来表征世界原点 (Center Base)
    geometries = [pcd]
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(world_frame)

    # 绘制机械臂末端 (因为点云被平移了，所以这些末端坐标也需要应用平移)
    # 点云生成器内部把基座中点 (mid_pos) 平移到了 [0,0,0]
    T_LB_RB = T_LB_H @ T_H_RB
    mid_pos = (T_LB_RB[:3, 3] / 2.0).astype(np.float32)
    T_CB_LB = np.eye(4, dtype=np.float32)
    T_CB_LB[:3, 3] = -mid_pos

    # 左臂基座
    left_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    left_base_frame.transform(T_CB_LB)
    geometries.append(left_base_frame)

    # 左臂末端
    T_LB_LE = eef_to_matrix(left_eef)
    left_end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    left_end_frame.transform(T_CB_LB @ T_LB_LE)
    geometries.append(left_end_frame)

    # 左相机 (Eye-in-hand)
    left_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    left_cam_frame.transform(T_CB_LB @ T_LB_LE @ T_LE_LC)
    geometries.append(left_cam_frame)

    # 右臂基座
    right_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    right_base_frame.transform(T_CB_LB @ T_LB_RB)
    geometries.append(right_base_frame)

    # 右臂末端
    T_RB_RE = eef_to_matrix(right_eef)
    right_end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    right_end_frame.transform(T_CB_LB @ T_LB_RB @ T_RB_RE)
    geometries.append(right_end_frame)

    # 右相机 (Eye-in-hand)
    right_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    right_cam_frame.transform(T_CB_LB @ T_LB_RB @ T_RB_RE @ T_RE_RC)
    geometries.append(right_cam_frame)

    # 头顶相机 (Head Camera)
    T_CB_H = T_CB_LB @ T_LB_H
    head_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    head_cam_frame.transform(T_CB_H)
    geometries.append(head_cam_frame)

    print("🖼️ 正在弹出 Open3D 可视化窗口...")
    o3d.visualization.draw_geometries(geometries, window_name=f"Frame {args.frame} Point Cloud")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单帧 HDF5 点云可视化工具")
    parser.add_argument("--hdf5_path", type=str, default="datasets/pick_place_d435/episode_2.hdf5", help="要读取的 hdf5 文件绝对或相对路径")
    parser.add_argument("--frame", type=int, default=1, help="要可视化的帧索引 (默认为第 0 帧)")
    args = parser.parse_args()
    main(args)
