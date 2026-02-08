#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
将HDF5数据集转换为Zarr格式,包含点云生成
用法: python convert_hdf5_to_zarr.py --num_episodes 100 --max_episodes 5 (debug模式)
"""

import os
import h5py
import zarr
import numpy as np
import cv2
import json
import argparse
import shutil
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # 上一级目录
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
DATASETS_ZARR_DIR = os.path.join(BASE_DIR, "datasets_zarr")
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_results")
INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, "D405_intrinsics.json")

# 点云配置
MAX_DEPTH_Head = 1.0  # 米
MAX_DEPTH_Hand = 0.6  # 米
FPS_SAMPLE_POINTS = 1024  # 点云采样点数

# 工作空间裁剪 (相对于左臂基座坐标系)
USE_WORKSPACE_CROP = True
WORKSPACE_X_RANGE = [-0.4, 0.5]
WORKSPACE_Y_RANGE = [-0.5, 3.0]
WORKSPACE_Z_RANGE = [-0.2, 1.0]

# 关键帧检测
GRIPPER_DELTA = 0.05  # 夹爪变化阈值
MIN_INTERVAL = 20  # 最小关键帧间隔

# ================= 标定加载函数 =================

def load_intrinsics(camera_name):
    """加载相机内参"""
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    return d['fx'], d['fy'], d['cx'], d['cy']

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

# ================= 点云生成函数 =================

def eef_to_matrix(eef_pose):
    """将end-effector pose转换为4x4变换矩阵"""
    if eef_pose is None or len(eef_pose) < 6:
        return np.eye(4)
    t = np.array(eef_pose[:3])
    r = R.from_euler('xyz', eef_pose[3:6]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def depth_to_point_cloud(depth_img, color_img, fx, fy, cx, cy, max_depth=None):
    """将深度图和彩色图转换为点云"""
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    valid = depth_img > 0
    if max_depth is not None:
        valid = valid & (depth_img < max_depth * 1000)
    
    z = depth_img[valid].astype(np.float32) / 1000.0
    u = u[valid]
    v = v[valid]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    b = color_img[valid, 0].astype(np.float32) / 255.0
    g = color_img[valid, 1].astype(np.float32) / 255.0
    r = color_img[valid, 2].astype(np.float32) / 255.0
    
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    
    return np.hstack((xyz, rgb))

def transform_point_cloud(cloud, T):
    """变换点云"""
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:]
    
    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.hstack((xyz, ones))
    xyz_trans = (T @ xyz_homo.T).T
    
    return np.hstack((xyz_trans[:, :3], rgb))

def crop_point_cloud(cloud_np, x_range, y_range, z_range):
    """裁剪点云"""
    xyz = cloud_np[:, :3]
    mask = (
        (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
        (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1]) &
        (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    )
    return cloud_np[mask]

def numpy_to_o3d(cloud_np):
    """转换numpy数组到open3d点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
    return pcd

def generate_point_cloud_single_frame(head_depth, head_color, left_depth, left_color, 
                                     right_depth, right_color, left_eef, right_eef,
                                     intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC,
                                     T_LB_H):
    """
    生成单帧点云 (在左臂基座坐标系下)
    返回: (N, 6) numpy array, N <= FPS_SAMPLE_POINTS
    """
    clouds_global = []
    
    # 1. Head Camera
    fx, fy, cx, cy = intrinsics['head']
    pc_head = depth_to_point_cloud(head_depth, head_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Head)
    if len(pc_head) > 0:
        clouds_global.append(pc_head)
    
    # 2. Left Wrist Camera
    fx, fy, cx, cy = intrinsics['left']
    pc_left = depth_to_point_cloud(left_depth, left_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    if len(pc_left) > 0:
        T_LB_LE = eef_to_matrix(left_eef)
        T_total_left = T_H_LB @ T_LB_LE @ T_LE_LC
        pc_left_global = transform_point_cloud(pc_left, T_total_left)
        clouds_global.append(pc_left_global)
    
    # 3. Right Wrist Camera
    fx, fy, cx, cy = intrinsics['right']
    pc_right = depth_to_point_cloud(right_depth, right_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    if len(pc_right) > 0:
        T_RB_RE = eef_to_matrix(right_eef)
        # 变换路径: Cam -> End -> Base -> HeadCam
        # P_Head = T_H_RB @ T_RB_RE @ T_RE_RC @ P_Cam
        T_total_right = T_H_RB @ T_RB_RE @ T_RE_RC
        pc_right_global = transform_point_cloud(pc_right, T_total_right)
        clouds_global.append(pc_right_global)
    
    if len(clouds_global) == 0:
        # 返回空点云
        return np.zeros((FPS_SAMPLE_POINTS, 6), dtype=np.float32)
    
    # 4. 合并并转换到左臂基座坐标系
    # 注意: T_LB_H实际就是Head->LeftBase, 和pointcloud_from_hdf5.py中用法一致
    merged_cloud = np.vstack(clouds_global)
    merged_cloud = transform_point_cloud(merged_cloud, T_LB_H)
    
    # 5. 工作空间裁剪
    if USE_WORKSPACE_CROP:
        merged_cloud = crop_point_cloud(merged_cloud, WORKSPACE_X_RANGE, 
                                       WORKSPACE_Y_RANGE, WORKSPACE_Z_RANGE)
    
    if len(merged_cloud) == 0:
        return np.zeros((FPS_SAMPLE_POINTS, 6), dtype=np.float32)
    
    # 6. 下采样
    pcd = numpy_to_o3d(merged_cloud)
    
    # 去噪
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 体素下采样
    pcd_voxel = pcd_clean.voxel_down_sample(voxel_size=0.005)
    
    # FPS采样
    if len(pcd_voxel.points) > FPS_SAMPLE_POINTS:
        pcd_fps = pcd_voxel.farthest_point_down_sample(FPS_SAMPLE_POINTS)
    else:
        pcd_fps = pcd_voxel
    
    # 转换回numpy
    pts = np.asarray(pcd_fps.points)
    clrs = np.asarray(pcd_fps.colors)
    result = np.hstack((pts, clrs)).astype(np.float32)
    
    # Pad到固定大小
    if len(result) < FPS_SAMPLE_POINTS:
        padding = np.zeros((FPS_SAMPLE_POINTS - len(result), 6), dtype=np.float32)
        result = np.vstack((result, padding))
    
    return result

# ================= HDF5数据读取 =================

def decode_jpeg(data):
    """解码JPEG数据"""
    img_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转换为RGB

def load_hdf5_episode(hdf5_path):
    """
    读取单个HDF5文件
    返回: dict with keys: eef, images (head, left_wrist, right_wrist), depths, qpos
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 读取end-effector数据
        eef_data = f['observations/eef'][()]  # (T, 14)
        qpos_data = f['observations/qpos'][()]  # (T, 14)
        
        # 读取图像 (JPEG编码)
        head_imgs = [decode_jpeg(d) for d in f['observations/images/head'][()]]
        left_imgs = [decode_jpeg(d) for d in f['observations/images/left_wrist'][()]]
        right_imgs = [decode_jpeg(d) for d in f['observations/images/right_wrist'][()]]
        
        # 读取深度
        head_depths = f['observations/images_depth/head'][()]
        left_depths = f['observations/images_depth/left_wrist'][()]
        right_depths = f['observations/images_depth/right_wrist'][()]
        
        return {
            'eef': eef_data,
            'qpos': qpos_data,
            'head_images': np.array(head_imgs),
            'left_images': np.array(left_imgs),
            'right_images': np.array(right_imgs),
            'head_depths': head_depths,
            'left_depths': left_depths,
            'right_depths': right_depths,
        }

# ================= 关键帧检测 =================

def transform_right_endpose_to_left_base(right_eef_array, T_H_LB, T_H_RB):
    """
    将右臂末端姿态从右臂基座坐标系转换到左臂基座坐标系
    right_eef_array: (N, 7) [x, y, z, rx, ry, rz, gripper]
    T_H_LB: (4, 4) Head到左臂基座的变换矩阵 (注意: 文件名head_base_to_left的含义)
    T_H_RB: (4, 4) Head到右臂基座的变换矩阵
    返回: (N, 7) 在左臂基座坐标系下的姿态
    
    变换链: Head -> RightBase -> RightEnd, 然后转到LeftBase
    即: T_LB_RE = T_H_LB @ T_H_RB @ T_RB_RE
    """
    N = len(right_eef_array)
    result = np.zeros_like(right_eef_array)
    
    for i in range(N):
        # 提取右臂末端在右臂基座系下的姿态
        T_RB_RE = eef_to_matrix(right_eef_array[i])
        
        # 转换到左臂基座系: Head -> RightBase -> RightEnd, 再转到LeftBase
        T_LB_RE = T_H_LB @ T_H_RB @ T_RB_RE
        
        # 提取位置
        result[i, :3] = T_LB_RE[:3, 3]
        
        # 提取旋转(转换为欧拉角)
        rot_matrix = T_LB_RE[:3, :3]
        result[i, 3:6] = R.from_matrix(rot_matrix).as_euler('xyz')
        
        # 夹爪值不变
        result[i, 6] = right_eef_array[i, 6]
    
    return result

def get_keyframe_mask(eef_data, gripper_delta=0.05, min_interval=5):
    """
    生成关键帧mask (只基于夹爪开合,不考虑暂停)
    eef_data: (T, 14) [left(7), right(7)]
    """
    T = len(eef_data)
    mask = np.zeros(T, dtype=bool)
    
    # 提取夹爪状态
    left_gripper = eef_data[:, 6]  # 第7维
    right_gripper = eef_data[:, 13]  # 第14维
    
    # 计算夹爪变化
    left_diff = np.abs(np.diff(left_gripper, prepend=left_gripper[0]))
    right_diff = np.abs(np.diff(right_gripper, prepend=right_gripper[0]))
    
    # 第一帧和最后一帧总是关键帧
    mask[0] = True
    mask[-1] = True
    
    last_keyframe_idx = 0
    for i in range(1, T - 1):
        # 检查夹爪是否有显著变化
        is_gripper_change = (left_diff[i] > gripper_delta) or (right_diff[i] > gripper_delta)
        
        # 强制最小间隔
        if (i - last_keyframe_idx) > min_interval and is_gripper_change:
            mask[i] = True
            last_keyframe_idx = i
    
    return mask

# ================= 主转换函数 =================

def convert_task_to_zarr(task_name, task_dir, max_episodes=None):
    """
    将单个任务的HDF5数据转换为Zarr格式
    
    Args:
        task_name: 任务名称 (文件夹名)
        task_dir: 任务文件夹路径
        max_episodes: 用于debug,只转换前N个episode (None表示转换全部)
    """
    # 自动扫描HDF5文件
    print(f"\n{'='*80}")
    print(f"🎯 任务: {task_name}")
    print(f"{'='*80}")
    print(f"📁 数据目录: {task_dir}")
    
    hdf5_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.hdf5')])
    print(f"🔍 找到 {len(hdf5_files)} 个HDF5文件")
    
    if len(hdf5_files) == 0:
        print(f"⚠️  任务 {task_name} 没有HDF5文件,跳过")
        return
    
    # 输出路径
    os.makedirs(DATASETS_ZARR_DIR, exist_ok=True)
    save_dir = os.path.join(DATASETS_ZARR_DIR, f"{task_name}.zarr")
    
    if os.path.exists(save_dir):
        print(f"⚠️  删除已存在的文件: {save_dir}")
        shutil.rmtree(save_dir)
    
    # 创建Zarr根
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # 加载标定数据
    print("\n📁 加载标定文件...")
    T_LE_LC = load_calibration_matrix("left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix("right_eye_in_hand.npy")
    T_LB_H = load_calibration_matrix("head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix("head_base_to_right_refined_icp.txt")
    
    if np.array_equal(T_LB_H, np.eye(4)):
        T_LB_H = load_calibration_matrix("head_base_to_left.npy")
    if np.array_equal(T_RB_H, np.eye(4)):
        T_RB_H = load_calibration_matrix("head_base_to_right.npy")
    
    # 注意: 文件名head_base_to_left实际表示 Head->LeftBase 的变换
    # 文件名head_base_to_right实际表示 RightBase->Head 的变换 (需要取逆得到Head->RightBase)
    # 和pointcloud_from_hdf5.py保持一致
    T_H_LB = T_LB_H
    T_H_RB = np.linalg.inv(T_RB_H)
    
    intrinsics = {
        'head': load_intrinsics('head'),
        'left': load_intrinsics('left'),
        'right': load_intrinsics('right')
    }
    print("✅ 标定文件加载完成")
    
    # 初始化Zarr数据集
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_datasets = {}
    
    total_count = 0
    
    # 确定要处理的文件列表
    files_to_process = hdf5_files[:max_episodes] if max_episodes is not None else hdf5_files
    
    print(f"\n🔄 开始转换 (共 {len(files_to_process)} episodes)...")
    print(f"第一个文件: {files_to_process[0]}")
    if len(files_to_process) > 1:
        print(f"最后一个文件: {files_to_process[-1]}")
    
    for hdf5_filename in tqdm(files_to_process, desc=f"Converting {task_name}"):
        # 构建文件路径
        hdf5_path = os.path.join(task_dir, hdf5_filename)
        
        if not os.path.exists(hdf5_path):
            print(f"\n⚠️  跳过不存在的文件: {hdf5_path}")
            continue
        
        try:
            # 读取HDF5数据
            data = load_hdf5_episode(hdf5_path)
            eef_data = data['eef']
            qpos_data = data['qpos']
            
            T = len(eef_data)
            if T < 2:
                print(f"\n⚠️  {hdf5_filename} 太短,跳过")
                continue
            
            # 分离左右臂
            left_eef = eef_data[:, :7]
            right_eef = eef_data[:, 7:14]
            
            # 生成点云 (每一帧) - 显示帧级别进度
            point_clouds = []
            print(f"\n  📊 {hdf5_filename}: 生成 {T} 帧点云...")
            for t in tqdm(range(T), desc=f"  Processing frames", leave=False, ncols=80):
                pc = generate_point_cloud_single_frame(
                    data['head_depths'][t], data['head_images'][t],
                    data['left_depths'][t], data['left_images'][t],
                    data['right_depths'][t], data['right_images'][t],
                    left_eef[t], right_eef[t],
                    intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H
                )
                point_clouds.append(pc)
            
            point_clouds = np.array(point_clouds)  # (T, 1024, 6)
            
            # 组织图像 (4个相机: head, left, right, 还需要一个front?)
            # 根据目标格式: (T, 4, 240, 320, 3)
            # 假设我们resize到240x320 (图像已经是RGB格式)
            def resize_images(imgs):
                return np.array([cv2.resize(img, (320, 240)) for img in imgs])
            
            head_resized = resize_images(data['head_images'])
            left_resized = resize_images(data['left_images'])
            right_resized = resize_images(data['right_images'])
            
            # 创建4个相机的图像 (如果只有3个,复制一个)
            images = np.stack([head_resized, head_resized, left_resized, right_resized], axis=1)  # (T, 4, 240, 320, 3)
            
            # 计算关键帧mask
            keyframe_mask = get_keyframe_mask(eef_data, GRIPPER_DELTA, MIN_INTERVAL)
            
            # 准备episode数据 (state[t] + action[t] -> state[t+1])
            ep_state = qpos_data[:-1]  # (T-1, 14)
            ep_action = qpos_data[1:]  # (T-1, 14) 下一个状态作为action
            ep_point_cloud = point_clouds[:-1]  # (T-1, 1024, 6)
            ep_images = images[:-1]  # (T-1, 4, 240, 320, 3)
            ep_keyframe_mask = keyframe_mask[:-1]  # (T-1,)
            ep_left_endpose = eef_data[:-1, :7]  # (T-1, 7) 左臂已经在左臂基座系
            # 右臂: 先转到Head系,再转到LeftBase系 (和点云变换一致)
            ep_right_endpose = transform_right_endpose_to_left_base(eef_data[:-1, 7:14], T_H_LB, T_H_RB)
            
            # 第一次初始化Zarr数据集
            if not zarr_datasets:
                print("\n📦 初始化Zarr数据集...")
                chunks = {
                    "state": (100, 14),
                    "action": (100, 14),
                    "point_cloud": (100, FPS_SAMPLE_POINTS, 6),
                    "images": (100, 4, 240, 320, 3),
                    "keyframe_mask": (100,),
                    "left_endpose": (100, 7),
                    "right_endpose": (100, 7),
                    "episode_ends": (100,)
                }
                
                zarr_datasets["state"] = zarr_data.create_dataset(
                    "state", shape=(0, 14), maxshape=(None, 14), 
                    chunks=chunks["state"], dtype=np.float64, compressor=compressor
                )
                zarr_datasets["action"] = zarr_data.create_dataset(
                    "action", shape=(0, 14), maxshape=(None, 14),
                    chunks=chunks["action"], dtype=np.float64, compressor=compressor
                )
                zarr_datasets["point_cloud"] = zarr_data.create_dataset(
                    "point_cloud", shape=(0, FPS_SAMPLE_POINTS, 6), maxshape=(None, FPS_SAMPLE_POINTS, 6),
                    chunks=chunks["point_cloud"], dtype=np.float32, compressor=compressor
                )
                zarr_datasets["images"] = zarr_data.create_dataset(
                    "images", shape=(0, 4, 240, 320, 3), maxshape=(None, 4, 240, 320, 3),
                    chunks=chunks["images"], dtype=np.uint8, compressor=compressor
                )
                zarr_datasets["keyframe_mask"] = zarr_data.create_dataset(
                    "keyframe_mask", shape=(0,), maxshape=(None,),
                    chunks=chunks["keyframe_mask"], dtype=bool, compressor=compressor
                )
                zarr_datasets["left_endpose"] = zarr_data.create_dataset(
                    "left_endpose", shape=(0, 7), maxshape=(None, 7),
                    chunks=chunks["left_endpose"], dtype=np.float64, compressor=compressor
                )
                zarr_datasets["right_endpose"] = zarr_data.create_dataset(
                    "right_endpose", shape=(0, 7), maxshape=(None, 7),
                    chunks=chunks["right_endpose"], dtype=np.float64, compressor=compressor
                )
                zarr_datasets["episode_ends"] = zarr_meta.create_dataset(
                    "episode_ends", shape=(0,), maxshape=(None,),
                    chunks=chunks["episode_ends"], dtype=np.int64, compressor=compressor
                )
            
            # 追加数据到Zarr
            zarr_datasets["state"].append(ep_state)
            zarr_datasets["action"].append(ep_action)
            zarr_datasets["point_cloud"].append(ep_point_cloud)
            zarr_datasets["images"].append(ep_images)
            zarr_datasets["keyframe_mask"].append(ep_keyframe_mask)
            zarr_datasets["left_endpose"].append(ep_left_endpose)
            zarr_datasets["right_endpose"].append(ep_right_endpose)
            
            total_count += len(ep_state)
            zarr_datasets["episode_ends"].append([total_count])
            
        except Exception as e:
            print(f"\n❌ {hdf5_filename} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 任务 {task_name} 转换完成!")
    print(f"   总帧数: {total_count}")
    print(f"   Episodes: {len(zarr_datasets['episode_ends'][:])}")
    print(f"   保存路径: {save_dir}")
    
    # 打印统计
    keyframe_count = np.sum(zarr_datasets["keyframe_mask"][:])
    print(f"   关键帧数: {keyframe_count} ({keyframe_count/total_count*100:.2f}%)")
    print(f"{'='*80}\n")


def convert_all_tasks(max_episodes=None, task_filter=None):
    """
    转换datasets目录下所有任务
    
    Args:
        max_episodes: 每个任务最多转换多少个episode (None表示全部)
        task_filter: 任务名称过滤器 (None表示全部任务, 或指定任务名列表)
    """
    print("\n" + "="*80)
    print("🚀 HDF5 to Zarr 批量转换工具")
    print("="*80)
    
    # 扫描datasets目录下的所有子文件夹
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ 数据目录不存在: {DATASETS_DIR}")
        return
    
    # 获取所有包含HDF5文件的子文件夹
    task_dirs = []
    for item in os.listdir(DATASETS_DIR):
        item_path = os.path.join(DATASETS_DIR, item)
        if os.path.isdir(item_path):
            # 检查是否包含HDF5文件
            hdf5_files = [f for f in os.listdir(item_path) if f.endswith('.hdf5')]
            if len(hdf5_files) > 0:
                task_dirs.append((item, item_path))
    
    if len(task_dirs) == 0:
        print(f"❌ 在 {DATASETS_DIR} 下未找到包含HDF5文件的任务文件夹")
        return
    
    # 应用过滤器
    if task_filter is not None:
        if isinstance(task_filter, str):
            task_filter = [task_filter]
        task_dirs = [(name, path) for name, path in task_dirs if name in task_filter]
        
        if len(task_dirs) == 0:
            print(f"❌ 没有匹配的任务: {task_filter}")
            return
    
    print(f"\n📋 发现 {len(task_dirs)} 个任务:")
    for i, (task_name, _) in enumerate(task_dirs, 1):
        print(f"   {i}. {task_name}")
    
    print(f"\n💾 输出目录: {DATASETS_ZARR_DIR}")
    
    # 逐个转换任务
    success_count = 0
    failed_tasks = []
    
    for task_name, task_path in task_dirs:
        try:
            convert_task_to_zarr(task_name, task_path, max_episodes)
            success_count += 1
        except Exception as e:
            print(f"\n❌ 任务 {task_name} 转换失败: {e}")
            import traceback
            traceback.print_exc()
            failed_tasks.append(task_name)
    
    # 最终总结
    print("\n" + "="*80)
    print("📊 转换总结")
    print("="*80)
    print(f"✅ 成功: {success_count}/{len(task_dirs)} 个任务")
    if failed_tasks:
        print(f"❌ 失败的任务: {', '.join(failed_tasks)}")
    print(f"💾 输出目录: {DATASETS_ZARR_DIR}")
    print("="*80 + "\n")

# ================= 主程序 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将HDF5数据集转换为Zarr格式 (含点云生成)")
    parser.add_argument("--max_episodes", type=int, default=None, help="每个任务最多转换多少个episodes (None表示全部)")
    parser.add_argument("--task", type=str, default=None, help="指定要转换的任务名称 (默认转换所有任务)")
    
    args = parser.parse_args()
    
    convert_all_tasks(
        max_episodes=args.max_episodes,
        task_filter=args.task
    )
