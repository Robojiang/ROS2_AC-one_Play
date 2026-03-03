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
import torch
import torch.nn.functional as F

# ================= 配置 =================
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # 上一级目录
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_utils.pointcloud_generator import PointCloudGenerator

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
DATASETS_ZARR_DIR = os.path.join(BASE_DIR, "datasets_zarr")
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_results")
INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, "D405_intrinsics.json")

# ⚡ GPU加速配置
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

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
    
def convert_pose_to_ghost_format(pose_rpy_7d):
    """
    Args: pose_rpy_7d (N, 7) [x, y, z, rx, ry, rz, gripper]
    Returns: (N, 7) [x, y, z, w, x, y, z] (GHOST: Pos + Quat(w-first))
    """
    N = len(pose_rpy_7d)
    result = np.zeros((N, 7))
    
    # 提取位置
    result[:, :3] = pose_rpy_7d[:, :3]
    
    # 提取RPY并转四元数
    # R.from_euler expects (N, 3)
    eulers = pose_rpy_7d[:, 3:6]
    quats = R.from_euler('xyz', eulers).as_quat() # [x, y, z, w]
    
    # 重排为 [w, x, y, z]
    result[:, 3] = quats[:, 3] # w
    result[:, 4] = quats[:, 0] # x
    result[:, 5] = quats[:, 1] # y
    result[:, 6] = quats[:, 2] # z
    
    return result

# ================= 主转换函数 =================

def convert_task_to_zarr(task_name, task_dir, max_episodes=None):
    """
    将单个任务的HDF5数据转换为Zarr格式
    
    Args:
        task_name: 任务名称 (文件夹名)
        task_dir: 任务文件夹路径
        max_episodes: 用于debug,只转换前N个episode (None表示转换全部)
    """
    import math

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
    
    # ⚡ 初始化GPU点云生成器 (统一使用推理所用的类，配置已在其内部定义)
    pc_generator = PointCloudGenerator(
        intrinsics=intrinsics,
        device=str(DEVICE)
    )
    
    print(f"\n⚡ GPU加速点云生成器已初始化")
    print(f"   设备: {pc_generator.device}")
    print(f"   降采样尺寸: {pc_generator.downsample_size} (原始: 640x480)")
    print(f"   目标点数: {pc_generator.fps_sample_points}")
    print(f"   采样方式: {'随机采样 (快速)' if pc_generator.use_random_sampling else 'FPS采样 (慢但均匀)'}")
    
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
            left_eefs_trans = []
            right_eefs_trans = []
            print(f"\n  📊 {hdf5_filename}: 生成 {T} 帧点云...")
            for t in tqdm(range(T), desc=f"  Processing frames", leave=False, ncols=80):
                pc, trans_eef = pc_generator.generate(
                    head_depth=data['head_depths'][t],
                    head_color=data['head_images'][t],
                    left_depth=data['left_depths'][t],
                    left_color=data['left_images'][t],
                    right_depth=data['right_depths'][t],
                    right_color=data['right_images'][t],
                    left_eef=left_eef[t],
                    right_eef=right_eef[t],
                    intrinsics=intrinsics,
                    T_H_LB=T_H_LB,
                    T_H_RB=T_H_RB,
                    T_LE_LC=T_LE_LC,
                    T_RE_RC=T_RE_RC,
                    T_LB_H=T_LB_H
                )
                point_clouds.append(pc)
                left_eefs_trans.append(trans_eef['left'])
                right_eefs_trans.append(trans_eef['right'])
            
            point_clouds = np.array(point_clouds)  # (T, N_POINTS, 6)
            left_eefs_trans = np.array(left_eefs_trans)
            right_eefs_trans = np.array(right_eefs_trans)
            
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
            
            # --- 修复：GHOST 数据集需要 Pos + Quaternion (7D) ---
            # 直接使用 generated 产出的统一坐标系下的左基座 endpose（[xyz rpy gripper] 转为 [xyz wxyz]）
            ep_left_endpose = convert_pose_to_ghost_format(left_eefs_trans[:-1])
            ep_right_endpose = convert_pose_to_ghost_format(right_eefs_trans[:-1])
            
            # --- Debug: 打印前3帧的维度和数据范围 ---
            if total_count == 0:
                print(f"[DEBUG] EP_LEFT_ENDPOSE shape: {ep_left_endpose.shape}")
                print(f"[DEBUG] EP_LEFT_ENDPOSE[0]: {ep_left_endpose[0]}")
            
            # 第一次初始化Zarr数据集
            if not zarr_datasets:
                print("\n📦 初始化Zarr数据集...")
                chunks = {
                    "state": (100, 14),
                    "action": (100, 14),
                    "point_cloud": (100, pc_generator.fps_sample_points, 6),
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
                    "point_cloud", shape=(0, pc_generator.fps_sample_points, 6), maxshape=(None, pc_generator.fps_sample_points, 6),
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
    
    if not zarr_datasets:
        print(f"\n❌ 任务 {task_name} 转换失败，没有成功处理任何有效数据!")
        print(f"{'='*80}\n")
        return

    print(f"\n✅ 任务 {task_name} 转换完成!")
    print(f"   总帧数: {total_count}")
    print(f"   Episodes: {len(zarr_datasets['episode_ends'][:])}")
    print(f"   保存路径: {save_dir}")
    
    # 打印统计
    keyframe_count = np.sum(zarr_datasets["keyframe_mask"][:])
    print(f"   关键帧数: {keyframe_count} ({keyframe_count/total_count*100:.2f}%)" if total_count > 0 else "   关键帧数: 0 (0.00%)")
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
