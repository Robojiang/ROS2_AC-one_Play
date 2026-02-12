#!/usr/bin/env python3
"""
测试GPU加速点云生成器的性能
"""

import os
import sys
import time
import numpy as np
import h5py
import json
import cv2
import torch
from scipy.spatial.transform import Rotation as R

# 添加路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "code_tools"))

# 导入GPU生成器 (从convert_hdf5_to_zarr.py)
from convert_hdf5_to_zarr import (
    GPUPointCloudGenerator,
    load_intrinsics,
    load_calibration_matrix,
    eef_to_matrix,
    decode_jpeg
)

# 配置
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_results")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TEST_EPISODE = os.path.join(DATASETS_DIR, "episode_0.hdf5")
NUM_FRAMES_TEST = 50  # 测试50帧

def load_test_data():
    """加载测试数据"""
    print(f"加载测试数据: {TEST_EPISODE}")
    
    with h5py.File(TEST_EPISODE, 'r') as f:
        eef_data = f['observations/eef'][()][:NUM_FRAMES_TEST]
        
        # 解码图像
        head_imgs = [decode_jpeg(d) for d in f['observations/images/head'][()][:NUM_FRAMES_TEST]]
        left_imgs = [decode_jpeg(d) for d in f['observations/images/left_wrist'][()][:NUM_FRAMES_TEST]]
        right_imgs = [decode_jpeg(d) for d in f['observations/images/right_wrist'][()][:NUM_FRAMES_TEST]]
        
        # 深度
        head_depths = f['observations/images_depth/head'][()][:NUM_FRAMES_TEST]
        left_depths = f['observations/images_depth/left_wrist'][()][:NUM_FRAMES_TEST]
        right_depths = f['observations/images_depth/right_wrist'][()][:NUM_FRAMES_TEST]
        
    return {
        'eef': eef_data,
        'head_images': np.array(head_imgs),
        'left_images': np.array(left_imgs),
        'right_images': np.array(right_imgs),
        'head_depths': head_depths,
        'left_depths': left_depths,
        'right_depths': right_depths,
    }

def load_calibration():
    """加载标定数据"""
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
    
    return T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H, intrinsics

def benchmark_gpu():
    """测试GPU版本性能"""
    print("\n" + "="*80)
    print("⚡ GPU加速点云生成器性能测试")
    print("="*80)
    
    # 加载数据
    data = load_test_data()
    T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H, intrinsics = load_calibration()
    
    # 初始化GPU生成器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    generator = GPUPointCloudGenerator(
        intrinsics=intrinsics,
        downsample_size=(160, 120),
        device=device
    )
    
    # 预热
    print("\n预热 (3帧)...")
    for i in range(3):
        left_eef = data['eef'][i, :7]
        right_eef = data['eef'][i, 7:14]
        
        generator.generate_frame(
            head_depth=data['head_depths'][i],
            head_color=data['head_images'][i],
            left_depth=data['left_depths'][i],
            left_color=data['left_images'][i],
            right_depth=data['right_depths'][i],
            right_color=data['right_images'][i],
            left_eef=left_eef,
            right_eef=right_eef,
            T_H_LB=T_H_LB,
            T_H_RB=T_H_RB,
            T_LE_LC=T_LE_LC,
            T_RE_RC=T_RE_RC,
            T_LB_H=T_LB_H,
            max_depth_head=1.0,
            max_depth_hand=0.6,
            use_workspace_crop=True,
            workspace_x_range=[-0.4, 0.5],
            workspace_y_range=[-0.5, 3.0],
            workspace_z_range=[-0.2, 1.0]
        )
    
    # 正式测试
    print(f"\n开始测试 ({NUM_FRAMES_TEST} 帧)...")
    start_time = time.time()
    
    for i in range(NUM_FRAMES_TEST):
        left_eef = data['eef'][i, :7]
        right_eef = data['eef'][i, 7:14]
        
        pc = generator.generate_frame(
            head_depth=data['head_depths'][i],
            head_color=data['head_images'][i],
            left_depth=data['left_depths'][i],
            left_color=data['left_images'][i],
            right_depth=data['right_depths'][i],
            right_color=data['right_images'][i],
            left_eef=left_eef,
            right_eef=right_eef,
            T_H_LB=T_H_LB,
            T_H_RB=T_H_RB,
            T_LE_LC=T_LE_LC,
            T_RE_RC=T_RE_RC,
            T_LB_H=T_LB_H,
            max_depth_head=1.0,
            max_depth_hand=0.6,
            use_workspace_crop=True,
            workspace_x_range=[-0.4, 0.5],
            workspace_y_range=[-0.5, 3.0],
            workspace_z_range=[-0.2, 1.0]
        )
        
        if i == 0:
            print(f"   首帧点云形状: {pc.shape}")
    
    elapsed = time.time() - start_time
    fps = NUM_FRAMES_TEST / elapsed
    
    print("\n" + "="*80)
    print("📊 性能报告")
    print("="*80)
    print(f"总帧数: {NUM_FRAMES_TEST}")
    print(f"总耗时: {elapsed:.3f} 秒")
    print(f"平均速度: {fps:.2f} FPS")
    print(f"单帧耗时: {elapsed/NUM_FRAMES_TEST*1000:.2f} ms")
    print(f"实时控制要求: {'✅ 满足 (>10Hz)' if fps > 10 else '❌ 不满足 (<10Hz)'}")
    print("="*80)

if __name__ == '__main__':
    benchmark_gpu()
