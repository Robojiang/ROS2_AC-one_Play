"""
标定数据加载模块
"""

import os
import json
import numpy as np
from pathlib import Path


def load_intrinsics(intrinsics_file, camera_name):
    """加载相机内参"""
    with open(intrinsics_file, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    return d['fx'], d['fy'], d['cx'], d['cy']


def load_calibration_matrix(calibration_dir, filename):
    """加载标定矩阵"""
    path = os.path.join(calibration_dir, filename)
    if os.path.exists(path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.txt'):
            return np.loadtxt(path).reshape(4, 4)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)


def load_calibration_data(args):
    """
    加载所有标定数据
    
    Args:
        args: 命令行参数（包含 calibration_dir）
        
    Returns:
        dict: 包含所有标定数据的字典
    """
    calibration_dir = Path(args.calibration_dir)
    intrinsics_file = calibration_dir / "D405_intrinsics.json"
    
    print(f"[INFO] 加载标定数据: {calibration_dir}")
    
    # 加载标定矩阵
    T_LE_LC = load_calibration_matrix(calibration_dir, "left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix(calibration_dir, "right_eye_in_hand.npy")
    T_LB_H = load_calibration_matrix(calibration_dir, "head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix(calibration_dir, "head_base_to_right_refined_icp.txt")
    
    T_H_LB = T_LB_H
    T_H_RB = np.linalg.inv(T_RB_H)
    
    # 加载内参
    intrinsics = {
        'head': load_intrinsics(intrinsics_file, 'head'),
        'left': load_intrinsics(intrinsics_file, 'left'),
        'right': load_intrinsics(intrinsics_file, 'right')
    }
    
    print("[INFO] 标定数据加载完成")
    
    return {
        'intrinsics': intrinsics,
        'T_H_LB': T_H_LB,
        'T_H_RB': T_H_RB,
        'T_LE_LC': T_LE_LC,
        'T_RE_RC': T_RE_RC,
        'T_LB_H': T_LB_H
    }
