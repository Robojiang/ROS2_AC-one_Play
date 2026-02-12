#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试点云生成器帧率（支持GPU加速）
用法：python code_tools/test_pointcloud_fps.py --n 100
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
import json
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_utils.pointcloud_generator import PointCloudGenerator

# ====== 配置 ======
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'calibration_results')
INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, 'D405_intrinsics.json')

# 默认分辨率
IMG_W, IMG_H = 640, 480

# 随机生成一帧假数据（可替换为真机采集数据）
def random_depth_color_pair():
    depth = (np.random.uniform(0.3, 1.0, (IMG_H, IMG_W)) * 1000).astype(np.uint16)  # 0.3~1.0米
    color = np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
    return depth, color

def load_intrinsics():
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    return {
        'head': tuple(all_data['head'][k] for k in ['fx', 'fy', 'cx', 'cy']),
        'left': tuple(all_data['left'][k] for k in ['fx', 'fy', 'cx', 'cy']),
        'right': tuple(all_data['right'][k] for k in ['fx', 'fy', 'cx', 'cy'])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='测试帧数')
    parser.add_argument('--device', type=str, default=None, help='cuda/cpu/None自动')
    args = parser.parse_args()

    intrinsics = load_intrinsics()
    pcg = PointCloudGenerator(
        intrinsics=intrinsics,
        fps_sample_points=1024,
        downsample_size=(160, 120),
        device=args.device
    )

    # 随机生成一组位姿和标定矩阵
    left_eef = np.zeros(7)
    right_eef = np.zeros(7)
    T_H_LB = np.eye(4)
    T_H_RB = np.eye(4)
    T_LE_LC = np.eye(4)
    T_RE_RC = np.eye(4)
    T_LB_H = np.eye(4)

    # 生成一帧假数据用于 warmup
    head_depth, head_color = random_depth_color_pair()
    left_depth, left_color = random_depth_color_pair()
    right_depth, right_color = random_depth_color_pair()
    pcg.generate(head_depth, head_color, left_depth, left_color, right_depth, right_color,
                 left_eef, right_eef, intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H)

    print(f"\n开始测试 {args.n} 帧点云生成...\n")
    t0 = time.time()
    for i in range(args.n):
        head_depth, head_color = random_depth_color_pair()
        left_depth, left_color = random_depth_color_pair()
        right_depth, right_color = random_depth_color_pair()
        pc = pcg.generate(head_depth, head_color, left_depth, left_color, right_depth, right_color,
                          left_eef, right_eef, intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H)
        if (i+1) % 10 == 0:
            print(f"  已生成 {i+1} 帧... shape={pc.shape}")
    t1 = time.time()
    fps = args.n / (t1 - t0)
    print(f"\n平均帧率: {fps:.2f} FPS ({args.n} 帧, 总耗时 {t1-t0:.2f} 秒)")

if __name__ == '__main__':
    main()
