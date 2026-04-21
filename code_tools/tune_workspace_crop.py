#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Workspace 裁剪范围调试与调优工具
读取 HDF5 数据集某一帧，关闭裁剪功能生成全量点云，
并叠加一个可视化的裁剪包围盒 (Bounding Box)。
支持 Shift+LeftClick 在 3D 界面上选取点来读取具体的 XYZ 坐标！
"""

import os
import sys
import argparse
import json
import cv2
import h5py
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("❌ 未安装 Open3D，请运行: pip install open3d")
    sys.exit(1)

# 将上一级目录加入路径，以便导入 inference_utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_utils.pointcloud_generator import PointCloudGenerator

CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration_results")
# 默认内参文件，可以根据实际情况在命令行修改
INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, "D435_intrinsics.json")

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
    print(f"❌ 缺少标定文件: {filename}, 将使用单位矩阵！")
    return np.eye(4)

def decode_jpeg(data):
    """解码JPEG数据"""
    img_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def main():
    parser = argparse.ArgumentParser(description="点云 Workspace 裁剪范围可视化与调试脚本")
    # 找一个现有的 HDF5 文件进行测试，可以自动寻找也可以手动指定
    parser.add_argument("--hdf5_path", type=str, default="", help="指定用于测试的HDF5文件路径 (如果不指定，将自动在 datasets 里找一个)")
    parser.add_argument("--frame", type=int, default=0, help="测试用的数据帧索引 (默认第0帧)")
    
    # 当前裁剪参数 (你可以随时通过命令行修改它们来实时预览)
    parser.add_argument("--x_range", type=float, nargs=2, default=[-1.0, 0.3], help="X轴范围 [min, max]")
    parser.add_argument("--y_range", type=float, nargs=2, default=[-3.0, 3.0], help="Y轴范围 [min, max]")
    parser.add_argument("--z_range", type=float, nargs=2, default=[-0.2, 1.0], help="Z轴范围 [min, max]")
    
    args = parser.parse_args()

    # 1. 自动寻找 HDF5 文件
    hdf5_file = args.hdf5_path
    if not hdf5_file:
        datasets_dir = os.path.join(BASE_DIR, "datasets")
        found = False
        for root, _, files in os.walk(datasets_dir):
            for f in files:
                if f.endswith(".hdf5"):
                    hdf5_file = os.path.join(root, f)
                    found = True
                    break
            if found: break
        if not found:
            print("❌ 没有找到任何 HDF5 文件，请手动通过 --hdf5_path 指定！")
            sys.exit(1)
            
    print(f"\n📂 使用测试数据: {hdf5_file} (帧: {args.frame})")

    # 2. 加载标定数据
    print("加载标定参数...")
    T_LE_LC = load_calibration_matrix("left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix("right_eye_in_hand.npy")
    T_LB_H = load_calibration_matrix("head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix("head_base_to_right_refined_icp.txt")
    
    if np.array_equal(T_LB_H, np.eye(4)):
        T_LB_H = load_calibration_matrix("head_base_to_left.npy")
    if np.array_equal(T_RB_H, np.eye(4)):
        T_RB_H = load_calibration_matrix("head_base_to_right.npy")
        
    T_H_LB = np.linalg.inv(T_LB_H)  # 修复这里：需同样取逆
    T_H_RB = np.linalg.inv(T_RB_H)

    try:
        intrinsics = {
            'head': load_intrinsics('head'),
            'left': load_intrinsics('left'),
            'right': load_intrinsics('right')
        }
    except Exception as e:
        print(f"❌ 相机内参加载失败，请检查 {INTRINSICS_FILE}: {e}")
        sys.exit(1)

    # 3. 读取 HDF5 单帧数据
    t = args.frame
    try:
        with h5py.File(hdf5_file, 'r') as f:
            left_eef = f['observations/eef'][t, :7]
            right_eef = f['observations/eef'][t, 7:14]
            head_color = decode_jpeg(f['observations/images/head'][t])
            left_color = decode_jpeg(f['observations/images/left_wrist'][t])
            right_color = decode_jpeg(f['observations/images/right_wrist'][t])
            head_depth = f['observations/images_depth/head'][t]
            left_depth = f['observations/images_depth/left_wrist'][t]
            right_depth = f['observations/images_depth/right_wrist'][t]
    except Exception as e:
        print(f"❌ HDF5格式读取失败: {e}")
        sys.exit(1)

    # 4. 初始化 PointCloudGenerator
    # ⚠️ 关键点: 强制关闭裁剪 use_workspace_crop=False，并大幅增加采样点数，让我们看到完整的环境！
    print("生成全量点云...")
    pc_gen = PointCloudGenerator(
        intrinsics=intrinsics,
        use_workspace_crop=False,    # <--- 不裁剪，看全貌！
        fps_sample_points=50000,     # <--- 提高点数使边界清晰
        voxel_size=0.005             # <--- 测试时网格稍微大一点没关系
    )

    pc_data, _ = pc_gen.generate(
        head_depth, head_color, left_depth, left_color, right_depth, right_color,
        left_eef, right_eef, intrinsics, 
        T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H
    )

    # 清除由于点数不足填充的0
    mask = np.any(pc_data != 0, axis=1)
    valid_pc = pc_data[mask]

    points = np.asarray(valid_pc[:, :3], dtype=np.float64)
    colors = np.asarray(valid_pc[:, 3:6], dtype=np.float64)

    # 5. 为了清晰显示裁剪范围，我们将框外的点云颜色变暗（偏红），框内的保持鲜艳原色
    # 因为 draw_geometries_with_editing 严格要求只能传入单一 PointCloud，不能混入 BoundingBox
    in_x = (points[:, 0] >= args.x_range[0]) & (points[:, 0] <= args.x_range[1])
    in_y = (points[:, 1] >= args.y_range[0]) & (points[:, 1] <= args.y_range[1])
    in_z = (points[:, 2] >= args.z_range[0]) & (points[:, 2] <= args.z_range[1])
    in_box = in_x & in_y & in_z

    colors[~in_box] = colors[~in_box] * 0.2 + np.array([0.2, 0.0, 0.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\n" + "="*60)
    print("✅ 准备完毕！即将在 Open3D 窗口中显示未经裁剪的全量点云。")
    print("📦 【颜色鲜亮】的部分是你当前的裁剪保留范围 (Crop Range)。")
    print("🟥 【颜色暗淡/偏红】的部分是超出范围、将被丢弃的点。")
    print("\n✨ 神奇功能 (坐标拾取):")
    print("  1. 按住键盘上的 [Shift] 键")
    print("  2. 用鼠标 [左键点击] 你想看到边界的物体上的点")
    print("  3. 终端会立即打印出你所点击点的真实坐标！")
    print("  4. 搜集完坐标后，你可以关闭窗口并修改 `workspace_x_range` 了。")
    print("="*60 + "\n")

    # 使用专门用于选点的可视化器
    o3d.visualization.draw_geometries_with_editing(
        [pcd], 
        window_name="Workspace Crop Tuner (Shift+Click to pick points!)",
        width=1280, height=800
    )


if __name__ == "__main__":
    main()
