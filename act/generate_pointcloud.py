#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从采集的RGB+深度数据生成点云
用法: python generate_pointcloud.py --frame 0 --camera head
"""

import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d


def load_intrinsics(intrinsics_path):
    """加载相机内参"""
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def create_pointcloud(rgb_image, depth_image, intrinsics):
    """
    从RGB图和深度图生成点云
    
    参数:
        rgb_image: (H, W, 3) BGR图像
        depth_image: (H, W) uint16深度图，单位毫米
        intrinsics: 相机内参字典
    
    返回:
        o3d.geometry.PointCloud
    """
    # 提取内参
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    height, width = depth_image.shape
    
    # 转换深度单位：毫米 -> 米
    depth_m = depth_image.astype(np.float32) / 1000.0
    
    # 创建像素坐标网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # 反投影到3D空间
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 筛选有效点（深度值 > 0）
    valid = z > 0
    points = np.stack([x[valid], y[valid], z[valid]], axis=1)
    
    # RGB转换为0-1范围
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    colors = rgb_image_rgb[valid].astype(np.float32) / 255.0
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def main():
    parser = argparse.ArgumentParser(description='生成点云')
    parser.add_argument('--data_root', type=str, default='data_points_test',
                        help='数据根目录')
    parser.add_argument('--frame', type=int, default=0,
                        help='帧序号')
    parser.add_argument('--camera', type=str, default='head',
                        choices=['head', 'left_wrist', 'right_wrist'],
                        help='相机名称')
    parser.add_argument('--save', type=str, default='',
                        help='保存点云路径 (例如: output.ply)')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化点云')
    
    args = parser.parse_args()
    
    # 构建文件路径
    frame_prefix = f"{args.frame:04d}_{args.camera}"
    rgb_path = os.path.join(args.data_root, 'colors', f"{frame_prefix}_camera.jpg")
    depth_path = os.path.join(args.data_root, 'depths', f"{frame_prefix}_camera.npy")
    intrinsics_path = os.path.join(args.data_root, 'intrinsics', f"{args.camera}_intrinsics.json")
    
    # 检查文件存在
    for path in [rgb_path, depth_path, intrinsics_path]:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return
    
    print(f"📂 加载数据...")
    print(f"   RGB: {rgb_path}")
    print(f"   Depth: {depth_path}")
    print(f"   Intrinsics: {intrinsics_path}")
    
    # 加载数据
    rgb_image = cv2.imread(rgb_path)
    depth_image = np.load(depth_path)
    intrinsics = load_intrinsics(intrinsics_path)
    
    print(f"   RGB shape: {rgb_image.shape}")
    print(f"   Depth shape: {depth_image.shape}, dtype: {depth_image.dtype}")
    print(f"   Depth range: {depth_image.min()} - {depth_image.max()} mm")
    
    # 生成点云
    print(f"\n🔧 生成点云...")
    pcd = create_pointcloud(rgb_image, depth_image, intrinsics)
    print(f"   点云包含 {len(pcd.points)} 个点")
    
    # 保存
    if args.save:
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"✅ 点云已保存: {args.save}")
    
    # 可视化
    if args.visualize:
        print(f"\n👁️  可视化点云...")
        print("   - 鼠标左键拖拽: 旋转")
        print("   - 鼠标滚轮: 缩放")
        print("   - 按 'Q' 退出")
        o3d.visualization.draw_geometries([pcd],
                                         window_name=f"Point Cloud - {args.camera} Frame {args.frame}",
                                         width=1280, height=720)


if __name__ == '__main__':
    main()
