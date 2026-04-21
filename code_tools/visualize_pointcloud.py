#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单点云可视化脚本 - 极致精简重写版
读取 Zarr 数据集中的点云数据，并用 Open3D 进行无限循环播放
"""

import zarr
import numpy as np
import argparse
import sys
import time

try:
    import open3d as o3d
except ImportError:
    print("❌ 未安装 Open3D，请运行: pip install open3d")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Zarr Point Cloud Viewer")
    parser.add_argument("--zarr_path", type=str, default="datasets_zarr/pick_place_d435.zarr", help="Zarr dataset path")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    args = parser.parse_args()

    print(f"�� Loading Zarr: {args.zarr_path}")
    try:
        root = zarr.open(args.zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error opening zarr: {e}")
        sys.exit(1)

    pc_array = root['data/point_cloud']
    ep_ends = root['meta/episode_ends'][:]
    
    start_idx = 0 if args.episode == 0 else ep_ends[args.episode - 1]
    end_idx = ep_ends[args.episode]

    print(f"🎬 Playing Episode {args.episode} (Frames {start_idx} ~ {end_idx - 1}, Total: {end_idx - start_idx})")

    # 1. 提取真正的第一帧 (用这一帧来初始化 Open3D，彻底避免空点云或包围盒警告)
    first_frame = pc_array[start_idx]
    # Open3D 内部几何体极其严格地要求数据必须是 np.float64 格式
    init_pts = np.asarray(first_frame[:, :3], dtype=np.float64)
    init_cols = np.asarray(first_frame[:, 3:6], dtype=np.float64)

    # 2. 初始化 Open3D 窗口和点云对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Zarr Point Cloud Viewer", width=1280, height=800)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(init_pts)
    pcd.colors = o3d.utility.Vector3dVector(init_cols)
    vis.add_geometry(pcd)

    # 3. 设置渲染风格：浅色背景、大号点径
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.9, 0.9, 0.9])  # 改为浅灰色背景，方便查看暗色点云

    # 4. 主渲染循环 (极简安全模式：不绑按键、直接轮播)
    frame_idx = start_idx
    print("\n💡 正在播放... (直接关闭弹出的 3D 窗口即可退出)")
    
    try:
        while True:
            # 拉取当前帧
            frame_data = pc_array[frame_idx]
            pts = np.asarray(frame_data[:, :3], dtype=np.float64)
            cols = np.asarray(frame_data[:, 3:6], dtype=np.float64)

            # 更新数据，避免重新创建对象
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            
            # 告诉 Open3D 对象改变了
            vis.update_geometry(pcd)

            # 抓取界面绘制和系统事件 (如果返回 False 说明用户手点了右上角关闭)
            if not vis.poll_events():
                break
            vis.update_renderer()

            # 帧递增，到末尾循环
            frame_idx += 1
            if frame_idx >= end_idx:
                frame_idx = start_idx
                
            # 模拟约 30 FPS 的播放速度
            time.sleep(0.033)

    except KeyboardInterrupt:
        pass
    finally:
        print("👋 播放结束，释放窗口。")
        vis.destroy_window()

if __name__ == "__main__":
    main()
