#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
调试脚本: 检查点云和机械臂末端位置的对应关系
"""

import zarr
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def visualize_frame(zarr_path, frame_idx=0):
    """可视化单帧点云和机械臂末端位置"""
    root = zarr.open(zarr_path, 'r')
    
    # 读取数据
    pc = root['data/point_cloud'][frame_idx]  # (1024, 6)
    left_endpose = root['data/left_endpose'][frame_idx]  # (7,)
    right_endpose = root['data/right_endpose'][frame_idx]  # (7,)
    
    print(f"帧 {frame_idx} 数据:")
    print(f"  Left End: pos={left_endpose[:3]}, rot={left_endpose[3:6]}, gripper={left_endpose[6]}")
    print(f"  Right End: pos={right_endpose[:3]}, rot={right_endpose[3:6]}, gripper={right_endpose[6]}")
    
    # 过滤有效点云
    valid_mask = np.abs(pc).sum(axis=1) > 0
    pc_valid = pc[valid_mask]
    print(f"  有效点云: {len(pc_valid)} 点")
    print(f"  点云范围: X[{pc_valid[:,0].min():.3f}, {pc_valid[:,0].max():.3f}]")
    print(f"            Y[{pc_valid[:,1].min():.3f}, {pc_valid[:,1].max():.3f}]")
    print(f"            Z[{pc_valid[:,2].min():.3f}, {pc_valid[:,2].max():.3f}]")
    
    # 创建可视化
    geometries = []
    
    # 1. 世界坐标系 (左臂基座)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    geometries.append(world_frame)
    
    # 2. 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_valid[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc_valid[:, 3:])
    geometries.append(pcd)
    
    # 3. 左臂基座原点 (红色球体)
    left_base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    left_base_sphere.paint_uniform_color([1, 0, 0])  # 红色
    left_base_sphere.translate([0, 0, 0])
    geometries.append(left_base_sphere)
    
    # 4. 左臂末端坐标系和原点 (红色)
    left_pos = left_endpose[:3]
    left_rot = R.from_euler('xyz', left_endpose[3:6]).as_matrix()
    T_left = np.eye(4)
    T_left[:3, :3] = left_rot
    T_left[:3, 3] = left_pos
    
    left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    left_frame.transform(T_left)
    geometries.append(left_frame)
    
    left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    left_sphere.paint_uniform_color([1, 0, 0])  # 红色
    left_sphere.translate(left_pos)
    geometries.append(left_sphere)
    
    # 5. 右臂基座位置 (和点云变换一致)
    # 读取标定矩阵
    import os
    calib_dir = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/calibration_results"
    T_LB_H = np.loadtxt(os.path.join(calib_dir, "head_base_to_left_refined_icp.txt"))
    T_RB_H = np.loadtxt(os.path.join(calib_dir, "head_base_to_right_refined_icp.txt"))
    
    # 计算右臂基座在左臂基座系下的位置
    # 和点云变换一致: LeftBase -> Head -> RightBase
    T_H_LB = np.linalg.inv(T_LB_H)  # LeftBase -> Head  
    T_H_RB = np.linalg.inv(T_RB_H)  # Head -> RightBase
    # RightBase在LeftBase系: T_LB_H @ T_H_RB (即 Head->RightBase 的位置再转到LeftBase系)
    # 但RightBase的原点在Head系是 T_H_RB的平移部分,转到LeftBase就是 T_LB_H @ [T_H_RB的origin]
    # 等价于: T_LB_RB = T_LB_H @ T_H_RB, 取origin
    T_LB_RB = T_LB_H @ T_H_RB
    
    right_base_pos = T_LB_RB[:3, 3]
    
    # 右臂基座坐标系 (绿色)
    right_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    right_base_frame.transform(T_LB_RB)
    geometries.append(right_base_frame)
    
    right_base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    right_base_sphere.paint_uniform_color([0, 1, 0])  # 绿色
    right_base_sphere.translate(right_base_pos)
    geometries.append(right_base_sphere)
    
    # 6. 右臂末端坐标系和原点 (绿色)
    # right_endpose 现在已经在左臂基座系了,直接使用
    right_pos = right_endpose[:3]
    right_rot = R.from_euler('xyz', right_endpose[3:6]).as_matrix()
    T_right = np.eye(4)
    T_right[:3, :3] = right_rot
    T_right[:3, 3] = right_pos
    
    right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    right_frame.transform(T_right)
    geometries.append(right_frame)
    
    right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    right_sphere.paint_uniform_color([0, 1, 0])  # 绿色
    right_sphere.translate(right_pos)
    geometries.append(right_sphere)
    
    print(f"\n可视化说明:")
    print(f"  - 白色坐标系: 世界坐标系 (左臂基座)")
    print(f"    * X轴(红): 向前")
    print(f"    * Y轴(绿): 向左")
    print(f"    * Z轴(蓝): 向上")
    print(f"  - 大红球: 左臂基座原点")
    print(f"  - 小红球 + 坐标系: 左臂末端")
    print(f"  - 绿色坐标系: 右臂基座")
    print(f"  - 大绿球: 右臂基座原点")
    print(f"  - 小绿球 + 坐标系: 右臂末端 (已转换到左臂基座系)")
    print(f"  - 彩色点云: 三相机融合点云")
    print(f"\n位置信息 (左臂基座系):")
    print(f"  - 右臂基座: X={right_base_pos[0]:.3f}, Y={right_base_pos[1]:.3f}, Z={right_base_pos[2]:.3f}")
    print(f"  - 左臂末端: X={left_pos[0]:.3f}, Y={left_pos[1]:.3f}, Z={left_pos[2]:.3f}")
    print(f"  - 右臂末端: X={right_pos[0]:.3f}, Y={right_pos[1]:.3f}, Z={right_pos[2]:.3f}")
    print(f"\nY轴分析:")
    print(f"  左臂末端Y = {left_pos[1]:.3f}")
    print(f"  右臂末端Y = {right_pos[1]:.3f}")
    print(f"  差值 = {right_pos[1] - left_pos[1]:.3f}")
    print(f"  => 右臂末端在左臂末端的 {'右侧(Y负方向)' if right_pos[1] < left_pos[1] else '左侧(Y正方向)'}")
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries, 
        window_name=f"Frame {frame_idx} - Point Cloud + End Effectors",
        width=1280, 
        height=720
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, default="test_conversion.zarr")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    
    visualize_frame(args.zarr_path, args.frame)
