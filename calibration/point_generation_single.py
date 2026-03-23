#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import json
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# ================= 配置 =================
DATA_ROOT = "data_points_test"
RESULTS_DIR = "calibration_results"
INTRINSICS_FILE = "calibration_results/D435_intrinsics.json"
OUTPUT_DIR = "point_cloud_output"
USE_ICP_REFINEMENT = True # 设置为 True 启用 ICP 修正

# ================= 辅助函数 =================

def load_intrinsics(camera_name):
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    fx, fy = d['fx'], d['fy']
    cx, cy = d['cx'], d['cy']
    return fx, fy, cx, cy

def load_calibration_matrix(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)

def pose_to_matrix(pose):
    # pose: [x, y, z, rx, ry, rz]
    if pose is None: return np.eye(4)
    t = np.array(pose[:3])
    r = R.from_euler('xyz', pose[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def depth_to_point_cloud(depth_img, color_img, fx, fy, cx, cy):
    """
    输入:
      depth_img: (H, W) uint16, unit mm
      color_img: (H, W, 3) BGR uint8
    输出:
      points: (N, 6) [x, y, z, r, g, b] in meters
    """
    h, w = depth_img.shape
    # 创建网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤无效深度 (0)
    valid = (depth_img > 0) & (depth_img < 3000) # 过滤掉太远的点(>3m)
    
    z = depth_img[valid].astype(np.float32) / 1000.0
    u = u[valid]
    v = v[valid]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 颜色 (BGR -> RGB, 0-1)
    b = color_img[valid, 0].astype(np.float32) / 255.0
    g = color_img[valid, 1].astype(np.float32) / 255.0
    r = color_img[valid, 2].astype(np.float32) / 255.0
    
    # 拼接 (N, 6)
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    return np.hstack((xyz, rgb))

def transform_point_cloud(cloud, T):
    """cloud: (N, 6), T: (4, 4)"""
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:]
    
    # 齐次变换
    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.hstack((xyz, ones))
    
    xyz_trans = (T @ xyz_homo.T).T
    xyz_new = xyz_trans[:, :3]
    
    return np.hstack((xyz_new, rgb))

def numpy_to_o3d(cloud_np):
    """转换 (N, 6) numpy 数组 到 open3d.geometry.PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
    return pcd

def visualize_clouds(clouds_dict):
    """
    使用 Open3D 可视化。
    clouds_dict: {"Window Name": point_cloud_numpy_array}
    """
    for name, cloud_np in clouds_dict.items():
        if cloud_np is None or len(cloud_np) == 0:
            continue
            
        print(f"👀 显示: {name} (按 'Q' 关闭当前窗口进入下一个，拖动鼠标旋转)")
        pcd = numpy_to_o3d(cloud_np)
        
        # 创建坐标轴方便参考
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        
        o3d.visualization.draw_geometries([pcd, axes], window_name=name, width=800, height=600)

def visualize_merged(clouds_list, frames_list=[]):
    """
    显示合并后的点云场景
    frames_list: [(T_matrix, label_string), ...]
    """
    print(f"👀 显示: Merged Result (按 'Q' 关闭)")
    geometries = []
    
    # 1. World Origin (Head Camera)
    axes_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(axes_world)

    # 2. Point Clouds
    for cloud_obj in clouds_list:
        if isinstance(cloud_obj, np.ndarray):
            pcd = numpy_to_o3d(cloud_obj)
        else:
            pcd = cloud_obj
        geometries.append(pcd)
        
    # 3. Coordinate Frames
    for T, label in frames_list:
        # T is 4x4
        if T is not None:
            # Create a small coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            frame.transform(T)
            geometries.append(frame)
            
            # Text label (Open3D doesn't support 3D text easily in simple visualizer, usually we just rely on axes)
            # Red=X, Green=Y, Blue=Z
            print(f"  -> Added Frame: {label} at pos {T[:3, 3]}")

    o3d.visualization.draw_geometries(geometries, window_name="Merged Point Cloud", width=1280, height=720)


def run_icp_refinement(source_np, target_np, title="ICP"):
    """
    运行 ICP 配准
    输入: source_np (N, 6), target_np (M, 6)
    输出: transformation (4, 4), aligned_source (pcd)
    """
    print(f"\n[{title}] Running ICP...")
    
    # 转换并预处理
    source = numpy_to_o3d(source_np)
    target = numpy_to_o3d(target_np)
    
    # 降采样并计算法向量 (对于 point-to-plane ICP 很重要)
    voxel_size = 0.005
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # 初始对齐假设为 Identity (因为我们已经有了初步标定)
    trans_init = np.eye(4)
    threshold = 0.02 # 2cm 距离阈值
    
    # 使用 Point-to-Plane ICP (通常比 Point-to-Point 收敛更好)
    try:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
    except Exception as e:
        print(f"ICP Error: {e}, falling back to Point-to-Point")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
    print(f"[{title}] ICP Fitness: {reg_p2l.fitness:.4f}, RMSE: {reg_p2l.inlier_rmse:.4f}")
    print(f"[{title}] Correction Matrix:\n{reg_p2l.transformation}")
    
    # 转换回 numpy (N, 6) 以便统一处理
    pcd_transformed = source.transform(reg_p2l.transformation)
    pts = np.asarray(pcd_transformed.points)
    clrs = np.asarray(pcd_transformed.colors)
    
    return reg_p2l.transformation, np.hstack((pts, clrs))


# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. 加载标定结果
    # Eye-in-Hand: T_Graip_Cam (OpenCV default output: Camera in Gripper frame)
    # P_Grip = T_LC_LE * P_Cam
    T_LC_LE = load_calibration_matrix("left_eye_in_hand.npy")
    T_RC_RE = load_calibration_matrix("right_eye_in_hand.npy")
    
    # Eye-to-Hand: T_Base_Cam (OpenCV default output: Camera in Base frame)
    # P_Base = T_LB_H_raw * P_HeadCam
    T_LB_H_raw = load_calibration_matrix("head_base_to_left.npy")
    T_RB_H_raw = load_calibration_matrix("head_base_to_right.npy")
    
    # 2. 加载内参
    intrinsics = {}
    for name in ['head', 'left', 'right']:
        intrinsics[name] = load_intrinsics(name)

    # 3. 只处理第一帧
    color_files = sorted(glob.glob(os.path.join(DATA_ROOT, "colors", "*_head_camera.jpg")))
    if not color_files:
        print("❌ 没有找到任何帧数据，请检查 data/colors 文件夹")
        return

    # 处理第一帧
    f = color_files[2]
    basename = os.path.basename(f)
    seq_id = basename.split('_')[0]
    print(f"\nProcessing First Frame {seq_id} ...")

    # 加载 Pose
    try:
        with open(os.path.join(DATA_ROOT, "poses", f"{seq_id}_left.json"), 'r') as jf:
            pose_l_data = json.load(jf)['pose']
        with open(os.path.join(DATA_ROOT, "poses", f"{seq_id}_right.json"), 'r') as jf:
            pose_r_data = json.load(jf)['pose']
    except FileNotFoundError:
        print("  ⚠️ 对应的 Pose 文件缺失，将跳过 Eye-in-Hand 拼接")
        pose_l_data = None
        pose_r_data = None

    # Pose: T_Base_End (End-Effector in Base frame)
    # P_Base = T_Base_End * P_End
    T_LB_LE = pose_to_matrix(pose_l_data)
    T_RB_RE = pose_to_matrix(pose_r_data)

    clouds_local = {} # 存储各自相机坐标系下的点云
    clouds_global = [] # 存储统一到 Head 系下的点云
    frames_to_show = [] # 存储需要显示的坐标系

    # Reference Dictionaries to store raw transforms for ICP correction
    T_HeadCam_Base_map = {} # 'left': T, 'right': T
    
    # --- 处理 Head ---
    # Head Camera is Global Origin
    name = "head"
    c_path = os.path.join(DATA_ROOT, "colors", f"{seq_id}_{name}_camera.jpg")
    d_path = os.path.join(DATA_ROOT, "depths", f"{seq_id}_{name}_camera.npy")
    
    head_cloud_np = None # For ICP target
    
    if os.path.exists(c_path) and os.path.exists(d_path):
        rgb = cv2.imread(c_path)
        depth = np.load(d_path)
        fx, fy, cx, cy = intrinsics[name]
        pc = depth_to_point_cloud(depth, rgb, fx, fy, cx, cy)
        clouds_local[name] = pc
        clouds_global.append(pc)
        head_cloud_np = pc # Store for ICP
        frames_to_show.append((np.eye(4), "Head_Camera"))

    # --- 处理 Left ---
    # 目标: P_Head
    # 路径: Cam -> End (EyeInHand result) -> Base -> Head (EyeToHand result is Base->HeadCam)
    #
    # 1. P_End = T_End_Cam @ P_Cam  (T_End_Cam 就是 calibration result "left_eye_in_hand.npy")
    # 2. P_Base = T_Base_End @ P_End
    # 3. P_Head = inv(T_Base_Head) @ P_Base (T_Base_Head 就是 "head_base_to_left.npy")
    
    name = "left"
    c_path = os.path.join(DATA_ROOT, "colors", f"{seq_id}_{name}_wrist_camera.jpg")
    d_path = os.path.join(DATA_ROOT, "depths", f"{seq_id}_{name}_wrist_camera.npy")
    
    left_cloud_global_np = None
    
    if os.path.exists(c_path) and os.path.exists(d_path) and pose_l_data is not None:
        rgb = cv2.imread(c_path)
        depth = np.load(d_path)
        fx, fy, cx, cy = intrinsics[name]
        pc_local = depth_to_point_cloud(depth, rgb, fx, fy, cx, cy)
        clouds_local[name] = pc_local

        T_End_Cam = T_LC_LE  # calibration result is Cam->End (or End->Cam? Let's check debug result)
        # Debug script said: T_base_target = T_base_end @ T_calib @ T_cam_target
        # So T_calib IS T_End_Cam. Correct.
        
        T_Base_End = T_LB_LE
        
        # Head Calibration Result: "head_base_to_left.npy"
        # Debug script said: T_end_target = inv(T_base_end) @ T_calib @ T_cam_target
        # This implies T_calib was treated as T_Base_Cam to check the invariant.
        # And since that check failed (high std dev), this matrix is suspicious.
        # But mathematically it represents T_Base_HeadCam.
        T_Base_HeadCam = T_LB_H_raw 
        
        T_HeadCam_Base = np.linalg.inv(T_Base_HeadCam)
        T_HeadCam_Base_map['left'] = T_HeadCam_Base # Store for ICP
        
        # P_Head = T_HeadCam_Base * T_Base_End * T_End_Cam * P_Cam
        T_total = T_HeadCam_Base @ T_Base_End @ T_End_Cam
        
        # 临时修复：因为 Head 标定可能有 12cm 误差，我们可以先手动补偿一个大概的位移看看
        # 或者暂时只信赖它的旋转，平移置零（如果是在调试阶段）
        # T_total = T_Base_End @ T_End_Cam # 只看 Base 系下的拼接
        
        pc_global = transform_point_cloud(pc_local, T_total)
        clouds_global.append(pc_global)
        left_cloud_global_np = pc_global
        frames_to_show.append((T_total, "Left_Camera"))

    # --- 处理 Right ---
    name = "right"
    c_path = os.path.join(DATA_ROOT, "colors", f"{seq_id}_{name}_wrist_camera.jpg")
    d_path = os.path.join(DATA_ROOT, "depths", f"{seq_id}_{name}_wrist_camera.npy")
    
    right_cloud_global_np = None
    
    if os.path.exists(c_path) and os.path.exists(d_path) and pose_r_data is not None:
        rgb = cv2.imread(c_path)
        depth = np.load(d_path)
        fx, fy, cx, cy = intrinsics[name]
        pc_local = depth_to_point_cloud(depth, rgb, fx, fy, cx, cy)
        clouds_local[name] = pc_local

        T_End_Cam = T_RC_RE
        T_Base_End = T_RB_RE
        T_Base_HeadCam = T_RB_H_raw
        
        T_HeadCam_Base = np.linalg.inv(T_Base_HeadCam)
        T_HeadCam_Base_map['right'] = T_HeadCam_Base # Store for ICP
        
        T_total = T_HeadCam_Base @ T_Base_End @ T_End_Cam
        
        pc_global = transform_point_cloud(pc_local, T_total)
        clouds_global.append(pc_global)
        right_cloud_global_np = pc_global
        frames_to_show.append((T_total, "Right_Camera"))

    # --- ICP Refinement Step ---
    print("\n" + "="*50)
    
    if USE_ICP_REFINEMENT:
        print("🛠️  ICP 自动校准步骤 (ENABLED)")
    else:
        print("⏭️  ICP 自动校准已跳过 (DISABLED)")

    icp_clouds = []
    
    if USE_ICP_REFINEMENT and head_cloud_np is not None:
        icp_clouds.append(head_cloud_np)
        
        # 1. Left Refinement
        if left_cloud_global_np is not None and 'left' in T_HeadCam_Base_map:
            print("正在校准 Left Arm (Eye-to-Base)...")
            T_icp_L, pcd_L_new = run_icp_refinement(left_cloud_global_np, head_cloud_np, title="Left->Head")
            icp_clouds.append(pcd_L_new)
            
            # 计算新的 T_HeadCam_Base
            # T_new = T_icp @ T_old
            T_HB_old = T_HeadCam_Base_map['left']
            T_HB_new = T_icp_L @ T_HB_old
            
            # 原始标定文件存的是 T_Base_Head (= inv(T_HB))
            T_BH_new = np.linalg.inv(T_HB_new)
            
            print(f"👉 建议修正 'head_base_to_left' 矩阵为:\n{T_BH_new}")
            save_path = os.path.join(RESULTS_DIR, "head_base_to_left_refined_icp.npy")
            np.save(save_path, T_BH_new)
            np.savetxt(save_path.replace('.npy', '.txt'), T_BH_new, fmt='%.8f')
            print(f"   已保存建议矩阵到: {save_path} (.npy & .txt)")

        # 2. Right Refinement
        if right_cloud_global_np is not None and 'right' in T_HeadCam_Base_map:
            print("正在校准 Right Arm (Eye-to-Base)...")
            T_icp_R, pcd_R_new = run_icp_refinement(right_cloud_global_np, head_cloud_np, title="Right->Head")
            icp_clouds.append(pcd_R_new)
            
            T_HB_old = T_HeadCam_Base_map['right']
            T_HB_new = T_icp_R @ T_HB_old
            
            T_BH_new = np.linalg.inv(T_HB_new)
            
            print(f"👉 建议修正 'head_base_to_right' 矩阵为:\n{T_BH_new}")
            save_path = os.path.join(RESULTS_DIR, "head_base_to_right_refined_icp.npy")
            np.save(save_path, T_BH_new)
            np.savetxt(save_path.replace('.npy', '.txt'), T_BH_new, fmt='%.8f')
            print(f"   已保存建议矩阵到: {save_path} (.npy & .txt)")
            
        print("="*50 + "\n")
        
        # 询问是否显示 ICP 结果
        print("准备显示 ICP 修正后的聚合效果...")
        visualize_merged(icp_clouds, frames_list=[])
        
        # 关键修复: 更新 clouds_global 以便后续的 Sampling 使用修正后的数据
        if len(icp_clouds) > 1:
            print("🔄 已将修正后的 ICP 点云应用到后续合并流程中")
            clouds_global = icp_clouds

    # --- DP3 采样处理 (FPS/Voxel) ---

    final_points = None
    if clouds_global:
        # 合并所有点云
        merged_cloud = np.vstack(clouds_global)
        
        print(f"\n✅ 原始合并点云点数: {len(merged_cloud)}")
        
        # --- 空间裁剪 (Workspace Crop) ---
        # 按照 pointcloud_generator.py 的默认范围进行裁剪
        workspace_x_range = (-0.4, 0.5)
        workspace_y_range = (-0.5, 3.0)
        workspace_z_range = (-0.2, 1.0)
        
        xyz = merged_cloud[:, :3]
        mask = (
            (xyz[:, 0] >= workspace_x_range[0]) & (xyz[:, 0] <= workspace_x_range[1]) &
            (xyz[:, 1] >= workspace_y_range[0]) & (xyz[:, 1] <= workspace_y_range[1]) &
            (xyz[:, 2] >= workspace_z_range[0]) & (xyz[:, 2] <= workspace_z_range[1])
        )
        merged_cloud = merged_cloud[mask]
        print(f"✂️  空间裁剪后点数: {len(merged_cloud)}")
        
        if len(merged_cloud) == 0:
            print("❌ 裁剪后点云为空，请检查裁剪范围或坐标系对齐是否正确。")
            return
            
        # 为了给 DP3 使用，通常需要降采样 (例如 4096 或 2048 点)
        # 直接由几十万点做 FPS 会非常慢，建议先体素下采样再 FPS
        
        pcd = numpy_to_o3d(merged_cloud)
        
        # 1. 移除离群点 (统计滤波)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"🧹 去噪后点数: {len(pcd.points)}")
        
        # 2. 体素下采样 (Voxel Grid) - 快速减少点数
        voxel_size = 0.005 # 5mm
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"📦 体素下采样({voxel_size}m)后点数: {len(pcd_down.points)}")
        
        # 3. 最远点采样 (FPS) - 确保点云分布均匀
        TARGET_POINTS = 4096
        if len(pcd_down.points) > TARGET_POINTS:
            pcd_fps = pcd_down.farthest_point_down_sample(TARGET_POINTS)
            print(f"🎯 FPS采样后点数: {len(pcd_fps.points)}")
            final_pcd = pcd_fps
        else:
            final_pcd = pcd_down
            
        final_points = np.hstack((np.asarray(final_pcd.points), np.asarray(final_pcd.colors)))

    # --- 实时显示 ---
    
    # 1. 显示单独的点云 (Local Coordinates)
    #    这里只能看每个相机自己拍的效果，用来验证对齐、去畸变和内参是否基本正确（比如墙要是直的）
    print("准备显示由于 [Head, Left, Right] 单独的点云，请依次按 'Q' 查看下一个...")
    # visualize_clouds(clouds_local)
    
    # 2. 显示合成后的点云 (Global Coordinates in Head Frame)
    if final_points is not None:
        print("\n✅ 正在显示合成+采样后的点云结果 (Ready for DP3)...")
        # 将 final_points 转换为 O3D 对象
        # 同时传入 frame 列表可视化
        visualize_merged([final_points], frames_list=frames_to_show)
    elif clouds_global:
        print("\n✅ 正在显示合成点云 (没有进行采样)...")
        visualize_merged(clouds_global, frames_list=frames_to_show)
    else:
        print("没有点云生成")

if __name__ == "__main__":
    main()
