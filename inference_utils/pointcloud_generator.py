"""
点云生成模块
负责从深度图和彩色图生成点云
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class PointCloudGenerator:
    """点云生成器"""
    
    def __init__(self, 
                 max_depth_head=1.0,
                 max_depth_hand=0.6,
                 fps_sample_points=1024,
                 use_workspace_crop=True,
                 workspace_x_range=(-0.4, 0.5),
                 workspace_y_range=(-0.5, 3.0),
                 workspace_z_range=(-0.2, 1.0),
                 voxel_size=0.005):
        """
        Args:
            max_depth_head: 头部相机最大深度(米)
            max_depth_hand: 手腕相机最大深度(米)
            fps_sample_points: FPS采样点数
            use_workspace_crop: 是否裁剪工作空间
            workspace_x/y/z_range: 工作空间范围
            voxel_size: 体素下采样大小
        """
        self.max_depth_head = max_depth_head
        self.max_depth_hand = max_depth_hand
        self.fps_sample_points = fps_sample_points
        self.use_workspace_crop = use_workspace_crop
        self.workspace_x_range = workspace_x_range
        self.workspace_y_range = workspace_y_range
        self.workspace_z_range = workspace_z_range
        self.voxel_size = voxel_size
    
    @staticmethod
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
    
    @staticmethod
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
        
        # RGB归一化到[0, 1]
        r = color_img[valid, 0].astype(np.float32) / 255.0
        g = color_img[valid, 1].astype(np.float32) / 255.0
        b = color_img[valid, 2].astype(np.float32) / 255.0
        
        xyz = np.stack((x, y, z), axis=1)
        rgb = np.stack((r, g, b), axis=1)
        
        return np.hstack((xyz, rgb))
    
    @staticmethod
    def transform_point_cloud(cloud, T):
        """变换点云"""
        xyz = cloud[:, :3]
        rgb = cloud[:, 3:]
        
        ones = np.ones((xyz.shape[0], 1))
        xyz_homo = np.hstack((xyz, ones))
        xyz_trans = (T @ xyz_homo.T).T
        
        return np.hstack((xyz_trans[:, :3], rgb))
    
    def crop_point_cloud(self, cloud_np):
        """裁剪点云"""
        xyz = cloud_np[:, :3]
        mask = (
            (xyz[:, 0] >= self.workspace_x_range[0]) & (xyz[:, 0] <= self.workspace_x_range[1]) &
            (xyz[:, 1] >= self.workspace_y_range[0]) & (xyz[:, 1] <= self.workspace_y_range[1]) &
            (xyz[:, 2] >= self.workspace_z_range[0]) & (xyz[:, 2] <= self.workspace_z_range[1])
        )
        return cloud_np[mask]
    
    @staticmethod
    def numpy_to_o3d(cloud_np):
        """转换numpy数组到open3d点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
        return pcd
    
    def generate(self, head_depth, head_color, left_depth, left_color,
                right_depth, right_color, left_eef, right_eef,
                intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H):
        """
        生成单帧点云 (在左臂基座坐标系下)
        
        Returns:
            (N, 6) numpy array, N <= fps_sample_points
        """
        clouds_global = []
        
        # 1. Head Camera
        fx, fy, cx, cy = intrinsics['head']
        pc_head = self.depth_to_point_cloud(head_depth, head_color, fx, fy, cx, cy, 
                                           max_depth=self.max_depth_head)
        if len(pc_head) > 0:
            clouds_global.append(pc_head)
        
        # 2. Left Wrist Camera
        fx, fy, cx, cy = intrinsics['left']
        pc_left = self.depth_to_point_cloud(left_depth, left_color, fx, fy, cx, cy,
                                           max_depth=self.max_depth_hand)
        if len(pc_left) > 0:
            T_LB_LE = self.eef_to_matrix(left_eef)
            T_total_left = T_H_LB @ T_LB_LE @ T_LE_LC
            pc_left_global = self.transform_point_cloud(pc_left, T_total_left)
            clouds_global.append(pc_left_global)
        
        # 3. Right Wrist Camera
        fx, fy, cx, cy = intrinsics['right']
        pc_right = self.depth_to_point_cloud(right_depth, right_color, fx, fy, cx, cy,
                                            max_depth=self.max_depth_hand)
        if len(pc_right) > 0:
            T_RB_RE = self.eef_to_matrix(right_eef)
            T_total_right = T_H_RB @ T_RB_RE @ T_RE_RC
            pc_right_global = self.transform_point_cloud(pc_right, T_total_right)
            clouds_global.append(pc_right_global)
        
        if len(clouds_global) == 0:
            return np.zeros((self.fps_sample_points, 6), dtype=np.float32)
        
        # 4. 合并并转换到左臂基座坐标系
        merged_cloud = np.vstack(clouds_global)
        merged_cloud = self.transform_point_cloud(merged_cloud, T_LB_H)
        
        # 5. 工作空间裁剪
        if self.use_workspace_crop:
            merged_cloud = self.crop_point_cloud(merged_cloud)
        
        if len(merged_cloud) == 0:
            return np.zeros((self.fps_sample_points, 6), dtype=np.float32)
        
        # 6. 下采样
        pcd = self.numpy_to_o3d(merged_cloud)
        
        # 去噪
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 体素下采样
        pcd_voxel = pcd_clean.voxel_down_sample(voxel_size=self.voxel_size)
        
        # FPS采样
        if len(pcd_voxel.points) > self.fps_sample_points:
            pcd_fps = pcd_voxel.farthest_point_down_sample(self.fps_sample_points)
        else:
            pcd_fps = pcd_voxel
        
        # 转换回numpy
        pts = np.asarray(pcd_fps.points)
        clrs = np.asarray(pcd_fps.colors)
        result = np.hstack((pts, clrs)).astype(np.float32)
        
        # Pad到固定大小
        if len(result) < self.fps_sample_points:
            padding = np.zeros((self.fps_sample_points - len(result), 6), dtype=np.float32)
            result = np.vstack((result, padding))
        
        return result
