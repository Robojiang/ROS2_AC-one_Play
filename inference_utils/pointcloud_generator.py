"""
点云生成模块 (GPU 加速版)
核心逻辑参考 self.convert_hdf5_to_zarr.py 以确保一致性
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import cv2

class PointCloudGenerator:
    """GPU加速点云生成器，接口兼容原用法"""
    def __init__(self,
                 intrinsics=None,
                 max_depth_head=1.0,
                 max_depth_hand=0.6,
                 fps_sample_points=1024,
                 use_workspace_crop=True,
                 workspace_x_range=(-0.4, 0.5),
                 workspace_y_range=(-0.5, 3.0),
                 workspace_z_range=(-0.2, 1.0),
                 voxel_size=0.005,
                 downsample_size=(160, 120),
                 device=None):
        self.max_depth_head = max_depth_head
        self.max_depth_hand = max_depth_hand
        self.fps_sample_points = fps_sample_points
        self.use_workspace_crop = use_workspace_crop
        self.workspace_x_range = workspace_x_range
        self.workspace_y_range = workspace_y_range
        self.workspace_z_range = workspace_z_range
        self.voxel_size = voxel_size
        self.downsample_size = downsample_size
        self.device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # 兼容推理/训练两种用法，若初始化时未提供内参，将在第一次generate时初始化
        if intrinsics is not None:
            self.intrinsics = intrinsics
            self._init_ray_dirs()
        else:
            self.intrinsics = None
            self.ray_dirs = None

    def _init_ray_dirs(self):
        self.ray_dirs = {}
        w, h = self.downsample_size
        for cam_name, (fx, fy, cx, cy) in self.intrinsics.items():
            # 调整内参到降采样分辨率
            scale_x = w / 640.0
            scale_y = h / 480.0
            fx_scaled = fx * scale_x
            fy_scaled = fy * scale_y
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            
            # 生成像素坐标网格
            u, v = torch.meshgrid(
                torch.arange(w, device=self.device, dtype=torch.float32),
                torch.arange(h, device=self.device, dtype=torch.float32),
                indexing='xy')
            
            # 计算射线方向
            x_over_z = (u - cx_scaled) / fx_scaled
            y_over_z = (v - cy_scaled) / fy_scaled
            self.ray_dirs[cam_name] = (x_over_z, y_over_z)

    @staticmethod
    def eef_to_matrix(eef_pose):
        if eef_pose is None or len(eef_pose) < 6:
            return np.eye(4)
        t = np.array(eef_pose[:3])
        r = R.from_euler('xyz', eef_pose[3:6]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        return T

    def depth_to_pointcloud(self, depth_img, color_img, cam_name, max_depth=None):
        # 1. 降采样
        depth_small = cv2.resize(depth_img, self.downsample_size, interpolation=cv2.INTER_NEAREST)
        color_small = cv2.resize(color_img, self.downsample_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. 转为torch tensor (mm -> m)
        depth_t = torch.from_numpy(depth_small).to(self.device).float() / 1000.0
        color_t = torch.from_numpy(color_small).to(self.device).float() / 255.0
        
        # 3. 有效性掩码
        valid = depth_t > 0
        if max_depth is not None:
            valid = valid & (depth_t < max_depth)
            
        # 4. 射线方向
        x_over_z, y_over_z = self.ray_dirs[cam_name]
        z = depth_t
        x = x_over_z * z
        y = y_over_z * z
        
        # 5. 展平并过滤
        x_flat = x[valid]
        y_flat = y[valid]
        z_flat = z[valid]
        
        r_flat = color_t[:, :, 0][valid]
        g_flat = color_t[:, :, 1][valid]
        b_flat = color_t[:, :, 2][valid]
        
        xyz = torch.stack([x_flat, y_flat, z_flat], dim=1)
        rgb = torch.stack([r_flat, g_flat, b_flat], dim=1)
        
        return torch.cat([xyz, rgb], dim=1)

    def transform_pointcloud(self, cloud, T):
        T_t = torch.from_numpy(T).to(self.device).float()
        xyz = cloud[:, :3]
        rgb = cloud[:, 3:]
        ones = torch.ones((xyz.shape[0], 1), device=self.device)
        xyz_homo = torch.cat([xyz, ones], dim=1)
        xyz_trans = (T_t @ xyz_homo.T).T
        return torch.cat([xyz_trans[:, :3], rgb], dim=1)

    def crop_pointcloud(self, cloud, x_range, y_range, z_range):
        xyz = cloud[:, :3]
        mask = (
            (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
            (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1]) &
            (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
        )
        return cloud[mask]

    def generate(self, head_depth, head_color, left_depth, left_color,
                right_depth, right_color, left_eef, right_eef,
                intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H):
        
        # 若需要，重新初始化内参
        try:
             # 判断内参是否变化
             if (not hasattr(self, 'intrinsics')) or (self.intrinsics != intrinsics):
                self.intrinsics = intrinsics
                self._init_ray_dirs()
        except:
             # 如果字典比较失败(如有numpy array)，直接重新初始化
             self.intrinsics = intrinsics
             self._init_ray_dirs()

        clouds = []
        # 1. Head Camera
        pc_head = self.depth_to_pointcloud(head_depth, head_color, 'head', max_depth=self.max_depth_head)
        if len(pc_head) > 0:
            clouds.append(pc_head)
        # 2. Left Wrist Camera
        pc_left = self.depth_to_pointcloud(left_depth, left_color, 'left', max_depth=self.max_depth_hand)
        if len(pc_left) > 0:
            T_LB_LE = self.eef_to_matrix(left_eef)
            T_total_left = T_H_LB @ T_LB_LE @ T_LE_LC
            pc_left = self.transform_pointcloud(pc_left, T_total_left)
            clouds.append(pc_left)
        # 3. Right Wrist Camera
        pc_right = self.depth_to_pointcloud(right_depth, right_color, 'right', max_depth=self.max_depth_hand)
        if len(pc_right) > 0:
            T_RB_RE = self.eef_to_matrix(right_eef)
            T_total_right = T_H_RB @ T_RB_RE @ T_RE_RC
            pc_right = self.transform_pointcloud(pc_right, T_total_right)
            clouds.append(pc_right)
        
        # --- 合并 ---
        if len(clouds) == 0:
            return np.zeros((self.fps_sample_points, 6), dtype=np.float32)
        merged = torch.cat(clouds, dim=0)
        
        # --- 统一转换到基座坐标系 ---
        merged = self.transform_pointcloud(merged, T_LB_H)
        
        # --- 空间裁剪 (GPU) ---
        if self.use_workspace_crop:
            merged = self.crop_pointcloud(merged, self.workspace_x_range, self.workspace_y_range, self.workspace_z_range)
        
        if len(merged) == 0:
            return np.zeros((self.fps_sample_points, 6), dtype=np.float32)

        # ====== 后处理路径 (Open3D CPU) ======
        # 参考 convert_hdf5_to_zarr.py 的做法
        merged_cpu = merged.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        if merged_cpu.shape[0] > 0:
            pcd.points = o3d.utility.Vector3dVector(merged_cpu[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(merged_cpu[:, 3:])
            
            # 1. 统计离群点去除 (Denoise)
            pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # 2. 体素降采样 (Voxel Downsample)
            pcd_voxel = pcd_clean.voxel_down_sample(voxel_size=self.voxel_size)
            
            # 3. 随机/FPS采样到固定点数
            pts = np.asarray(pcd_voxel.points)
            clrs = np.asarray(pcd_voxel.colors)
            
            if len(pts) > self.fps_sample_points:
                # 随机采样 (比FPS快，效果接近)
                indices = np.random.choice(len(pts), self.fps_sample_points, replace=False)
                pts = pts[indices]
                clrs = clrs[indices]
                result = np.hstack((pts, clrs)).astype(np.float32)
            else:
                # 点数不足，全取
                if len(pts) > 0:
                     result = np.hstack((pts, clrs)).astype(np.float32)
                else:
                     result = np.zeros((0, 6), dtype=np.float32)
        else:
            result = np.zeros((0, 6), dtype=np.float32)

        # Padding
        if len(result) < self.fps_sample_points:
            padding = np.zeros((self.fps_sample_points - len(result), 6), dtype=np.float32)
            if len(result) > 0:
                result = np.vstack((result, padding))
            else:
                result = padding
            
        return result
