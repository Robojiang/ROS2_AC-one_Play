"""
脚本: policy/GHOST/scripts/visualize_ghost_beacon.py
功能: 验证 GHOSTBeaconPolicy 的 Beacon 提取和 Heatmap 生成逻辑
可视化内容:
1. 场景点云与 Real Trident (Dim 7, RGB Color coded)
2. 基于关键帧生成的 Heatmap (Dim 8 & 9)
   - Red Channel: Right Arm Heatmap
   - Green Channel: Left Arm Heatmap
   - Blue Channel: Real Trident Indicator (1.0 for Trident, 0.0 for Scene)

该脚本验证几何逻辑是否正确: 
- 红色高亮区域应位于下一个关键帧的右臂抓取位置
- 绿色高亮区域应位于下一个关键帧的左臂抓取位置
- 蓝色点云为当前的机械臂状态
"""

import sys
import os
import torch
import numpy as np
import zarr
import argparse
import cv2

# Add paths
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path) # policy/GHOST/scripts
ghost_dir = os.path.dirname(scripts_dir) # policy/GHOST
policy_dir = os.path.dirname(ghost_dir) # policy
root_dir = os.path.dirname(policy_dir)

sys.path.append(ghost_dir)
sys.path.append(os.path.join(ghost_dir, 'dataset')) 
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from ghost_beacon_policy import GHOSTBeaconPolicy
from dataset.ghost_keyframe_dataset import GHOSTKeyframeDataset

def visualize_beacon_policy(zarr_path, episode_idx=0, save_video=False):
    print(f"Loading dataset from: {zarr_path}")
    
    dataset = GHOSTKeyframeDataset(
        zarr_path=zarr_path,
        horizon=16, 
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0
    )
    
    root = zarr.open(zarr_path, mode='r')
    episode_ends = root['meta/episode_ends']
    
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0]
    else:
        start_idx = episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx]
        
    print(f"\n=== Visualizing Episode {episode_idx} (Frames {start_idx} to {end_idx-1}) ===")
    
    shape_meta = {
        'action': {'shape': [14]},
        'obs': {
            'point_cloud': {'shape': [1024, 6]},
            'agent_pos': {'shape': [32]},
            'target_keypose': {'shape': [18]}
        }
    }
    
    class DummyScheduler:
        pass
    
    print("\n=== Initializing GHOSTBeaconPolicy ===")
    policy = GHOSTBeaconPolicy(
        shape_meta=shape_meta,
        noise_scheduler=DummyScheduler(),
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        use_pc_color=True,
        use_aux_points=True,
        aux_point_num=200, 
        aux_length=0.2,
        aux_trident_side_len=0.15, # Updated to 0.15 as per config
        aux_trident_max_width=0.08,
        aux_radius=0.01,
        use_keyframe_prediction=True,
        keyframe_noise_std=0.02,
        beacon_sigma=0.3 # Adjust this for visualization clarity
    )
    
    print("Fitting normalizer...")
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    def render_view(x, y, colors, img_size, label, cx, cy):
        img = np.full((img_size, img_size, 3), 30, dtype=np.uint8) 
        scale = img_size / 2.0 
        offset = img_size / 2
        
        u = ((x - cx) * scale + offset).astype(int)
        v = ((y - cy) * scale + offset).astype(int)
        v = img_size - v
        
        valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u = u[valid]
        v = v[valid]
        c = (colors[valid] * 255).astype(np.uint8)
        
        for j in range(len(u)):
             # BGR for OpenCV
             cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
             
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return img

    video_writer = None
    fps = 10
    center_x, center_y, center_z = 0.5, 0.0, 0.5
    
    print("\n=== Starting Loop ===")
    print("Red Channel   = Right Arm Target Heatmap")
    print("Green Channel = Left Arm Target Heatmap")
    print("Blue Channel  = Real Trident Indicator")
    
    for i in range(start_idx, end_idx):
        try:
            sample = dataset[i]
        except IndexError:
            break
            
        # Add batch dim
        obs_pc = sample['obs']['point_cloud'][0:1].unsqueeze(0) # (1, 1, N, 6)
        agent_pos = sample['obs']['agent_pos'][0:1].unsqueeze(0) # (1, 1, 32)
        target_keypose = sample['obs']['target_keypose'][0:1].unsqueeze(0) # (1, 1, 18)
        
        with torch.no_grad():
            # 1. Generate Aux Points (Real Trident)
            real_aux = policy._generate_aux_points(agent_pos) 
            
            # P1 Input Construction for Visualization
            pc_xyz = obs_pc[..., :3]
            aux_xyz = real_aux # Raw
            
            # 2. Get Predicted Beacons (Simulated by adding noise to GT, but using the NEW direct-beacon logic)
            # In the new policy, we don't predict pose. We predict beacons.
            # So for visualization, we should:
            #   a. Extract GT 18D Pose
            #   b. Convert to GT Beacons (Raw)
            #   c. Apply Normalization (to mimic what network sees) -> Add Noise -> Unnormalize?
            #      Or just Add Noise to Raw?
            #      Let's follow training logic: GT Beacons -> Normalize -> Add Noise (Simulate Pred) -> Unnormalize
            
            # 2. Get Predicted Beacons (Simulated)
            # a. GT Pose (Raw from dataset)
            # sample['obs']['target_keypose'] is already RAW because we read from dataset and haven't normalized it manually yet.
            target_pose_raw = sample['obs']['target_keypose'][0:1].unsqueeze(0)
            
            # b. GT Beacons (Raw)
            gt_beacons_left, gt_beacons_right = policy._get_beacons_from_pose(target_pose_raw)
            # gt_beacons: (1, 1, 2, 3)
            
            # c. Normalize Beacons & PC for Heatmap Computation (To match training logic)
            # Training computes heatmap in NORMALIZED space.
            if 'point_cloud' in policy.normalizer.params_dict:
                params = policy.normalizer['point_cloud'].params_dict
                # Ensure on CPU for numpy conversion later, but on GPU for computation
                scale = params['scale'].to(gt_beacons_left.device)[:3].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                offset = params['offset'].to(gt_beacons_left.device)[:3].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
                # Normalize Beacons (x * scale + offset)
                gt_beacons_left_norm = gt_beacons_left * scale + offset
                gt_beacons_right_norm = gt_beacons_right * scale + offset
                
                # Add Noise in Normalized Space
                vis_noise_std = policy.keyframe_noise_std
                vis_beacons_left_norm = gt_beacons_left_norm + torch.randn_like(gt_beacons_left_norm) * vis_noise_std
                vis_beacons_right_norm = gt_beacons_right_norm + torch.randn_like(gt_beacons_right_norm) * vis_noise_std
                
                # Normalize Scene PC & Aux for Heatmap
                # pc_xyz is (1, 1, N, 3)
                pc_xyz_norm = pc_xyz * scale + offset
                
                # aux_xyz is (1, 1, K, 3)
                aux_xyz_norm = aux_xyz * scale + offset
                
                all_xyz_norm = torch.cat([pc_xyz_norm, aux_xyz_norm], dim=2)
                
                # 3. Compute Heatmaps in NORMALIZED space
                heat_left = policy._compute_heatmap(all_xyz_norm, vis_beacons_left_norm, sigma=policy.beacon_sigma)
                heat_right = policy._compute_heatmap(all_xyz_norm, vis_beacons_right_norm, sigma=policy.beacon_sigma)
                
                # For display of the white dots, we need to unnormalize the noisy beacons back to RAW
                vis_beacons_left_raw = (vis_beacons_left_norm - offset) / scale
                vis_beacons_right_raw = (vis_beacons_right_norm - offset) / scale
                
            else:
                 # No normalization fallback
                 vis_beacons_left_raw = gt_beacons_left
                 vis_beacons_right_raw = gt_beacons_right
                 all_xyz_raw = torch.cat([pc_xyz, aux_xyz], dim=2)
                 heat_left = policy._compute_heatmap(all_xyz_raw, vis_beacons_left_raw, sigma=policy.beacon_sigma)
                 heat_right = policy._compute_heatmap(all_xyz_raw, vis_beacons_right_raw, sigma=policy.beacon_sigma)

            # 4. Prepare Colors for Visualization
            # Points: all_xyz (RAW)
            all_xyz = torch.cat([pc_xyz, aux_xyz], dim=2)
            
            # Channels: HeatR, HeatL, Indicator
            # Points: all_xyz
            # Channels: HeatR, HeatL, Indicator
            
            N_scene = pc_xyz.shape[2]
            N_aux = aux_xyz.shape[2]
            
            ind_scene = torch.zeros((1, 1, N_scene, 1))
            ind_aux = torch.ones((1, 1, N_aux, 1))
            indicator = torch.cat([ind_scene, ind_aux], dim=2)
            
            heat_left_flat = heat_left.squeeze().numpy()
            heat_right_flat = heat_right.squeeze().numpy()
            indicator_flat = indicator.squeeze().numpy()
            
            xyz_flat = all_xyz.squeeze().numpy()
            
            # Debug: Also show Beacons as white dots?
            beacons_l_flat = vis_beacons_left_raw.cpu().squeeze().numpy() # (2, 3)
            beacons_r_flat = vis_beacons_right_raw.cpu().squeeze().numpy() # (2, 3)
            
            # Color Mapping:
            # R = Right Heat
            # G = Left Heat
            # B = Indicator
            
            colors = np.zeros((len(xyz_flat), 3))
            colors[:, 0] = heat_right_flat # R
            colors[:, 1] = heat_left_flat  # G
            colors[:, 2] = indicator_flat  # B
            
            # Enhance visibility of heatmaps
            # Apply color map? No, direct channel mapping is fine but heat might be faint
            # Normalize heat for viz?
            colors[:, 0] = np.clip(colors[:, 0] * 3.0, 0, 1) 
            colors[:, 1] = np.clip(colors[:, 1] * 3.0, 0, 1)
            
            # Add Beacon points for verification (White)
            beacon_pts = np.vstack([beacons_l_flat, beacons_r_flat])
            beacon_cols = np.ones((len(beacon_pts), 3))
            
            final_xyz = np.vstack([xyz_flat, beacon_pts])
            final_cols = np.vstack([colors, beacon_cols])
            
        # Update center
        scene_pts = xyz_flat[:N_scene]
        valid_mask = (scene_pts[:, 2] > -0.5) & (scene_pts[:, 2] < 1.5)
        if valid_mask.any() and i == start_idx:
             center_x = np.mean(scene_pts[valid_mask, 0])
             center_y = np.mean(scene_pts[valid_mask, 1])
             center_z = np.mean(scene_pts[valid_mask, 2])
             
        img_size = 500
        
        img_xy = render_view(final_xyz[:, 0], final_xyz[:, 1], final_cols, img_size, "Top View (XY)", center_x, center_y)
        img_xz = render_view(final_xyz[:, 0], final_xyz[:, 2], final_cols, img_size, "Front View (XZ)", center_x, center_z)
        img_yz = render_view(final_xyz[:, 1], final_xyz[:, 2], final_cols, img_size, "Side View (YZ)", center_y, center_z)
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        cv2.putText(combined, "R: Right Heat (Target), G: Left Heat (Target), B: Real Trident", (20, img_size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(combined, "White Dots: Target Beacon Centers", (20, img_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("GHOST Beacon Visualization", combined)
        
        if save_video:
            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(f"ghost_beacon_vis_ep{episode_idx}.mp4", fourcc, fps, (w, h))
            video_writer.write(combined)
            
        key = cv2.waitKey(50)
        if key == ord('q') or key == 27:
            break
            
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, 
                        default="policy/VGC/data/stack_blocks_two-demo_3d_vision_hard-100-ppi.zarr",
                        help="Path to zarr dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save_video", default=True, type=bool, help="Whether to save the visualization as video")
    
    args = parser.parse_args()
    
    visualize_beacon_policy(args.zarr_path, args.episode, args.save_video)
