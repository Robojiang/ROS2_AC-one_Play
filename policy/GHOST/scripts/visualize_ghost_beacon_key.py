"""
脚本: policy/GHOST/scripts/visualize_ghost_beacon_key.py
功能: 验证 GHOSTBeaconKeyPolicy 的完整逻辑
可视化内容:
1. 场景点云 (Scene)
2. 实时机械臂三叉戟 (Real Trident) -> 蓝色
3. 目标关键帧虚影 (Ghost Trident) -> 黄色
4. Beacon Heatmap -> 叠加在所有点云上的红/绿光晕
   - 绿色光晕: 左臂 Heatmap (即将在该位置操作)
   - 红色光晕: 右臂 Heatmap (即将在该位置操作)
   - 强烈的白色实心点: Heatmap 的中心 (Beacon 位置)

此脚本用于验证:
- Ghost 是否正确生成在 Target Keypose 位置。
- Heatmap 是否以此 Ghost 的指尖中心为圆心扩散。
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
ghost_dir = os.path.dirname(os.path.dirname(current_file_path)) # policy/GHOST
policy_dir = os.path.dirname(ghost_dir) # policy
root_dir = os.path.dirname(policy_dir)

sys.path.append(ghost_dir)
sys.path.append(os.path.join(ghost_dir, 'dataset')) # For ghost_keyframe_dataset
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from ghost_beacon_key_policy import GHOSTBeaconKeyPolicy
from dataset.ghost_keyframe_dataset import GHOSTKeyframeDataset

def visualize_beacon_key_policy(zarr_path, episode_idx=0, save_video=False):
    print(f"Loading dataset from: {zarr_path}")
    
    # 1. Dataset
    dataset = GHOSTKeyframeDataset(
        zarr_path=zarr_path,
        horizon=16, 
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0
    )
    
    # Meta info
    root = zarr.open(zarr_path, mode='r')
    episode_ends = root['meta/episode_ends']
    
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0] - 1 # -1 to avoid boundary issues if any
    else:
        start_idx = episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx] - 1
        
    print(f"\n=== Visualizing Episode {episode_idx} (Frames {start_idx} to {end_idx}) ===")
    
    # 2. Policy Setup
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
    
    print("\n=== Initializing GHOSTBeaconKeyPolicy ===")
    policy = GHOSTBeaconKeyPolicy(
        shape_meta=shape_meta,
        noise_scheduler=DummyScheduler(),
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        use_pc_color=True,
        use_aux_points=True,
        aux_point_num=200, 
        aux_length=0.2,
        beacon_sigma=0.3, # Sigma parameter
        keyframe_noise_std=0.03, # Noise for viz
        use_keyframe_prediction=True
    )
    
    print("Fitting normalizer...")
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    # 3. View Renderer
    def render_view(points, colors, img_size, label, cx, cy, view_axis='xy'):
        # view_axis: 'xy', 'xz', 'yz'
        img = np.full((img_size, img_size, 3), 30, dtype=np.uint8) # Dark BG
        scale = img_size / 2.0 
        offset = img_size / 2
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        if view_axis == 'xy':
            u_raw, v_raw = x, y
            cu, cv = cx, cy
        elif view_axis == 'xz':
            u_raw, v_raw = x, z
            cu, cv = cx, cy # passed in z as cy
        elif view_axis == 'yz':
            u_raw, v_raw = y, z
            cu, cv = cx, cy # passed in z as cy
            
        u = ((u_raw - cu) * scale + offset).astype(int)
        v = ((v_raw - cv) * scale + offset).astype(int)
        v = img_size - v # Flip Y for image coords
        
        valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u = u[valid]
        v = v[valid]
        c = (colors[valid] * 255).astype(np.uint8)
        
        # Draw points
        for j in range(len(u)):
            cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
            
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return img

    video_writer = None
    fps = 10
    
    # Dynamic center logic
    center_x, center_y, center_z = 0.5, 0.0, 0.5
    
    print("\n=== Visualization Legend ===")
    print("Blue Points   : Real Trident (Current Agent State)")
    print("Yellow Points : Ghost Trident (Predicted Keyframe State)")
    print("Green Glow    : Left Arm Beacon Heatmap")
    print("Red Glow      : Right Arm Beacon Heatmap")
    print("White Dots    : Beacon Center Points")
    
    # Loop
    for i in range(start_idx, end_idx):
        try:
            sample = dataset[i]
        except IndexError:
            break
            
        # Get Data (Raw Tensors)
        # Add batch dimension: (1, ...)
        obs_pc = sample['obs']['point_cloud'][0:1] # (1, N, 6) -> We need (1, 1, N, 6) for policy funcs if they expect T
        # Policy inputs usually (B, T, ...). 
        # _generate_aux_points expects (B, T, D)
        
        obs_pc = obs_pc.unsqueeze(0) # (1, 1, N, 6)
        agent_pos = sample['obs']['agent_pos'][0:1].unsqueeze(0) # (1, 1, 32)
        target_keypose = sample['obs']['target_keypose'][0:1].unsqueeze(0) # (1, 1, 18) - RAW
        
        with torch.no_grad():
            # A. Generate P1 Input Geometry (Real Trident)
            # ------------------------------------------------
            real_aux = policy._generate_aux_points(agent_pos) # (1, 1, K, 3) RAW
            
            # B. Generate P2 Input Geometry (Ghost Trident)
            # ------------------------------------------------
            # Simulate prediction: Normalize -> Add Noise -> Unnormalize
            if 'target_keypose' in policy.normalizer.params_dict:
                kp_norm = policy.normalizer['target_keypose'].normalize(target_keypose)
                kp_norm = kp_norm + torch.randn_like(kp_norm) * policy.keyframe_noise_std
                kp_raw = policy.normalizer['target_keypose'].unnormalize(kp_norm)
            else:
                kp_raw = target_keypose
                
            ghost_aux = policy._generate_trident_from_pose(kp_raw) # (1, 1, K, 3) RAW
            
            # C. Combine Points for Heatmap Calc
            # ------------------------------------------------
            scene_xyz = obs_pc[..., :3] # (1, 1, N, 3)
            # Combine: Scene + Real + Ghost
            all_xyz = torch.cat([scene_xyz, real_aux, ghost_aux], dim=2) # (1, 1, Total, 3) RAW
            
            # D. Calculate Heatmap
            # ------------------------------------------------
            # Policy expects RAW xyz and RAW pose for physical distance calculation
            heat_l, heat_r, b_l, b_r = policy._calculate_beacon_heatmap(all_xyz, kp_raw, sigma=policy.beacon_sigma)
            # heat: (1, 1, Total, 1)
            
            
        # --- Visualization Data Preparation ---
        
        # 1. Flatten Points
        scene_pts = scene_xyz.squeeze().numpy()
        real_pts = real_aux.squeeze().numpy()
        ghost_pts = ghost_aux.squeeze().numpy()
        
        total_pts = np.vstack([scene_pts, real_pts, ghost_pts])
        
        # 2. Base Colors
        N_scene = len(scene_pts)
        N_real = len(real_pts)
        N_ghost = len(ghost_pts)
        
        # Build colors directly from masks and heatmaps
        # R channel: aux mask (1 for aux/real trident points, 0 otherwise)
        # G channel: heat_l (left arm heat)
        # B channel: heat_r (right arm heat)
        # This avoids using raw scene RGB which can be visually distracting.
        h_l_flat = heat_l.squeeze().cpu().numpy()
        h_r_flat = heat_r.squeeze().cpu().numpy()

        # Safety: ensure heat arrays length matches total points
        L_heat = len(h_l_flat)
        L_pts = len(total_pts)
        if L_heat != L_pts:
            # If lengths mismatch for any reason, crop/pad heat arrays to match
            if L_heat > L_pts:
                h_l_flat = h_l_flat[:L_pts]
                h_r_flat = h_r_flat[:L_pts]
            else:
                h_l_flat = np.pad(h_l_flat, (0, L_pts - L_heat))
                h_r_flat = np.pad(h_r_flat, (0, L_pts - L_heat))

        # Aux mask: scene=0, real_aux=1, ghost=0
        indicator = np.concatenate([
            np.zeros(N_scene, dtype=float),
            np.ones(N_real, dtype=float),
            np.zeros(N_ghost, dtype=float)
        ], axis=0)

        cols = np.stack([indicator, h_l_flat, h_r_flat], axis=1)
        cols = np.clip(cols, 0.0, 1.0)
        
        # 4. Add Beacon Dots (White)
        beacon_pts = np.vstack([b_l.squeeze().numpy(), b_r.squeeze().numpy()])
        beacon_cols = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        
        # But render_view only draws 2px dots. Beacons should be larger? 
        # We can just add them to the list, render_view will draw them. 
        # Maybe add a few points around beacon to make it look bigger?
        # Or just trust the 2px dot + Glow surrounding it.
        
        final_pts = np.vstack([total_pts, beacon_pts])
        final_cols = np.vstack([cols, beacon_cols])
        
        # --- Render ---
        
        # Update center if Scene valid
        valid_mask = (scene_pts[:, 2] > -0.5) & (scene_pts[:, 2] < 1.5)
        if valid_mask.any() and i == start_idx:
             center_x = np.mean(scene_pts[valid_mask, 0])
             center_y = np.mean(scene_pts[valid_mask, 1])
             center_z = np.mean(scene_pts[valid_mask, 2])
             
        img_size = 600 # High res
        
        img_xy = render_view(final_pts, final_cols, img_size, "Top (XY)", center_x, center_y, 'xy')
        img_xz = render_view(final_pts, final_cols, img_size, "Front (XZ)", center_x, center_z, 'xz')
        img_yz = render_view(final_pts, final_cols, img_size, "Side (YZ)", center_y, center_z, 'yz')
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        # Legend Overlay
        legend_txt = f"Frame: {i} | Blue=Real, Yellow=Ghost, Red/Green Glow=Heatmap"
        cv2.putText(combined, legend_txt, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("GHOST Beacon Key Viz", combined)
        
        if save_video:
            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                save_path = f"ghost_beacon_key_vis_ep{episode_idx}.mp4"
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                print(f"Recording video to {save_path}")
            video_writer.write(combined)
            
        key = cv2.waitKey(50) # 50fps max display
        if key == ord('q') or key == 27:
            break
            
    if video_writer:
        video_writer.release()
        print("Video Saved.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, 
                        default="policy/VGC/data/stack_blocks_two-demo_3d_vision_hard-100-ppi.zarr",
                        help="Path to dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save_video", action='store_true', default=True)
    
    args = parser.parse_args()
    
    visualize_beacon_key_policy(args.zarr_path, args.episode, args.save_video)
