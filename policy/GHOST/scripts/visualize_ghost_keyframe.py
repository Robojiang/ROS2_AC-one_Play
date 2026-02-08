"""
脚本: policy/GHOST/scripts/visualize_ghost_keyframe.py
功能: 验证 GHOSTKeyframePolicy 和 GHOSTKeyframeDataset 的集成
可视化内容:
1. 场景点云 (原始颜色)
2. 实时机械臂三叉戟 (Real Trident) -> 红/绿/蓝
3. 目标关键帧"虚拟影子" (Ghost Shadow) -> 黄/青/洋红 (基于Dataset生成的Target Keypose)

该脚本不进行训练，仅通过Policy内部的生成函数验证几何逻辑是否正确 (即影子是否出现在预期的下一个关键帧位置)。
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
sys.path.append(os.path.join(ghost_dir, 'dataset')) # For ghost_keyframe_dataset
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from ghost_keyframe_policy import GHOSTKeyframePolicy
from dataset.ghost_keyframe_dataset import GHOSTKeyframeDataset

def visualize_keyframe_policy(zarr_path, episode_idx=0, save_video=False):
    print(f"Loading dataset from: {zarr_path}")
    
    # 1. Create Dataset
    # We use horizon=16 just like config
    dataset = GHOSTKeyframeDataset(
        zarr_path=zarr_path,
        horizon=16, 
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0
    )
    
    # Get raw meta data for loop bounds
    root = zarr.open(zarr_path, mode='r')
    episode_ends = root['meta/episode_ends']
    
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0]
    else:
        start_idx = episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx]
        
    print(f"\n=== Visualizing Episode {episode_idx} (Frames {start_idx} to {end_idx-1}) ===")
    
    # 2. Create Policy (Dummy instantiation to access helper methods)
    shape_meta = {
        'action': {'shape': [14]},
        'obs': {
            'point_cloud': {'shape': [1024, 6]},
            'agent_pos': {'shape': [32]},
            'target_keypose': {'shape': [18]}
        }
    }
    
    # We don't need a real scheduler or model for visualization of inputs
    class DummyScheduler:
        pass
    
    print("\n=== Initializing GHOSTKeyframePolicy to test geometry generation ===")
    policy = GHOSTKeyframePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DummyScheduler(), # Won't be used
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        use_pc_color=True,
        use_aux_points=True,
        aux_point_num=200, # Same as config
        aux_length=0.2,
        aux_trident_side_len=0.1,
        aux_trident_max_width=0.08,
        aux_radius=0.01,
        use_keyframe_prediction=True,
        keyframe_noise_std=0.1 # Use default noise std
    )
    
    # Fit Normalizer (Required for correctly adding noise in normalized space)
    print("Fitting normalizer to support noise visualization...")
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    # Helper for View Rendering
    def render_view(x, y, colors, img_size, label, cx, cy):
        img = np.full((img_size, img_size, 3), 30, dtype=np.uint8) # Dark background
        scale = img_size / 2.0 
        offset = img_size / 2
        
        # Center the view
        u = ((x - cx) * scale + offset).astype(int)
        v = ((y - cy) * scale + offset).astype(int)
        v = img_size - v
        
        valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u = u[valid]
        v = v[valid]
        c = (colors[valid] * 255).astype(np.uint8)
        
        # Draw points
        for j in range(len(u)):
             # Draw filled circle
             cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
             
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return img

    # Loop Vars
    video_writer = None
    fps = 10 # Slow down to see clearly
    
    # Center calculation (dynamic but stabilized)
    center_x, center_y, center_z = 0.5, 0.0, 0.5
    
    print("\n=== Starting Visualization Loop ===")
    print("Red/Green/Blue = Real Trident (Current)")
    print("Yellow/Cyan/Magenta = Ghost Trident (Target Keyframe)")
    
    for i in range(start_idx, end_idx):
        try:
            # Get sample (returns tensors)
            sample = dataset[i]
        except IndexError:
            break
            
        # Extract Tensors (Take first timestep of horizon)
        # sample['obs']['point_cloud']: (T, N, 6)
        obs_pc = sample['obs']['point_cloud'][0] # (N, 6)
        agent_pos = sample['obs']['agent_pos'][0].unsqueeze(0).unsqueeze(0) # (1, 1, 32)
        target_keypose = sample['obs']['target_keypose'][0].unsqueeze(0).unsqueeze(0) # (1, 1, 18)
        
        # 1. Generate Real Trident (Points)
        with torch.no_grad():
            real_aux = policy._generate_aux_points(agent_pos) # (1, 1, K, 3)
            real_pts = real_aux.squeeze().numpy()
            
            # Generate Ghost Trident (Points) with visualization noise
            # Mimic the training logic: Normalize -> Add Noise -> Unnormalize
            target_keypose_norm = policy.normalizer['target_keypose'].normalize(target_keypose)
            
            # Use larger noise for visualization clarity (e.g. 0.3)
            vis_noise_std = 0.05 
            noise = torch.randn_like(target_keypose_norm) * vis_noise_std
            noisy_keypose_norm = target_keypose_norm + noise
            
            noisy_keypose_raw = policy.normalizer['target_keypose'].unnormalize(noisy_keypose_norm)
            
            ghost_aux = policy._generate_trident_from_pose(noisy_keypose_raw) # (1, 1, K, 3)
            ghost_pts = ghost_aux.squeeze().numpy()
            
        # 2. Prepare Colors
        # Real Colors (RGB for Center/L/R)
        n_pts = len(real_pts)
        n_center = int(policy.aux_point_num * 0.5)
        n_side = (policy.aux_point_num - n_center) // 2
        
        # Colors for one hand part
        c_red = np.array([[1.0, 0.0, 0.0]])
        c_green = np.array([[0.0, 1.0, 0.0]])
        c_blue = np.array([[0.0, 0.0, 1.0]])
        
        # Colors for Ghost (Yellow, Cyan, Magenta)
        c_yellow = np.array([[1.0, 1.0, 0.0]])
        c_cyan = np.array([[0.0, 1.0, 1.0]])
        c_magenta = np.array([[1.0, 0.0, 1.0]])
        
        def make_trident_colors(n_center, n_side, c_c, c_l, c_r, total_pts):
            c_hand = np.vstack([
                np.repeat(c_c, n_center, axis=0),
                np.repeat(c_l, n_side, axis=0),
                np.repeat(c_r, n_side, axis=0)
            ])
            # Assuming 2 hands
            if total_pts > 0 and len(c_hand) > 0:
                n_hands = total_pts // len(c_hand)
                cols = np.tile(c_hand, (n_hands, 1))
                return cols[:total_pts]
            return np.zeros((total_pts, 3))
            
        real_colors = make_trident_colors(n_center, n_side, c_red, c_green, c_blue, len(real_pts))
        ghost_colors = make_trident_colors(n_center, n_side, c_yellow, c_cyan, c_magenta, len(ghost_pts))
        
        # 3. Combine with Scene
        scene_pts = obs_pc[:, :3].numpy()
        scene_rgb = obs_pc[:, 3:6].numpy()
        
        # Update center if valid points exist
        valid_mask = (scene_pts[:, 2] > -0.5) & (scene_pts[:, 2] < 1.5)
        if valid_mask.any() and i == start_idx:
             center_x = np.mean(scene_pts[valid_mask, 0])
             center_y = np.mean(scene_pts[valid_mask, 1])
             center_z = np.mean(scene_pts[valid_mask, 2])
        
        # Stack all
        all_xyz = np.vstack([scene_pts, real_pts, ghost_pts])
        all_rgb = np.vstack([scene_rgb, real_colors, ghost_colors])
        
        # 4. Render
        img_size = 500
        
        img_xy = render_view(all_xyz[:, 0], all_xyz[:, 1], all_rgb, img_size, "Top View (XY) [Y=Right]", center_x, center_y)
        img_xz = render_view(all_xyz[:, 0], all_xyz[:, 2], all_rgb, img_size, "Front View (XZ) [Z=Up]", center_x, center_z)
        img_yz = render_view(all_xyz[:, 1], all_xyz[:, 2], all_rgb, img_size, "Side View (YZ)", center_y, center_z)
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        # Add Legend
        cv2.putText(combined, "RGB: Real Trident (Current)", (20, img_size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(combined, "YCM: Ghost Trident (Target Keyframe)", (20, img_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(combined, f"Frame: {i - start_idx}/{end_idx - start_idx}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show
        cv2.imshow("GHOST Keyframe Visualization", combined)
        
        if save_video:
            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(f"ghost_keyframe_vis_ep{episode_idx}.mp4", fourcc, fps, (w, h))
            video_writer.write(combined)
            
        key = cv2.waitKey(20)
        if key == ord('q') or key == 27:
            break
            
    if video_writer:
        video_writer.release()
        print("Video saved.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, 
                        default="policy/VGC/data/stack_blocks_two-demo_3d_vision_hard-100-ppi.zarr",
                        help="Path to zarr dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save_video", default=True, type=bool, help="Whether to save the visualization as video")
    
    args = parser.parse_args()
    
    visualize_keyframe_policy(args.zarr_path, args.episode, args.save_video)
