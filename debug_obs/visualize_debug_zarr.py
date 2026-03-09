import sys
import os
import torch
import numpy as np
import zarr
import argparse
import cv2
import pickle

# --- Path Setup ---
# Current script: debug_obs/visualize_debug_zarr.py
# Root: ..
current_file_path = os.path.abspath(__file__)
debug_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(debug_dir)

# Policy paths
ghost_policy_path = os.path.join(project_root, 'policy', 'GHOST')
dp3_path = os.path.join(project_root, 'policy', 'DP3', '3D-Diffusion-Policy')

if ghost_policy_path not in sys.path:
    sys.path.append(ghost_policy_path)
if dp3_path not in sys.path:
    sys.path.append(dp3_path)

try:
    from ghost_policy import GHOSTPolicy
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
except ImportError as e:
    print(f"[ERROR] Import failed. Make sure paths are correct.\n{e}")
    sys.exit(1)

def visualize_zarr_v3(zarr_path, start_frame=0, save_video=False):
    """
    Visualize Zarr dataset compatible with Zarr v3.
    """
    if not os.path.exists(zarr_path):
        print(f"[ERROR] File not found: {zarr_path}")
        return

    print(f"[INFO] Loading Zarr from: {zarr_path}")
    
    # 1. Open Zarr Group (Compatible with v2 and v3)
    try:
        # mode='r' is standard for both versions
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"[ERROR] Failed to open zarr file: {e}")
        return

    # 2. Read Data
    # inference_dp3.py writes columns directly into 'data' group
    # Structure:
    # root['data']['point_cloud'] (T, N, 6)
    # root['data']['agent_pos'] (T, 32)
    # root['data']['state'] (T, 14)
    # root['data']['left_endpose'] (T, 7)
    # root['data']['right_endpose'] (T, 7)
    # root['meta']['episode_ends']
    
    try:
        data_group = root['data']
        point_cloud_arr = data_group['point_cloud'][:]
        agent_pos_arr = data_group['agent_pos'][:]
        state_arr = data_group['state'][:] # Gripper info is here (indices 6 and 13)
        
        # Checking shapes
        print(f"Loaded Data Shapes:")
        print(f"  Point Cloud: {point_cloud_arr.shape}")
        print(f"  Agent Pos:   {agent_pos_arr.shape}")
        print(f"  State:       {state_arr.shape}")
        
    except KeyError as e:
        print(f"[ERROR] Key not found in Zarr: {e}")
        print("Available keys in data:", list(root['data'].keys()) if 'data' in root else 'No data group')
        return

    # Metadata
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends'][:]
        total_frames = episode_ends[-1]
    else:
        # Fallback if meta is missing (e.g. simple debug dump)
        total_frames = len(point_cloud_arr)
        print(f"[WARN] No episode_ends found, using total length: {total_frames}")

    # 3. Setup Policy for Aux Points
    print("\n[INFO] Initializing GHOST Policy for Aux Points generation...")
    shape_meta = {
        'action': {'shape': [14]},
        'obs': {
            'point_cloud': {'shape': [1024, 6]},
            'agent_pos': {'shape': [32]}
        }
    }
    
    # Mock Scheduler (not used for aux point generation but required for init)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    policy = GHOSTPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=16, 
        n_action_steps=8,
        n_obs_steps=2,
        use_pc_color=True,
        use_aux_points=True,
        aux_point_num=80,
    )
    
    # 4. Setup Normalizer
    # GHOST logic uses normalizer to scale gripper widths for animation logic usually
    print("[INFO] Fitting Normalizer...")
    normalizer = LinearNormalizer()
    norm_data = {
        'agent_pos': agent_pos_arr,
        'action': data_group['action'][:] if 'action' in data_group else np.zeros_like(state_arr),
        'point_cloud': point_cloud_arr
    }
    normalizer.fit(data=norm_data, last_n_dims=1, mode='limits')
    policy.set_normalizer(normalizer)

    # 5. Visualization Loop
    print("\n[INFO] Starting Visualization (Press 'q' to quit)")
    
    # View params
    center_x, center_y, center_z = 0.5, 0.0, 0.4
    center_calculated = False
    img_size = 500
    video_writer = None
    fps = 30
    
    for i in range(start_frame, total_frames):
        # Get frame data
        pc_frame = point_cloud_arr[i] # (N, 6)
        agent_pos_frame = agent_pos_arr[i] # (32,)
        state_frame = state_arr[i] # (14,)
        
        # Calculate Aux Points
        # Expand timestamps for policy input
        # Policy expects (B, T, D) but _generate_aux_points takes (B, T, D) or (B, D)?
        # Looking at previous code: agent_pos_tensor = (1, 1, 32)
        agent_pos_tensor = torch.from_numpy(agent_pos_frame).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            aux_pts = policy._generate_aux_points(agent_pos_tensor)
        aux_pts = aux_pts.cpu().numpy().squeeze() # (K, 3)
        if len(aux_pts.shape) == 1: aux_pts = aux_pts.reshape(-1, 3)

        # Aux Colors
        n_center = int(policy.aux_point_num * 0.5)
        n_side = (policy.aux_point_num - n_center) // 2
        
        c_red = np.array([[1.0, 0.0, 0.0]])   # Center
        c_green = np.array([[0.0, 1.0, 0.0]]) # Left
        c_blue = np.array([[0.0, 0.0, 1.0]])  # Right
        
        colors_block = np.vstack([
            np.repeat(c_red, n_center, axis=0),
            np.repeat(c_green, n_side, axis=0),
            np.repeat(c_blue, n_side, axis=0)
        ])
        
        # Check num hands (simple heuristic based on points count)
        n_pts = len(aux_pts)
        if n_pts > 0 and len(colors_block) > 0:
             n_hands = n_pts // len(colors_block)
             aux_cols = np.tile(colors_block, (n_hands, 1))
             aux_cols = aux_cols[:n_pts]
        else:
             aux_cols = np.zeros((n_pts, 3))

        # Combine PC
        scene_xyz = pc_frame[:, :3]
        scene_rgb = pc_frame[:, 3:6]
        
        # Filter outliers for cleaner view center calculation
        if not center_calculated and len(scene_xyz) > 0:
            valid_mask = (scene_xyz[:, 2] > -0.5) & (scene_xyz[:, 2] < 1.5)
            if valid_mask.any():
                center_x = np.mean(scene_xyz[valid_mask, 0])
                center_y = np.mean(scene_xyz[valid_mask, 1])
                center_z = np.mean(scene_xyz[valid_mask, 2])
                center_calculated = True
                print(f"[INFO] Auto-centered view at: {center_x:.2f}, {center_y:.2f}, {center_z:.2f}")

        full_xyz = np.vstack([scene_xyz, aux_pts])
        full_rgb = np.vstack([scene_rgb, aux_cols])

        # Renderer Helper
        def render_plane(x_axis, y_axis, labels, cx, cy):
            img = np.full((img_size, img_size, 3), 40, dtype=np.uint8) # Dark grey bg
            scale = img_size / 2.0
            offset = img_size / 2
            
            u = ((x_axis - cx) * scale + offset).astype(int)
            v = ((y_axis - cy) * scale + offset).astype(int)
            v = img_size - v # Flip Y
            
            valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
            
            # Draw circles
            u_v = u[valid]
            v_v = v[valid]
            c_v = (full_rgb[valid] * 255).astype(np.uint8)
            
            # BGR for OpenCV
            for idx in range(len(u_v)):
                 # c_v is RGB, cv2 needs BGR
                 cv2.circle(img, (u_v[idx], v_v[idx]), 2, 
                            (int(c_v[idx, 2]), int(c_v[idx, 1]), int(c_v[idx, 0])), -1)
            
            cv2.putText(img, labels, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return img

        # Render 3 Views
        img_xy = render_plane(full_xyz[:, 0], full_xyz[:, 1], "Top (XY)", center_x, center_y)
        img_xz = render_plane(full_xyz[:, 0], full_xyz[:, 2], "Front (XZ)", center_x, center_z)
        img_yz = render_plane(full_xyz[:, 1], full_xyz[:, 2], "Side (YZ)", center_y, center_z)
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        # Add Text Info
        left_grip = state_frame[6]
        right_grip = state_frame[13]
        info1 = f"Frame: {i} | Pts: {len(scene_xyz)}"
        info2 = f"L.Grip: {left_grip:.3f} | R.Grip: {right_grip:.3f}"
        
        cv2.putText(combined, info1, (20, img_size - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(combined, info2, (20, img_size - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("Debug Observation Viewer (Zarr v3 compatible)", combined)
        
        # Save Video
        if save_video:
            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_name = os.path.join(os.path.dirname(zarr_path), f"vis_{os.path.basename(zarr_path)}.mp4")
                video_writer = cv2.VideoWriter(vid_name, fourcc, fps, (w, h))
                print(f"[INFO] Recording to {vid_name}")
            video_writer.write(combined)
            
        key = cv2.waitKey(33) # ~30fps
        if key == ord('q') or key == 27:
            print("[INFO] User Quit")
            break
            
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/arx/haitao_codes/ROS2_AC-one_Play/debug_obs/debug_DP3_1773047769.zarr", help="Path to .zarr file or .pkl file")
    parser.add_argument("--video", action="store_true", help="Save video")
    
    args = parser.parse_args()
    
    target_path = args.path
    
    # Auto-find latest if not provided
    if not target_path:
        # Check debug_obs folder
        if os.path.exists(debug_dir):
            files = [os.path.join(debug_dir, f) for f in os.listdir(debug_dir) if f.endswith('.zarr')]
            if files:
                target_path = max(files, key=os.path.getmtime)
                print(f"[INFO] Auto-selected latest Zarr: {target_path}")
            else:
                print("[ERROR] No zarr files found in debug_obs/ and no path provided.")
                sys.exit(1)
        else:
             print(f"[ERROR] debug_obs dir not found: {debug_dir}")
             sys.exit(1)
             
    visualize_zarr_v3(target_path, save_video=args.video)
