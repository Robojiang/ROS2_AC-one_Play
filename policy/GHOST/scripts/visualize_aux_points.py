"""
可视化脚本: 动态检查经过GHOSTDataset处理后的虚拟点云
验证归一化后传入policy的点云和虚拟点云是否正确
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
scripts_dir = os.path.dirname(current_file_path)
ghost_dir = os.path.dirname(scripts_dir)
policy_dir = os.path.dirname(ghost_dir)
project_root = os.path.dirname(policy_dir)

sys.path.append(ghost_dir)
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from ghost_policy import GHOSTPolicy
from dataset.ghost_dataset import GHOSTDataset

def visualize_dataset_dynamic(zarr_path, episode_idx=0, save_video=False):
    """
    动态播放GHOSTDataset处理后的数据
    验证归一化后的点云和虚拟点云是否正确
    """
    print(f"Loading dataset from: {zarr_path}")
    
    # 1. 使用GHOSTDataset加载数据(模拟训练流程)
    print("\n=== Creating GHOSTDataset (simulating training data loading) ===")
    dataset = GHOSTDataset(
        zarr_path=zarr_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0,
    )
    
    # 2. 获取episode范围
    root = zarr.open(zarr_path, mode='r')
    episode_ends = root['meta/episode_ends']
    state_data = root['data/state']
    left_endpose = root['data/left_endpose']
    right_endpose = root['data/right_endpose']
    
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0]
    else:
        start_idx = episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx]
    
    print(f"\n=== Visualizing Episode {episode_idx} (Frames {start_idx} to {end_idx-1}) ===")
    print(f"Total frames: {end_idx - start_idx}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # 3. 创建policy用于生成虚拟点云
    print("\n=== Creating GHOST Policy ===")
    shape_meta = {
        'action': {'shape': [14]},
        'obs': {
            'point_cloud': {'shape': [1024, 6]},
            'agent_pos': {'shape': [32]}
        }
    }
    
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
        aux_length=0.3,
        aux_trident_side_len=0.15,
        aux_trident_max_width=0.08,
        aux_radius=0.01,
    )
    
    # Video writer setup
    video_writer = None
    fps = 30
    
    def render_view(x, y, colors, img_size, label):
        img = np.full((img_size, img_size, 3), 50, dtype=np.uint8)
        scale = img_size / 1.5
        offset = img_size / 2
        
        u = (x * scale + offset).astype(int)
        v = (y * scale + offset).astype(int)
        v = img_size - v
        
        valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u = u[valid]
        v = v[valid]
        c = (colors[valid] * 255).astype(np.uint8)
        
        for j in range(len(u)):
            cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
            
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img
    
    # 4. 动态循环播放
    print("\n=== Starting Dynamic Visualization ===")
    print("Press 'q' or ESC to quit")
    
    for i in range(start_idx, end_idx):
        # 从dataset获取样本(Dataset返回的是原始数据，未归一化)
        try:
            sample = dataset[i]
        except IndexError:
            print(f"Frame {i} out of dataset range, skipping...")
            continue
        
        # sample结构: {'obs': {'point_cloud': (T, N, 6), 'agent_pos': (T, 32)}, 'action': (T, 14)}
        point_cloud_seq = sample['obs']['point_cloud']  # (T, N, 6) - 原始数据
        agent_pos_seq = sample['obs']['agent_pos']  # (T, 32) - 原始数据
        
        # 取序列中的第一帧
        scene_pc = point_cloud_seq[0].numpy()  # (N, 6)
        agent_pos = agent_pos_seq[0].numpy()  # (32,)
        
        # 从agent_pos中提取末端位置
        left_pos = agent_pos[14:17]   # 原始位置
        right_pos = agent_pos[23:26]  # 原始位置
        
        # 从原始zarr获取gripper值(用于显示)
        state = state_data[i]
        left_pose_raw = left_endpose[i]  # [x, y, z, qw, qx, qy, qz]
        right_pose_raw = right_endpose[i]
        
        left_pos_raw = left_pose_raw[0:3]
        right_pos_raw = right_pose_raw[0:3]
        
        # 生成虚拟点云(使用原始agent_pos)
        agent_pos_tensor = torch.from_numpy(agent_pos).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
        
        with torch.no_grad():
            aux_pts = policy._generate_aux_points(agent_pos_tensor)
        
        aux_pts = aux_pts.squeeze(0).squeeze(0).numpy()  # (K, 3)
        
        # 手动生成颜色用于可视化 (Red=Center, Green=Left, Blue=Right)
        # 需与Policy内部生成逻辑保持一致
        n_center = int(policy.aux_point_num * 0.5)
        n_side = (policy.aux_point_num - n_center) // 2
        
        c_red = np.array([[1.0, 0.0, 0.0]])
        c_green = np.array([[0.0, 1.0, 0.0]])
        c_blue = np.array([[0.0, 0.0, 1.0]])
        
        # 单只手的颜色分布
        colors_per_hand = np.vstack([
            np.repeat(c_red, n_center, axis=0),
            np.repeat(c_green, n_side, axis=0),
            np.repeat(c_blue, n_side, axis=0)
        ])
        
        # 根据点数推断手的数量并复制颜色
        if len(aux_pts) > 0 and len(colors_per_hand) > 0:
            num_hands = len(aux_pts) // len(colors_per_hand)
            aux_cols = np.tile(colors_per_hand, (num_hands, 1))
            # 截断以防万一
            aux_cols = aux_cols[:len(aux_pts)]
        else:
            aux_cols = np.zeros_like(aux_pts)
        
        # 合并场景点云和虚拟点云
        scene_xyz = scene_pc[:, :3]
        scene_rgb = scene_pc[:, 3:6]
        
        combined_xyz = np.vstack([scene_xyz, aux_pts])
        combined_rgb = np.vstack([scene_rgb, aux_cols])
        
        # 自动计算视图中心 (基于场景点云的均值)
        # 只要有一帧计算过就行，不用每帧变
        center_x = 0.0
        center_y = 0.0 
        center_z = 0.5
        
        # 简单过滤离群值
        valid_mask = (scene_xyz[:, 2] > -0.5) & (scene_xyz[:, 2] < 1.5)
        if valid_mask.any():
            center_x = np.mean(scene_xyz[valid_mask, 0])
            center_y = np.mean(scene_xyz[valid_mask, 1])
            center_z = np.mean(scene_xyz[valid_mask, 2])
            
        def render_view(x, y, colors, img_size, label, cx, cy):
            img = np.full((img_size, img_size, 3), 50, dtype=np.uint8)
            scale = img_size / 2.0  # Zoom 适应视野 (1米范围铺满一半屏幕)
            offset = img_size / 2
            
            # 使用传入的中心进行平移
            u = ((x - cx) * scale + offset).astype(int)
            v = ((y - cy) * scale + offset).astype(int)
            v = img_size - v
            
            valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
            u = u[valid]
            v = v[valid]
            c = (colors[valid] * 255).astype(np.uint8)
            
            for j in range(len(u)):
                cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
                
            cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img
        
        img_size = 500
        
        # 渲染视图 (使用动态计算的中心)
        img_xy = render_view(combined_xyz[:, 0], combined_xyz[:, 1], combined_rgb, img_size, "Top (XY)", center_x, center_y)
        img_xz = render_view(combined_xyz[:, 0], combined_xyz[:, 2], combined_rgb, img_size, "Front (XZ)", center_x, center_z)
        img_yz = render_view(combined_xyz[:, 1], combined_xyz[:, 2], combined_rgb, img_size, "Side (YZ)", center_y, center_z)
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        # 添加末端效应器标记
        for arm_name, pos, color_bgr in [("Left", left_pos, (0, 255, 0)), ("Right", right_pos, (255, 0, 0))]:
            # Top view marker
            scale = img_size / 2.0
            offset = img_size / 2
            
            u_xy = int((pos[0] - center_x) * scale + offset)
            v_xy = int(img_size - ((pos[1] - center_y) * scale + offset))
            
            if 0 <= u_xy < img_size and 0 <= v_xy < img_size:
                cv2.circle(img_xy, (u_xy, v_xy), 8, color_bgr, 2)
                cv2.putText(img_xy, arm_name[0], (u_xy+10, v_xy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            # Front view marker
            u_xz = int((pos[0] - center_x) * scale + offset)
            v_xz = int(img_size - ((pos[2] - center_z) * scale + offset))
            if 0 <= u_xz < img_size and 0 <= v_xz < img_size:
                cv2.circle(img_xz, (u_xz, v_xz), 8, color_bgr, 2)
            
            # Side view marker
            u_yz = int((pos[1] - center_y) * scale + offset)
            v_yz = int(img_size - ((pos[2] - center_z) * scale + offset))
            if 0 <= u_yz < img_size and 0 <= v_yz < img_size:
                cv2.circle(img_yz, (u_yz, v_yz), 8, color_bgr, 2)
        
        combined = np.hstack([img_xy, img_xz, img_yz])
        
        # 添加信息文本
        frame_num = i - start_idx
        info_text = [
            f"Episode: {episode_idx}, Frame: {frame_num}/{end_idx-start_idx}",
            f"Dataset Sample Index: {i}",
            f"Left End (Agent Pos): [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]",
            f"Right End (Agent Pos): [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]",
            f"Left End (Raw Zarr): [{left_pos_raw[0]:.3f}, {left_pos_raw[1]:.3f}, {left_pos_raw[2]:.3f}]",
            f"Right End (Raw Zarr): [{right_pos_raw[0]:.3f}, {right_pos_raw[1]:.3f}, {right_pos_raw[2]:.3f}]",
            f"Left Gripper: {state[6]:.3f}, Right Gripper: {state[13]:.3f}",
            f"Aux Points: {len(aux_pts)}, Scene Points: {len(scene_xyz)}",
            "Data: Raw Metric Space",
            "Aux: Red=Center, Green=Left, Blue=Right"
        ]
        
        y_offset = 80
        for k, text in enumerate(info_text):
            cv2.putText(combined, text, (20, y_offset + k*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 初始化video writer
        if save_video and video_writer is None:
            height, width = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = f"ghost_dataset_ep{episode_idx}.mp4"
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            print(f"Recording video to {filename}...")
        
        # 显示
        cv2.imshow("GHOST Dataset - Dynamic Visualization", combined)
        
        # 保存视频帧
        if video_writer is not None:
            video_writer.write(combined)
        
        # 等待按键
        key = cv2.waitKey(30)  # 30ms延迟
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("\nUser interrupted. Exiting...")
            break
    
    # 清理
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved successfully!")
    
    cv2.destroyAllWindows()
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GHOST Dataset with Aux Points Dynamically")
    parser.add_argument("--zarr_path", type=str, 
                        default="policy/VGC/data/stack_blocks_two-demo_3d_vision_hard-100-ppi.zarr",
                        help="Path to zarr dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--save_video", type=bool, default=True, help="Save as MP4 video")
    
    args = parser.parse_args()
    
    visualize_dataset_dynamic(args.zarr_path, args.episode, args.save_video)
