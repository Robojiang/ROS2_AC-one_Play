import sys
import os
import torch
import numpy as np
import zarr
import argparse

# Add paths
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path) # policy/GHOST/scripts
ghost_dir = os.path.dirname(scripts_dir) # policy/GHOST
policy_dir = os.path.dirname(ghost_dir) # policy
root_dir = os.path.dirname(policy_dir)

sys.path.append(ghost_dir)
sys.path.append(os.path.join(ghost_dir, 'dataset'))
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from dataset.ghost_keyframe_dataset import GHOSTKeyframeDataset

def check_gripper_values(zarr_path):
    print(f"Loading dataset from: {zarr_path}")
    
    # Open Zarr directly first for fast check
    root = zarr.open(zarr_path, mode='r')
    
    # Try to access state directly instead of agent_pos (which is a derived key in Dataset)
    if 'data/state' in root:
        state = root['data/state'][:]
        print(f"State Shape: {state.shape}")
        
        # Left Gripper: Index 6
        # Right Gripper: Index 13
        
        left_grip = state[:, 6]
        right_grip = state[:, 13]
    elif 'data/agent_pos' in root:
        agent_pos = root['data/agent_pos'][:]
        print(f"Agent Pos Shape: {agent_pos.shape}")
        left_grip = agent_pos[:, 6]
        right_grip = agent_pos[:, 13]
    else:
        print("Could not find 'data/state' or 'data/agent_pos' in Zarr group")
        print("Keys:", list(root['data'].keys()))
        return
        
    print("\n=== Left Gripper (Index 6) Stats ===")
    print(f"Min: {left_grip.min()}")
    print(f"Max: {left_grip.max()}")
    print(f"Mean: {left_grip.mean()}")
    print(f"Std: {left_grip.std()}")
    print(f"Unique values (first 10): {np.unique(left_grip)[:10]}")
    
    print("\n=== Right Gripper (Index 13) Stats ===")
    print(f"Min: {right_grip.min()}")
    print(f"Max: {right_grip.max()}")
    print(f"Mean: {right_grip.mean()}")
    print(f"Std: {right_grip.std()}")
    print(f"Unique values (first 10): {np.unique(right_grip)[:10]}")

    # Check Dataset __getitem__ output
    print("\n=== Checking Dataset Output (First 5 episodes) ===")
    dataset = GHOSTKeyframeDataset(
        zarr_path=zarr_path,
        horizon=16, 
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0
    )
    
    # Normalizer
    normalizer = dataset.get_normalizer()
    print("\n=== Normalizer Stats for agent_pos ===")
    if 'agent_pos' in normalizer.params_dict:
        params = normalizer.params_dict['agent_pos']
        scale = params['scale']
        offset = params['offset']
        if isinstance(scale, torch.Tensor):
            scale = scale.numpy()
            offset = offset.numpy()
            
        print(f"Left Gripper (Idx 6) - Scale: {scale.flatten()[6]}, Offset: {offset.flatten()[6]}")
        print(f"Right Gripper (Idx 13) - Scale: {scale.flatten()[13]}, Offset: {offset.flatten()[13]}")
    
    for i in range(0, 100, 20):
        try:
            sample = dataset[i]
            obs_agent_pos = sample['obs']['agent_pos'] # (T, D)
            # Take first step
            step_pos = obs_agent_pos[0]
            l_val = step_pos[6].item()
            r_val = step_pos[13].item()
            print(f"Sample {i}: Left={l_val:.4f}, Right={r_val:.4f}")
        except:
            break

if __name__ == "__main__":
    zarr_path = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/datasets_zarr/pick_place_d405.zarr"
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    
    check_gripper_values(zarr_path)
