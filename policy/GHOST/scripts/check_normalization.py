"""
检查Normalization的时机和影响
验证: 
1. agent_pos传入_generate_aux_points时是否已经normalized
2. 场景点云是否normalized
3. 虚拟点云的normalization策略是否正确
"""
import sys
import os
import torch
import numpy as np
import zarr
import argparse

# Add paths
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
ghost_dir = os.path.dirname(scripts_dir)
policy_dir = os.path.dirname(ghost_dir)

sys.path.append(ghost_dir)
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from dataset.ghost_dataset import GHOSTDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

def check_normalization(zarr_path):
    """
    检查数据在各个阶段的状态
    """
    print("="*80)
    print("Normalization Check")
    print("="*80)
    
    # 1. Load Dataset
    print("\n[1] Loading Dataset...")
    dataset = GHOSTDataset(
        zarr_path=zarr_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.0,
    )
    
    # Get one sample BEFORE normalization
    sample = dataset[0]
    
    print(f"Sample Keys: {sample.keys()}")
    print(f"  obs keys: {sample['obs'].keys()}")
    
    raw_agent_pos = sample['obs']['agent_pos'] # (T, 32)
    raw_pc = sample['obs']['point_cloud'] # (T, N, 6)
    raw_action = sample['action'] # (T, 14)
    
    print(f"\n[RAW DATA - Before Normalization]")
    print(f"  Agent Pos Shape: {raw_agent_pos.shape}")
    print(f"  Agent Pos Sample (t=0): {raw_agent_pos[0, :5]}...")
    print(f"  Agent Pos Range: [{raw_agent_pos.min():.3f}, {raw_agent_pos.max():.3f}]")
    print(f"  Left End Pos (indices 14:17): {raw_agent_pos[0, 14:17]}")
    print(f"  Right End Pos (indices 23:26): {raw_agent_pos[0, 23:26]}")
    print(f"")
    print(f"  Point Cloud Shape: {raw_pc.shape}")
    print(f"  Point Cloud XYZ Range: X[{raw_pc[..., 0].min():.3f}, {raw_pc[..., 0].max():.3f}], "
          f"Y[{raw_pc[..., 1].min():.3f}, {raw_pc[..., 1].max():.3f}], "
          f"Z[{raw_pc[..., 2].min():.3f}, {raw_pc[..., 2].max():.3f}]")
    print(f"  Point Cloud RGB Range: [{raw_pc[..., 3:].min():.3f}, {raw_pc[..., 3:].max():.3f}]")
    
    # 2. Get Normalizer
    print(f"\n[2] Creating Normalizer...")
    normalizer = dataset.get_normalizer(mode='limits')
    
    print(f"Normalizer has params for: {list(normalizer.params_dict.keys())}")
    
    if 'agent_pos' in normalizer.params_dict:
        agent_pos_stats = normalizer.params_dict['agent_pos']
        print(f"\n  Agent Pos Normalizer:")
        print(f"    Input Min: {agent_pos_stats['input_stats']['min'][:5]}...")
        print(f"    Input Max: {agent_pos_stats['input_stats']['max'][:5]}...")
        if 'output_stats' in agent_pos_stats:
            print(f"    Output Min: {agent_pos_stats['output_stats']['min']}")
            print(f"    Output Max: {agent_pos_stats['output_stats']['max']}")
        if 'scale' in agent_pos_stats:
            print(f"    Scale: {agent_pos_stats['scale'][:5]}...")
            print(f"    Offset: {agent_pos_stats['offset'][:5]}...")
        
    if 'point_cloud' in normalizer.params_dict:
        pc_stats = normalizer.params_dict['point_cloud']
        print(f"\n  Point Cloud Normalizer:")
        print(f"    Input Min: {pc_stats['input_stats']['min']}")
        print(f"    Input Max: {pc_stats['input_stats']['max']}")
        if 'output_stats' in pc_stats:
            print(f"    Output Min: {pc_stats['output_stats']['min']}")
            print(f"    Output Max: {pc_stats['output_stats']['max']}")
        if 'scale' in pc_stats:
            print(f"    Scale: {pc_stats['scale']}")
            print(f"    Offset: {pc_stats['offset']}")
    else:
        print(f"\n  Point Cloud: NOT NORMALIZED (no normalizer params)")
    
    if 'action' in normalizer.params_dict:
        action_stats = normalizer.params_dict['action']
        print(f"\n  Action Normalizer:")
        print(f"    Input Min: {action_stats['input_stats']['min'][:5]}...")
        print(f"    Input Max: {action_stats['input_stats']['max'][:5]}...")
    
    # 3. Apply Normalization
    print(f"\n[3] Applying Normalization (simulating forward pass)...")
    
    batch = {
        'obs': {
            'agent_pos': raw_agent_pos.unsqueeze(0), # (1, T, 32)
            'point_cloud': raw_pc.unsqueeze(0), # (1, T, N, 6)
        },
        'action': raw_action.unsqueeze(0) # (1, T, 14)
    }
    
    nobs = normalizer.normalize(batch['obs'])
    naction = normalizer['action'].normalize(batch['action'])
    
    norm_agent_pos = nobs['agent_pos']
    norm_pc = nobs['point_cloud']
    
    print(f"\n[NORMALIZED DATA]")
    print(f"  Normalized Agent Pos Range: [{norm_agent_pos.min():.3f}, {norm_agent_pos.max():.3f}]")
    print(f"  Normalized Left End Pos (14:17): {norm_agent_pos[0, 0, 14:17]}")
    print(f"  Normalized Right End Pos (23:26): {norm_agent_pos[0, 0, 23:26]}")
    print(f"")
    print(f"  Normalized PC XYZ Range: X[{norm_pc[..., 0].min():.3f}, {norm_pc[..., 0].max():.3f}], "
          f"Y[{norm_pc[..., 1].min():.3f}, {norm_pc[..., 1].max():.3f}], "
          f"Z[{norm_pc[..., 2].min():.3f}, {norm_pc[..., 2].max():.3f}]")
    print(f"  Normalized PC RGB Range: [{norm_pc[..., 3:].min():.3f}, {norm_pc[..., 3:].max():.3f}]")
    
    # 4. Check what's passed to _generate_aux_points
    print(f"\n[4] Checking _generate_aux_points Input...")
    print(f"  In forward(): uses batch['obs']['agent_pos'] (RAW, NOT NORMALIZED)")
    print(f"  This means aux points are generated in METRIC space")
    print(f"  Then aux points are normalized using normalizer['point_cloud']")
    
    # Simulate aux point generation
    print(f"\n[5] Simulating Aux Point Generation...")
    from ghost_policy import GHOSTPolicy
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    
    shape_meta = {
        'action': {'shape': [14]},
        'obs': {
            'point_cloud': {'shape': [1024, 6]},
            'agent_pos': {'shape': [32]}
        }
    }
    
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
    
    # Set normalizer
    policy.set_normalizer(normalizer)
    
    # Generate using RAW agent pos
    with torch.no_grad():
        aux_pts_raw, aux_cols = policy._generate_aux_points(batch['obs']['agent_pos'])
    
    print(f"\n  Aux Points (RAW, Metric Space):")
    print(f"    Shape: {aux_pts_raw.shape}")
    print(f"    Range: X[{aux_pts_raw[..., 0].min():.3f}, {aux_pts_raw[..., 0].max():.3f}], "
          f"Y[{aux_pts_raw[..., 1].min():.3f}, {aux_pts_raw[..., 1].max():.3f}], "
          f"Z[{aux_pts_raw[..., 2].min():.3f}, {aux_pts_raw[..., 2].max():.3f}]")
    
    # Check if aux points are near end effector positions
    left_end_raw = batch['obs']['agent_pos'][0, 0, 14:17]
    right_end_raw = batch['obs']['agent_pos'][0, 0, 23:26]
    
    print(f"\n  End Effector Positions (RAW):")
    print(f"    Left: {left_end_raw.numpy()}")
    print(f"    Right: {right_end_raw.numpy()}")
    
    # Check proximity
    aux_pts_flat = aux_pts_raw[0, 0].numpy() # (K, 3)
    left_distances = np.linalg.norm(aux_pts_flat - left_end_raw.numpy(), axis=1)
    right_distances = np.linalg.norm(aux_pts_flat - right_end_raw.numpy(), axis=1)
    
    print(f"\n  Aux Points Distance to Ends:")
    print(f"    Min Distance to Left End: {left_distances.min():.4f}")
    print(f"    Min Distance to Right End: {right_distances.min():.4f}")
    print(f"    => Aux points should be VERY CLOSE to end effectors (distance ~0)")
    
    # Now normalize aux points
    if 'point_cloud' in normalizer.params_dict:
        aux_pts_normalized = normalizer['point_cloud'].normalize(aux_pts_raw)
        print(f"\n  Aux Points (NORMALIZED):")
        print(f"    Range: X[{aux_pts_normalized[..., 0].min():.3f}, {aux_pts_normalized[..., 0].max():.3f}], "
              f"Y[{aux_pts_normalized[..., 1].min():.3f}, {aux_pts_normalized[..., 1].max():.3f}], "
              f"Z[{aux_pts_normalized[..., 2].min():.3f}, {aux_pts_normalized[..., 2].max():.3f}]")
    
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("1. agent_pos传入_generate_aux_points时是 RAW (未normalized)")
    print("2. 虚拟点云生成在 METRIC 空间 (真实世界坐标)")
    print("3. 虚拟点云随后被 normalizer['point_cloud'] 归一化")
    print("4. 场景点云在batch['obs']中被normalize")
    print("5. 两者在相同的归一化空间中concat,送入网络")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str,
                        default="policy/VGC/data/stack_blocks_two-demo_3d_vision_easy-100-ppi.zarr")
    args = parser.parse_args()
    
    check_normalization(args.zarr_path)
