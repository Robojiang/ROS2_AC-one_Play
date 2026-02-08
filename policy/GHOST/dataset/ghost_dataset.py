import sys
import os
import torch
import numpy as np
import copy
import zarr
from typing import Dict

# Add DP3 path for utilities
current_file_path = os.path.abspath(__file__)
aim_dataset_dir = os.path.dirname(current_file_path) # policy/AIM/dataset
aim_dir = os.path.dirname(aim_dataset_dir) # policy/AIM
policy_dir = os.path.dirname(aim_dir) # policy

dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

def quat_to_rot6d(quat):
    """
    Convert quaternion to 6D rotation representation.
    quat: (..., 4) in [w, x, y, z] format (as stored in zarr)
    Returns: (..., 6) first two columns of rotation matrix
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    
    # First column of rotation matrix
    r11 = 1 - 2 * (yy + zz)
    r21 = 2 * (xy + zw)
    r31 = 2 * (xz - yw)
    
    # Second column of rotation matrix
    r12 = 2 * (xy - zw)
    r22 = 1 - 2 * (xx + zz)
    r32 = 2 * (yz + xw)
    
    rot6d = np.stack([r11, r21, r31, r12, r22, r32], axis=-1)
    return rot6d

class GHOSTDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        
        # Resolve zarr path (Assume relative to VGC root if not abs, similar to VGC dataset convention)
        if not os.path.isabs(zarr_path):
             # Try to find it relative to workspace root or typical paths
             # Check if it exists relative to current working directory
             if not os.path.exists(zarr_path):
                 # Check if it exists relative to VGC directory (policy/VGC)
                 vgc_zarr_path = os.path.join(policy_dir, 'VGC', zarr_path)
                 if os.path.exists(vgc_zarr_path):
                     zarr_path = vgc_zarr_path
                     print(f"[GHOSTDataset] Resolved path to VGC dir: {zarr_path}")
                 else:
                     print(f"[GHOSTDataset] Warning: Zarr path not found: {zarr_path} or {vgc_zarr_path}")
            
        print(f"[GHOSTDataset] Loading dataset from: {zarr_path}")
        
        # Load keys required for AIM
        # We need: state, action, point_cloud, endpose (for constructing agent_pos)
        keys_to_load = ["state", "action", "point_cloud", "left_endpose", "right_endpose"]
        
        # Copy from path
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, 
            keys=keys_to_load
        )
        
        
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # Construct data for normalization matching __getitem__ structure
        state = self.replay_buffer["state"]
        left_endpose = self.replay_buffer["left_endpose"]
        right_endpose = self.replay_buffer["right_endpose"]
        
        # Convert to 9D
        left_pos = left_endpose[..., :3]
        left_rot = quat_to_rot6d(left_endpose[..., 3:])
        left_9d = np.concatenate([left_pos, left_rot], axis=-1)
        
        right_pos = right_endpose[..., :3]
        right_rot = quat_to_rot6d(right_endpose[..., 3:])
        right_9d = np.concatenate([right_pos, right_rot], axis=-1)
        
        # Agent Pos: [Joint(14), Left(9), Right(9)] -> 32D
        # joint_action in VGC Dataset is "state" for robot joints?
        # Inspecting VGC code, "state" key usually holds joint positions.
        # Assuming state is (N, 14)
        agent_pos = np.concatenate([state, left_9d, right_9d], axis=-1)
        
        data = {
            'action': self.replay_buffer['action'],
            'point_cloud': self.replay_buffer['point_cloud'], # Normalize all channels (XYZ+RGB) like DP3
            'agent_pos': agent_pos
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # DP3 usually does NOT normalize point cloud via this Normalizer (it does it in Encoder centering)
        # But if we want to be safe, we can keep it.
        # However, checking DP3 BaseDataset, it usually doesn't normalize 'point_cloud' here.
        # Let's clean it up to avoid scaling PC if not expected.
        # But if we pass it here, it creates params.
        
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample
        
        # Process Agent Pos
        state = data['state'] # (T, 14)
        left_endpose = data['left_endpose'] # (T, 7)
        right_endpose = data['right_endpose'] # (T, 7)
        
        left_pos = left_endpose[..., :3]
        left_rot = quat_to_rot6d(left_endpose[..., 3:])
        left_9d = np.concatenate([left_pos, left_rot], axis=-1)
        
        right_pos = right_endpose[..., :3]
        right_rot = quat_to_rot6d(right_endpose[..., 3:])
        right_9d = np.concatenate([right_pos, right_rot], axis=-1)
        
        agent_pos = np.concatenate([state, left_9d, right_9d], axis=-1) # (T, 32)

        point_cloud = data['point_cloud'] # (T, N, 6) usually
        
        torch_data = {
            'obs': {
                'point_cloud': torch.from_numpy(point_cloud).float(),
                'agent_pos': torch.from_numpy(agent_pos).float(),
            },
            'action': torch.from_numpy(data['action']).float(),
        }
        return torch_data
