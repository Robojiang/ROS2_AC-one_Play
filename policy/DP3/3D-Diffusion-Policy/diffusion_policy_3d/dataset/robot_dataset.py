import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb


class RobotDataset(BaseDataset):

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
        
        # 智能路径解析逻辑 (从 GHOST 移植)
        # 1. 如果是绝对路径，直接使用
        if os.path.isabs(zarr_path):
            pass
        else:
            # 2. 尝试相对于当前脚本目录 (原逻辑)
            current_file_path = os.path.abspath(__file__)
            dataset_dir = os.path.dirname(current_file_path)
            candidate_1 = os.path.join(dataset_dir, zarr_path)
            
            # 3. 尝试相对于工作区根目录 (arx_data/ROS2_AC-one_Play)
            # 假设当前文件在 policy/DP3/.../dataset/robot_dataset.py
            # 往上走 5 层到达 arx_data/ROS2_AC-one_Play
            workspace_root = os.path.abspath(os.path.join(dataset_dir, "../../../../.."))
            candidate_2 = os.path.join(workspace_root, zarr_path)
            
            # 4. 尝试相对于 policy 目录
            candidate_3 = os.path.join(workspace_root, "policy", zarr_path)

            if os.path.exists(candidate_1):
                zarr_path = candidate_1
                print(f"[RobotDataset] Resolved path (dataset rel): {zarr_path}")
            elif os.path.exists(candidate_2):
                zarr_path = candidate_2
                print(f"[RobotDataset] Resolved path (workspace rel): {zarr_path}")
            elif os.path.exists(candidate_3):
                zarr_path = candidate_3
                print(f"[RobotDataset] Resolved path (policy rel): {zarr_path}")
            else:
                print(f"[RobotDataset] Warning: Zarr path not found in typical locations: {zarr_path}")
                # Fallback to original logic to show error later if needed, or keep relative
                # zarr_path = candidate_1 

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["state", "action", "point_cloud"])  # 'img'
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
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][
            :,
        ].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)  # (T, 1024, 6)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, 1024, 6
                "agent_pos": agent_pos,  # T, D_pos
            },
            "action": sample["action"].astype(np.float32),  # T, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
