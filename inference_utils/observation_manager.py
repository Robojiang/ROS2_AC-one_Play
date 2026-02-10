"""
观测管理模块
负责维护观测历史缓冲区和动作队列
"""

import collections
import numpy as np
import torch


class ObservationManager:
    """观测管理器"""
    
    def __init__(self, policy_type, n_obs_steps, n_action_steps):
        """
        Args:
            policy_type: 策略类型 ('DP3' 或 'GHOST')
            n_obs_steps: 观测历史长度
            n_action_steps: 每次推理生成的动作数
        """
        self.policy_type = policy_type
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        # 观测历史缓冲区
        self.obs_buffer = collections.deque(maxlen=n_obs_steps)
        
        # 动作队列
        self.action_queue = collections.deque()
    
    def reset(self):
        """重置观测缓冲区和动作队列"""
        self.obs_buffer.clear()
        self.action_queue.clear()
        print(f"[INFO] 观测管理器已重置")
    
    def add_observation(self, obs_dict, point_cloud):
        """
        添加新观测到缓冲区
        
        Args:
            obs_dict: ROS观测字典 (包含 qpos, eef, images等)
            point_cloud: 生成的点云 (N, 6)
            
        Returns:
            curr_obs: 当前观测字典
        """
        curr_obs = {
            'point_cloud': point_cloud,
            'joint_action': {'vector': obs_dict['qpos']},
            'endpose': {
                'left_endpose': obs_dict['eef'][:7],
                'right_endpose': obs_dict['eef'][7:14]
            }
        }
        
        # DP3 需要 agent_pos
        if self.policy_type == 'DP3':
            curr_obs['agent_pos'] = obs_dict['qpos']
        
        self.obs_buffer.append(curr_obs)
        return curr_obs
    
    def is_ready_for_inference(self):
        """检查是否有足够的观测历史进行推理"""
        if self.policy_type == 'DP3':
            return len(self.obs_buffer) >= self.n_obs_steps
        else:  # GHOST 内部管理历史
            return len(self.obs_buffer) > 0
    
    def prepare_dp3_batch(self):
        """准备 DP3 推理的 batch 数据"""
        if not self.is_ready_for_inference():
            return None
        
        obs_list = list(self.obs_buffer)
        point_clouds = np.stack([o['point_cloud'] for o in obs_list], axis=0)  # (T, N, 6)
        agent_positions = np.stack([o['agent_pos'] for o in obs_list], axis=0)  # (T, 14)
        
        batch = {
            'obs': {
                'point_cloud': torch.from_numpy(point_clouds).float().cuda().unsqueeze(0),
                'agent_pos': torch.from_numpy(agent_positions).float().cuda().unsqueeze(0)
            }
        }
        return batch
    
    def prepare_ghost_input(self):
        """准备 GHOST 推理的输入数据"""
        if not self.is_ready_for_inference():
            return None
        
        curr_obs = self.obs_buffer[-1]  # GHOST 使用最新观测
        
        ghost_input = {
            'joint_action': curr_obs['joint_action'],
            'endpose': curr_obs['endpose'],
            'pointcloud': curr_obs['point_cloud']
        }
        return ghost_input
    
    def add_actions(self, actions):
        """
        添加新动作到队列
        
        Args:
            actions: numpy array (horizon, action_dim) 或 (action_dim,)
        """
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        n_steps = min(self.n_action_steps, len(actions))
        for i in range(n_steps):
            self.action_queue.append(actions[i])
        
        return n_steps
    
    def get_next_action(self):
        """
        从队列中取出下一个动作
        
        Returns:
            action: numpy array (action_dim,) 或 None
        """
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        return None
    
    def has_actions(self):
        """检查动作队列是否为空"""
        return len(self.action_queue) > 0
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'obs_buffer_size': len(self.obs_buffer),
            'action_queue_size': len(self.action_queue),
            'is_ready': self.is_ready_for_inference()
        }
