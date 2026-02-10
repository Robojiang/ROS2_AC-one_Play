import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import copy

# Add DP3 path to import base modules
current_file_path = os.path.abspath(__file__)
ghost_policy_dir = os.path.dirname(current_file_path) # policy/ghost
policy_dir = os.path.dirname(ghost_policy_dir) # policy
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZRGB, PointNetEncoderXYZ
try:
    from diffusion_policy_3d.model.vision.pointnet2_clean import PointNet2Encoder
except ImportError:
    PointNet2Encoder = None
    
from diffusion_policy_3d.common.model_util import print_params
from termcolor import cprint

class GHOSTPolicy(BasePolicy):
    def __init__(self, 
                 shape_meta: dict,
                 noise_scheduler,
                 horizon, 
                 n_action_steps, 
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 use_pc_color=False,
                 pointnet_type="pointnet",
                 pointcloud_encoder_cfg=None,
                 use_aux_points=True,
                 aux_point_num=50, # Number of points to generate per gripper
                 aux_length=0.2,   # Length of the laser beam
                 aux_radius=0.01,  # Radius of the cylinder
                 aux_trident_side_len=0.15,
                 aux_trident_max_width=0.08,
                 **kwargs):
        super().__init__()
        
        self.condition_type = condition_type
        self.use_pc_color = use_pc_color
        self.use_aux_points = use_aux_points
        self.aux_point_num = aux_point_num
        self.aux_length = aux_length
        self.aux_radius = aux_radius
        self.aux_trident_side_len = aux_trident_side_len
        self.aux_trident_max_width = aux_trident_max_width
        
        # 1. Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta['obs']
        self.state_dim = obs_shape_meta['agent_pos']['shape'][0]
        
        # 2. Configure PointNet Encoder
        base_channels = 6 if use_pc_color else 3
        in_channels = base_channels + 1 if use_aux_points else base_channels
        
        cprint(f"[GHOSTPolicy] Input Channels: {in_channels} (Color: {use_pc_color}, Aux: {use_aux_points})", "cyan")
        
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {}
        
        # Override in_channels
        enc_cfg = copy.deepcopy(pointcloud_encoder_cfg)
        enc_cfg['in_channels'] = in_channels
        
        if pointnet_type == "pointnet":
            # We use PointNetEncoderXYZRGB generically as it allows setting in_channels
            self.obs_encoder = PointNetEncoderXYZRGB(**enc_cfg)
        elif pointnet_type == "pointnet++":
            if PointNet2Encoder is None:
                raise ImportError("PointNet2Encoder not found. Please check diffusion_policy_3d/model/vision/pointnet2_clean.py")
            cprint("[GHOSTPolicy] Using PointNet++ Encoder", "yellow")
            self.obs_encoder = PointNet2Encoder(**enc_cfg)
        else:
            raise ValueError(f"Unknown pointnet_type: {pointnet_type}")
        
        # Output dim of encoder
        # PointNetEncoderXYZRGB output dim is defined by 'out_channels' in cfg or default 1024
        self.obs_feature_dim = enc_cfg.get('out_channels', 1024)
        
        # 3. Diffusion Model
        # Input to diffusion is action + condition (if not global)
        input_dim = self.action_dim
        global_cond_dim = None
        
        # Add explicit Proprioception Embedding (MLP)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.Mish(),
            nn.Linear(256, self.obs_feature_dim)
        )
        
        combined_feature_dim = self.obs_feature_dim * 2 # PointNet + Proprio
        
        if obs_as_global_cond:
            # We treat the extracted point features as global condition
            global_cond_dim = combined_feature_dim
            if not "cross_attention" in condition_type:
                 global_cond_dim *= n_obs_steps
        else:
            input_dim += combined_feature_dim
            
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )
        
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        
        print_params(self)
        
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _rot6d_to_mat(self, d6):
        """
        Converts 6D rotation representation to 3x3 rotation matrix.
        d6: (..., 6)
        Returns: (..., 3, 3)
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)
    
    def _generate_aux_points(self, agent_pos):
        """
        Generate auxiliary points (Trident) for the gripper based on agent_pos.
        Trident: Center (Red), Left Finger (Green), Right Finger (Blue).
        agent_pos: (B, T, D_state)
        Returns: 
           points: (B, T, num_aux, 3) XYZ coordinates
        """
        B, T, D = agent_pos.shape
        states_list = [] # List of tuples (pos, rot6d, gripper_width)
        
        # Parse Agent Pos (VGC Format: 32D = 14 Joint + 9 Left + 9 Right)
        if D == 32:
             # Left Hand
             # Joint State: 0-14. Left indices usually 0-6 joints, 6 gripper.
             left_gripper_val = agent_pos[..., 6].clip(0, 1) # (B, T)
             left_pos = agent_pos[..., 14:17] # (B, T, 3)
             left_rot6d = agent_pos[..., 17:23] # (B, T, 6)
             states_list.append((left_pos, left_rot6d, left_gripper_val))
             
             # Right Hand
             # Right indices usually 7-13 joints, 13 gripper.
             right_gripper_val = agent_pos[..., 13].clip(0, 1) # (B, T)
             right_pos = agent_pos[..., 23:26]
             right_rot6d = agent_pos[..., 26:32]
             states_list.append((right_pos, right_rot6d, right_gripper_val))
        else:
             # Fallback
             if D >= 9:
                 pos = agent_pos[..., -9:-6]
                 rot = agent_pos[..., -6:]
                 # Default open gripper (1.0)
                 gripper = torch.ones((B, T), device=agent_pos.device, dtype=agent_pos.dtype)
                 states_list.append((pos, rot, gripper))

        if not states_list:
            return torch.zeros((B, T, 0, 3), device=agent_pos.device)

        output_pcs = []
        
        # --- Generate Local Trident Geometry ---
        # 1. Center Prong - Along X axis
        n_center = int(self.aux_point_num * 0.5)
        n_side = (self.aux_point_num - n_center) // 2
        
        dists_c = torch.linspace(0, self.aux_length, n_center, device=agent_pos.device, dtype=agent_pos.dtype)
        # Using cylinder noise for center too?
        # Let's keep it simple: Line + Noise
        pts_c = torch.stack([dists_c, torch.zeros_like(dists_c), torch.zeros_like(dists_c)], dim=-1) # (N, 3)
        # Add slight cylinder noise
        angle_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        rad_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * self.aux_radius
        pts_c += torch.stack([torch.zeros_like(dists_c), rad_c*torch.cos(angle_c), rad_c*torch.sin(angle_c)], dim=-1)
        
        # Side geometry foundation (Lines along X)
        dists_s = torch.linspace(0, self.aux_trident_side_len, n_side, device=agent_pos.device, dtype=agent_pos.dtype)
        base_s = torch.stack([dists_s, torch.zeros_like(dists_s), torch.zeros_like(dists_s)], dim=-1) # (N, 3)
        # Noise for sides
        angle_s = torch.rand(n_side, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        rad_s = torch.rand(n_side, device=agent_pos.device, dtype=agent_pos.dtype) * self.aux_radius
        noise_s = torch.stack([torch.zeros_like(dists_s), rad_s*torch.cos(angle_s), rad_s*torch.sin(angle_s)], dim=-1)
        base_s_noisy = base_s + noise_s
        
        # Repeat for all B, T
        pts_c_batch = repeat(pts_c, 'n c -> (b t) n c', b=B, t=T)
        base_s_batch = repeat(base_s_noisy, 'n c -> (b t) n c', b=B, t=T)
        
        for pos, rot6d, gripper in states_list:
            # pos: (B, T, 3)
            # gripper: (B, T)
            
            pos_flat = rearrange(pos, 'b t c -> (b t) c')
            rot6d_flat = rearrange(rot6d, 'b t c -> (b t) c')
            gripper_flat = rearrange(gripper, 'b t -> (b t)')
            
            rot_mat = self._rot6d_to_mat(rot6d_flat) # (BT, 3, 3)

            # Apply dynamic width
            offset_y = gripper_flat * self.aux_trident_max_width # (BT,)
            
            pts_l = base_s_batch.clone()
            pts_l[:, :, 1] += offset_y.unsqueeze(1)  # (BT, n_side, 1)
            
            pts_r = base_s_batch.clone()
            pts_r[:, :, 1] -= offset_y.unsqueeze(1)
            
            # Combine Local
            tool_local = torch.cat([pts_c_batch, pts_l, pts_r], dim=1) # (BT, N, 3)
            
            # Transform to Global
            # rot_mat: (BT, 3, 3) where rows are basis vectors (u, v, w)
            # tool_local: (BT, N, 3)
            # We want x*u + y*v + z*w, which corresponds to tool_local @ rot_mat (if rows are u,v,w)
            # Previous incorrect: tool_local @ rot_mat.T
            tool_global = torch.bmm(tool_local, rot_mat) + pos_flat.unsqueeze(1)
            
            output_pcs.append(tool_global)
        
        all_aux_pts = torch.cat(output_pcs, dim=1) 
        
        all_aux_pts = rearrange(all_aux_pts, '(b t) n c -> b t n c', b=B, t=T)
        
        return all_aux_pts

    def forward(self, batch):
        # Normalize dict
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        # Prepare Input
        # Point Cloud: (B, T, N, 3 or 6)
        pc = nobs['point_cloud']
        # Agent Pos: (B, T, D)
        agent_pos = nobs['agent_pos']
        
        # Identify if we need to slice pc (Dataset returns 6D now, but we might only want 3D)
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        
        # 1. Generate Aux Points
        if self.use_aux_points:
            # Check if 'point_cloud' has a normalizer
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) 
            
            # Normalize Aux Points manually by slicing params (Efficient & Clean)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(aux_pts_xyz.device)
                offset = params['offset'].to(aux_pts_xyz.device)
                
                # Normalize XYZ using the first 3 dims
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            
            if self.use_pc_color:
                # Generate black color (0,0,0) and normalize it
                aux_pts_rgb = torch.zeros_like(aux_pts_xyz)
                if 'point_cloud' in self.normalizer.params_dict:
                     aux_pts_rgb = aux_pts_rgb * scale[3:] + offset[3:]
                aux_feats.append(aux_pts_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1) # (B, T, K, C)
            
            # 2. Add Indicator Channel
            scene_indicator = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            pc_with_ind = torch.cat([pc, scene_indicator], dim=-1)
            
            aux_indicator = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            aux_pc_with_ind = torch.cat([aux_pc, aux_indicator], dim=-1)
            
            # Concat
            full_pc = torch.cat([pc_with_ind, aux_pc_with_ind], dim=2) # (B, T, N+K, C+1)
            
        else:
             full_pc = pc
             
        # Flatten for processing
        full_pc_flat = rearrange(full_pc, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        
        # Reshape
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        # --- Proprioception Encoding ---
        # agent_pos: (B, T, D_state)
        # We need to make sure this is properly encoded
        proprio_features = self.proprio_mlp(agent_pos) # (B, T, D_emb)
        
        # Combine
        combined_features = torch.cat([point_features, proprio_features], dim=-1) # (B, T, 2*D_emb)

        # Select steps for conditioning
        if self.obs_as_global_cond:
             # Flatten T
             n_obs = self.n_obs_steps
             global_cond = combined_features[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        
        # Diffusion Loss
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device).long()
        
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        
        pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        
        loss = F.mse_loss(pred, noise)
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict

    def get_action(self, batch):
        # ... logic similar to forward but sampling ...
        # Normalize dict
        nobs = self.normalizer.normalize(batch['obs'])
        
        # Prepare Input
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        
        if self.use_aux_points:
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) 
            
            # Normalize Aux Points manually by slicing params
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(aux_pts_xyz.device)
                offset = params['offset'].to(aux_pts_xyz.device)
                
                # Normalize XYZ
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            
            if self.use_pc_color:
                # Generate black color (0,0,0) and normalize it
                aux_pts_rgb = torch.zeros_like(aux_pts_xyz)
                if 'point_cloud' in self.normalizer.params_dict:
                     aux_pts_rgb = aux_pts_rgb * scale[3:] + offset[3:]
                aux_feats.append(aux_pts_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1)
            
            scene_indicator = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            pc_with_ind = torch.cat([pc, scene_indicator], dim=-1)
            
            aux_indicator = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            aux_pc_with_ind = torch.cat([aux_pc, aux_indicator], dim=-1)
            
            full_pc = torch.cat([pc_with_ind, aux_pc_with_ind], dim=2) 
        else:
             full_pc = pc
             
        full_pc_flat = rearrange(full_pc, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        global_cond = combined_features
        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = global_cond[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')

        # Sampling
        naction = torch.randn((B, self.horizon, self.action_dim), device=pc.device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        action = self.normalizer['action'].unnormalize(naction)
        return action
