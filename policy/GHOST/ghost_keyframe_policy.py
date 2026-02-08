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
import transforms3d

class GHOSTKeyframePolicy(BasePolicy):
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
                 use_keyframe_prediction=True,
                 keyframe_model_prob=0.0, # Probability to use predicted keyframe
                 keyframe_pred_loss_weight=1.0,
                 keyframe_noise_std=0.1, # Standard deviation of noise added to GT keyframe
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
        self.use_keyframe_prediction = use_keyframe_prediction
        self.keyframe_model_prob = keyframe_model_prob
        self.keyframe_pred_loss_weight = keyframe_pred_loss_weight
        self.keyframe_noise_std = keyframe_noise_std
        
        # 1. Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta['obs']
        self.state_dim = obs_shape_meta['agent_pos']['shape'][0]
        
        # 2. Configure PointNet Encoders
        base_channels = 6 if use_pc_color else 3
        # P1 Input: Base + Aux (Indicator=1D for Real vs Env) -> Total 7D (if base=6) or 4D
        p1_channels = base_channels + 1 if use_aux_points else base_channels
        
        # P2 Input: Base + Aux + Shadow
        # If we use 8D: [XYZ RGB RealInd ShadowInd]
        # Real Ind: 1 if real, 0 if env/shadow
        # Shadow Ind: 1 if shadow, 0 if env/real
        p2_channels = p1_channels + 1 if use_keyframe_prediction else p1_channels
        
        cprint(f"[GHOSTKeyframePolicy] P1 Channels: {p1_channels}, P2 Channels: {p2_channels}", "cyan")
        
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {}
        
        # Setup Encoders
        if PointNet2Encoder is None and pointnet_type == "pointnet++":
            raise ImportError("PointNet2Encoder not found")

        # --- P2 (Actor Encoder) ---
        enc_cfg_p2 = copy.deepcopy(pointcloud_encoder_cfg)
        enc_cfg_p2['in_channels'] = p2_channels
        
        
        self.obs_encoder = PointNetEncoderXYZRGB(**enc_cfg_p2)
        
            
        self.obs_feature_dim = enc_cfg_p2.get('out_channels', 1024)
            
        # --- P1 (Keyframe Predictor) ---
        if self.use_keyframe_prediction:
            enc_cfg_p1 = copy.deepcopy(pointcloud_encoder_cfg)
            enc_cfg_p1['in_channels'] = p1_channels # Standard 7D
            
            if pointnet_type == "pointnet":
                self.keyframe_encoder = PointNetEncoderXYZRGB(**enc_cfg_p1)
            elif pointnet_type == "pointnet++":
                self.keyframe_encoder = PointNet2Encoder(**enc_cfg_p1)
                
            self.keyframe_feature_dim = enc_cfg_p1.get('out_channels', 1024)
            # Head to predict 18D Pose (Left 9 + Right 9)
            self.keyframe_head = nn.Sequential(
                nn.Linear(self.keyframe_feature_dim, 256),
                nn.Mish(),
                nn.Linear(256, 18) # 18D Keypose
            )
            cprint("[GHOSTKeyframePolicy] Keyframe Prediction Module Enabled (P1)", "yellow")

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
    
    def _generate_trident_from_pose(self, pose_18d):
        """
        Generate auxiliary points (Trident) for the keyframe pose.
        pose_18d: (B, T, 18) [Left 9D, Right 9D] or (B, 18)
        Returns: 
           points: (B, T, num_aux, 3) XYZ coordinates
        """
        if pose_18d.dim() == 2:
            pose_18d = pose_18d.unsqueeze(1)
            
        B, T, _ = pose_18d.shape
        states_list = []
        
        # Left: 0-9
        left_pos = pose_18d[..., 0:3]
        left_rot = pose_18d[..., 3:9]
        left_grip = torch.ones((B, T), device=pose_18d.device) # Default open for shadow
        states_list.append((left_pos, left_rot, left_grip))
        
        # Right: 9-18
        right_pos = pose_18d[..., 9:12]
        right_rot = pose_18d[..., 12:18]
        right_grip = torch.ones((B, T), device=pose_18d.device)
        states_list.append((right_pos, right_rot, right_grip))
        
        output_pcs = []
        
        # --- Precompute Tool Geometry ---
        n_center = int(self.aux_point_num * 0.5)
        n_side = (self.aux_point_num - n_center) // 2
        
        dists_c = torch.linspace(0, self.aux_length, n_center, device=pose_18d.device, dtype=pose_18d.dtype)
        pts_c = torch.stack([dists_c, torch.zeros_like(dists_c), torch.zeros_like(dists_c)], dim=-1)
        angle_c = torch.rand(n_center, device=pose_18d.device, dtype=pose_18d.dtype) * 2 * np.pi
        rad_c = torch.rand(n_center, device=pose_18d.device, dtype=pose_18d.dtype) * self.aux_radius
        pts_c += torch.stack([torch.zeros_like(dists_c), rad_c*torch.cos(angle_c), rad_c*torch.sin(angle_c)], dim=-1)
        
        dists_s = torch.linspace(0, self.aux_trident_side_len, n_side, device=pose_18d.device, dtype=pose_18d.dtype)
        base_s = torch.stack([dists_s, torch.zeros_like(dists_s), torch.zeros_like(dists_s)], dim=-1)
        angle_s = torch.rand(n_side, device=pose_18d.device, dtype=pose_18d.dtype) * 2 * np.pi
        rad_s = torch.rand(n_side, device=pose_18d.device, dtype=pose_18d.dtype) * self.aux_radius
        noise_s = torch.stack([torch.zeros_like(dists_s), rad_s*torch.cos(angle_s), rad_s*torch.sin(angle_s)], dim=-1)
        base_s_noisy = base_s + noise_s
        
        pts_c_batch = repeat(pts_c, 'n c -> (b t) n c', b=B, t=T)
        base_s_batch = repeat(base_s_noisy, 'n c -> (b t) n c', b=B, t=T)
        
        for pos, rot6d, gripper in states_list:
            pos_flat = rearrange(pos, 'b t c -> (b t) c')
            rot6d_flat = rearrange(rot6d, 'b t c -> (b t) c')
            gripper_flat = rearrange(gripper, 'b t -> (b t)')
            
            rot_mat = self._rot6d_to_mat(rot6d_flat)
            
            offset_y = gripper_flat * self.aux_trident_max_width
            
            pts_l = base_s_batch.clone()
            pts_l[:, :, 1] += offset_y.unsqueeze(1)
            pts_r = base_s_batch.clone()
            pts_r[:, :, 1] -= offset_y.unsqueeze(1)
            
            tool_local = torch.cat([pts_c_batch, pts_l, pts_r], dim=1)
            tool_global = torch.bmm(tool_local, rot_mat) + pos_flat.unsqueeze(1)
            output_pcs.append(tool_global)
            
        all_aux_pts = torch.cat(output_pcs, dim=1)
        all_aux_pts = rearrange(all_aux_pts, '(b t) n c -> b t n c', b=B, t=T)
        return all_aux_pts

    def _generate_aux_points(self, agent_pos):
        B, T, D = agent_pos.shape
        states_list = [] # List of tuples (pos, rot6d, gripper_width)
        
        if D == 32:
             left_gripper_val = agent_pos[..., 6].clip(0, 1) # (B, T)
             left_pos = agent_pos[..., 14:17] # (B, T, 3)
             left_rot6d = agent_pos[..., 17:23] # (B, T, 6)
             states_list.append((left_pos, left_rot6d, left_gripper_val))
             
             right_gripper_val = agent_pos[..., 13].clip(0, 1) # (B, T)
             right_pos = agent_pos[..., 23:26]
             right_rot6d = agent_pos[..., 26:32]
             states_list.append((right_pos, right_rot6d, right_gripper_val))
        else:
             if D >= 9:
                 pos = agent_pos[..., -9:-6]
                 rot = agent_pos[..., -6:]
                 gripper = torch.ones((B, T), device=agent_pos.device, dtype=agent_pos.dtype)
                 states_list.append((pos, rot, gripper))

        if not states_list:
            return torch.zeros((B, T, 0, 3), device=agent_pos.device)

        output_pcs = []
        n_center = int(self.aux_point_num * 0.5)
        
        dists_c = torch.linspace(0, self.aux_length, n_center, device=agent_pos.device, dtype=agent_pos.dtype)
        pts_c = torch.stack([dists_c, torch.zeros_like(dists_c), torch.zeros_like(dists_c)], dim=-1) # (N, 3)
        angle_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        rad_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * self.aux_radius
        pts_c += torch.stack([torch.zeros_like(dists_c), rad_c*torch.cos(angle_c), rad_c*torch.sin(angle_c)], dim=-1)
        
        n_side = (self.aux_point_num - n_center) // 2
        dists_s = torch.linspace(0, self.aux_trident_side_len, n_side, device=agent_pos.device, dtype=agent_pos.dtype)
        base_s = torch.stack([dists_s, torch.zeros_like(dists_s), torch.zeros_like(dists_s)], dim=-1) # (N, 3)
        angle_s = torch.rand(n_side, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        rad_s = torch.rand(n_side, device=agent_pos.device, dtype=agent_pos.dtype) * self.aux_radius
        noise_s = torch.stack([torch.zeros_like(dists_s), rad_s*torch.cos(angle_s), rad_s*torch.sin(angle_s)], dim=-1)
        base_s_noisy = base_s + noise_s
        
        pts_c_batch = repeat(pts_c, 'n c -> (b t) n c', b=B, t=T)
        base_s_batch = repeat(base_s_noisy, 'n c -> (b t) n c', b=B, t=T)
        
        for pos, rot6d, gripper in states_list:
            pos_flat = rearrange(pos, 'b t c -> (b t) c')
            rot6d_flat = rearrange(rot6d, 'b t c -> (b t) c')
            gripper_flat = rearrange(gripper, 'b t -> (b t)')
            
            rot_mat = self._rot6d_to_mat(rot6d_flat) # (BT, 3, 3)
            offset_y = gripper_flat * self.aux_trident_max_width # (BT,)
            
            pts_l = base_s_batch.clone()
            pts_l[:, :, 1] += offset_y.unsqueeze(1)  # (BT, n_side, 1)
            pts_r = base_s_batch.clone()
            pts_r[:, :, 1] -= offset_y.unsqueeze(1)
            
            tool_local = torch.cat([pts_c_batch, pts_l, pts_r], dim=1) # (BT, N, 3)
            tool_global = torch.bmm(tool_local, rot_mat) + pos_flat.unsqueeze(1)
            output_pcs.append(tool_global)
        
        all_aux_pts = torch.cat(output_pcs, dim=1) 
        all_aux_pts = rearrange(all_aux_pts, '(b t) n c -> b t n c', b=B, t=T)
        return all_aux_pts

    def forward(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        loss_dict = {}
        
        # --- Step 1: Generate Standard Aux Points (Real Trident) ---
        if self.use_aux_points:
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) 
            
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(aux_pts_xyz.device)
                offset = params['offset'].to(aux_pts_xyz.device)
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue)
                # aux_pts_xyz is (B, T, 2*K, 3)
                # First K points are Left, Last K points are Right
                K = aux_pts_xyz.shape[2] // 2
                
                # Create raw colors [0, 1]
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                
                aux_pts_rgb = torch.cat([
                    cols_left.expand(B, T, K, 3),
                    cols_right.expand(B, T, K, 3)
                ], dim=2)
                
                # Normalize colors to match scene statistics if normalizer exists
                if 'point_cloud' in self.normalizer.params_dict:
                     aux_pts_rgb = aux_pts_rgb * scale[3:] + offset[3:]
                aux_feats.append(aux_pts_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1) # (B, T, K, C)
            
            p1_ind_env = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            p1_ind_real = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            
            pc_p1 = torch.cat([pc, p1_ind_env], dim=-1)
            aux_p1 = torch.cat([aux_pc, p1_ind_real], dim=-1)
            
            full_pc_p1 = torch.cat([pc_p1, aux_p1], dim=2) # 7D
        else:
            full_pc_p1 = pc
            aux_pc = None

        # --- Step 2: Keyframe Prediction (P1) ---
        full_pc_p2 = full_pc_p1 # Default
        loss_p1 = 0.0
        
        if self.use_keyframe_prediction:
            # Run P1
            flat_p1_input = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_feats = self.keyframe_encoder(flat_p1_input)
            pred_keypose_norm = self.keyframe_head(p1_feats)
            pred_keypose_norm = rearrange(pred_keypose_norm, '(b t) d -> b t d', b=B, t=T)
            
            # Loss for P1
            target_keypose_norm = nobs['target_keypose']
            loss_p1 = F.mse_loss(pred_keypose_norm, target_keypose_norm)
            loss_dict['loss_keyframe'] = loss_p1.item()
            
            # --- Step 3: Virtual Shadow Generation ---
            use_pred = False
            if self.training:
                if torch.rand(1).item() < self.keyframe_model_prob:
                    use_pred = True
            else:
                use_pred = True 
            
            if use_pred:
                shadow_pose_norm = pred_keypose_norm.detach() 
            else:
                shadow_pose_norm = target_keypose_norm
                
            # Add noise to keyframe (whether predicted or GT) during training
            # This helps the policy be robust to keyframe prediction errors (or prevent overfitting to perfect predictions)
            if self.training and self.keyframe_noise_std > 0:
                noise = torch.randn_like(shadow_pose_norm) * self.keyframe_noise_std
                shadow_pose_norm = shadow_pose_norm + noise
            
            shadow_pose_raw = self.normalizer['target_keypose'].unnormalize(shadow_pose_norm)
            
            shadow_xyz = self._generate_trident_from_pose(shadow_pose_raw)
            
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(shadow_xyz.device)
                offset = params['offset'].to(shadow_xyz.device)
                shadow_xyz = shadow_xyz * scale[:3] + offset[:3]
            
            shadow_feats = [shadow_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue) for Shadow (Same as Real Trident)
                K_shadow = shadow_xyz.shape[2] // 2
                
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=shadow_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=shadow_xyz.device).view(1, 1, 1, 3)
                
                shadow_rgb = torch.cat([
                    cols_left.expand(B, T, K_shadow, 3),
                    cols_right.expand(B, T, K_shadow, 3)
                ], dim=2)

                if 'point_cloud' in self.normalizer.params_dict:
                     shadow_rgb = shadow_rgb * scale[3:] + offset[3:]
                shadow_feats.append(shadow_rgb)
            shadow_pc = torch.cat(shadow_feats, dim=-1)
            
            # --- Step 4: Construct P2 Input (8D) ---
            # Env: [..., 0, 0], Real: [..., 1, 0], Shadow: [..., 0, 1]
            p2_zeros_col = torch.zeros(full_pc_p1.shape[:-1] + (1,), device=full_pc_p1.device)
            full_pc_p2_base = torch.cat([full_pc_p1, p2_zeros_col], dim=-1)
            
            shadow_ind_real = torch.zeros((B, T, shadow_pc.shape[2], 1), device=shadow_pc.device)
            shadow_ind_shadow = torch.ones((B, T, shadow_pc.shape[2], 1), device=shadow_pc.device)
            full_shadow_pc = torch.cat([shadow_pc, shadow_ind_real, shadow_ind_shadow], dim=-1)
            
            full_pc_p2 = torch.cat([full_pc_p2_base, full_shadow_pc], dim=2)
            
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = combined_features[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device).long()
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        
        diff_loss = F.mse_loss(pred, noise)
        loss_dict['loss_diffusion'] = diff_loss.item()
        
        final_loss = diff_loss
        if self.use_keyframe_prediction:
            final_loss += self.keyframe_pred_loss_weight * loss_p1
        
        loss_dict['loss'] = final_loss.item()
        return final_loss, loss_dict

    def get_action(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        
        if self.use_aux_points:
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) 
            
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(aux_pts_xyz.device)
                offset = params['offset'].to(aux_pts_xyz.device)
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue)
                # aux_pts_xyz is (B, T, 2*K, 3)
                # First K points are Left, Last K points are Right
                K = aux_pts_xyz.shape[2] // 2
                
                # Create raw colors [0, 1]
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                
                aux_pts_rgb = torch.cat([
                    cols_left.expand(B, T, K, 3),
                    cols_right.expand(B, T, K, 3)
                ], dim=2)
                
                if 'point_cloud' in self.normalizer.params_dict:
                     aux_pts_rgb = aux_pts_rgb * scale[3:] + offset[3:]
                aux_feats.append(aux_pts_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1)
            
            p1_ind_env = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            p1_ind_real = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            
            pc_p1 = torch.cat([pc, p1_ind_env], dim=-1)
            aux_p1 = torch.cat([aux_pc, p1_ind_real], dim=-1)
            
            full_pc_p1 = torch.cat([pc_p1, aux_p1], dim=2)
        else:
             full_pc_p1 = pc
             aux_pc = None

        full_pc_p2 = full_pc_p1
        
        if self.use_keyframe_prediction:
            flat_p1_input = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_feats = self.keyframe_encoder(flat_p1_input)
            pred_keypose_norm = self.keyframe_head(p1_feats)
            pred_keypose_norm = rearrange(pred_keypose_norm, '(b t) d -> b t d', b=B, t=T)
            
            shadow_pose_norm = pred_keypose_norm
            shadow_pose_raw = self.normalizer['target_keypose'].unnormalize(shadow_pose_norm)
            
            shadow_xyz = self._generate_trident_from_pose(shadow_pose_raw)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(shadow_xyz.device)
                offset = params['offset'].to(shadow_xyz.device)
                shadow_xyz = shadow_xyz * scale[:3] + offset[:3]
            
            shadow_feats = [shadow_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue) for Shadow (Same as Real Trident)
                K_shadow = shadow_xyz.shape[2] // 2
                
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=shadow_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=shadow_xyz.device).view(1, 1, 1, 3)
                
                shadow_rgb = torch.cat([
                    cols_left.expand(B, T, K_shadow, 3),
                    cols_right.expand(B, T, K_shadow, 3)
                ], dim=2)

                if 'point_cloud' in self.normalizer.params_dict:
                     shadow_rgb = shadow_rgb * scale[3:] + offset[3:]
                shadow_feats.append(shadow_rgb)
            shadow_pc = torch.cat(shadow_feats, dim=-1)
            
            p2_zeros_col = torch.zeros(full_pc_p1.shape[:-1] + (1,), device=full_pc_p1.device)
            full_pc_p2_base = torch.cat([full_pc_p1, p2_zeros_col], dim=-1)
            
            shadow_ind_real = torch.zeros((B, T, shadow_pc.shape[2], 1), device=shadow_pc.device)
            shadow_ind_shadow = torch.ones((B, T, shadow_pc.shape[2], 1), device=shadow_pc.device)
            full_shadow_pc = torch.cat([shadow_pc, shadow_ind_real, shadow_ind_shadow], dim=-1)
            
            full_pc_p2 = torch.cat([full_pc_p2_base, full_shadow_pc], dim=2)
             
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        global_cond = combined_features
        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = global_cond[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')

        naction = torch.randn((B, self.horizon, self.action_dim), device=pc.device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        action = self.normalizer['action'].unnormalize(naction)
        return action
