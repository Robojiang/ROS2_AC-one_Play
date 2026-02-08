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

class GHOSTBeaconKeyPolicy(BasePolicy):
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
                 beacon_sigma=0.3, # Sigma for heatmap calculation (in normalized space or raw, depending on usage)
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
        self.beacon_sigma = beacon_sigma # Store beacon sigma
        self.num_inference_steps = num_inference_steps
        
        # 1. Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta['obs']
        self.state_dim = obs_shape_meta['agent_pos']['shape'][0]
        
        # 2. Configure PointNet Encoders
        base_channels = 6 if use_pc_color else 3
        # P1 Input: Base + Aux (Indicator=1D for Real vs Env) -> Total 7D (if base=6) or 4D
        p1_channels = base_channels + 1 if use_aux_points else base_channels
        
        # P2 Input: Base + Aux + Ghost + HeatL + HeatR
        # Base: XYZ (3) + RGB (3) = 6
        # Real Ind: 1 
        # Ghost Ind: 1
        # Heatmap Left: 1
        # Heatmap Right: 1
        # Total: 6 + 1 + 1 + 1 + 1 = 10
        p2_channels = p1_channels + 3 if use_keyframe_prediction else p1_channels
        
        cprint(f"[GHOSTBeaconKeyPolicy] P1 Channels: {p1_channels}, P2 Channels: {p2_channels}", "cyan")
        
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
            cprint("[GHOSTBeaconKeyPolicy] Keyframe Prediction Module Enabled (P1)", "yellow")

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
        elif D == 9: # Single arm
             gripper_val = agent_pos[..., 8].clip(0, 1)
             pos = agent_pos[..., 0:3]
             rot6d = agent_pos[..., 2:8] # Warning: Indices might be wrong for 9D
             # Assuming standard layout if 9D is used, but for now robust to 32D
             pass
        elif D >= 14: # Maybe 14D action?
             pass

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

    def _calculate_beacon_heatmap(self, point_cloud, pose_18d, sigma=0.3):
        """
        Calculate heatmap values for each point in point_cloud relative to ghost trident tip.
        Ghost Trident Tip is defined as position + aux_length along local X axis.
        
        Args:
            point_cloud: (B, T, N, 3)
            pose_18d: (B, T, 18) - 18D Pose of Ghost
            sigma: Standard deviation for Gaussian heatmap
        
        Returns:
            heat_l: (B, T, N, 1)
            heat_r: (B, T, N, 1)
        """
        B, T, N, _ = point_cloud.shape
        
        def get_single_arm_beacon(pos, rot6d):
            # pos: (B, T, 3)
            # rot6d: (B, T, 6)
            rot_mat = self._rot6d_to_mat(rot6d) # (B, T, 3, 3)

            # so here the local beacon should be [aux_length, 0, 0].
            beacon_local = torch.tensor([self.aux_length/2, 0.0, 0.0], device=pos.device, dtype=pos.dtype)
            beacon_local = beacon_local.view(1, 1, 3).expand(B, T, 3)

            # Apply rotation consistently with _generate_trident_from_pose which uses
            # row-vector multiplication: tool_global = tool_local @ rot_mat
            rot_mat_flat = rearrange(rot_mat, 'b t i j -> (b t) i j')
            beacon_local_row = rearrange(beacon_local, 'b t c -> (b t) 1 c')

            beacon_vec_global = torch.bmm(beacon_local_row, rot_mat_flat).squeeze(1) # (BT, 3)
            beacon_vec_global = rearrange(beacon_vec_global, '(b t) c -> b t c', b=B, t=T)

            beacon_pos = pos + beacon_vec_global
            return beacon_pos

        # Left Arm (0-9)
        l_pos = pose_18d[..., 0:3]
        l_rot = pose_18d[..., 3:9]
        beacon_l = get_single_arm_beacon(l_pos, l_rot) # (B, T, 3)
        
        # Right Arm (9-18)
        r_pos = pose_18d[..., 9:12]
        r_rot = pose_18d[..., 12:18]
        beacon_r = get_single_arm_beacon(r_pos, r_rot) # (B, T, 3)
        
        # Compute Distances & Heatmaps
        # point_cloud: (B, T, N, 3)
        # beacon: (B, T, 3) -> (B, T, 1, 3)
        
        dist_sq_l = torch.sum((point_cloud - beacon_l.unsqueeze(2))**2, dim=-1, keepdim=True) # (B, T, N, 1)
        dist_sq_r = torch.sum((point_cloud - beacon_r.unsqueeze(2))**2, dim=-1, keepdim=True) # (B, T, N, 1)
        
        heat_l = torch.exp(-dist_sq_l / (2 * sigma**2))
        heat_r = torch.exp(-dist_sq_r / (2 * sigma**2))
        
        return heat_l, heat_r, beacon_l, beacon_r

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
                
                # Apply XYZ normalization
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue) - Consistent with ghost_keyframe
                K = aux_pts_xyz.shape[2] // 2
                
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                
                aux_rgb = torch.cat([
                    cols_left.expand(B, T, K, 3),
                    cols_right.expand(B, T, K, 3)
                ], dim=2)
                
                # Normalize colors to match scene statistics
                if 'point_cloud' in self.normalizer.params_dict:
                    aux_rgb = aux_rgb * scale[3:6] + offset[3:6]
                aux_feats.append(aux_rgb)
            
            aux_pcd = torch.cat(aux_feats, dim=-1)
            
            # Construct Masks [RealMask]
            # Scene points: 0, Aux points: 1
            mask_scene = torch.zeros((B, T, N, 1), device=pc.device)
            mask_aux = torch.ones((B, T, aux_pts_xyz.shape[2], 1), device=pc.device)
            
            pc_with_mask = torch.cat([pc, mask_scene], dim=-1)
            aux_with_mask = torch.cat([aux_pcd, mask_aux], dim=-1)
            
            full_pc_p1 = torch.cat([pc_with_mask, aux_with_mask], dim=2) # (B, T, N+K, 7)
        else:
            # If no aux points, just add mask 0
            mask_scene = torch.zeros((B, T, N, 1), device=pc.device)
            full_pc_p1 = torch.cat([pc, mask_scene], dim=-1)

        # --- Step 2: Keyframe Prediction (P1) ---
        full_pc_p2 = None # Will construct new one
        loss_p1 = 0.0
        
        pred_keypose_norm = None
        target_keypose_norm = nobs['target_keypose']
        
        if self.use_keyframe_prediction:
            full_pc_flat = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_features = self.keyframe_encoder(full_pc_flat) # (BT, D)
            pred_keypose_norm = self.keyframe_head(p1_features) # (BT, 18)
            pred_keypose_norm = rearrange(pred_keypose_norm, '(b t) d -> b t d', b=B, t=T)
            
            loss_p1 = F.mse_loss(pred_keypose_norm, target_keypose_norm)
            loss_dict['loss_keyframe'] = loss_p1.item()
            
            # --- Construct P2 Input with Ghost Trident & Heatmaps ---
            
            # 1. Decide which pose to use for Ghost generation
            use_pred_prob = self.keyframe_model_prob
            # Scale prob by curriculum if needed (handled outside often, or here)
            # Assuming simple probability switch:
            use_pred = np.random.rand() < use_pred_prob
            
            if self.training:
                if use_pred:
                    pose_to_gen = pred_keypose_norm.detach() # Detach gradient? Usually yes for P2 input
                else:
                    pose_to_gen = target_keypose_norm
                   
            else:
                pose_to_gen = pred_keypose_norm # Inference always use prediction
            
             # Add noise to GT during training
            if self.training and self.keyframe_noise_std > 0:
                noise = torch.randn_like(pose_to_gen) * self.keyframe_noise_std
                pose_to_gen = pose_to_gen + noise
            
            # 2. Unnormalize Pose for Geometry Generation
            # We need physical coordinates to generate the shape
            pose_18d_raw = self.normalizer['target_keypose'].unnormalize(pose_to_gen)
            
            # 3. Generate Ghost Trident Points (Raw)
            ghost_pts_raw = self._generate_trident_from_pose(pose_18d_raw)
            
            # 4. Normalize Ghost Points (to match Scene/Real normalization)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(ghost_pts_raw.device)
                offset = params['offset'].to(ghost_pts_raw.device)
                ghost_pts_norm = ghost_pts_raw * scale[:3] + offset[:3]
            else:
                ghost_pts_norm = ghost_pts_raw
                
            # 5. Prepare Ghost Features (Colors & Masks)
            # Ghost Color: match real aux colors so network sees ghost as "future real"
            # Left: Red, Right: Blue (same convention as aux)
            K = ghost_pts_norm.shape[2] // 2
            cols_left = torch.tensor([1.0, 0.0, 0.0], device=ghost_pts_norm.device, dtype=ghost_pts_norm.dtype).view(1, 1, 1, 3)
            cols_right = torch.tensor([0.0, 0.0, 1.0], device=ghost_pts_norm.device, dtype=ghost_pts_norm.dtype).view(1, 1, 1, 3)
            ghost_rgb = torch.cat([
                cols_left.expand(B, T, K, 3),
                cols_right.expand(B, T, K, 3)
            ], dim=2)

            # Normalize colors to match scene statistics (if normalizer provided)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(ghost_rgb.device)
                offset = params['offset'].to(ghost_rgb.device)
                ghost_rgb = ghost_rgb * scale[3:6] + offset[3:6]
            
            ghost_pcd = torch.cat([ghost_pts_norm, ghost_rgb], dim=-1)
            
            # 6. Calculate Heatmaps
            # We calculate heatmap for: Scene + Real + Ghost
            # Combine XYZ first
            # full_pc_p1 coordinates are at index 0:3
            p1_xyz = full_pc_p1[..., 0:3]
            
            all_xyz = torch.cat([p1_xyz, ghost_pts_norm], dim=2) # (B, T, TotalPoints, 3)
            
            # Heatmap Beacon: We use the `pose_to_gen` (Normalized? No, Function expects 18D Pose for Geoshaping)
            # wait, `_calculate_beacon_heatmap` takes `pose_18d` and computes beacon position.
            # If `all_xyz` is NORMALIZED, we should calculate beacon position in NORMALIZED space or unnormalize points.
            # Usually Heatmap is calculated in Euclidean metric space (Raw) for physical meaning of Sigma.
            
            # Let's UNNORMALIZE points, compute heat, then use heat as feature.
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(all_xyz.device)
                offset = params['offset'].to(all_xyz.device)
                # Denormalize (unnormalize): raw = (normalized - offset) / scale
                all_xyz_raw = (all_xyz - offset[:3]) / scale[:3]
            else:
                all_xyz_raw = all_xyz
                
            heat_l, heat_r, beacon_l, beacon_r = self._calculate_beacon_heatmap(all_xyz_raw, pose_18d_raw, sigma=self.beacon_sigma)
            
            # 7. Assemble P2 Input
            # Structure: [XYZ(3), RGB(3), RealMask(1), GhostMask(1), HeatL(1), HeatR(1)]
            
            # P1 (Scene+Real) Features:
            # - XYZ(3): p1_xyz
            # - RGB(3): full_pc_p1[..., 3:6]
            # - RealMask(1): full_pc_p1[..., 6]
            # - GhostMask(1): 0 
            
            p1_feats = full_pc_p1
            # Append GhostMask=0 to P1
            mask_ghost_p1 = torch.zeros((B, T, p1_feats.shape[2], 1), device=pc.device)
            p1_block = torch.cat([p1_feats, mask_ghost_p1], dim=-1)
            
            # Ghost Features:
            # - XYZ(3): ghost_pts_norm
            # - RGB(3): ghost_rgb
            # - RealMask(1): 0
            # - GhostMask(1): 1
            
            mask_real_ghost = torch.zeros((B, T, ghost_pts_norm.shape[2], 1), device=pc.device)
            mask_ghost_ghost = torch.ones((B, T, ghost_pts_norm.shape[2], 1), device=pc.device)
            
            ghost_block = torch.cat([ghost_pcd, mask_real_ghost, mask_ghost_ghost], dim=-1)
            
            # Concatenate blocks
            full_block = torch.cat([p1_block, ghost_block], dim=2)
            
            # Append Heatmaps
            full_pc_p2 = torch.cat([full_block, heat_l, heat_r], dim=-1) # (B, T, N_total, 10)
            
        else:
             # Fallback if Keyframe Pred disabled (should not happen for this policy class usually)
             full_pc_p2 = full_pc_p1 # Incorrect dimensions if P2 expects 10 channels. Handle appropriately if logic allows.
             # Just padding zeros for missing channels?
             pad = torch.zeros((B, T, full_pc_p1.shape[2], 3), device=pc.device)
             full_pc_p2 = torch.cat([full_pc_p1, pad], dim=-1)

        # --- Step 3: Action Prediction (P2) ---
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        if self.obs_as_global_cond:
            n_obs = self.n_obs_steps
            global_cond = combined_features[:, :n_obs, :]
            if not "cross_attention" in self.condition_type:
                global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        else:
             # Just pass None, handled in input_dim
             global_cond = None
             # But we need to concat to input? Not typical for obs_as_global_cond=False in this codebase?
             # Base implementation usually assumes global_cond if separated.
             pass
        
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device).long()
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        
        pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        
        diff_loss = F.mse_loss(pred, noise)

        # Combine diffusion loss and keyframe loss (if present) into final loss
        final_loss = diff_loss
        if self.use_keyframe_prediction:
            final_loss = final_loss + self.keyframe_pred_loss_weight * loss_p1

        # Build loss dict for logging (scalar values)
        loss_dict = {
            'loss_diffusion': diff_loss.item(),
            'loss_keyframe': loss_p1.item() if isinstance(loss_p1, torch.Tensor) else float(loss_p1),
            'loss': final_loss.item()
        }

        return final_loss, loss_dict

    def get_action(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        
        # --- Step 1: P1 Input --- 
        if self.use_aux_points:
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos)
            
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(aux_pts_xyz.device)
                offset = params['offset'].to(aux_pts_xyz.device)
                
                # Apply XYZ normalization
                aux_pts_xyz = aux_pts_xyz * scale[:3] + offset[:3]
                
            aux_feats = [aux_pts_xyz]
            if self.use_pc_color:
                # Colorize Left (Red) and Right (Blue) - Consistent with ghost_keyframe
                K = aux_pts_xyz.shape[2] // 2
                
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                
                aux_rgb = torch.cat([
                    cols_left.expand(B, T, K, 3),
                    cols_right.expand(B, T, K, 3)
                ], dim=2)
                
                # Normalize colors to match scene statistics
                if 'point_cloud' in self.normalizer.params_dict:
                    aux_rgb = aux_rgb * scale[3:6] + offset[3:6]
                aux_feats.append(aux_rgb)
            
            aux_pcd = torch.cat(aux_feats, dim=-1)
            
            mask_scene = torch.zeros((B, T, N, 1), device=pc.device)
            mask_aux = torch.ones((B, T, aux_pts_xyz.shape[2], 1), device=pc.device)
            
            pc_with_mask = torch.cat([pc, mask_scene], dim=-1)
            aux_with_mask = torch.cat([aux_pcd, mask_aux], dim=-1)
            
            full_pc_p1 = torch.cat([pc_with_mask, aux_with_mask], dim=2)
        else:
             mask_scene = torch.zeros((B, T, N, 1), device=pc.device)
             full_pc_p1 = torch.cat([pc, mask_scene], dim=-1)

        # --- Step 2: Prediction ---
        if self.use_keyframe_prediction:
            full_pc_flat = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_features = self.keyframe_encoder(full_pc_flat)
            pred_keypose_norm = self.keyframe_head(p1_features)
            pred_keypose_norm = rearrange(pred_keypose_norm, '(b t) d -> b t d', b=B, t=T)
            
            # Predict only -> No GT mixing
            pose_to_gen = pred_keypose_norm
            pose_18d_raw = self.normalizer['target_keypose'].unnormalize(pose_to_gen)
            
            # Generate Ghost
            ghost_pts_raw = self._generate_trident_from_pose(pose_18d_raw)
            
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(ghost_pts_raw.device)
                offset = params['offset'].to(ghost_pts_raw.device)
                ghost_pts_norm = ghost_pts_raw * scale[:3] + offset[:3]
            else:
                ghost_pts_norm = ghost_pts_raw
                
            # Make ghost color match real aux: left red, right blue
            K = ghost_pts_norm.shape[2] // 2
            cols_left = torch.tensor([1.0, 0.0, 0.0], device=ghost_pts_norm.device, dtype=ghost_pts_norm.dtype).view(1, 1, 1, 3)
            cols_right = torch.tensor([0.0, 0.0, 1.0], device=ghost_pts_norm.device, dtype=ghost_pts_norm.dtype).view(1, 1, 1, 3)
            ghost_rgb = torch.cat([
                cols_left.expand(B, T, K, 3),
                cols_right.expand(B, T, K, 3)
            ], dim=2)

            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(ghost_rgb.device)
                offset = params['offset'].to(ghost_rgb.device)
                ghost_rgb = ghost_rgb * scale[3:6] + offset[3:6]
            
            ghost_pcd = torch.cat([ghost_pts_norm, ghost_rgb], dim=-1)
            
            # Heatmaps
            p1_xyz = full_pc_p1[..., 0:3]
            all_xyz = torch.cat([p1_xyz, ghost_pts_norm], dim=2)
            
            # Denormalize for Heat (raw = (normalized - offset) / scale)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(all_xyz.device)
                offset = params['offset'].to(all_xyz.device)
                all_xyz_raw = (all_xyz - offset[:3]) / scale[:3]
            else:
                all_xyz_raw = all_xyz
                 
            heat_l, heat_r, beacon_l, beacon_r = self._calculate_beacon_heatmap(all_xyz_raw, pose_18d_raw, sigma=self.beacon_sigma)
            
            # Assemble P2
            mask_ghost_p1 = torch.zeros((B, T, full_pc_p1.shape[2], 1), device=pc.device)
            p1_block = torch.cat([full_pc_p1, mask_ghost_p1], dim=-1)
            
            mask_real_ghost = torch.zeros((B, T, ghost_pts_norm.shape[2], 1), device=pc.device)
            mask_ghost_ghost = torch.ones((B, T, ghost_pts_norm.shape[2], 1), device=pc.device)
            ghost_block = torch.cat([ghost_pcd, mask_real_ghost, mask_ghost_ghost], dim=-1)
            
            full_block = torch.cat([p1_block, ghost_block], dim=2)
            full_pc_p2 = torch.cat([full_block, heat_l, heat_r], dim=-1)
            
        else:
             pad = torch.zeros((B, T, full_pc_p1.shape[2], 3), device=pc.device)
             full_pc_p2 = torch.cat([full_pc_p1, pad], dim=-1)
             
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        global_cond = combined_features
        if self.obs_as_global_cond:
            n_obs = self.n_obs_steps
            global_cond = combined_features[:, :n_obs, :]
            if not "cross_attention" in self.condition_type:
                global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        naction = torch.randn((B, self.horizon, self.action_dim), device=pc.device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        action = self.normalizer['action'].unnormalize(naction)
        return action
