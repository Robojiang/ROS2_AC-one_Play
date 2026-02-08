import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
import sys

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
import numpy as np

class GHOSTBeaconPolicy(BasePolicy):
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
                 aux_point_num=50, 
                 aux_length=0.2,   
                 aux_radius=0.01,  
                 aux_trident_side_len=0.15,
                 aux_trident_max_width=0.08, # Used for calculating beacon position
                 use_keyframe_prediction=True,
                 keyframe_model_prob=0.0, 
                 keyframe_pred_loss_weight=1.0,
                 keyframe_noise_std=0.1,
                 beacon_sigma=0.1, # Sigma for Gaussian heatmap
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
        self.beacon_sigma = beacon_sigma
        
        # 1. Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta['obs']
        self.state_dim = obs_shape_meta['agent_pos']['shape'][0]
        
        # 2. Configure PointNet Encoders
        base_channels = 6 if use_pc_color else 3
        
        # P1 Input: Base + Aux Indicator (7D)
        p1_channels = base_channels + 1 if use_aux_points else base_channels
        
        # P2 Input: P1 Input + Left Heatmap + Right Heatmap (9D)
        # 7 (Base+Aux) + 1 (Heat L) + 1 (Heat R) = 9
        p2_channels = p1_channels + 2 if use_keyframe_prediction else p1_channels
        
        cprint(f"[GHOSTBeaconPolicy] P1 Channels: {p1_channels}, P2 Channels: {p2_channels}", "cyan")
        
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {}
        
        # Setup Encoders
        if PointNet2Encoder is None and pointnet_type == "pointnet++":
            raise ImportError("PointNet2Encoder not found")

        # --- P2 (Actor Encoder with Heatmaps) ---
        enc_cfg_p2 = copy.deepcopy(pointcloud_encoder_cfg)
        enc_cfg_p2['in_channels'] = p2_channels
        
        self.obs_encoder = PointNetEncoderXYZRGB(**enc_cfg_p2)
        
        self.obs_feature_dim = enc_cfg_p2.get('out_channels', 1024)
            
        # --- P1 (Keyframe Predictor) ---
        if self.use_keyframe_prediction:
            enc_cfg_p1 = copy.deepcopy(pointcloud_encoder_cfg)
            enc_cfg_p1['in_channels'] = p1_channels
            
            if pointnet_type == "pointnet":
                self.keyframe_encoder = PointNetEncoderXYZRGB(**enc_cfg_p1)
            elif pointnet_type == "pointnet++":
                self.keyframe_encoder = PointNet2Encoder(**enc_cfg_p1)
                
            self.keyframe_feature_dim = enc_cfg_p1.get('out_channels', 1024)
            # Head to predict 12D Beacons directly (Left Tip 1, Left Tip 2, Right Tip 1, Right Tip 2)
            # 2 Hands * 2 Tips/Hand * 3 Coords = 12
            self.keyframe_head = nn.Sequential(
                nn.Linear(self.keyframe_feature_dim, 256),
                nn.Mish(),
                nn.Linear(256, 12) 
            )
            cprint("[GHOSTBeaconPolicy] Beacon Keypoint Prediction Module Enabled (P1) - Predicting 12D Coords", "yellow")

        # 3. Diffusion Model
        input_dim = self.action_dim
        global_cond_dim = None
        
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.Mish(),
            nn.Linear(256, self.obs_feature_dim)
        )
        
        combined_feature_dim = self.obs_feature_dim * 2 
        
        if obs_as_global_cond:
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
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)

    def _get_beacons_from_pose(self, pose_18d):
        """
        Extract beacon points (gripper tips) from the 18D pose.
        Beacons are the endpoints of the 'short edges' of the trident.
        """
        if pose_18d.dim() == 2:
            pose_18d = pose_18d.unsqueeze(1)
        B, T, _ = pose_18d.shape
        
        # Helper to process one gripper
        def process_gripper(pos, rot6d, gripper_width):
             # pos: (B, T, 3), rot6d: (B, T, 6)
             rot_mat = self._rot6d_to_mat(rot6d) # (B, T, 3, 3)
             
             # Local geometry for short edge endpoints
             # Left Tip: [side_len, +width, 0]
             # Right Tip: [side_len, -width, 0]
             
             # If width is 0 (e.g. gripper closed or user config), the two points will merge on the axis.
             # Ensure width has a minimum value for visualization/heatmap separation if desired, 
             # but here we respect the config.
             
             side_len = self.aux_trident_side_len
             width = gripper_width * self.aux_trident_max_width
             
             # Create local points (B, T, 2, 3)
             zeros = torch.zeros_like(width)
             
             # Important: The coordinate system here assumes X is forward (along the tool axis),
             # Y is Left/Right (Width), Z is Up/Down.
             # Check if your gripper geometry matches this.
             
             l_tip_local = torch.stack([torch.full_like(width, side_len), width, zeros], dim=-1)
             r_tip_local = torch.stack([torch.full_like(width, side_len), -width, zeros], dim=-1)
             
             tips_local = torch.stack([l_tip_local, r_tip_local], dim=2) # (B, T, 2, 3)
             
             # Apply Rotation
             # tips_local shape: (B, T, 2, 3)
             # rot_mat shape: (B, T, 3, 3)
             # _rot6d_to_mat returns row-stacked basis vectors (Local -> Global)
             # Correct transformation is v_global = v_local @ R + pos
             
             tips_global = torch.matmul(tips_local, rot_mat) # (B, T, 2, 3)
             
             # Add Position
             tips_global += pos.unsqueeze(2)
             
             return tips_global # (B, T, 2, 3)
         
        # Left Arm (0-9)
        l_pos = pose_18d[..., 0:3]
        l_rot = pose_18d[..., 3:9]
        l_grip = torch.ones((B, T), device=pose_18d.device)
        beacons_left = process_gripper(l_pos, l_rot, l_grip)
        
        # Right Arm (9-18)
        r_pos = pose_18d[..., 9:12]
        r_rot = pose_18d[..., 12:18]
        r_grip = torch.ones((B, T), device=pose_18d.device)
        beacons_right = process_gripper(r_pos, r_rot, r_grip)
        
        return beacons_left, beacons_right

    def _compute_heatmap(self, pc, beacons, sigma=0.1):
        """
        Compute heatmap for points based on distance to beacons.
        pc: (B, T, N, 3)
        beacons: (B, T, K, 3)
        Returns: (B, T, N, 1) - Max heat value across all beacons
        """
        B, T, N, _ = pc.shape
        K = beacons.shape[2]
        
        # Expand for broadcasting
        # pc: (B, T, N, 1, 3)
        pc_exp = pc.unsqueeze(3)
        # beacons: (B, T, 1, K, 3)
        beacons_exp = beacons.unsqueeze(2)
        
        # Dist squared: (B, T, N, K)
        dist_sq = torch.sum((pc_exp - beacons_exp) ** 2, dim=-1)
        
        # Gaussian: exp(-d^2 / (2*sigma^2))
        heat = torch.exp(-dist_sq / (2 * sigma**2))
        
        # Max over beacons (combine tips into one blob)
        heat_max, _ = torch.max(heat, dim=-1, keepdim=True)
        return heat_max

    # Reuse Trident/Aux generation logic from Keyframe Policy verbatim?
    # Yes, for P1 input (Real Trident), we use the same logic.
    # I will copy `_generate_aux_points` from previous file.
    def _generate_aux_points(self, agent_pos):
        B, T, D = agent_pos.shape
        states_list = [] 
        
        if D == 32:
             left_gripper_val = agent_pos[..., 6].clip(0, 1) 
             left_pos = agent_pos[..., 14:17] 
             left_rot6d = agent_pos[..., 17:23] 
             states_list.append((left_pos, left_rot6d, left_gripper_val))
             
             right_gripper_val = agent_pos[..., 13].clip(0, 1) 
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
        pts_c = torch.stack([dists_c, torch.zeros_like(dists_c), torch.zeros_like(dists_c)], dim=-1)
        angle_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        rad_c = torch.rand(n_center, device=agent_pos.device, dtype=agent_pos.dtype) * self.aux_radius
        pts_c += torch.stack([torch.zeros_like(dists_c), rad_c*torch.cos(angle_c), rad_c*torch.sin(angle_c)], dim=-1)
        
        n_side = (self.aux_point_num - n_center) // 2
        dists_s = torch.linspace(0, self.aux_trident_side_len, n_side, device=agent_pos.device, dtype=agent_pos.dtype)
        base_s = torch.stack([dists_s, torch.zeros_like(dists_s), torch.zeros_like(dists_s)], dim=-1)
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

    def forward(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        if not self.use_pc_color and pc.shape[-1] > 3:
            pc = pc[..., :3]

        B, T, N, C = pc.shape
        loss_dict = {}
        
        # --- Step 1: P1 Input Construction (Scene + Real Trident) ---
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
                K = aux_pts_xyz.shape[2] // 2
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                aux_pts_rgb = torch.cat([cols_left.expand(B, T, K, 3), cols_right.expand(B, T, K, 3)], dim=2)
                if 'point_cloud' in self.normalizer.params_dict:
                     aux_pts_rgb = aux_pts_rgb * scale[3:] + offset[3:]
                aux_feats.append(aux_pts_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1)
            
            # Indicator Channel: Env=0, Real Trident=1
            p1_ind_env = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            p1_ind_real = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            
            pc_p1 = torch.cat([pc, p1_ind_env], dim=-1)
            aux_p1 = torch.cat([aux_pc, p1_ind_real], dim=-1)
            
            full_pc_p1 = torch.cat([pc_p1, aux_p1], dim=2) # 7D
        else:
            full_pc_p1 = pc
            aux_pc = None

        # --- Step 2: Beacon Prediction (P1) & Heatmap Generation ---
        loss_p1 = 0.0
        
        if self.use_keyframe_prediction:
            # A. Prepare Ground Truth Beacons from Target Keypose (18D)
            target_keypose_norm = nobs['target_keypose'] # (B, T, 18)
            
            # We need Raw Pose to compute physical beacon locations
            target_pose_raw = self.normalizer['target_keypose'].unnormalize(target_keypose_norm)
            gt_beacons_left, gt_beacons_right = self._get_beacons_from_pose(target_pose_raw) # (B, T, 2, 3) each
            
            # Normalize GT Beacons for Loss Calculation? 
            # Usually regression losses are better in normalized space, but points are in metric space.
            # Let's normalize them using the Point Cloud normalizer if available (Spatial Normalization)
            if 'point_cloud' in self.normalizer.params_dict:
                params = self.normalizer['point_cloud'].params_dict
                scale = params['scale'].to(gt_beacons_left.device)[:3]
                offset = params['offset'].to(gt_beacons_left.device)[:3]
                
                # Correct Normalization: x * scale + offset
                gt_beacons_left_norm = gt_beacons_left * scale + offset
                gt_beacons_right_norm = gt_beacons_right * scale + offset
            else:
                gt_beacons_left_norm = gt_beacons_left
                gt_beacons_right_norm = gt_beacons_right
                
            # Flatten to 12D for Loss: [Left_Tip1, Left_Tip2, Right_Tip1, Right_Tip2]
            # gt_beacons_left: (B, T, 2, 3) -> (B, T, 6)
            gt_vec_left = rearrange(gt_beacons_left_norm, 'b t n c -> b t (n c)')
            gt_vec_right = rearrange(gt_beacons_right_norm, 'b t n c -> b t (n c)')
            gt_vec_12d = torch.cat([gt_vec_left, gt_vec_right], dim=-1) # (B, T, 12)

            # B. Predict Beacons
            flat_p1_input = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_feats = self.keyframe_encoder(flat_p1_input)
            pred_vec_12d = self.keyframe_head(p1_feats)
            pred_vec_12d = rearrange(pred_vec_12d, '(b t) d -> b t d', b=B, t=T)
            
            # Loss for P1 (MSE on Coordinates)
            loss_p1 = F.mse_loss(pred_vec_12d, gt_vec_12d)
            loss_dict['loss_keyframe'] = loss_p1.item()
            
            # C. Determine Beacons for Heatmap (Curriculum + Noise)
            use_pred = False
            if self.training:
                if torch.rand(1).item() < self.keyframe_model_prob:
                    use_pred = True
            else:
                use_pred = True 
            
            if use_pred:
                beacons_vec_for_heat = pred_vec_12d.detach()
            else:
                beacons_vec_for_heat = gt_vec_12d

            if self.training and self.keyframe_noise_std > 0:
                noise = torch.randn_like(beacons_vec_for_heat) * self.keyframe_noise_std
                beacons_vec_for_heat = beacons_vec_for_heat + noise
            
            # Reshape back to (B, T, 2, 3)
            # 12D = [L_p1(3), L_p2(3), R_p1(3), R_p2(3)]
            beacons_left_norm = rearrange(beacons_vec_for_heat[..., :6], 'b t (n c) -> b t n c', n=2, c=3)
            beacons_right_norm = rearrange(beacons_vec_for_heat[..., 6:], 'b t (n c) -> b t n c', n=2, c=3)
            
            # D. Compute Heatmaps
            # Note: _compute_heatmap expects points in the SAME coordinate space as input PC.
            # PC input to `_compute_heatmap` is `current_xyz`.
            # If `point_cloud` was normalized (which it is in `forward`), then `current_xyz` is normalized.
            # And `beacons_left_norm` is also normalized. So we are good!
            
            current_xyz = full_pc_p1[..., :3]
            
            heat_left = self._compute_heatmap(current_xyz, beacons_left_norm, sigma=self.beacon_sigma)
            heat_right = self._compute_heatmap(current_xyz, beacons_right_norm, sigma=self.beacon_sigma)
            
            # 5. Append Heatmaps (Dimensions 8 & 9)
            full_pc_p2 = torch.cat([full_pc_p1, heat_left, heat_right], dim=-1)
            
        else:
            full_pc_p2 = full_pc_p1

        # --- Step 3: Action Prediction (P2) ---
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = combined_features[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        else:
             global_cond = combined_features
        
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
        
        # --- Step 1: P1 Input --- 
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
                K = aux_pts_xyz.shape[2] // 2
                cols_left = torch.tensor([1.0, 0.0, 0.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                cols_right = torch.tensor([0.0, 0.0, 1.0], device=aux_pts_xyz.device).view(1, 1, 1, 3)
                aux_pts_rgb = torch.cat([cols_left.expand(B, T, K, 3), cols_right.expand(B, T, K, 3)], dim=2)
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

        # --- Step 2: Prediction ---
        if self.use_keyframe_prediction:
            flat_p1_input = rearrange(full_pc_p1, 'b t n c -> (b t) n c')
            p1_feats = self.keyframe_encoder(flat_p1_input)
            pred_vec_12d = self.keyframe_head(p1_feats)
            pred_vec_12d = rearrange(pred_vec_12d, '(b t) d -> b t d', b=B, t=T)
            
            # Predict 12D vectors directly -> These are normalized beacons
            beacons_vec = pred_vec_12d
            
            # Reshape back to (B, T, 2, 3)
            beacons_left = rearrange(beacons_vec[..., :6], 'b t (n c) -> b t n c', n=2, c=3)
            beacons_right = rearrange(beacons_vec[..., 6:], 'b t (n c) -> b t n c', n=2, c=3)
            
            # Note: During inference we don't need to unnormalize/renormalize if the prediction target was already normalized.
            # In training, we normalized the GT beacons using PC stats. 
            # So the prediction `pred_vec_12d` is already in PC-normalized space.
            # We can directly use it for heatmap.
            
            current_xyz = full_pc_p1[..., :3]
            heat_left = self._compute_heatmap(current_xyz, beacons_left, sigma=self.beacon_sigma)
            heat_right = self._compute_heatmap(current_xyz, beacons_right, sigma=self.beacon_sigma)
            
            full_pc_p2 = torch.cat([full_pc_p1, heat_left, heat_right], dim=-1)
            
        else:
            full_pc_p2 = full_pc_p1
             
        full_pc_flat = rearrange(full_pc_p2, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        global_cond = combined_features
        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = combined_features[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')

        naction = torch.randn((B, self.horizon, self.action_dim), device=pc.device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        action = self.normalizer['action'].unnormalize(naction)
        return action
