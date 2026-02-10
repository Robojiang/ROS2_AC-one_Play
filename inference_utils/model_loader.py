"""
统一模型加载器 - 支持 DP3 和 GHOST，normalizer 已集成在模型中
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加 3D-Diffusion-Policy 到 sys.path (用于导入 DP3 模型)
dp3_path = os.path.join(os.path.dirname(__file__), '..', 'policy', 'DP3', '3D-Diffusion-Policy')
if dp3_path not in sys.path:
    sys.path.insert(0, os.path.abspath(dp3_path))


class PolicyWrapper:
    """策略包装器基类，提供统一接口"""
    
    def __init__(self, model, config):
        """
        Args:
            model: 模型实例 (DP3 模型内部已包含 normalizer)
            config: OmegaConf 配置对象
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def predict_action(self, obs_dict):
        """预测动作 - 子类实现"""
        raise NotImplementedError


class DP3Wrapper(PolicyWrapper):
    """DP3 策略包装器 - DP3 模型内部已集成 normalizer 和 scheduler"""
    
    def __init__(self, model, config):
        super().__init__(model, config)
        # config 是顶层配置，policy 相关参数在 config.policy 下
        self.num_inference_steps = config.policy.num_inference_steps
        self.n_obs_steps = config.n_obs_steps
        self.n_action_steps = config.n_action_steps
        
    def predict_action(self, obs_dict):
        """
        预测动作序列
        
        Args:
            obs_dict: 包含点云和机器人状态
                - 'point_cloud': (B, To, N, 6) torch.Tensor 点云序列 (To=观察步数)
                - 'agent_pos': (B, To, D) torch.Tensor 机器人状态序列
        
        Returns:
            actions: (B, horizon, action_dim) numpy.ndarray 动作序列
        """
        # DP3 模型的 predict_action 方法已经处理了所有细节
        # 包括归一化、去噪、反归一化等
        with torch.no_grad():
            # 将输入移动到正确的设备
            obs_dict_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in obs_dict.items()
            }
            
            # 调用模型的 predict_action 方法 - 返回字典 {'action': tensor}
            result = self.model.predict_action(obs_dict_device)
            actions = result['action']  # 提取 action tensor
            
        return actions.cpu().numpy()


class GHOSTWrapper(PolicyWrapper):
    """GHOST 策略包装器"""
    
    def __init__(self, model, config):
        super().__init__(model, config)
        # TODO: 根据 GHOST 模型特性初始化
        
    def predict_action(self, obs_dict):
        """
        预测动作序列 - GHOST 实现
        
        Args:
            obs_dict: 包含点云和机器人状态
        
        Returns:
            actions: 动作序列
        """
        # TODO: 实现 GHOST 推理逻辑
        raise NotImplementedError("GHOST inference not implemented yet")


def load_policy_model(policy_name, task_name, ckpt_name, root_dir="weights"):
    """
    统一加载接口 - 根据策略名称自动加载模型和 normalizer
    
    Args:
        policy_name: 'DP3' 或 'GHOST'
        task_name: 任务名称，例如 'pick_place_d405'
        ckpt_name: checkpoint 文件名，例如 'latest.ckpt'
        root_dir: 权重根目录
    
    Returns:
        policy_wrapper: PolicyWrapper 实例（DP3Wrapper 或 GHOSTWrapper）
    """
    # 1. 构建 checkpoint 路径
    ckpt_path = Path(root_dir) / task_name / policy_name / ckpt_name
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading {policy_name} model from: {ckpt_path}")
    
    # 2. 加载 checkpoint (weights_only=False 因为包含 OmegaConf 配置对象)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # 3. 提取配置 (normalizer 已集成在模型中，不需要单独提取)
    config = checkpoint.get('cfg', None)
    
    if config is None:
        raise ValueError(f"Config not found in checkpoint: {ckpt_path}")
    
    print(f"  Config: {type(config)}")
    print(f"  Normalizer: integrated in model (auto-loaded)")
    
    # 4. 根据策略名称加载不同模型
    if policy_name.upper() == 'DP3':
        return _load_dp3_model(checkpoint, config)
    elif policy_name.upper() == 'GHOST':
        return _load_ghost_model(checkpoint, config)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def _load_dp3_model(checkpoint, config):
    """加载 DP3 模型 - normalizer 已集成在模型中"""
    from diffusion_policy_3d.policy.dp3 import DP3
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    
    # 1. 创建 noise scheduler
    noise_scheduler_config = config.policy.noise_scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=noise_scheduler_config.num_train_timesteps,
        beta_schedule=noise_scheduler_config.beta_schedule,
        clip_sample=noise_scheduler_config.clip_sample,
        prediction_type=noise_scheduler_config.prediction_type,
    )
    
    # 2. 创建模型实例
    model = DP3(
        shape_meta=config.policy.shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=config.policy.horizon,
        n_action_steps=config.policy.n_action_steps,
        n_obs_steps=config.policy.n_obs_steps,
        num_inference_steps=config.policy.num_inference_steps,
        obs_as_global_cond=config.policy.obs_as_global_cond,
        diffusion_step_embed_dim=config.policy.diffusion_step_embed_dim,
        down_dims=config.policy.down_dims,
        kernel_size=config.policy.kernel_size,
        n_groups=config.policy.n_groups,
        condition_type=config.policy.condition_type,
        use_down_condition=config.policy.use_down_condition,
        use_mid_condition=config.policy.use_mid_condition,
        use_up_condition=config.policy.use_up_condition,
        encoder_output_dim=config.policy.encoder_output_dim,
        crop_shape=config.policy.crop_shape,
        use_pc_color=config.policy.use_pc_color,
        pointnet_type=config.policy.pointnet_type,
        pointcloud_encoder_cfg=config.policy.pointcloud_encoder_cfg,
    )
    
    # 3. 加载权重 (包含 normalizer 的参数)
    state_dict = checkpoint.get('state_dicts', {}).get('model', None)
    if state_dict is None:
        raise ValueError("Model state_dict not found in checkpoint")
    
    model.load_state_dict(state_dict)
    
    # 3. 设置为评估模式并移动到 GPU
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"  DP3 model loaded on {device}")
    print(f"  Normalizer keys: {list(model.normalizer.params_dict.keys())}")
    
    # 4. 创建 wrapper
    return DP3Wrapper(model, config)


def _load_ghost_model(checkpoint, config):
    """加载 GHOST 模型"""
    # TODO: 实现 GHOST 加载逻辑
    raise NotImplementedError("GHOST model loading not implemented yet")


def test_model_loading(policy_name="DP3", task_name="pick_place_d405", ckpt_name="latest.ckpt"):
    """
    测试模型加载是否成功
    
    用法:
        python -c "from inference_utils.model_loader import test_model_loading; test_model_loading()"
    """
    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60)
    
    try:
        # 1. 加载模型
        policy_wrapper = load_policy_model(policy_name, task_name, ckpt_name)
        
        print("\n✓ Model loaded successfully!")
        print(f"  Model device: {policy_wrapper.device}")
        print(f"  Config type: {type(policy_wrapper.config)}")
        
        # 2. 测试 normalizer (已集成在模型中)
        if hasattr(policy_wrapper.model, 'normalizer'):
            print(f"\n✓ Normalizer found in model")
            print(f"  Normalizer keys: {list(policy_wrapper.model.normalizer.params_dict.keys())}")
        else:
            print("\n⚠ No normalizer found in model")
        
        # 3. 测试模型推理（使用假数据）
        print("\n✓ Testing inference with dummy data...")
        
        if policy_name.upper() == 'DP3':
            # 创建假的点云和机器人状态
            # DP3 使用 (B, To, N, 6) 格式: B=batch, To=obs_steps, N=points, 6=xyz+rgb
            To = policy_wrapper.n_obs_steps
            dummy_obs = {
                'point_cloud': torch.randn(1, To, 512, 6),  # (B, To, N, 6)
                'agent_pos': torch.randn(1, To, 14)  # (B, To, D)
            }
            
            actions = policy_wrapper.predict_action(dummy_obs)
            print(f"  Action shape: {actions.shape}")
            print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

# ✓ Model loaded successfully!
#   Model device: cuda:0
#   Normalizer keys: ['action', 'agent_pos', 'point_cloud']
# ✓ Testing inference with dummy data...
#   Action shape: (1, 6, 14)
#   Action range: [-0.804, 1.036]
# ✓ All tests passed!

if __name__ == "__main__":
    # 测试加载
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        policy_name = sys.argv[1]
        task_name = sys.argv[2] if len(sys.argv) > 2 else "pick_place_d405"
        ckpt_name = sys.argv[3] if len(sys.argv) > 3 else "750.ckpt"
    else:
        policy_name = "DP3"
        task_name = "pick_place_d405"
        ckpt_name = "750.ckpt"
    
    test_model_loading(policy_name, task_name, ckpt_name)
