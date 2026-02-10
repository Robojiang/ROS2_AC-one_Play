"""
模型加载模块
负责加载 DP3 和 GHOST 策略
避免不必要的 huggingface/transformers 依赖
"""

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf


class DP3InferenceWrapper:
    """DP3推理包装器"""
    
    def __init__(self, policy_model, cfg):
        self.policy = policy_model
        self.cfg = cfg
        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
    
    def get_action(self, batch):
        """
        执行推理
        
        Args:
            batch: dict with 'obs' key containing 'point_cloud' and 'agent_pos'
            
        Returns:
            actions: tensor (horizon, action_dim)
        """
        with torch.no_grad():
            result = self.policy.predict_action(batch['obs'])
            actions = result['action']  # (B, horizon, action_dim)
        return actions


def load_dp3_policy(args, root_dir):
    """
    加载DP3策略 - 直接从checkpoint加载，避免huggingface依赖
    
    Args:
        args: 命令行参数
        root_dir: 项目根目录
        
    Returns:
        (policy, config): 策略对象和配置
    """
    print(f"[INFO] 加载DP3策略...")
    
    # 1. 搜索权重文件
    weights_dir = root_dir / "weights" / args.task_name / "DP3"
    if not weights_dir.exists():
        raise FileNotFoundError(f"权重目录不存在: {weights_dir}")
    
    ckpt_files = sorted(weights_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"在 {weights_dir} 中未找到 .ckpt 文件")
    
    if args.ckpt_name:
        ckpt_path = weights_dir / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[WARN] 指定的权重文件不存在: {ckpt_path}，使用最新权重")
            ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = ckpt_files[-1]
    
    print(f"[INFO] 加载权重: {ckpt_path}")
    
    # 2. 加载 checkpoint (包含配置和权重)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    state_dict = checkpoint['state_dicts']['model']
    
    print(f"[INFO] 从checkpoint加载配置")
    print(f"  - horizon: {cfg.horizon}")
    print(f"  - n_obs_steps: {cfg.n_obs_steps}")
    print(f"  - n_action_steps: {cfg.n_action_steps}")
    
    # 3. 添加 DP3 路径
    dp3_path = str(root_dir / "policy" / "DP3" / "3D-Diffusion-Policy")
    if dp3_path not in sys.path:
        sys.path.insert(0, dp3_path)
    
    # 4. 直接导入必要的模块
    #    transformers 已从环境中移除，diffusers 可以正常导入
    try:
        print("[INFO] 导入模型依赖...")
        
        # 直接导入（transformers 已卸载，不会触发冲突）
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        print("[INFO] ✓ 成功导入 DDPMScheduler")
        
        # 导入 DP3 模型
        from diffusion_policy_3d.policy.dp3 import DP3
        print("[INFO] ✓ 成功导入 DP3 模型")
        
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        raise RuntimeError(
            f"DP3 依赖导入失败: {e}\n\n"
            "这个错误通常是因为环境依赖冲突。DP3 推理实际上不需要 transformers 和 huggingface_hub，\n"
            "但 diffusers 库在导入时会自动加载它们。\n\n"
            "临时解决方案：\n"
            "1. 卸载不兼容的包：pip uninstall transformers tokenizers -y\n"
            "2. 或者使用 Python 3.10 的 conda 环境（但会与 ROS2 Jazzy 冲突）\n"
            "3. 或者重新编译 numcodecs/zarr 以支持 Python 3.12\n\n"
            "长期方案：将 DP3 模型和 DDPMScheduler 的代码独立出来，完全移除 diffusers 依赖"
        )
    
    # 5. 创建噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.policy.noise_scheduler.num_train_timesteps,
        beta_schedule=cfg.policy.noise_scheduler.beta_schedule,
        clip_sample=cfg.policy.noise_scheduler.clip_sample,
        prediction_type=cfg.policy.noise_scheduler.prediction_type
    )
    
    # 6. 实例化模型（使用checkpoint中保存的所有参数）
    print("[INFO] 实例化 DP3 模型...")
    
    # 从 cfg.policy 中提取所有模型参数
    policy_cfg = cfg.policy
    
    policy_model = DP3(
        shape_meta=cfg.shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        n_obs_steps=cfg.n_obs_steps,
        num_inference_steps=policy_cfg.num_inference_steps,
        obs_as_global_cond=policy_cfg.obs_as_global_cond,
        # 以下参数也从 checkpoint 配置中获取，确保完全一致
        diffusion_step_embed_dim=policy_cfg.get('diffusion_step_embed_dim', 256),
        down_dims=policy_cfg.get('down_dims', [256, 512, 1024]),
        kernel_size=policy_cfg.get('kernel_size', 5),
        n_groups=policy_cfg.get('n_groups', 8),
        condition_type=policy_cfg.get('condition_type', 'film'),
        use_down_condition=policy_cfg.get('use_down_condition', True),
        use_mid_condition=policy_cfg.get('use_mid_condition', True),
        use_up_condition=policy_cfg.get('use_up_condition', True),
        encoder_output_dim=policy_cfg.get('encoder_output_dim', 256),
        crop_shape=policy_cfg.get('crop_shape', None),
        use_pc_color=policy_cfg.use_pc_color,
        pointnet_type=policy_cfg.pointnet_type,
        pointcloud_encoder_cfg=policy_cfg.pointcloud_encoder_cfg
    )
    
    # 7. 加载权重
    print("[INFO] 加载模型权重...")
    policy_model.load_state_dict(state_dict)
    policy_model.cuda()
    policy_model.eval()
    
    # 8. 创建推理包装器
    wrapper = DP3InferenceWrapper(policy_model, cfg)
    
    print("[INFO] ✓ DP3策略加载完成")
    
    return wrapper, cfg


def load_ghost_policy(args, root_dir):
    """
    加载GHOST策略
    
    Args:
        args: 命令行参数
        root_dir: 项目根目录
        
    Returns:
        (wrapper, config): GHOST包装器和配置
    """
    print(f"[INFO] 加载GHOST策略...")
    print(f"[WARN] GHOST 策略加载功能待实现")
    print(f"[WARN] GHOST 原有的 deploy_policy.py 是为仿真设计的")
    print(f"[WARN] 请基于实际需求重新实现 GHOST 真机推理逻辑")
    
    # TODO: 实现 GHOST 真机加载逻辑
    raise NotImplementedError("GHOST 策略加载功能待实现，需要根据实际需求重写")


def load_policy_model(args, root_dir):
    """
    根据策略类型加载对应的模型
    
    Args:
        args: 命令行参数
        root_dir: 项目根目录
        
    Returns:
        (policy, config): 策略对象和配置
    """
    if args.policy == 'DP3':
        return load_dp3_policy(args, root_dir)
    elif args.policy == 'GHOST':
        return load_ghost_policy(args, root_dir)
    else:
        raise ValueError(f"不支持的策略类型: {args.policy}")
