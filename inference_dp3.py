# -- coding: UTF-8
"""
DP3推理脚本 - 基于原始inference.py的多进程架构
只替换推理部分，保持ROS进程不变
"""
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ACT_DIR = ROOT / "act"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))
    
os.chdir(str(ROOT))

import argparse
import yaml
import rclpy
import torch
import threading
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
from functools import partial
import signal

from act.utils.ros_operator import RosOperator, Rate
from act.utils.setup_loader import setup_loader

# 导入推理工具
from inference_utils.pointcloud_generator import PointCloudGenerator
from inference_utils.model_loader import load_policy_model
from inference_utils.calibration import load_calibration_data
from scipy.spatial.transform import Rotation as R
import pickle # Added for debug saving

def save_debug_data(data_list, policy_name):
    """保存调试数据到Zarr文件 (便于查看器检查)"""
    if not data_list:
        return

    save_dir = ROOT / "debug_obs"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    policy_safe = policy_name.replace('/', '_')
    filename = f"debug_{policy_safe}_{timestamp}.zarr"
    filepath = save_dir / filename
    
    print(f"\n[INFO] 正在保存 {len(data_list)} 帧调试数据到 {filepath} (Zarr格式) ...")
    try:
        import zarr
        import numpy as np
        
        # 1. 整理数据为列表
        # step_data keys: qpos, raw_eef, point_cloud, agent_pos, transformed_eef
        qpos_list = [d['qpos'] for d in data_list]
        point_cloud_list = [d['point_cloud'] for d in data_list]
        agent_pos_list = [d['agent_pos'] for d in data_list]
        
        # 提取 Left/Right Endpose (并转换为 7D: Pos + Quat[w,x,y,z])
        # transformed_eef['left'] 是 [x, y, z, rx, ry, rz] (Euler)
        left_endpose_list = []
        right_endpose_list = []
        
        for d in data_list:
            # Left
            l_raw = d['transformed_eef']['left']
            l_quat = euler_to_w_last_quat(l_raw[3:6]) # Euler -> [w,x,y,z]
            l_pose_7d = np.concatenate([l_raw[:3], l_quat])
            left_endpose_list.append(l_pose_7d)
            
            # Right 
            r_raw = d['transformed_eef']['right']
            r_quat = euler_to_w_last_quat(r_raw[3:6])
            r_pose_7d = np.concatenate([r_raw[:3], r_quat])
            right_endpose_list.append(r_pose_7d)
        
        # Stack arrays
        state = np.array(qpos_list) # (T, 14)
        point_cloud = np.array(point_cloud_list) # (T, N, 6)
        agent_pos = np.array(agent_pos_list) # (T, D)
        left_endpose = np.array(left_endpose_list) # (T, 7)
        right_endpose = np.array(right_endpose_list) # (T, 7)
        
        # 构造 Dummy Action (因为 Inference 阶段可能只存了观测，或者 action 需要额外收集)
        # 既然是为了检查格式，存一个全0的 action 即可
        action = np.zeros_like(state) 
        
        # 2. 创建 Zarr Group (使用 zarr.open 兼容新旧版本)
        # 注意: Zarr 3.x 以后建议使用 zarr.open(path, mode='w')
        root_group = zarr.open(str(filepath), mode='w')
        
        # 3. 创建 Data Group
        data_group = root_group.create_group('data')
        
        # 4. 写入数据集
        # 直接赋值以兼容 Zarr 新版本 (避免使用 create_dataset 和 compressor 参数带来的版本差异)
        # Zarr 会自动推断 chunks 和 default compressor
        data_group['state'] = state
        data_group['action'] = action
        data_group['point_cloud'] = point_cloud
        data_group['agent_pos'] = agent_pos
        data_group['left_endpose'] = left_endpose
        data_group['right_endpose'] = right_endpose

        # 5. Meta Group (Episode Ends)
        meta_group = root_group.create_group('meta')
        episode_ends = np.array([len(data_list)], dtype=np.int64)
        meta_group['episode_ends'] = episode_ends
        
        print(f"[INFO] 保存成功: {filepath}")
        
    except ImportError:
        print("[ERROR] 缺少 zarr 库，尝试回退到 Pickle 保存...")
        # Fallback to pickle
        pkl_path = filepath.with_suffix('.pkl')
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_list, f)
            print(f"[INFO] 已回退保存为: {pkl_path}")
        except Exception as e:
             print(f"[ERROR] Pickle 保存也失败: {e}")

    except Exception as e:
        print(f"[ERROR] Zarr 保存失败: {e}")
        import traceback
        traceback.print_exc()

def euler_to_w_last_quat(euler):
    """
    Euler angles (xyz) to Quaternion [x, y, z, w] (scipy default)
    Then convert to [w, x, y, z] to match GHOST dataset format.
    """
    q = R.from_euler('xyz', euler).as_quat() # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]]) # -> [w, x, y, z]

def quat_to_rot6d(quat):
    """
    Convert quaternion to 6D rotation representation.
    quat: (..., 4) in [w, x, y, z] format
    Returns: (..., 6)
    """
    quat = np.array(quat)
    # Ensure it's floats
    if quat.dtype != np.float32 and quat.dtype != np.float64:
        quat = quat.astype(np.float32)
        
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    
    # First column
    r11 = 1 - 2 * (yy + zz)
    r21 = 2 * (xy + zw)
    r31 = 2 * (xz - yw)
    
    # Second column
    r12 = 2 * (xy - zw)
    r22 = 1 - 2 * (xx + zz)
    r32 = 2 * (yz + xw)
    
    rot6d = np.stack([r11, r21, r31, r12, r22, r32], axis=-1)
    return rot6d

def get_ghost_agent_pos(qpos, left_pos, left_quat, right_pos, right_quat):
    """
    构造 GHOST 策略所需的 32维 agent_pos
    Structure: [qpos(14), left_9d(9), right_9d(9)]
    where 9d = [pos(3), rot6d(6)]
    Total: 14 + 18 = 32
    """
    # Left (9D)
    left_rot6d = quat_to_rot6d(left_quat)
    left_9d = np.concatenate([left_pos, left_rot6d])
    
    # Right (9D)
    right_rot6d = quat_to_rot6d(right_quat)
    right_9d = np.concatenate([right_pos, right_rot6d])
    
    return np.concatenate([qpos, left_9d, right_9d])

SAFE_INIT_POSITION = [0.0, 0, 0, 0, 0.0, 0.0, 0]


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading yaml: {e}")
        return None


def make_shm_name_dict(shapes):
    """创建共享内存名称字典"""
    shm_name_dict = {}
    for cam in shapes["cameras"]:
        shm_name_dict[f"{cam}_rgb"] = f"shm_rgb_{cam}"
        shm_name_dict[f"{cam}_depth"] = f"shm_depth_{cam}"
    shm_name_dict["qpos"] = "shm_qpos"
    shm_name_dict["eef"] = "shm_eef"
    shm_name_dict["action"] = "shm_action"
    return shm_name_dict


def create_shm_dict(shm_name_dict, shapes):
    """创建共享内存"""
    shm_dict = {}
    for cam in shapes["cameras"]:
        # RGB
        rgb_shape = shapes[f"{cam}_rgb"]
        size = np.prod(rgb_shape) * np.dtype(np.uint8).itemsize
        shm = SharedMemory(name=shm_name_dict[f"{cam}_rgb"], create=True, size=size)
        shm_dict[f"{cam}_rgb"] = (shm, rgb_shape, np.uint8)
        
        # Depth
        depth_shape = shapes[f"{cam}_depth"]
        size = np.prod(depth_shape) * np.dtype(np.uint16).itemsize
        shm = SharedMemory(name=shm_name_dict[f"{cam}_depth"], create=True, size=size)
        shm_dict[f"{cam}_depth"] = (shm, depth_shape, np.uint16)
    
    # qpos
    qpos_shape = shapes["qpos"]
    size = np.prod(qpos_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["qpos"], create=True, size=size)
    shm_dict["qpos"] = (shm, qpos_shape, np.float32)
    
    # eef
    eef_shape = shapes["eef"]
    size = np.prod(eef_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["eef"], create=True, size=size)
    shm_dict["eef"] = (shm, eef_shape, np.float32)
    
    # action
    action_shape = (14,)
    size = np.prod(action_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["action"], create=True, size=size)
    shm_dict["action"] = (shm, action_shape, np.float32)
    
    return shm_dict


def connect_shm_dict(shm_name_dict, shapes):
    """连接到已创建的共享内存"""
    shm_dict = {}
    for cam in shapes["cameras"]:
        # RGB
        shm = SharedMemory(name=shm_name_dict[f"{cam}_rgb"], create=False)
        shm_dict[f"{cam}_rgb"] = (shm, shapes[f"{cam}_rgb"], np.uint8)
        
        # Depth
        shm = SharedMemory(name=shm_name_dict[f"{cam}_depth"], create=False)
        shm_dict[f"{cam}_depth"] = (shm, shapes[f"{cam}_depth"], np.uint16)
    
    shm = SharedMemory(name=shm_name_dict["qpos"], create=False)
    shm_dict["qpos"] = (shm, shapes["qpos"], np.float32)
    
    shm = SharedMemory(name=shm_name_dict["eef"], create=False)
    shm_dict["eef"] = (shm, shapes["eef"], np.float32)
    
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, (14,), np.float32)
    
    return shm_dict


def cleanup_shm(names):
    """清理共享内存"""
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def move_to_target(ros_operator, target_pos, steps=100):
    """平滑移动到目标位置"""
    import math
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array(target_pos)
    for i in range(steps):
        t = i / (steps - 1)
        s = (1 - math.cos(math.pi * t)) / 2
        pos = start + (end - start) * s
        ros_operator.follow_arm_publish_continuous(pos.tolist(), pos.tolist())


def move_to_safe_position(ros_operator, steps=200):
    """移动到安全位置"""
    import math
    print("\n[INFO] 正在移动到安全位置...")
    try:
        obs = ros_operator.get_observation()
        if obs is None:
            ros_operator.follow_arm_publish_continuous(SAFE_INIT_POSITION, SAFE_INIT_POSITION)
            return
        
        left_start = np.array(obs['qpos'][:7])
        right_start = np.array(obs['qpos'][7:14])
        target = np.array(SAFE_INIT_POSITION)
        
        for i in range(steps):
            t = i / (steps - 1)
            s = (1 - math.cos(math.pi * t)) / 2
            left_pos = left_start + (target - left_start) * s
            right_pos = right_start + (target - right_start) * s
            ros_operator.follow_arm_publish_continuous(left_pos.tolist(), right_pos.tolist())
        
        print("[INFO] 已到达安全位置")
    except Exception as e:
        print(f"[WARN] 归位出错: {e}")


def init_robot(ros_operator, connected_event, start_event):
    """初始化机器人（遵循原始逻辑）"""
    init0 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, -2.8]
    init1 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    
    print("正在移动到初始位置...")
    # 移动到初始位置
    move_to_target(ros_operator, init0, steps=100)
    
    # 通知主进程：机器人已初始化
    connected_event.set()
    
    print("移动到初始位置完成，等待用户确认...")
    # 等待用户确认
    start_event.wait()
    
    # 最终姿态
    ros_operator.follow_arm_publish_continuous(init1, init1)

    print("[INFO] 机器人初始化完成，开始推理")


def signal_handler(sig, frame, ros_operator):
    """信号处理器"""
    print('\n[INFO] 捕获到退出信号，正在安全关闭...')
    move_to_safe_position(ros_operator, steps=100)
    print('[INFO] 安全关闭完成')
    sys.exit(0)


def ros_process(args, meta_queue, connected_event, start_event, shm_ready_event):
    """ROS进程 - 遵循原始架构"""
    setup_loader(ACT_DIR)
    rclpy.init()
    
    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)
    
    def _spin_loop(node):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
    
    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()
    
    # 注册信号处理
    handler = partial(signal_handler, ros_operator=ros_operator)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # 初始化机器人
    init_robot(ros_operator, connected_event, start_event)
    
    # 等待完整观测数据（这是关键！）
    print("[INFO] 等待完整观测数据...")
    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if obs:
            # 发送shapes信息
            shapes = {
                "cameras": args.camera_names,
                "qpos": obs["qpos"].shape,
                "eef": obs["eef"].shape
            }
            for cam in args.camera_names:
                shapes[f"{cam}_rgb"] = obs["images"][cam].shape
                shapes[f"{cam}_depth"] = obs["images_depth"][cam].shape
            
            meta_queue.put(shapes)
            print("[INFO] 观测数据已就绪")
            break
        rate.sleep()
    
    # 接收共享内存名称并创建
    shm_name_dict = meta_queue.get()
    cleanup_shm(shm_name_dict.values())
    shm_dict = create_shm_dict(shm_name_dict, shapes)
    shm_ready_event.set()
    
    # 主循环：观测 → 写入共享内存 → 读取动作 → 执行
    rate = Rate(args.frame_rate)
    try:
        while rclpy.ok():
            obs = ros_operator.get_observation()
            if not obs:
                rate.sleep()
                continue
            
            # 写入观测到共享内存
            for cam in args.camera_names:
                # RGB
                shm, shape, dtype = shm_dict[f"{cam}_rgb"]
                np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                np_array[:] = obs["images"][cam]
                
                # Depth
                shm, shape, dtype = shm_dict[f"{cam}_depth"]
                np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                np_array[:] = obs["images_depth"][cam]
            
            # qpos
            shm, shape, dtype = shm_dict["qpos"]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["qpos"]
            
            # eef
            shm, shape, dtype = shm_dict["eef"]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["eef"]
            
            # 读取动作
            shm, shape, dtype = shm_dict["action"]
            action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            
            # 执行动作
            if np.any(action):
                left_action = action[:7]
                right_action = action[7:14]
                ros_operator.follow_arm_publish(left_action, right_action)
            
            rate.sleep()
            
    except Exception as e:
        print(f"[ERROR] ros_process异常: {e}")
    finally:
        print("[INFO] ros_process正在清理...")
        move_to_safe_position(ros_operator, steps=100)
        rclpy.shutdown()
        for shm, _, _ in shm_dict.values():
            shm.close()
            shm.unlink()


def inference_process(args, shm_dict, shapes, calibration_data, ros_proc):
    """推理进程 - 使用统一模型加载器"""
    print("[INFO] 推理进程启动")
    
    # 1. 加载策略（使用新的 model_loader）
    policy = load_policy_model(
        policy_name=args.policy,
        task_name=args.task_name,
        ckpt_name=args.ckpt_name or 'latest.ckpt',
        root_dir=str(ROOT / 'weights')
    )
    
    n_obs_steps = policy.n_obs_steps
    n_action_steps = policy.n_action_steps
    
    print(f"[INFO] 模型加载完成: n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
    
    # 2. 创建点云生成器
    pc_generator = PointCloudGenerator()
    
    # 3. 观测缓存 - 用于构建历史观测
    obs_buffer = []  # 存储 (point_cloud, agent_pos) 元组
    
    # === RHC (Receding Horizon Control) 变量 ===
    current_action_chunk = None # 存储当前预测的一整块动作
    action_execution_idx = 0    # 当前块执行到了第几步
    EXECUTION_HORIZON = 8       # 每执行8步重新推理一次 (可根据推理速度调整)
    
    # === Debug 数据录制 ===
    debug_data_list = []
    
    step_count = 0
    print(f"[INFO] 开始推理循环 (max_steps={args.max_publish_step})")
    if args.debug:
        print("[WARN] DEBUG模式：不执行动作，将录制每一个step的观测数据(image, qpos, point_cloud, agent_pos)")
    
    try:
        while ros_proc.is_alive():
            try:
                # 从共享内存读取观测
                obs_dict = {"images": {}, "images_depth": {}, "qpos": None, "eef": None}
                
                for cam in args.camera_names:
                    # RGB
                    shm, shape, dtype = shm_dict[f"{cam}_rgb"]
                    obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                    
                    # Depth
                    shm, shape, dtype = shm_dict[f"{cam}_depth"]
                    obs_dict["images_depth"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                
                shm, shape, dtype = shm_dict["qpos"]
                obs_dict["qpos"] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                
                shm, shape, dtype = shm_dict["eef"]
                obs_dict["eef"] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                
                # 生成点云 (获取统一坐标系下的末端位姿)
                point_cloud, transformed_eef = pc_generator.generate(
                    head_depth=obs_dict['images_depth']['head'],
                    head_color=obs_dict['images']['head'],
                    left_depth=obs_dict['images_depth']['left_wrist'],
                    left_color=obs_dict['images']['left_wrist'],
                    right_depth=obs_dict['images_depth']['right_wrist'],
                    right_color=obs_dict['images']['right_wrist'],
                    left_eef=obs_dict['eef'][:7],
                    right_eef=obs_dict['eef'][7:14],
                    intrinsics=calibration_data['intrinsics'],
                    T_H_LB=calibration_data['T_H_LB'],
                    T_H_RB=calibration_data['T_H_RB'],
                    T_LE_LC=calibration_data['T_LE_LC'],
                    T_RE_RC=calibration_data['T_RE_RC'],
                    T_LB_H=calibration_data['T_LB_H']
                )
                
                # --- 构造 Agent Pos ---
                if args.policy.startswith('GHOST/'):
                    # GHOST 需要: [qpos(14), left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6)] = 32
                    # eef 数据来自 transformed_eef (已经在左臂基座系下)
                    # Left
                    l_pos = transformed_eef['left'][:3]
                    l_euler = transformed_eef['left'][3:6]
                    l_quat = euler_to_w_last_quat(l_euler) # Euler -> [w, x, y, z]
                    
                    # Right
                    r_pos = transformed_eef['right'][:3]
                    r_euler = transformed_eef['right'][3:6]
                    r_quat = euler_to_w_last_quat(r_euler) # Euler -> [w, x, y, z]
                    
                    agent_pos = get_ghost_agent_pos(
                        obs_dict['qpos'], 
                        l_pos, l_quat,
                        r_pos, r_quat
                    )
                else:
                    # DP3 (通常只需 qpos=14 或者 qpos+eef=28，视训练配置而定)
                    # 你之前的报错显示 DP3 正常跑通，所以如果不改 GHOST 逻辑，DP3 维持原状
                    # 这里假设 DP3 只需要 qpos (14)
                     agent_pos = obs_dict['qpos']

                # === 保存 Debug 数据 ===
                if args.debug:
                    step_data = {
                        'step': step_count,
                        'timestamp': time.time(),
                        'qpos': obs_dict['qpos'], # (14,)
                        'raw_eef': obs_dict['eef'], # (14,)
                        'point_cloud': point_cloud, # (N, 6)
                        'agent_pos': agent_pos,     # (32,) or (14,)
                        'transformed_eef': transformed_eef # dict
                    }
                    # 为了不让内存爆炸，只存关键数据，图片太大就不存了，除非你有需求
                    # 如果你想存图片，把下面的注释打开，但会导致写入变慢和文件巨大
                    # step_data['images'] = obs_dict['images']
                    debug_data_list.append(step_data)
                    
                    # 每100步打印一次
                    if step_count % 30 == 0:
                        print(f"[DEBUG] Recorded step {step_count} | PC Points: {len(point_cloud)} | AgentPos dim: {agent_pos.shape}")

                # 添加到观测缓存
                obs_buffer.append((point_cloud, agent_pos))
                if len(obs_buffer) > n_obs_steps:
                    obs_buffer.pop(0)
                
                # 如果缓存不足，用最新观测填充
                while len(obs_buffer) < n_obs_steps:
                    obs_buffer.insert(0, obs_buffer[0])
                
                # === Receding Horizon Control 逻辑 ===
                # 1. 如果没有动作块
                # 2. 或者已经执行了 EXECUTION_HORIZON 步
                # 3. 或者当前块已经快执行完了 (保护措施)
                need_inference = (current_action_chunk is None) or \
                                 (action_execution_idx >= EXECUTION_HORIZON) or \
                                 (action_execution_idx >= len(current_action_chunk))

                if need_inference:
                    # 构建模型输入：(B=1, To, N, 6) 和 (B=1, To, D)
                    point_clouds = []
                    agent_poses = []
                    
                    for pc, ap in obs_buffer[-n_obs_steps:]:
                        point_clouds.append(pc)  # (N, 6)
                        agent_poses.append(ap)   # (D,)
                    
                    # 堆叠并转换为 torch tensor
                    point_cloud_batch = torch.from_numpy(np.stack(point_clouds)).float().unsqueeze(0)  # (1, To, N, 6)
                    agent_pos_batch = torch.from_numpy(np.stack(agent_poses)).float().unsqueeze(0)      # (1, To, D)
                    
                    model_input = {
                        'point_cloud': point_cloud_batch,
                        'agent_pos': agent_pos_batch
                    }
                    
                    # 推理
                    if args.debug: t0 = time.time()
                    actions = policy.predict_action(model_input)  # (1, horizon, action_dim)
                    if args.debug and step_count % 30 == 0:
                         print(f"[DEBUG] 推理耗时: {(time.time()-t0)*1000:.1f}ms")

                    current_action_chunk = actions[0]  # (horizon, action_dim)
                    action_execution_idx = 0
                    
                # === 获取并执行当前动作 ===
                if current_action_chunk is not None:
                    # 防止索引越界
                    safe_idx = min(action_execution_idx, len(current_action_chunk) - 1)
                    action = current_action_chunk[safe_idx]
                    action_execution_idx += 1
                    
                    if args.debug:
                        if step_count % 30 == 0:
                            left_curr = obs_dict['qpos'][:7]
                            right_curr = obs_dict['qpos'][7:14]
                            left_dp = action[:7]
                            right_dp = action[7:14]
                            
                            print(f"\n[DEBUG] Step {step_count}:")
                            print(f"  Current Left : {np.round(left_curr, 3)}")
                            print(f"  Action  Left : {np.round(left_dp, 3)}")
                            print(f"  Delta   Left : {np.round(left_dp - left_curr, 3)}")
                            print("-" * 40)
                            print(f"  Current Right: {np.round(right_curr, 3)}")
                            print(f"  Action  Right: {np.round(right_dp, 3)}")
                            print(f"  Delta   Right: {np.round(right_dp - right_curr, 3)}")
                    else:
                        # 写入共享内存
                        shm, shape, dtype = shm_dict["action"]
                        np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                        np_array[:] = action
                    
                    step_count += 1
                    if step_count >= args.max_publish_step:
                        break
                
                time.sleep(1.0 / args.frame_rate)
                
            except KeyboardInterrupt:
                print("\n[INFO] 用户中断推理")
                break
            except Exception as e:
                print(f"[ERROR] 推理出错: {e}")
                import traceback
                traceback.print_exc()
                break
    finally:
        # === 保存录制的数据 ===
        # 这个 finally 位于 while 循环外部，确保只在整个流程结束时保存一次
        if args.debug:
            save_debug_data(debug_data_list, args.policy)
    
    print("[INFO] 推理进程结束")
    
    # === 保存录制的数据 === (已移至 finally 块中处理)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='DP3', choices=['DP3', 'GHOST/base', 'GHOST/key'])
    parser.add_argument('--task_name', type=str, default='pick_place_d405')
    # parser.add_argument('--ckpt_name', type=str, help='Checkpoint filename (e.g., 750.ckpt, latest.ckpt)')
    parser.add_argument('--ckpt_name', type=str, default='3000.ckpt', help='Checkpoint filename (e.g., 750.ckpt, latest.ckpt)')
    parser.add_argument('--debug', action='store_true', default=True)
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_publish_step', type=int, default=1000)
    parser.add_argument('--frame_rate', type=int, default=60)
    parser.add_argument('--calibration_dir', type=str, default=str(ROOT / 'calibration_results'))
    parser.add_argument('--data', type=str, default=str(ROOT / 'act/data/config.yaml'))
    parser.add_argument('--camera_names', nargs='+', default=['head', 'left_wrist', 'right_wrist'])
    parser.add_argument('--use_depth_image', action='store_true', default=True)
    parser.add_argument('--use_base', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print(f"{args.policy} 推理脚本 (基于原始多进程架构)")
    print(f"策略: {args.policy}")
    print(f"任务: {args.task_name}")
    print(f"权重: {args.ckpt_name}")
    if args.debug:
        print("⚠️  DEBUG MODE - 不执行动作 ⚠️")
    print("="*80)
    
    # 加载标定数据
    from types import SimpleNamespace
    calib_args = SimpleNamespace(calibration_dir=args.calibration_dir)
    calibration_data = load_calibration_data(calib_args)
    
    # 创建同步对象
    meta_queue = mp.Queue()
    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()
    
    # 启动ROS进程
    print("[INFO] 启动ROS进程...")
    ros_proc = mp.Process(target=ros_process, args=(args, meta_queue, connected_event, start_event, shm_ready_event))
    ros_proc.start()
   
    # 等待ROS初始化
    connected_event.wait()
    print("[INFO] ROS初始化完成")
    
    # 等待用户确认
    input("按Enter开始推理...")
    start_event.set()
    
    # 等待shapes
    shapes = meta_queue.get()
    
    # 创建共享内存
    shm_name_dict = make_shm_name_dict(shapes)
    meta_queue.put(shm_name_dict)
    shm_ready_event.wait()
    
    # 连接共享内存
    shm_dict = connect_shm_dict(shm_name_dict, shapes)
    
    # 启动推理
    try:
        inference_process(args, shm_dict, shapes, calibration_data, ros_proc)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        for shm, _, _ in shm_dict.values():
            shm.close()
        ros_proc.terminate()
        ros_proc.join()
        print("[INFO] 程序结束")


if __name__ == '__main__':
    main()
