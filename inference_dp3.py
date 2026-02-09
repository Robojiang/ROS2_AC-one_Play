# -- coding: UTF-8
import os
import sys
import time
import threading
import signal
from functools import partial
from pathlib import Path
import argparse
import collections

import numpy as np
import torch
import yaml
import h5py
import json
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# 强制刷新输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import rclpy
from rclpy.executors import MultiThreadedExecutor

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader
from multiprocessing.shared_memory import SharedMemory

# DP3策略导入
from policy.DP3.dp3_policy import DP3
from omegaconf import OmegaConf

# 设置打印输出行宽
np.set_printoptions(linewidth=200, suppress=True)

# 安全初始位置（机械臂归位姿态）
SAFE_INIT_POSITION = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, -2.8]

# 点云生成配置
MAX_DEPTH_Head = 1.0  # 米
MAX_DEPTH_Hand = 0.6  # 米
FPS_SAMPLE_POINTS = 1024  # 点云采样点数

# 工作空间裁剪 (相对于左臂基座坐标系)
USE_WORKSPACE_CROP = True
WORKSPACE_X_RANGE = [-0.4, 0.5]
WORKSPACE_Y_RANGE = [-0.5, 3.0]
WORKSPACE_Z_RANGE = [-0.2, 1.0]


def load_yaml(yaml_file):
    """加载YAML配置文件"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")
        return None


# ==================== 点云生成函数 ====================

def load_intrinsics(intrinsics_file, camera_name):
    """加载相机内参"""
    with open(intrinsics_file, 'r') as f:
        all_data = json.load(f)
    d = all_data[camera_name]
    return d['fx'], d['fy'], d['cx'], d['cy']


def load_calibration_matrix(calibration_dir, filename):
    """加载标定矩阵"""
    path = os.path.join(calibration_dir, filename)
    if os.path.exists(path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.txt'):
            return np.loadtxt(path).reshape(4, 4)
    print(f"❌ 缺少标定文件: {filename}")
    return np.eye(4)


def eef_to_matrix(eef_pose):
    """将end-effector pose转换为4x4变换矩阵"""
    if eef_pose is None or len(eef_pose) < 6:
        return np.eye(4)
    t = np.array(eef_pose[:3])
    r = R.from_euler('xyz', eef_pose[3:6]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def depth_to_point_cloud(depth_img, color_img, fx, fy, cx, cy, max_depth=None):
    """将深度图和彩色图转换为点云"""
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    valid = depth_img > 0
    if max_depth is not None:
        valid = valid & (depth_img < max_depth * 1000)
    
    z = depth_img[valid].astype(np.float32) / 1000.0
    u = u[valid]
    v = v[valid]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # RGB归一化到[0, 1]
    r = color_img[valid, 0].astype(np.float32) / 255.0
    g = color_img[valid, 1].astype(np.float32) / 255.0
    b = color_img[valid, 2].astype(np.float32) / 255.0
    
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    
    return np.hstack((xyz, rgb))


def transform_point_cloud(cloud, T):
    """变换点云"""
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:]
    
    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.hstack((xyz, ones))
    xyz_trans = (T @ xyz_homo.T).T
    
    return np.hstack((xyz_trans[:, :3], rgb))


def crop_point_cloud(cloud_np, x_range, y_range, z_range):
    """裁剪点云"""
    xyz = cloud_np[:, :3]
    mask = (
        (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
        (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1]) &
        (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    )
    return cloud_np[mask]


def numpy_to_o3d(cloud_np):
    """转换numpy数组到open3d点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_np[:, 3:])
    return pcd


def generate_point_cloud(head_depth, head_color, left_depth, left_color, 
                         right_depth, right_color, left_eef, right_eef,
                         intrinsics, T_H_LB, T_H_RB, T_LE_LC, T_RE_RC, T_LB_H):
    """
    生成单帧点云 (在左臂基座坐标系下)
    返回: (N, 6) numpy array, N <= FPS_SAMPLE_POINTS
    """
    clouds_global = []
    
    # 1. Head Camera
    fx, fy, cx, cy = intrinsics['head']
    pc_head = depth_to_point_cloud(head_depth, head_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Head)
    if len(pc_head) > 0:
        clouds_global.append(pc_head)
    
    # 2. Left Wrist Camera
    fx, fy, cx, cy = intrinsics['left']
    pc_left = depth_to_point_cloud(left_depth, left_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    if len(pc_left) > 0:
        T_LB_LE = eef_to_matrix(left_eef)
        T_total_left = T_H_LB @ T_LB_LE @ T_LE_LC
        pc_left_global = transform_point_cloud(pc_left, T_total_left)
        clouds_global.append(pc_left_global)
    
    # 3. Right Wrist Camera
    fx, fy, cx, cy = intrinsics['right']
    pc_right = depth_to_point_cloud(right_depth, right_color, fx, fy, cx, cy, max_depth=MAX_DEPTH_Hand)
    if len(pc_right) > 0:
        T_RB_RE = eef_to_matrix(right_eef)
        T_total_right = T_H_RB @ T_RB_RE @ T_RE_RC
        pc_right_global = transform_point_cloud(pc_right, T_total_right)
        clouds_global.append(pc_right_global)
    
    if len(clouds_global) == 0:
        # 返回空点云
        return np.zeros((FPS_SAMPLE_POINTS, 6), dtype=np.float32)
    
    # 4. 合并并转换到左臂基座坐标系
    merged_cloud = np.vstack(clouds_global)
    merged_cloud = transform_point_cloud(merged_cloud, T_LB_H)
    
    # 5. 工作空间裁剪
    if USE_WORKSPACE_CROP:
        merged_cloud = crop_point_cloud(merged_cloud, WORKSPACE_X_RANGE, 
                                       WORKSPACE_Y_RANGE, WORKSPACE_Z_RANGE)
    
    if len(merged_cloud) == 0:
        return np.zeros((FPS_SAMPLE_POINTS, 6), dtype=np.float32)
    
    # 6. 下采样
    pcd = numpy_to_o3d(merged_cloud)
    
    # 去噪
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 体素下采样
    pcd_voxel = pcd_clean.voxel_down_sample(voxel_size=0.005)
    
    # FPS采样
    if len(pcd_voxel.points) > FPS_SAMPLE_POINTS:
        pcd_fps = pcd_voxel.farthest_point_down_sample(FPS_SAMPLE_POINTS)
    else:
        pcd_fps = pcd_voxel
    
    # 转换回numpy
    pts = np.asarray(pcd_fps.points)
    clrs = np.asarray(pcd_fps.colors)
    result = np.hstack((pts, clrs)).astype(np.float32)
    
    # Pad到固定大小
    if len(result) < FPS_SAMPLE_POINTS:
        padding = np.zeros((FPS_SAMPLE_POINTS - len(result), 6), dtype=np.float32)
        result = np.vstack((result, padding))
    
    return result


# ==================== 机械臂控制函数 ====================

def move_to_target(ros_operator, target_pos, steps=50):
    """平滑移动到目标位置"""
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array(target_pos)
    
    for i in range(steps):
        alpha = (i + 1) / steps
        pos = start * (1 - alpha) + end * alpha
        ros_operator.follow_arm_publish_continuous(pos, pos)
        time.sleep(0.01)


def move_to_safe_position(ros_operator, steps=400):
    """将机械臂从当前位置平滑移动到安全初始位置"""
    print("\n[INFO] 正在将机械臂移动到安全初始位置...")
    try:
        current_qpos = ros_operator.get_observation()
        if current_qpos is None:
            print("[WARN] 无法获取当前位置，直接移动到安全位置")
            move_to_target(ros_operator, SAFE_INIT_POSITION, steps=steps)
            return
        
        current_pos = current_qpos['qpos'][:7]
        target_pos = np.array(SAFE_INIT_POSITION)
        
        # 平滑插值
        for i in range(steps):
            alpha = (i + 1) / steps
            pos = current_pos * (1 - alpha) + target_pos * alpha
            ros_operator.follow_arm_publish_continuous(pos, pos)
            time.sleep(0.01)
        
        print("[INFO] 机械臂已到达安全位置")
    except Exception as e:
        print(f"[ERROR] 移动到安全位置时出错: {e}")


def init_robot(ros_operator, use_base, connected_event, start_event):
    """初始化机器人"""
    init0 = SAFE_INIT_POSITION
    init1 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    
    # 发布初始位置（关节空间姿态）
    move_to_target(ros_operator, init0, steps=100)
    
    connected_event.set()
    start_event.wait()
    
    ros_operator.follow_arm_publish_continuous(init1, init1)
    if use_base:
        ros_operator.robot_base_shutdown()


def signal_handler(sig, frame, ros_operator, use_base):
    """信号处理器，确保退出时机械臂回到安全位置"""
    print('\n[INFO] 捕获到退出信号，正在安全关闭...')
    
    # 先让机械臂回到安全位置
    move_to_safe_position(ros_operator, steps=100)
    
    # 底盘给零
    if use_base:
        ros_operator.robot_base_shutdown()
    
    print('[INFO] 安全关闭完成')
    sys.exit(0)


# ==================== DP3推理进程 ====================

def inference_process(args, config, policy, calibration_data, ros_proc):
    """DP3推理进程"""
    print("[INFO] DP3推理进程启动")
    
    max_publish_step = args.max_publish_step
    n_obs_steps = config.n_obs_steps
    n_action_steps = config.n_action_steps
    
    # 观测历史缓冲区
    obs_buffer = collections.deque(maxlen=n_obs_steps)
    
    # 动作队列
    action_queue = collections.deque()
    
    step_count = 0
    
    # 等待ROS进程准备好
    time.sleep(2)
    
    print(f"[INFO] 开始推理循环 (max_steps={max_publish_step})")
    
    rate = Rate(args.frame_rate)
    
    while ros_proc.is_alive() and step_count < max_publish_step:
        try:
            # 1. 获取观测数据
            obs_dict = ros_proc.ros_operator.get_observation()
            
            if obs_dict is None:
                print("[DEBUG] 等待观测数据同步...")
                rate.sleep()
                continue
            
            # 2. 生成点云
            point_cloud = generate_point_cloud(
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
            
            # 3. 构建观测
            curr_obs = {
                'point_cloud': point_cloud,  # (N, 6)
                'agent_pos': obs_dict['qpos']  # (14,) - joint positions
            }
            
            obs_buffer.append(curr_obs)
            
            # 4. 如果观测历史不足，继续收集
            if len(obs_buffer) < n_obs_steps:
                print(f"[DEBUG] 收集观测历史 {len(obs_buffer)}/{n_obs_steps}")
                rate.sleep()
                continue
            
            # 5. 如果动作队列为空，执行推理
            if len(action_queue) == 0:
                # 准备batch
                obs_list = list(obs_buffer)
                
                # Stack observations
                point_clouds = np.stack([o['point_cloud'] for o in obs_list], axis=0)  # (T, N, 6)
                agent_positions = np.stack([o['agent_pos'] for o in obs_list], axis=0)  # (T, 14)
                
                # 转换为torch tensor并添加batch维度
                batch = {
                    'obs': {
                        'point_cloud': torch.from_numpy(point_clouds).float().cuda().unsqueeze(0),  # (1, T, N, 6)
                        'agent_pos': torch.from_numpy(agent_positions).float().cuda().unsqueeze(0)  # (1, T, 14)
                    }
                }
                
                # 推理
                with torch.no_grad():
                    actions = policy.get_action(batch)  # (1, horizon, action_dim)
                
                # 提取动作并放入队列
                actions = actions[0].cpu().numpy()  # (horizon, action_dim)
                for i in range(n_action_steps):
                    action_queue.append(actions[i])
                
                print(f"[INFO] 步骤 {step_count}: 生成 {n_action_steps} 个动作")
            
            # 6. 从队列中取出动作并执行
            action = action_queue.popleft()
            
            # 处理夹爪动作 (根据具体需求调整)
            # action: (14,) - [left_joints(7), right_joints(7)]
            left_action = action[:7]
            right_action = action[7:14]
            
            # 发布动作
            ros_proc.ros_operator.follow_arm_publish_continuous(left_action, right_action)
            
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"[INFO] 推理步骤: {step_count}/{max_publish_step}")
            
            rate.sleep()
            
        except KeyboardInterrupt:
            print("[INFO] 推理进程收到中断信号")
            break
        except Exception as e:
            print(f"[ERROR] 推理循环出错: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("[INFO] DP3推理进程结束")


# ==================== ROS进程 ====================

class RosProcess:
    """ROS进程封装"""
    def __init__(self, args, connected_event, start_event):
        self.args = args
        self.connected_event = connected_event
        self.start_event = start_event
        self.ros_operator = None
        self._alive = True
        
    def is_alive(self):
        return self._alive
        
    def run(self):
        """ROS进程主循环"""
        setup_loader(ROOT)
        rclpy.init()
        
        data = load_yaml(self.args.data)
        self.ros_operator = RosOperator(self.args, data, in_collect=False)
        
        def _spin_loop(node):
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.001)
        
        spin_thread = threading.Thread(target=_spin_loop, args=(self.ros_operator,), daemon=True)
        spin_thread.start()
        
        # 注册信号处理
        handler = partial(signal_handler, ros_operator=self.ros_operator, use_base=self.args.use_base)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        
        # 初始化机器人
        init_robot(self.ros_operator, self.args.use_base, self.connected_event, self.start_event)
        
        print("[INFO] ROS进程准备完成")
        
        # 保持运行
        try:
            while rclpy.ok():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[INFO] ROS进程收到中断信号")
        finally:
            self._alive = False
            self.ros_operator.destroy_node()
            rclpy.shutdown()
            spin_thread.join()


# ==================== 主函数 ====================

def load_dp3_policy(args):
    """加载DP3策略"""
    print(f"[INFO] 加载DP3策略: {args.ckpt_dir}")
    
    # 1. 加载hydra配置
    config_path = Path(args.ckpt_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    print(f"[INFO] 加载配置: {config_path}")
    
    # 2. 创建DP3策略
    policy = DP3(cfg, args)
    
    # 3. 加载权重
    ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")
    
    print(f"[INFO] 加载权重: {ckpt_path}")
    policy.policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    policy.policy.cuda()
    policy.policy.eval()
    
    print("[INFO] DP3策略加载完成")
    
    return policy, cfg


def load_calibration_data(args):
    """加载标定数据"""
    calibration_dir = Path(args.calibration_dir)
    intrinsics_file = calibration_dir / "D405_intrinsics.json"
    
    print(f"[INFO] 加载标定数据: {calibration_dir}")
    
    # 加载标定矩阵
    T_LE_LC = load_calibration_matrix(calibration_dir, "left_eye_in_hand.npy")
    T_RE_RC = load_calibration_matrix(calibration_dir, "right_eye_in_hand.npy")
    T_LB_H = load_calibration_matrix(calibration_dir, "head_base_to_left_refined_icp.txt")
    T_RB_H = load_calibration_matrix(calibration_dir, "head_base_to_right_refined_icp.txt")
    
    # 注意: 文件名head_base_to_left实际表示 Head->LeftBase 的变换
    T_H_LB = T_LB_H
    T_H_RB = np.linalg.inv(T_RB_H)
    
    # 加载内参
    intrinsics = {
        'head': load_intrinsics(intrinsics_file, 'head'),
        'left': load_intrinsics(intrinsics_file, 'left_wrist'),
        'right': load_intrinsics(intrinsics_file, 'right_wrist')
    }
    
    print("[INFO] 标定数据加载完成")
    
    return {
        'intrinsics': intrinsics,
        'T_H_LB': T_H_LB,
        'T_H_RB': T_H_RB,
        'T_LE_LC': T_LE_LC,
        'T_RE_RC': T_RE_RC,
        'T_LB_H': T_LB_H
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DP3策略真机推理脚本")
    
    # 推理设置
    parser.add_argument('--max_publish_step', type=int, default=1000, help='最大推理步数')
    parser.add_argument('--frame_rate', type=int, default=30, help='控制频率(Hz)')
    
    # 模型和权重
    parser.add_argument('--ckpt_dir', type=str, 
                        default=Path.joinpath(ROOT, 'weights/pick_place_d405/DP3'),
                        help='权重目录')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt',
                        help='权重文件名')
    
    # 标定数据
    parser.add_argument('--calibration_dir', type=str,
                        default=Path.joinpath(ROOT, 'calibration_results'),
                        help='标定数据目录')
    
    # 配置文件
    parser.add_argument('--data', type=str,
                        default=Path.joinpath(ROOT, 'act/data/config.yaml'),
                        help='ROS配置文件')
    
    # 相机设置
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='相机名称')
    parser.add_argument('--use_depth_image', action='store_true', default=True,
                        help='使用深度图')
    
    # 机器人设置
    parser.add_argument('--use_base', action='store_true', help='使用底盘')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("DP3策略真机推理脚本")
    print("="*80)
    
    # 1. 加载DP3策略
    policy, config = load_dp3_policy(args)
    
    # 2. 加载标定数据
    calibration_data = load_calibration_data(args)
    
    # 3. 创建同步事件
    connected_event = threading.Event()
    start_event = threading.Event()
    
    # 4. 启动ROS进程
    print("[INFO] 启动ROS进程...")
    ros_proc = RosProcess(args, connected_event, start_event)
    ros_thread = threading.Thread(target=ros_proc.run, daemon=False)
    ros_thread.start()
    
    # 等待ROS连接
    connected_event.wait()
    print("[INFO] ROS连接成功")
    
    # 等待用户确认开始
    input("按Enter键开始推理...")
    start_event.set()
    
    # 5. 启动推理进程
    print("[INFO] 启动推理进程...")
    try:
        inference_process(args, config, policy, calibration_data, ros_proc)
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号")
    except Exception as e:
        print(f"\n[ERROR] 推理过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        print("[INFO] 正在清理资源...")
        move_to_safe_position(ros_proc.ros_operator, steps=100)
        if args.use_base:
            ros_proc.ros_operator.robot_base_shutdown()
        print("[INFO] 程序结束")


if __name__ == '__main__':
    main()
