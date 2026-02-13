# -- coding: UTF-8
"""
夹爪深度诊断脚本 (带状态反馈)
"""
import sys
import os
import time
import numpy as np
import yaml
import rclpy
from pathlib import Path

# ================= 配置路径 =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ACT_DIR = ROOT / "act"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

from act.utils.setup_loader import setup_loader
setup_loader(ACT_DIR)
from act.utils.ros_operator import RosOperator

# ================= 参数 =================
DATA_CONFIG = str(ROOT / 'act/data/config.yaml')

def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    rclpy.init()
    
    class Args:
        def __init__(self):
            self.state_dim = 14
            self.arm_type = "airbot_play"
            self.use_depth_image = False
            self.use_base = False
            self.record = None
            self.frame_rate = 60 # 提高频率以便接收反馈
            # 关键修改：清空相机列表，避免因为没有图像数据导致 get_observation 返回 None
            self.camera_names = [] 
            self.ckpt_dir = '/tmp'
            self.ckpt_name = 'test'
            self.episode_path = '/tmp'
    
    args = Args()
    cfg_data = load_yaml(DATA_CONFIG)
    ros_operator = RosOperator(args, cfg_data, in_collect=False)
    
    print("[INFO] 等待设备连接并获取初始状态...")
    time.sleep(1)
    
    # 辅助函数：执行动作并监控反馈
    def move_and_check(target_left, target_right, steps=80):
        print(f"\n>>> 目标: 左={target_left:.2f}, 右={target_right:.2f}")
        
        left_cmd = np.zeros(7, dtype=np.float32)
        right_cmd = np.zeros(7, dtype=np.float32)
        
        # 保持其他关节为0，只动夹爪
        left_cmd[6] = target_left
        right_cmd[6] = target_right
        
        for i in range(steps):
            # 发送指令
            ros_operator.follow_arm_publish(left_cmd, right_cmd)
            rclpy.spin_once(ros_operator, timeout_sec=0.005)
            
            # 获取反馈
            obs = ros_operator.get_observation()
            if obs is not None:
                qpos = obs['qpos']
                curr_left = qpos[6]
                curr_right = qpos[13]
                
                # 每10帧打印一次
                if i % 10 == 0:
                     print(f"    [反馈] Left: {curr_left:.3f} | Right: {curr_right:.3f}")
            
            time.sleep(0.02)

    try:
        # 1. 初始读取
        for _ in range(10): 
            rclpy.spin_once(ros_operator, timeout_sec=0.1)
            
        print("\n=== 阶段 0: 尝试归零 ===")
        move_and_check(0.0, 0.0)

        # 2. 渐进式测试左爪
        print("\n=== 阶段 1: 左爪温和测试 (0 -> -1.0) ===")
        move_and_check(-1.0, 0.0)
        
        print("\n=== 阶段 2: 左爪尝试回弹 ( -1.0 -> 0.0) ===")
        move_and_check(0.0, 0.0)
        
        # 如果阶段2成功，说明小幅度没问题，测试大幅度
        print("\n=== 阶段 3: 左爪极限测试 (0 -> -2.8) ===")
        move_and_check(-2.8, 0.0)
        
        print("\n=== 阶段 4: 左爪极限回弹 (-2.8 -> 0.0) ===")
        move_and_check(0.0, 0.0)
        
        # 为了对比，看看正值有没有反应
        print("\n=== 阶段 5: 尝试正值 (如果是松开?) 0 -> 0.5 ===")
        move_and_check(0.5, 0.0)
        move_and_check(0.0, 0.0)

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        print("清理...")
        rclpy.shutdown()

if __name__ == "__main__":
    main()
