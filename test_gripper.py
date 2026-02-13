# -- coding: UTF-8
"""
夹爪独立测试脚本
用于诊断左右夹爪是否受控，以及ID映射是否正确
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

# 确保能导入 act
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

# 必须导入 setup_loader 并执行，否则可能找不到某些依赖
from act.utils.setup_loader import setup_loader
setup_loader(ACT_DIR)

from act.utils.ros_operator import RosOperator

# ================= 参数 =================
DATA_CONFIG = str(ROOT / 'act/data/config.yaml')
SAFE_INIT_POSITION = [0.0, 0, 0, 0, 0.0, 0.0, 0] 

# 定义一些测试用的动作
# 注意：基于你的描述，-2.8 可能是闭合/或者某个极限，0.0 是另一个状态
# 我们需要测试这几个值
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = -2.8  # 如果你的 safe position 里用了 -2.8，我们假设它是闭合
# 这里我们只是假设，实际运行看现象

def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    print(f"工作目录: {ROOT}")
    print(f"配置文件: {DATA_CONFIG}")
    
    # 初始化 ROS
    rclpy.init()
    
    # 模拟 args 对象，RosOperator 需要它
    class Args:
        def __init__(self):
            self.state_dim = 14
            self.arm_type = "airbot_play"
            self.use_depth_image = False  # 简化测试，不订阅深度图
            self.use_base = False
            self.record = None # 不记录
            self.frame_rate = 30
            self.camera_names = ['head', 'left_wrist', 'right_wrist']
            self.ckpt_dir = '/tmp'
            self.ckpt_name = 'test'
            self.episode_path = '/tmp'
    
    args = Args()
    cfg_data = load_yaml(DATA_CONFIG)
    
    print("[INFO] 正在初始化 RosOperator...")
    # in_collect=False 表示推理模式/非采集模式
    ros_operator = RosOperator(args, cfg_data, in_collect=False)
    
    # 等待一会，确保发布者连接上（简单的 sleep 替代 spin 线程，这是单线程测试）
    print("[INFO] 等待设备连接...")
    time.sleep(2)

    # 简单的 spin 替代，确保有回调处理（虽然这里主要是发布）
    # 在这个简单的测试中，我们手动调用几次 spin_once 也可以
    
    try:
        # 1. 初始状态：全部归零 (或者安全位置)
        print("\n=== 测试 1: 发送全 0 位置 (除了左夹爪保持 -2.8 如果那是初始值) ===")
        # 构造一个全 0 的动作，或者你之前用的初始动作
        # init0 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, -2.8] 
        # 我们先试最简单的，全0姿态，看夹爪反应
        
        # 你的 SAFE_INIT_POSITION 是 [0.0, 0, 0, 0, 0.0, 0.0, 0]
        # 注意第7位是 0。
        
        left_action = np.zeros(7, dtype=np.float32)
        right_action = np.zeros(7, dtype=np.float32)
        
        # 将左夹爪设为 0
        left_action[6] = 0.0
        # 将右夹爪设为 0
        right_action[6] = 0.0
        
        print(f"发送: Left={left_action}, Right={right_action}")
        for _ in range(50):
            ros_operator.follow_arm_publish(left_action, right_action)
            rclpy.spin_once(ros_operator, timeout_sec=0.01)
            time.sleep(0.02)
            
        print("请观察：左右夹爪是否都动了？位置是否一致？(理论上都在 0)")
        input("按 Enter 继续测试左夹爪...")

        # 2. 测试左夹爪
        print("\n=== 测试 2: 仅操作左夹爪 (设为 -2.8) ===")
        left_action[6] = -2.8  # 改变左夹爪
        right_action[6] = 0.0  # 右夹爪保持 0
        
        print(f"发送: Left Gripper=-2.8, Right Gripper=0.0")
        for _ in range(50):
            ros_operator.follow_arm_publish(left_action, right_action)
            rclpy.spin_once(ros_operator, timeout_sec=0.01)
            time.sleep(0.02)
            
        print("请观察：只有左夹爪动了吗？")
        input("按 Enter 继续测试右夹爪...")

        # 3. 测试右夹爪
        print("\n=== 测试 3: 仅操作右夹爪 (设为 -2.8, 左边复位到 0) ===")
        left_action[6] = 0.0   # 左夹爪复位
        right_action[6] = -2.8 # 右夹爪动作
        
        print(f"发送: Left Gripper=0.0, Right Gripper=-2.8")
        for _ in range(50):
            ros_operator.follow_arm_publish(left_action, right_action)
            rclpy.spin_once(ros_operator, timeout_sec=0.01)
            time.sleep(0.02)
            
        print("请观察：只有右夹爪动了吗？")
        input("按 Enter 结束测试并归位...")

    except KeyboardInterrupt:
        print("中断")
    
    finally:
        print("清理并退出...")
        rclpy.shutdown()

if __name__ == "__main__":
    main()
