#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
import yaml
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]     # 指向 ROS2_AC-one_Play
ROOT = PROJECT_ROOT / "act"        # 将 ROOT 指向 act 文件夹
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    sys.path.append(str(PROJECT_ROOT)) # 也添加项目根目录，方便访问 calibration_data_ark 等

# Load local message definitions & setup paths
# 必须先调用 setup_loader，否则 RosOperator 内部 import 消息会失败
from act.utils.ros_operator import RosOperator, Rate
from act.utils.setup_loader import setup_loader
setup_loader(ROOT)

import rclpy

# ================= 配置区域 =================
# 数据保存根目录
SAVE_DIR_ROOT = "calibration_data_ark"

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading yaml: {e}")
        return None

class calibration_args:
    """Mock args object mimicking argparse for RosOperator"""
    def __init__(self):
        self.config = os.path.join(ROOT, 'data/config.yaml')
        self.camera_names = ['head', 'left_wrist', 'right_wrist']
        self.use_depth_image = False
        self.use_base = False
        self.record = 'Distance'
        self.frame_rate = 30 # Calibration doesn't need high speed
        # Add attributes expected by RosOperator
        self.ckpt_dir = '' 
        self.ckpt_name = ''
        self.episode_path = ''

def main():
    # 0. 模式选择
    print("\n" + "="*50)
    print("  Ark X5 机械臂标定采集程序 (ROS2版)")
    print("="*50)
    print("请选择采集模式：")
    print("  [1] 左臂眼在手 (Left Eye-in-Hand)")
    print("  [2] 右臂眼在手 (Right Eye-in-Hand)")
    print("  [3] 固定相机 -> 左臂 (Head Camera -> Left Arm)")
    print("  [4] 固定相机 -> 右臂 (Head Camera -> Right Arm)")
    
    try:
        choice = input("请输入模式编号 (1/2/3/4): ").strip()
    except KeyboardInterrupt:
        print("\n\n退出。")
        return

    mode_config = {}
    save_subdir = ""
    
    # 根据 RosOperator 的命名习惯映射
    # RosOperator: obs_dict['images'][camera_name], obs_dict['eef']
    
    target_cam = ""
    target_arm_idx = 0 # 0 for left (0:6), 1 for right (7:13)
    
    if choice == '1':
        target_cam = 'left_wrist'
        target_arm_idx = 0 
        save_subdir = "left_eye_in_hand"
    elif choice == '2':
        target_cam = 'right_wrist'
        target_arm_idx = 1
        save_subdir = "right_eye_in_hand"
    elif choice == '3':
        target_cam = 'head'
        target_arm_idx = 0
        save_subdir = "head_base_to_left"
    elif choice == '4':
        target_cam = 'head'
        target_arm_idx = 1
        save_subdir = "head_base_to_right"
    else:
        print("无效输入")
        return

    # 1. 初始化 ROS & RosOperator
    rclpy.init()
    
    args = calibration_args()
    config = load_yaml(args.config)
    if config is None:
        print("无法加载配置文件。")
        return

    print("正在初始化 Robot Operator...")
    # in_collect=True 确保订阅控制器状态，虽然标定主要用 feedback
    # 注意：回退到 in_collect=True，这样不会发布 Topic，避免任何控制冲突
    ros_operator = RosOperator(args, config, in_collect=True)
    
    # 启动 spin 线程
    def _spin_loop(node):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    # == 关键修改：预热 2 秒 ==
    # RosOperator 使用的是队列 pop() 模式。如果刚启动还没有数据进来就去取，会报错。
    # 等待几秒让后台线程接收第一批数据。
    print("等待设备数据流预热 (2s)...")
    time.sleep(2.0)

    # 2. 准备保存路径
    save_path = os.path.join(SAVE_DIR_ROOT, save_subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"\n✅ 采集程序已就绪！保存路径: {save_path}")
    print("🎮 按键说明 (Ark 键盘):")
    print("   [键 1 (Index 0)]: 激活采集/标记重力补偿模式 (Start)")
    print("   [键 3 (Index 2)]: 保存当前数据 (Save)") 
    # collect.py 默认逻辑: 0是开始/init, 2是删除/save. 
    
    print("   [Ctrl+C]: 退出程序")

    count = 0
    active = False 

    try:
        while rclpy.ok():
            # Get observation (会等待数据同步)
            # 注意: ros_operator.get_observation() 内部会检查 deque 是否为空
            # 如果为空会打印日志并返回None，我们加一点延时防止刷屏
            obs = ros_operator.get_observation()
            
            if obs is None:
                # print("Waiting for data...", end='\r')
                time.sleep(0.1)
                continue

            # Extract Image
            if target_cam in obs['images']:
                frame = obs['images'][target_cam]
            else:
                print(f"Waiting for camera: {target_cam}", end='\r')
                continue
                
            # 确保图像是有效的 numpy array
            if frame is None or frame.size == 0:
                continue

            # Extract Pose
            # obs['eef'] is concatenation of left (7) and right (7)
            # Each is [x,y,z,r,p,y, gripper]
            start_idx = target_arm_idx * 7
            pose_7d = obs['eef'][start_idx : start_idx+7]
            pose_6d = pose_7d[:6] # x,y,z,r,p,y

            # Check Keys
            triggered_keys = []
            with ros_operator.joy_lock:
                triggered_keys = list(ros_operator.triggered_joys.keys())
                ros_operator.triggered_joys.clear()

            # Logic
            if 0 in triggered_keys: # Key 1
                active = True
                print("\n[状态] 激活! 请移动机械臂到标定点")
            
            if 2 in triggered_keys: # Key 3
                if not active:
                    print("\n[警告] 请先按 键1 激活!")
                else:
                    img_name = f"img_{count:04d}.jpg"
                    json_name = f"pose_{count:04d}.json"
                    
                    cv2.imwrite(os.path.join(save_path, img_name), frame)
                    
                    # Store as list for JSON
                    pose_list = pose_6d.tolist()
                    
                    # 补充: 保存 gripper 状态，有时可能需要确认夹爪是否夹紧
                    gripper_state = float(pose_7d[6])
                    
                    data = {
                        "id": count,
                        "pose": pose_list,
                        "gripper": gripper_state,
                        "units": "m, rad (x,y,z,r,p,y)",
                        "mode": save_subdir
                    }
                    with open(os.path.join(save_path, json_name), 'w') as f:
                        json.dump(data, f, indent=4)
                        
                    print(f"\n[保存] 第 {count} 组数据已保存。")
                    count += 1
                    
                    # Visual feedback
                    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,255,0), 10)
                    cv2.imshow("Ark Calibration", frame)
                    cv2.waitKey(200)

            # Display
            disp = frame.copy()
            status_text = "ACTIVE (Press 3 Save)" if active else "WAITING (Press 1)"
            status_color = (0, 255, 0) if active else (0, 0, 255)
            
            cv2.putText(disp, f"Mode: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            cv2.putText(disp, f"Count: {count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            p_str = f"XYZ: {pose_6d[0]:.2f} {pose_6d[1]:.2f} {pose_6d[2]:.2f}"
            cv2.putText(disp, p_str, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Ark Calibration", disp)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        ros_operator.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        spin_thread.join()

if __name__ == '__main__':
    main()
