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
ROOT = FILE.parents[1]  # parents[1] 指向项目根目录 ROS2_AC-one_Play
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Add act directory to path (so 'utils' can be imported directly)
ACT_DIR = ROOT / "act"
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

# Load local message definitions & setup paths
from act.utils.setup_loader import setup_loader
# 这里的 msg 目录其实在 act/msg 下，所以应该传入 ACT_DIR
setup_loader(ACT_DIR)

from act.utils.ros_operator import RosOperator, Rate

import rclpy


# ================= 配置 =================
DATA_ROOT = "data_points_test"
SAVE_DIR_COLORS = os.path.join(DATA_ROOT, "colors")
SAVE_DIR_DEPTHS = os.path.join(DATA_ROOT, "depths")
SAVE_DIR_POSES = os.path.join(DATA_ROOT, "poses")
SAVE_DIR_INTRINSICS = os.path.join(DATA_ROOT, "intrinsics")

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading yaml: {e}")
        return None

class ark_collector_args:
    """Mock args object for RosOperator"""
    def __init__(self):
        # 配置文件路径更正：在 act/data/config.yaml 而不是 data/config.yaml
        self.config = os.path.join(ROOT, 'act/data/config.yaml')
        self.camera_names = ['head', 'left_wrist', 'right_wrist']
        self.use_depth_image = True  # 启用深度图
        self.use_base = False
        self.record = 'Distance'
        self.frame_rate = 30
        self.ckpt_dir = '' 
        self.ckpt_name = ''
        self.episode_path = ''

def main():
    # 0. 准备保存目录
    for dir_path in [SAVE_DIR_COLORS, SAVE_DIR_DEPTHS, SAVE_DIR_POSES, SAVE_DIR_INTRINSICS]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 1. 初始化 ROS & RosOperator
    rclpy.init()
    
    args = ark_collector_args()
    config = load_yaml(args.config)
    if config is None:
        print("❌ 无法加载配置文件。")
        return

    print("正在初始化 Ark Robot Operator...")
    ros_operator = RosOperator(args, config, in_collect=True)
    
    # 启动 ROS spin 线程
    def _spin_loop(node):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    # 预热，等待数据流建立
    print("等待设备数据流预热 (2s)...")
    time.sleep(2.0)

    print("\n✅ 方舟数据采集系统已就绪！")
    print("📷 相机: head, left_wrist, right_wrist")
    print("🎮 控制:")
    print("   [Space]: 开始/暂停录制 (10Hz)")
    print("   [Q]: 退出程序")
    print("-" * 50)

    # 保存相机内参（用于后续点云生成）
    intrinsics_saved = False
    camera_names = ['head', 'left_wrist', 'right_wrist']

    count = 0
    recording = False
    last_save_time = 0
    SAVE_INTERVAL = 0.1  # 10Hz

    try:
        while rclpy.ok():
            # 获取观测数据
            obs = ros_operator.get_observation()
            
            if obs is None:
                time.sleep(0.01)
                continue

            # === 保存相机内参（只需保存一次，用于点云生成）===
            if not intrinsics_saved and config:
                for cam_name in camera_names:
                    cam_key = cam_name + '_camera' if cam_name != 'head' else 'head_camera'
                    if cam_key in config:
                        intrinsics = {
                            'fx': config[cam_key]['camera_matrix']['data'][0],
                            'fy': config[cam_key]['camera_matrix']['data'][4],
                            'cx': config[cam_key]['camera_matrix']['data'][2],
                            'cy': config[cam_key]['camera_matrix']['data'][5],
                            'distortion': config[cam_key]['distortion_coefficients']['data'],
                            'width': config[cam_key]['image_width'],
                            'height': config[cam_key]['image_height']
                        }
                        intrinsics_path = os.path.join(SAVE_DIR_INTRINSICS, f"{cam_name}_intrinsics.json")
                        with open(intrinsics_path, 'w') as f:
                            json.dump(intrinsics, f, indent=2)
                intrinsics_saved = True
                print("✅ 相机内参已保存")
                
                # 调试信息
                print(f"🔍 调试信息:")
                print(f"   obs keys: {obs.keys()}")
                if 'images' in obs:
                    print(f"   RGB cameras: {list(obs['images'].keys())}")
                if 'images_depth' in obs:
                    print(f"   Depth cameras: {list(obs['images_depth'].keys())}")
                else:
                    print(f"   ⚠️  'images_depth' 不在 obs 中")

            # === 准备显示和保存数据 ===
            displays_rgb = []
            displays_depth = []
            capture_data = {}  # {camera_name: (rgb, depth)}
            
            for cam_name in camera_names:
                # RGB
                if cam_name in obs['images']:
                    rgb_img = obs['images'][cam_name]
                else:
                    rgb_img = None
                
                # Depth (注意：obs 里的 key 是 'images_depth'，不是 'depth_images')
                depth_img = None
                if 'images_depth' in obs and cam_name in obs['images_depth']:
                    depth_img = obs['images_depth'][cam_name]
                
                # 显示逻辑：RGB + Depth 可视化
                if rgb_img is not None:
                    # RGB 预览
                    small_rgb = cv2.resize(rgb_img, (320, 240))
                    if recording:
                        cv2.circle(small_rgb, (300, 20), 8, (0, 0, 255), -1)  # 红点表示录制中
                    
                    # 显示相机名称和状态
                    display_name = cam_name.replace('_wrist', '')
                    status_color = (0, 255, 0) if depth_img is not None else (0, 165, 255)
                    status_text = f"{display_name} [D+]" if depth_img is not None else f"{display_name} [D-]"
                    cv2.putText(small_rgb, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    displays_rgb.append(small_rgb)
                    
                    # Depth 可视化
                    if depth_img is not None:
                        # 固定深度范围进行可视化（避免动态归一化导致的跳变）
                        # 设置深度范围：0mm - 1000mm
                        DEPTH_MIN = 0  # mm
                        DEPTH_MAX = 1000  # mm

                        # 裁剪到有效范围
                        depth_clipped = np.clip(depth_img, DEPTH_MIN, DEPTH_MAX)
                        
                        # 归一化到0-255
                        depth_normalized = ((depth_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
                        
                        # 应用伪彩色
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        small_depth = cv2.resize(depth_colored, (320, 240))
                        
                        # 添加深度统计信息
                        valid_depth = depth_img[depth_img > 0]
                        if len(valid_depth) > 0:
                            depth_mean = int(valid_depth.mean())
                            depth_std = int(valid_depth.std())
                            depth_text = f"Mean:{depth_mean}mm Std:{depth_std}mm"
                        else:
                            depth_text = "NO VALID DEPTH"
                        
                        cv2.putText(small_depth, depth_text, (5, 220), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        displays_depth.append(small_depth)
                        
                        # 保存数据
                        capture_data[cam_name] = (rgb_img, depth_img)
                    else:
                        # 无深度图，显示黑屏
                        black_depth = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(black_depth, "NO DEPTH", (80, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        displays_depth.append(black_depth)
                else:
                    # RGB 缺失，显示黑屏
                    black_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(black_rgb, f"{cam_name} NO RGB", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    displays_rgb.append(black_rgb)
                    
                    black_depth = np.zeros((240, 320, 3), dtype=np.uint8)
                    displays_depth.append(black_depth)

            # 显示预览：上排RGB，下排Depth
            if displays_rgb and displays_depth:
                row_rgb = np.hstack(displays_rgb)
                row_depth = np.hstack(displays_depth)
                combined = np.vstack([row_rgb, row_depth])
                cv2.imshow("Ark Data Collector Preview", combined)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                recording = not recording
                if recording:
                    print(f"\n▶️  开始录制... (目标 10Hz, 下一帧: {count})")
                else:
                    print(f"\n⏸️  暂停录制. (已保存: {count} 帧)")

            # === 自动采集逻辑 ===
            if recording:
                current_time = time.time()
                
                # 检查时间间隔 (10Hz)
                if current_time - last_save_time >= SAVE_INTERVAL:
                    # 检查数据完整性：所有相机都有数据
                    if len(capture_data) == len(camera_names):
                        # 获取机械臂位姿
                        # obs['eef'] = [left_7d, right_7d] = [x,y,z,r,p,y,gripper] * 2
                        pose_left = obs['eef'][0:7].tolist()   # [x,y,z,r,p,y,gripper]
                        pose_right = obs['eef'][7:14].tolist()
                        
                        # 保存所有相机的 RGB 和 Depth
                        for cam_name, (rgb, depth) in capture_data.items():
                            prefix = f"{count:04d}_{cam_name}"
                            
                            # RGB (BGR格式)
                            cv2.imwrite(os.path.join(SAVE_DIR_COLORS, f"{prefix}_camera.jpg"), rgb)
                            
                            # Depth (原始uint16深度值，单位：毫米，与RGB对齐)
                            # 可用于点云生成：depth_m = depth_array / 1000.0
                            depth_path = os.path.join(SAVE_DIR_DEPTHS, f"{prefix}_camera.npy")
                            np.save(depth_path, depth.astype(np.uint16))
                        
                        # 保存位姿
                        with open(os.path.join(SAVE_DIR_POSES, f"{count:04d}_left.json"), 'w') as f:
                            json.dump({
                                "pose": pose_left[:6],  # [x,y,z,r,p,y]
                                "gripper": pose_left[6],
                                "unit": "m, rad"
                            }, f, indent=2)
                        
                        with open(os.path.join(SAVE_DIR_POSES, f"{count:04d}_right.json"), 'w') as f:
                            json.dump({
                                "pose": pose_right[:6],
                                "gripper": pose_right[6],
                                "unit": "m, rad"
                            }, f, indent=2)
                        
                        freq = 1.0 / (current_time - last_save_time)
                        print(f"\r[REC] ✅ Saved Frame {count:04d} | Freq: {freq:.1f}Hz", end="")
                        
                        count += 1
                        last_save_time = current_time
                    else:
                        # 数据未对齐，等待下一帧
                        pass
                        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    finally:
        print(f"\n\n📊 采集完成！共保存 {count} 帧数据")
        print(f"   RGB: {SAVE_DIR_COLORS}")
        print(f"   Depth (uint16, mm): {SAVE_DIR_DEPTHS}")
        print(f"   Poses: {SAVE_DIR_POSES}")
        print(f"   Intrinsics: {SAVE_DIR_INTRINSICS}")
        print(f"\n💡 生成点云示例:")
        print(f"   depth = np.load('depths/0000_head_camera.npy')")
        print(f"   depth_m = depth.astype(float) / 1000.0  # 转换为米")
        
        ros_operator.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        spin_thread.join(timeout=2.0)

if __name__ == '__main__':
    main()
