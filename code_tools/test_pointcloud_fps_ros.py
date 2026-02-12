#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
基于 ROS 的实时点云生成帧率测试
- 直接使用 RosOperator 获取真实数据
- 统计点云生成耗时与 FPS
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ACT_DIR = ROOT / "act"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

import cv2
import rclpy
from act.utils.setup_loader import setup_loader
from act.utils.ros_operator import RosOperator, Rate
from inference_utils.pointcloud_generator import PointCloudGenerator
from inference_utils.calibration import load_calibration_data


def render_view(x, y, colors, img_size, label):
    """Simple 2D projection renderer using OpenCV"""
    img = np.full((img_size, img_size, 3), 30, dtype=np.uint8) # Dark background
    scale = img_size / 1.5  # Zoom factor
    offset_x = img_size / 2
    offset_y = img_size / 2
    
    # Project to image coordinates
    # We want (0,0) to be at center
    u = (x * scale + offset_x).astype(int)
    v = (y * scale + offset_y).astype(int)
    v = img_size - v # Flip Y axis to match image convention (top-left origin)
    
    # Filter valid points
    valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    u = u[valid]
    v = v[valid]
    if len(colors) > 0:
        c = (colors[valid] * 255).astype(np.uint8)
    else:
        c = np.zeros((len(u), 3), dtype=np.uint8)
    
    # Draw points
    # Fast drawing by accessing pixels directly is hard in Python, using circle loop
    # For 1024 points loop is fast enough
    for j in range(len(u)):
        # cv2 uses BGR
        color = (int(c[j][2]), int(c[j][1]), int(c[j][0]))
        cv2.circle(img, (u[j], v[j]), 2, color, -1)
        
    cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

def load_yaml(yaml_file):
    import yaml
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading yaml: {e}")
        return None


def wait_for_data(ros_operator, timeout=30.0):
    """等待所有必要的数据流就绪"""
    print(f"[INFO] 正在等待数据流... (超时: {timeout}s)")
    start_time = time.time()
    last_print = 0
    
    rate = Rate(10)  # 10Hz检查频率
    
    while rclpy.ok():
        if time.time() - start_time > timeout:
            print("[ERROR] 等待数据超时！")
            return False
            
        # 检查所有必要的队列是否非空
        missing = []
        
        # 1. 检查RGB图像
        if len(ros_operator.img_head_deque) == 0: missing.append("head_rgb")
        if len(ros_operator.img_left_deque) == 0: missing.append("left_rgb")
        if len(ros_operator.img_right_deque) == 0: missing.append("right_rgb")
        
        # 2. 检查深度图
        if ros_operator.args.use_depth_image:
            if len(ros_operator.img_head_depth_deque) == 0: missing.append("head_depth")
            if len(ros_operator.img_left_depth_deque) == 0: missing.append("left_depth")
            if len(ros_operator.img_right_depth_deque) == 0: missing.append("right_depth")
            
        # 3. 检查机械臂状态
        if len(ros_operator.feedback_left_arm_deque) == 0: missing.append("left_arm")
        if len(ros_operator.feedback_right_arm_deque) == 0: missing.append("right_arm")
        
        if not missing:
            print("[INFO] 所有数据流已就绪！")
            return True
            
        if time.time() - last_print > 2.0:  # 每2秒打印一次状态
            print(f"等待数据中... 缺失: {', '.join(missing)}")
            last_print = time.time()
            
        rate.sleep()
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='测试帧数')
    parser.add_argument('--frame_rate', type=int, default=30, help='采样频率')
    parser.add_argument('--calibration_dir', type=str, default=str(ROOT / 'calibration_results'))
    parser.add_argument('--data', type=str, default=str(ROOT / 'act/data/config.yaml'))
    parser.add_argument('--camera_names', nargs='+', default=['head', 'left_wrist', 'right_wrist'])
    parser.add_argument('--use_depth_image', action='store_true', default=True)
    parser.add_argument('--use_base', action='store_true')
    parser.add_argument('--visualize', action='store_true', help='【新增】实时可视化生成的点云')
    args = parser.parse_args()

    setup_loader(ACT_DIR)
    rclpy.init()

    config = load_yaml(args.data)
    if config is None:
        print("❌ 无法加载配置文件")
        return

    # in_collect=True 开启更多的机械臂状态订阅（防止 Slave 没开数据缺失，使用 Master 数据替代）
    ros_operator = RosOperator(args, config, in_collect=True)

    def _spin_loop(node):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    print("等待设备数据流预热 (2s)...")
    time.sleep(2.0)  # 关键：给 ROS 节点时间接收数据，避免 get_observation 刷屏报错

    # 标定数据
    from types import SimpleNamespace
    calib_args = SimpleNamespace(calibration_dir=args.calibration_dir)
    calibration_data = load_calibration_data(calib_args)

    # 点云生成器
    pc_generator = PointCloudGenerator()

    # 等待观测就绪
    if not wait_for_data(ros_operator):
        print("[ERROR] 无法获取完整观测数据，程序退出")
        return

    # 获取第一帧用于预热
    rate = Rate(args.frame_rate)
    obs = ros_operator.get_observation()
    if obs is None:
        print("[ERROR] get_observation() 返回 None")
        return

    # 预热
    _ = pc_generator.generate(
        head_depth=obs['images_depth']['head'],
        head_color=obs['images']['head'],
        left_depth=obs['images_depth']['left_wrist'],
        left_color=obs['images']['left_wrist'],
        right_depth=obs['images_depth']['right_wrist'],
        right_color=obs['images']['right_wrist'],
        left_eef=obs['eef'][:7],
        right_eef=obs['eef'][7:14],
        intrinsics=calibration_data['intrinsics'],
        T_H_LB=calibration_data['T_H_LB'],
        T_H_RB=calibration_data['T_H_RB'],
        T_LE_LC=calibration_data['T_LE_LC'],
        T_RE_RC=calibration_data['T_RE_RC'],
        T_LB_H=calibration_data['T_LB_H']
    )

    # 可视化初始化
    use_cv2_vis = args.visualize
    if use_cv2_vis:
        print("[INFO] 初始化 OpenCV 2D 投影窗口... (按 'q' 退出)")
        # cv2.namedWindow("PointCloud Views", cv2.WINDOW_NORMAL)
        
    print(f"[INFO] 开始测试 {args.n} 帧点云生成...")
    t0 = time.time()
    times = []
    count = 0

    try:
        while rclpy.ok() and count < args.n:
            obs = ros_operator.get_observation()
            if not obs:
                rate.sleep()
                continue

            t_start = time.time()
            pc = pc_generator.generate(
                head_depth=obs['images_depth']['head'],
                head_color=obs['images']['head'],
                left_depth=obs['images_depth']['left_wrist'],
                left_color=obs['images']['left_wrist'],
                right_depth=obs['images_depth']['right_wrist'],
                right_color=obs['images']['right_wrist'],
                left_eef=obs['eef'][:7],
                right_eef=obs['eef'][7:14],
                intrinsics=calibration_data['intrinsics'],
                T_H_LB=calibration_data['T_H_LB'],
                T_H_RB=calibration_data['T_H_RB'],
                T_LE_LC=calibration_data['T_LE_LC'],
                T_RE_RC=calibration_data['T_RE_RC'],
                T_LB_H=calibration_data['T_LB_H']
            )
            t_end = time.time()

            # 可视化更新 (2D Projection)
            if use_cv2_vis:
                valid_mask = np.abs(pc).sum(axis=1) > 1e-6
                valid_points = pc[valid_mask]
                
                if len(valid_points) > 0:
                    x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
                    colors = valid_points[:, 3:6]
                    
                    img_size = 400
                    # 1. Top View (XY)
                    img_xy = render_view(x, y, colors, img_size, "Top (XY)")
                    # 2. Front View (XZ) - Assuming Z is up, looking from Front
                    # Shift Z down slightly to center it (assuming workspace z is -0.2 to 1.0)
                    img_xz = render_view(x, z - 0.4, colors, img_size, "Front (XZ)")
                    # 3. Side View (YZ)
                    img_yz = render_view(y, z - 0.4, colors, img_size, "Side (YZ)")
                    
                    combined = np.hstack([img_xy, img_xz, img_yz])
                    cv2.imshow("PointCloud Views", combined)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("User requested exit.")
                        break
                else:
                    # Show black image if no points
                    blank = np.zeros((400, 1200, 3), dtype=np.uint8)
                    cv2.putText(blank, "NO POINTS", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    cv2.imshow("PointCloud Views", blank)
                    cv2.waitKey(1)

            times.append(t_end - t_start)
            count += 1

            if count % 10 == 0:
                valid_num = np.sum(np.abs(pc).sum(axis=1) > 1e-6)
                if valid_num > 0 and use_cv2_vis:
                     # 偶尔打印一下范围，方便调试
                     pass
                print(f"  帧 {count}... shape={pc.shape}, 有效={valid_num}")

            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if times:
            avg = sum(times) / len(times)
            fps = 1.0 / avg if avg > 0 else 0.0
            print(f"\n平均单帧耗时: {avg*1000:.2f} ms")
            print(f"平均生成帧率: {fps:.2f} FPS")
        ros_operator.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
