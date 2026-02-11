#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
可视化HDF5数据集中的三视角图像和action
用法: python visualize_data.py --file datasets/episode_0.hdf5 --fps 10
"""

import os
import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path
import io
from PIL import Image


def decode_jpeg(jpeg_bytes):
    """
    解码JPEG字节流为numpy图像
    
    Args:
        jpeg_bytes: JPEG压缩的字节流
        
    Returns:
        img: (H, W, 3) BGR格式图像
    """
    # 使用PIL解码
    img = Image.open(io.BytesIO(jpeg_bytes))
    img = np.array(img)
    # RGB转BGR (OpenCV格式)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def draw_text(img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """在图像上绘制文本"""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), thickness + 2)  # 黑色背景
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)  # 白色文字
    return img


def create_action_panel(action, width=640, height=200):
    """
    创建action信息面板
    
    Args:
        action: (14,) action数组
        width, height: 面板尺寸
        
    Returns:
        panel: (height, width, 3) BGR图像
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # 深灰色背景
    
    # 绘制标题
    draw_text(panel, "Action Values (14-DoF)", (10, 25), 
              font_scale=0.7, color=(0, 255, 255), thickness=2)
    
    # 绘制左臂数据 (0-6)
    y_offset = 50
    draw_text(panel, "Left Arm:", (10, y_offset), 
              font_scale=0.6, color=(100, 255, 100), thickness=1)
    for i in range(7):
        text = f"{i}: {action[i]:+.3f}"
        x_pos = 10 + (i % 4) * 150
        y_pos = y_offset + 25 + (i // 4) * 25
        color = (255, 200, 100) if i == 6 else (255, 255, 255)  # 夹爪用橙色
        draw_text(panel, text, (x_pos, y_pos), font_scale=0.5, color=color)
    
    # 绘制右臂数据 (7-13)
    y_offset = 125
    draw_text(panel, "Right Arm:", (10, y_offset), 
              font_scale=0.6, color=(100, 100, 255), thickness=1)
    for i in range(7, 14):
        text = f"{i}: {action[i]:+.3f}"
        x_pos = 10 + ((i-7) % 4) * 150
        y_pos = y_offset + 25 + ((i-7) // 4) * 25
        color = (255, 200, 100) if i == 13 else (255, 255, 255)  # 夹爪用橙色
        draw_text(panel, text, (x_pos, y_pos), font_scale=0.5, color=color)
    
    return panel


def visualize_episode(hdf5_path, fps=10, start_frame=0):
    """
    可视化HDF5文件中的数据
    
    Args:
        hdf5_path: HDF5文件路径
        fps: 播放帧率
        start_frame: 起始帧
    """
    print(f"📖 加载文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取数据
        head_images = f['observations/images/head'][()]  # (T, jpeg_bytes)
        left_images = f['observations/images/left_wrist'][()]
        right_images = f['observations/images/right_wrist'][()]
        actions = f['action'][()]  # (T, 14)
        
        total_frames = len(actions)
        print(f"✅ 数据加载完成")
        print(f"   总帧数: {total_frames}")
        print(f"   图像形状: {head_images.shape}")
        print(f"   动作形状: {actions.shape}")
        print(f"\n控制说明:")
        print("   空格键: 暂停/继续")
        print("   →键: 下一帧")
        print("   ←键: 上一帧")
        print("   q键: 退出")
        print("   s键: 保存当前帧\n")
        
        # 创建窗口
        window_name = f"Data Visualization - {Path(hdf5_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 720)
        
        frame_idx = start_frame
        paused = False
        delay = int(1000 / fps)  # 毫秒
        
        while True:
            # 解码当前帧的三张图像
            try:
                head_img = decode_jpeg(head_images[frame_idx])
                left_img = decode_jpeg(left_images[frame_idx])
                right_img = decode_jpeg(right_images[frame_idx])
            except Exception as e:
                print(f"⚠️  解码失败 frame {frame_idx}: {e}")
                frame_idx = (frame_idx + 1) % total_frames
                continue
            
            # 获取图像尺寸并调整
            h, w = head_img.shape[:2]
            target_h = 480
            target_w = int(w * target_h / h)
            
            head_img = cv2.resize(head_img, (target_w, target_h))
            left_img = cv2.resize(left_img, (target_w, target_h))
            right_img = cv2.resize(right_img, (target_w, target_h))
            
            # 在图像上标注视角名称和帧号
            draw_text(head_img, f"Head View - Frame {frame_idx}/{total_frames-1}", 
                     (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)
            draw_text(left_img, f"Left Wrist - Frame {frame_idx}/{total_frames-1}", 
                     (10, 30), font_scale=0.8, color=(0, 255, 0), thickness=2)
            draw_text(right_img, f"Right Wrist - Frame {frame_idx}/{total_frames-1}", 
                     (10, 30), font_scale=0.8, color=(255, 0, 255), thickness=2)
            
            # 水平拼接三个视角
            top_row = np.hstack([head_img, left_img, right_img])
            
            # 创建action面板
            action_panel = create_action_panel(actions[frame_idx], 
                                              width=top_row.shape[1], 
                                              height=240)
            
            # 垂直拼接
            display = np.vstack([top_row, action_panel])
            
            # 添加播放状态提示
            status = "PAUSED" if paused else "PLAYING"
            status_color = (0, 255, 255) if paused else (0, 255, 0)
            draw_text(display, status, (display.shape[1] - 150, 30), 
                     font_scale=0.8, color=status_color, thickness=2)
            
            # 显示
            cv2.imshow(window_name, display)
            
            # 处理按键
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("👋 退出")
                break
            elif key == ord(' '):  # 空格键暂停
                paused = not paused
                print(f"{'⏸️  暂停' if paused else '▶️  继续'}")
            elif key == 83 or key == ord('d'):  # 右箭头或'd'键
                frame_idx = min(frame_idx + 1, total_frames - 1)
                print(f"➡️  Frame {frame_idx}")
            elif key == 81 or key == ord('a'):  # 左箭头或'a'键
                frame_idx = max(frame_idx - 1, 0)
                print(f"⬅️  Frame {frame_idx}")
            elif key == ord('s'):  # 保存当前帧
                save_name = f"frame_{frame_idx:04d}.png"
                cv2.imwrite(save_name, display)
                print(f"💾 保存: {save_name}")
            elif not paused:
                # 自动播放
                frame_idx = (frame_idx + 1) % total_frames
                if frame_idx == 0:
                    print("🔄 循环播放")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="可视化HDF5数据集")
    parser.add_argument("--file", type=str, default="/home/arx/ROS2_AC-one_Play/act/datasets_fixed/episode_0.hdf5",  # /home/arx/ROS2_AC-one_Play/act/datasets_fixed/episode_0.hdf5
                       help="HDF5文件路径")                                       # datasets/episode_0.hdf5
    parser.add_argument("--fps", type=int, default=10, 
                       help="播放帧率")
    parser.add_argument("--start", type=int, default=0, 
                       help="起始帧")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"❌ 文件不存在: {args.file}")
        return
    
    visualize_episode(args.file, args.fps, args.start)


if __name__ == "__main__":
    main()
