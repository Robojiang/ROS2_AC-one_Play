#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用 RealSense SDK 测试深度图稳定性
需要安装: pip install pyrealsense2
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# 相机序列号（根据你的 realsense.sh 配置）
CAMERA_SERIALS = {
    'camera_h': '135222070706',
    'camera_l': '409122273564',
    'camera_r': '352122274412'
}

# 深度可视化参数
DEPTH_MIN = 0  # mm
DEPTH_MAX = 1000  # mm


def init_camera(serial_number):
    """初始化指定序列号的相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用指定相机
    config.enable_device(serial_number)
    
    # 配置流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动 pipeline
    profile = pipeline.start(config)
    
    # 创建对齐对象（将深度对齐到 RGB）
    align = rs.align(rs.stream.color)
    
    return pipeline, align


def visualize_depth(depth_image):
    """将深度图可视化为伪彩色"""
    # 裁剪到有效范围
    depth_clipped = np.clip(depth_image, DEPTH_MIN, DEPTH_MAX)
    
    # 归一化到 0-255
    depth_normalized = ((depth_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
    
    # 应用 Jet colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 添加统计信息
    valid_depth = depth_image[depth_image > 0]
    if len(valid_depth) > 0:
        depth_mean = int(valid_depth.mean())
        depth_std = int(valid_depth.std())
        text = f"Mean:{depth_mean}mm Std:{depth_std}mm"
    else:
        text = "NO VALID DEPTH"
    
    cv2.putText(depth_colored, text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return depth_colored


def main():
    print("正在初始化 RealSense 相机...")
    
    # 初始化所有3个相机
    cameras = {}
    for name, serial in CAMERA_SERIALS.items():
        try:
            pipeline, align = init_camera(serial)
            cameras[name] = {'pipeline': pipeline, 'align': align}
            print(f"✅ 相机 {name} (SN: {serial}) 已启动")
        except Exception as e:
            print(f"❌ 相机 {name} 启动失败: {e}")
    
    if len(cameras) == 0:
        print("没有可用的相机，退出")
        return
    
    print(f"\n成功启动 {len(cameras)} 个相机")
    print("按 'q' 退出")
    
    frame_count = 0
    
    try:
        while True:
            displays_rgb = []
            displays_depth = []
            
            # 从所有相机获取帧
            for name in ['camera_h', 'camera_l', 'camera_r']:
                if name not in cameras:
                    # 相机不可用，显示黑屏
                    black = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(black, f"{name} OFFLINE", (50, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    displays_rgb.append(black)
                    displays_depth.append(black.copy())
                    continue
                
                pipeline = cameras[name]['pipeline']
                align = cameras[name]['align']
                
                try:
                    # 等待帧
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                    
                    # 对齐深度到 RGB
                    aligned_frames = align.process(frames)
                    
                    # 获取对齐后的帧
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        raise Exception("Frame is None")
                    
                    # 转换为 numpy 数组
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())  # uint16, 单位: mm
                    
                    # RGB 预览
                    color_small = cv2.resize(color_image, (320, 240))
                    display_name = name.replace('camera_', '').upper()
                    cv2.putText(color_small, display_name, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    displays_rgb.append(color_small)
                    
                    # 深度图可视化
                    depth_colormap = visualize_depth(depth_image)
                    depth_small = cv2.resize(depth_colormap, (320, 240))
                    displays_depth.append(depth_small)
                    
                except Exception as e:
                    # 帧获取失败
                    black = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(black, f"{name} NO FRAME", (50, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    displays_rgb.append(black)
                    displays_depth.append(black.copy())
            
            # 拼接显示：上排RGB，下排Depth
            if displays_rgb and displays_depth:
                row_rgb = np.hstack(displays_rgb)
                row_depth = np.hstack(displays_depth)
                combined = np.vstack([row_rgb, row_depth])
                
                # 添加帧计数
                cv2.putText(combined, f"Frame: {frame_count}", (10, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("RealSense Direct Test - All Cameras", combined)
            
            frame_count += 1
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        for name, cam in cameras.items():
            cam['pipeline'].stop()
        cv2.destroyAllWindows()
        print("\n程序退出")


if __name__ == '__main__':
    main()
