#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import yaml
import json
import numpy as np
import time
import pyrealsense2 as rs

# ================= 配置 =================
CONFIG_FILE = "calibration/cam_config_client_101_with_depth.yaml"
SAVE_FILE = "calibration_data_ark/intrinsics.json"
# =======================================

def get_camera_intrinsics(serial_number, width=640, height=480):
    """连接指定SN的相机并读取内参"""
    print(f"🔄 尝试连接相机: {serial_number} (分辨率: {width}x{height})")
    
    # 查找设备
    ctx = rs.context()
    devices = ctx.query_devices()
    target_found = False
    
    for dev in devices:
        try:
            sn = dev.get_info(rs.camera_info.serial_number)
            if sn == str(serial_number):
                target_found = True
                break
        except Exception:
            continue
            
    if not target_found:
        print(f"⚠️  未找到相机 SN: {serial_number}，跳过")
        return None

    print(f"✅ 连接相机: {serial_number}")
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_device(str(serial_number))
        # 开启流以读取参数 (D405 通常支持 Color 流，或者使用 Depth 流的 RGB 映射)
        # 这里尝试开启 Color 流，并指定分辨率
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        cfg = pipeline.start(config)
        
        # 等待一小会儿确保流稳定
        time.sleep(0.5)
        
        # 获取内参
        # 注意: get_stream 返回的是 generic stream profile，需要 as_video_stream_profile
        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        
        pipeline.stop()
        
        # RealSense distortion coeffs 通常是 [k1, k2, p1, p2, k3] (Brown-Conrady)
        coeffs = intr.coeffs
        
        # 转换为列表方便 JSON 序列化
        intr_data = {
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy,
            "k1": coeffs[0] if len(coeffs) > 0 else 0.0,
            "k2": coeffs[1] if len(coeffs) > 1 else 0.0,
            "k3": coeffs[4] if len(coeffs) > 4 else 0.0, # 注意顺序 RS通常是 k1,k2,p1,p2,k3
            "p1": coeffs[2] if len(coeffs) > 2 else 0.0,
            "p2": coeffs[3] if len(coeffs) > 3 else 0.0,
            "k4": 0.0, "k5": 0.0, "k6": 0.0 # RS 一般不提供 k4-k6
        }
        return intr_data
        
    except Exception as e:
        print(f"❌ 读取失败 {serial_number}: {e}")
        try: pipeline.stop()
        except: pass
        return None

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 找不到配置文件: {CONFIG_FILE}")
        return

    # 1. 解析 YAML 获取序列号
    with open(CONFIG_FILE, 'r') as f:
        cfg = yaml.safe_load(f)

    cameras = {}
    resolutions = {} # 存储分辨率 (width, height)

    if 'head_camera' in cfg:
        cameras['head'] = str(cfg['head_camera']['serial_number'])
        h, w = cfg['head_camera'].get('image_shape', [480, 640])
        resolutions['head'] = (w, h)
        
    if 'left_wrist_camera' in cfg:
        cameras['left'] = str(cfg['left_wrist_camera']['serial_number'])
        h, w = cfg['left_wrist_camera'].get('image_shape', [480, 640])
        resolutions['left'] = (w, h)
        
    if 'right_wrist_camera' in cfg:
        cameras['right'] = str(cfg['right_wrist_camera']['serial_number'])
        h, w = cfg['right_wrist_camera'].get('image_shape', [480, 640])
        resolutions['right'] = (w, h)

    print(f"📋 待读取列表: {cameras}")
    
    # 2. 依次读取内参
    results = {}
    for name, sn in cameras.items():
        w, h = resolutions.get(name, (640, 480))
        data = get_camera_intrinsics(sn, width=w, height=h)
        if data:
            results[name] = data
            print(f"   -> {name} 读取成功")
    
    # 3. 保存结果
    if not os.path.exists(os.path.dirname(SAVE_FILE)):
        os.makedirs(os.path.dirname(SAVE_FILE))
        
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n💾 内参已保存至: {SAVE_FILE}")

if __name__ == "__main__":
    main()
