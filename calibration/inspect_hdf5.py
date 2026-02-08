#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import h5py
import numpy as np
import os

# 读取第一个 HDF5 文件
hdf5_path = "/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/datasets/episode_4.hdf5"

print(f"检查文件: {hdf5_path}")
print("=" * 80)

with h5py.File(hdf5_path, 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"\n[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [Dataset] {name}")
            print(f"    Shape: {obj.shape}")
            print(f"    Dtype: {obj.dtype}")
            if len(obj.shape) > 0 and obj.shape[0] > 0:
                # 显示第一个元素
                if obj.shape[0] < 5:
                    print(f"    Sample: {obj[:]}")
                else:
                    print(f"    First element: {obj[0]}")
    
    print("\n文件结构:")
    f.visititems(print_structure)
    
    # 尝试读取第一帧数据
    print("\n" + "=" * 80)
    print("尝试读取第一帧数据...")
    
    # 检查可能的键名
    keys = list(f.keys())
    print(f"\n顶层键: {keys}")
    
    # 通常的结构可能是 observations/images, actions, qpos 等
    if 'observations' in f:
        obs = f['observations']
        print(f"\nObservations 子键: {list(obs.keys())}")
        
        # 查找图像相关数据
        if 'images' in obs:
            images = obs['images']
            print(f"\nImages 子键: {list(images.keys())}")
            for img_key in images.keys():
                print(f"  {img_key}: shape={images[img_key].shape}, dtype={images[img_key].dtype}")
"""
文件结构:                                                                       
  [Dataset] action                                                              
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [-6.9066143e-01  1.4963379e+00  8.6194420e-01 -4.4499111e-01 
  1.2016296e-02 -1.6231823e-01  0.0000000e+00 -1.3351440e-03                    
  9.5540524e-01  8.5469627e-01 -5.8461094e-01 -1.9073486e-04                    
  5.1498413e-03  0.0000000e+00]                                                 
  [Dataset] action_base                                                         
    Shape: (551, 6)                                                             
    Dtype: float32                                                              
    First element: [0. 0. 0. 0. 0. 0.]                                          
  [Dataset] action_eef                                                          
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [ 1.1745295e-01 -1.7589086e-01  2.4112517e-02 -1.8477428e-01 
  1.0792499e+00 -7.1611607e-01  0.0000000e+00  9.4719797e-02                    
 -2.4674204e-04  1.3103172e-01  5.2929427e-03  6.8531990e-01                    
 -1.0862538e-03  0.0000000e+00]                                                 
  [Dataset] action_velocity                                                     
    Shape: (551, 4)                                                             
    Dtype: float32                                                              
    First element: [0. 0. 0. 0.]                                                
                                                                                
[Group] observations                                                            
  [Dataset] observations/base_velocity                                          
    Shape: (551, 4)                                                             
    Dtype: float32                                                              
    First element: [0. 0. 0. 0.]                                                
  [Dataset] observations/eef                                                    
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [ 1.1745295e-01 -1.7589086e-01  2.4112517e-02 -1.8477428e-01 
  1.0792499e+00 -7.1611607e-01 -9.2698097e-02  9.4719797e-02                    
 -2.4674204e-04  1.3103172e-01  5.2929427e-03  6.8531990e-01                    
 -1.0862538e-03 -6.5613747e-02]                                                 
  [Dataset] observations/effort                                                 
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [-9.5237732e-02 -2.2490845e+00  2.6300354e+00  9.8901081e-01 
 -2.4423599e-03 -7.3261261e-03  7.3261261e-03 -2.1978378e-02                    
 -1.1794872e+00  2.8205147e+00  1.4236870e+00 -2.4423599e-03                    
 -7.3261261e-03 -2.4423599e-03]                                                 
                                                                                
[Group] observations/images                                                     
  [Dataset] observations/images/head                                            
    Shape: (551, 18975)                                                         
    Dtype: uint8                                                                
    First element: [255 216 255 ...   0   0   0]                                
  [Dataset] observations/images/left_wrist                                      
    Shape: (551, 18975)                                                         
    Dtype: uint8                                                                
    First element: [255 216 255 ...   0   0   0]                                
  [Dataset] observations/images/right_wrist                                     
    Shape: (551, 18975)                                                         
    Dtype: uint8                                                                
    First element: [255 216 255 ...   0   0   0]                                
                                                                                
[Group] observations/images_depth                                               
  [Dataset] observations/images_depth/head                                      
    Shape: (551, 480, 640)                                                      
    Dtype: uint16                                                               
    First element: [[0 0 0 ... 0 0 0]                                           
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]                                                              
 ...                                                                            
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]]                                                             
  [Dataset] observations/images_depth/left_wrist                                
    Shape: (551, 480, 640)                                                      
    Dtype: uint16                                                               
    First element: [[0 0 0 ... 0 0 0]                                           
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]                                                              
 ...                                                                            
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]                                                              
 [0 0 0 ... 0 0 0]]                                                             
  [Dataset] observations/images_depth/right_wrist                               
    Shape: (551, 480, 640)                                                      
    Dtype: uint16                                                               
    First element: [[  0   0   0 ...   0   0   0]                               
 [  0   0   0 ...   0   0   0]                                                  
 [  0   0   0 ... 995   0   0]                                                  
 ...                                                                            
 [  0   0   0 ...   0   0   0]                                                  
 [  0   0   0 ...   0   0   0]                                                  
 [  0   0   0 ...   0   0   0]]                                                 
  [Dataset] observations/qpos                                                   
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [-6.9066143e-01  1.4963379e+00  8.6194420e-01 -4.4499111e-01 
  1.2016296e-02 -1.6231823e-01 -9.2698097e-02 -1.3351440e-03                    
  9.5540524e-01  8.5469627e-01 -5.8461094e-01 -1.9073486e-04                    
  5.1498413e-03 -6.5613747e-02]                                                 
  [Dataset] observations/qvel                                                   
    Shape: (551, 14)                                                            
    Dtype: float32                                                              
    First element: [ 0.02197838  0.39120865 -0.02197838  0.3186798   0.01099014 
-0.03296661                                                                     
 -0.47252655 -0.00439644  0.01318741  0.00439644 -0.01099014 -0.03296661        
 -0.01099014 -0.01099014]                                                       
  [Dataset] observations/robot_base                                             
    Shape: (551, 6)                                                             
    Dtype: float32                                                              
    First element: [0. 0. 0. 0. 0. 0.]                                          
                                                                                
================================================================================
尝试读取第一帧数据...                                                           
                                                                                
顶层键: ['action', 'action_base', 'action_eef', 'action_velocity', 'observations
']                                                                              
                                                                                
Observations 子键: ['base_velocity', 'eef', 'effort', 'images', 'images_depth', 
'qpos', 'qvel', 'robot_base']                                                   
                                                                                
Images 子键: ['head', 'left_wrist', 'right_wrist']                              
  head: shape=(551, 18975), dtype=uint8                                         
  left_wrist: shape=(551, 18975), dtype=uint8                                   
  right_wrist: shape=(551, 18975), dtype=uint8                                  
"""
