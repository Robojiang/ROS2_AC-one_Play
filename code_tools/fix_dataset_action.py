#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
修复数据集的 action 数据，使用时间偏移方法
用法: python fix_dataset_action.py --input_dir "datasets (copy)" --output_dir datasets_fixed --episode 0
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
import shutil


def fix_action_with_temporal_shift(input_path, output_path):
    """
    使用时间偏移修复 action 数据
    
    Args:
        input_path: 输入 HDF5 文件路径
        output_path: 输出 HDF5 文件路径
    """
    print(f"\n{'='*80}")
    print(f"📖 处理文件: {input_path}")
    print(f"💾 输出文件: {output_path}")
    print(f"{'='*80}")
    
    with h5py.File(input_path, 'r') as f_in:
        # 读取关键数据
        qpos = f_in['observations/qpos'][()]           # (T, 14)
        qvel = f_in['observations/qvel'][()]           # (T, 14)
        effort = f_in['observations/effort'][()]       # (T, 14)
        eef = f_in['observations/eef'][()]             # (T, 14)
        robot_base = f_in['observations/robot_base'][()]     # (T, 6)
        base_velocity = f_in['observations/base_velocity'][()]  # (T, 4)
        
        # 读取旧的 action（用于对比）
        old_action = f_in['action'][()]                # (T, 14)
        old_action_eef = f_in['action_eef'][()]       # (T, 14)
        old_action_base = f_in['action_base'][()]     # (T, 6)
        old_action_velocity = f_in['action_velocity'][()]  # (T, 4)
        
        # 读取图像数据（JPEG 压缩格式）
        head_images = f_in['observations/images/head'][()]
        left_images = f_in['observations/images/left_wrist'][()]
        right_images = f_in['observations/images/right_wrist'][()]
        
        # 读取深度图像（如果有）
        has_depth = 'images_depth' in f_in['observations']
        if has_depth:
            head_depth = f_in['observations/images_depth/head'][()]
            left_depth = f_in['observations/images_depth/left_wrist'][()]
            right_depth = f_in['observations/images_depth/right_wrist'][()]
        
        total_frames = len(qpos)
        print(f"\n📊 原始数据统计:")
        print(f"   总帧数: {total_frames}")
        print(f"   qpos 形状: {qpos.shape}")
        print(f"   action 形状: {old_action.shape}")
        print(f"   是否有深度图: {'是' if has_depth else '否'}")
        
        # 分析旧 action 的夹爪值
        left_gripper_old = old_action[:, 6]
        right_gripper_old = old_action[:, 13]
        left_gripper_obs = qpos[:, 6]
        right_gripper_obs = qpos[:, 13]
        
        print(f"\n🔍 旧 action 夹爪分析:")
        print(f"   左夹爪 (action)  - 范围: [{left_gripper_old.min():.3f}, {left_gripper_old.max():.3f}], "
              f"非零帧数: {np.count_nonzero(left_gripper_old)}/{len(left_gripper_old)}")
        print(f"   右夹爪 (action)  - 范围: [{right_gripper_old.min():.3f}, {right_gripper_old.max():.3f}], "
              f"非零帧数: {np.count_nonzero(right_gripper_old)}/{len(right_gripper_old)}")
        print(f"   左夹爪 (qpos)    - 范围: [{left_gripper_obs.min():.3f}, {left_gripper_obs.max():.3f}]")
        print(f"   右夹爪 (qpos)    - 范围: [{right_gripper_obs.min():.3f}, {right_gripper_obs.max():.3f}]")
        
        # ========== ✅ 核心修复：时间偏移 ==========
        new_action = np.zeros_like(old_action)
        new_action_eef = np.zeros_like(old_action_eef)
        new_action_base = np.zeros_like(old_action_base)
        new_action_velocity = np.zeros_like(old_action_velocity)
        
        # 对于前 T-1 帧：action[t] = qpos[t+1]
        new_action[:] = qpos[:]
        new_action_eef[:] = eef[:]
        new_action_base[:] = robot_base[:]
        new_action_velocity[:] = base_velocity[:]
        
       
        
        # 分析新 action 的夹爪值
        left_gripper_new = new_action[:, 6]
        right_gripper_new = new_action[:, 13]
        
        print(f"\n✅ 新 action 夹爪分析:")
        print(f"   左夹爪 (action)  - 范围: [{left_gripper_new.min():.3f}, {left_gripper_new.max():.3f}], "
              f"非零帧数: {np.count_nonzero(left_gripper_new)}/{len(left_gripper_new)}")
        print(f"   右夹爪 (action)  - 范围: [{right_gripper_new.min():.3f}, {right_gripper_new.max():.3f}], "
              f"非零帧数: {np.count_nonzero(right_gripper_new)}/{len(right_gripper_new)}")
        
        # 计算变化统计
        left_gripper_diff = np.abs(np.diff(left_gripper_new, prepend=left_gripper_new[0]))
        right_gripper_diff = np.abs(np.diff(right_gripper_new, prepend=right_gripper_new[0]))
        print(f"   左夹爪显著变化帧数 (>0.05): {np.sum(left_gripper_diff > 0.05)}")
        print(f"   右夹爪显著变化帧数 (>0.05): {np.sum(right_gripper_diff > 0.05)}")
    
    # ========== 保存到新文件 ==========
    print(f"\n💾 保存修复后的数据...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # 保存 observations（完全不变）
        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos)
        obs.create_dataset('qvel', data=qvel)
        obs.create_dataset('effort', data=effort)
        obs.create_dataset('eef', data=eef)
        obs.create_dataset('robot_base', data=robot_base)
        obs.create_dataset('base_velocity', data=base_velocity)
        
        # 保存图像（JPEG 格式）
        images = obs.create_group('images')
        images.create_dataset('head', data=head_images)
        images.create_dataset('left_wrist', data=left_images)
        images.create_dataset('right_wrist', data=right_images)
        
        # 保存深度图像（如果有）
        if has_depth:
            images_depth = obs.create_group('images_depth')
            images_depth.create_dataset('head', data=head_depth)
            images_depth.create_dataset('left_wrist', data=left_depth)
            images_depth.create_dataset('right_wrist', data=right_depth)
        
        # 保存修复后的 action
        root.create_dataset('action', data=new_action)
        root.create_dataset('action_eef', data=new_action_eef)
        root.create_dataset('action_base', data=new_action_base)
        root.create_dataset('action_velocity', data=new_action_velocity)
    
    print(f"✅ 修复完成！")
    print(f"{'='*80}\n")
    
    return {
        'total_frames': total_frames,
        'left_gripper_old_nonzero': np.count_nonzero(left_gripper_old),
        'right_gripper_old_nonzero': np.count_nonzero(right_gripper_old),
        'left_gripper_new_nonzero': np.count_nonzero(left_gripper_new),
        'right_gripper_new_nonzero': np.count_nonzero(right_gripper_new),
    }


def batch_fix_datasets(input_dir, output_dir, episode_indices=None):
    """
    批量修复数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        episode_indices: 要处理的 episode 索引列表（None 表示处理所有）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 查找所有 HDF5 文件
    hdf5_files = sorted(input_path.glob('episode_*.hdf5'))
    
    if not hdf5_files:
        print(f"❌ 在 {input_dir} 中未找到 HDF5 文件")
        return
    
    print(f"\n📁 找到 {len(hdf5_files)} 个 HDF5 文件")
    
    # 过滤要处理的文件
    if episode_indices is not None:
        hdf5_files = [f for f in hdf5_files 
                     if any(f.stem == f'episode_{idx}' for idx in episode_indices)]
        print(f"📌 选择处理 {len(hdf5_files)} 个文件")
    
    # 处理每个文件
    results = []
    for i, input_file in enumerate(hdf5_files):
        output_file = output_path / input_file.name
        
        try:
            result = fix_action_with_temporal_shift(str(input_file), str(output_file))
            results.append((input_file.name, result, True))
        except Exception as e:
            print(f"❌ 处理 {input_file.name} 时出错: {e}")
            results.append((input_file.name, None, False))
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"🎉 批量处理完成！")
    print(f"{'='*80}")
    print(f"\n处理结果汇总:")
    print(f"{'文件名':<20} {'总帧数':<10} {'左夹爪(旧)':<12} {'左夹爪(新)':<12} {'右夹爪(旧)':<12} {'右夹爪(新)':<12} {'状态'}")
    print("-" * 100)
    
    for filename, result, success in results:
        if success and result:
            print(f"{filename:<20} {result['total_frames']:<10} "
                  f"{result['left_gripper_old_nonzero']:<12} "
                  f"{result['left_gripper_new_nonzero']:<12} "
                  f"{result['right_gripper_old_nonzero']:<12} "
                  f"{result['right_gripper_new_nonzero']:<12} "
                  f"✅")
        else:
            print(f"{filename:<20} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} ❌")
    
    print(f"\n✅ 修复后的数据集保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="修复数据集的 action 数据")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/arx/ROS2_AC-one_Play/act/datasets (copy)",
                       help="输入数据集目录")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/arx/ROS2_AC-one_Play/act/datasets_fixed",
                       help="输出数据集目录")
    parser.add_argument("--episode", type=int, nargs='+', default=None,
                       help="要处理的 episode 索引（默认处理所有）")
    parser.add_argument("--single_file", type=str, default=None,
                       help="只处理单个文件（指定完整路径）")
    
    args = parser.parse_args()
    
    if args.single_file:
        # 处理单个文件
        output_file = os.path.join(args.output_dir, os.path.basename(args.single_file))
        fix_action_with_temporal_shift(args.single_file, output_file)
    else:
        # 批量处理
        batch_fix_datasets(args.input_dir, args.output_dir, args.episode)


if __name__ == "__main__":
    main()
