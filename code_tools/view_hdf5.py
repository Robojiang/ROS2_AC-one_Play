#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
读取HDF5文件中的左右夹爪开度数据并绘制折线图
用法: python plot_gripper_from_hdf5.py --file datasets/task_name/0.hdf5
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def read_gripper_data(hdf5_path):
    """
    读取HDF5文件中的夹爪数据
    
    Args:
        hdf5_path: HDF5文件路径
        
    Returns:
        left_gripper: (T,) 左臂夹爪开度
        right_gripper: (T,) 右臂夹爪开度
        timestamps: (T,) 时间步索引
    """
    print(f"📖 读取文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 检查可用的数据键
        print(f"   可用的观测数据: {list(f['observations'].keys())}")
        
        # 读取end-effector数据或qpos数据
        if 'eef' in f['observations']:
            data = f['observations/eef'][()]  # (T, 14)
            data_type = "eef (末端姿态)"
        elif 'qpos' in f['observations']:
            data = f['observations/qpos'][()]  # (T, 14)
            data_type = "qpos (关节角度)"
        else:
            raise ValueError("❌ HDF5文件中未找到 'eef' 或 'qpos' 数据")
        
        print(f"   数据类型: {data_type}")
        print(f"   数据形状: {data.shape}")
        
        # 提取夹爪数据 (第7维和第14维)
        left_gripper = data[:, 6]   # 左臂夹爪
        right_gripper = data[:, 13]  # 右臂夹爪
        
        timestamps = np.arange(len(left_gripper))
        
        print(f"   ✅ 读取完成，共 {len(timestamps)} 帧")
        print(f"   左夹爪范围: [{left_gripper.min():.3f}, {left_gripper.max():.3f}]")
        print(f"   右夹爪范围: [{right_gripper.min():.3f}, {right_gripper.max():.3f}]")
        
        return left_gripper, right_gripper, timestamps


def plot_gripper_data(left_gripper, right_gripper, timestamps, save_path=None):
    """
    绘制左右夹爪开度折线图
    
    Args:
        left_gripper: (T,) 左臂夹爪开度
        right_gripper: (T,) 右臂夹爪开度
        timestamps: (T,) 时间步索引
        save_path: 保存路径（None表示不保存）
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 子图1: 左夹爪
    axes[0].plot(timestamps, left_gripper, 'b-', linewidth=2, label='Left Gripper')
    axes[0].set_xlabel('Timestep', fontsize=12)
    axes[0].set_ylabel('Gripper Opening', fontsize=12)
    axes[0].set_title('Left Gripper Opening Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 标注开合状态（假设 < 2.5 为开，>= 2.5 为合）
    threshold = 2.5
    axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    axes[0].fill_between(timestamps, 0, threshold, alpha=0.1, color='green', label='Open')
    axes[0].fill_between(timestamps, threshold, left_gripper.max(), alpha=0.1, color='red', label='Closed')
    
    # 子图2: 右夹爪
    axes[1].plot(timestamps, right_gripper, 'r-', linewidth=2, label='Right Gripper')
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Gripper Opening', fontsize=12)
    axes[1].set_title('Right Gripper Opening Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    axes[1].axhline(y=threshold, color='b', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].fill_between(timestamps, 0, threshold, alpha=0.1, color='green')
    axes[1].fill_between(timestamps, threshold, right_gripper.max(), alpha=0.1, color='red')
    
    # 子图3: 左右对比
    axes[2].plot(timestamps, left_gripper, 'b-', linewidth=2, label='Left Gripper', alpha=0.7)
    axes[2].plot(timestamps, right_gripper, 'r-', linewidth=2, label='Right Gripper', alpha=0.7)
    axes[2].set_xlabel('Timestep', fontsize=12)
    axes[2].set_ylabel('Gripper Opening', fontsize=12)
    axes[2].set_title('Left vs Right Gripper Comparison', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    axes[2].axhline(y=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 图像已保存: {save_path}")
    
    plt.show()


def analyze_gripper_changes(left_gripper, right_gripper, threshold=0.05):
    """
    分析夹爪开合次数
    
    Args:
        left_gripper: (T,) 左臂夹爪开度
        right_gripper: (T,) 右臂夹爪开度
        threshold: 判断开合的阈值
    """
    print("\n" + "="*60)
    print("📊 夹爪动作分析")
    print("="*60)
    
    # 计算变化量
    left_diff = np.abs(np.diff(left_gripper, prepend=left_gripper[0]))
    right_diff = np.abs(np.diff(right_gripper, prepend=right_gripper[0]))
    
    # 统计显著变化次数
    left_changes = np.sum(left_diff > threshold)
    right_changes = np.sum(right_diff > threshold)
    
    print(f"左夹爪:")
    print(f"  总变化次数: {left_changes}")
    print(f"  平均值: {left_gripper.mean():.3f}")
    print(f"  标准差: {left_gripper.std():.3f}")
    print(f"  范围: [{left_gripper.min():.3f}, {left_gripper.max():.3f}]")
    
    print(f"\n右夹爪:")
    print(f"  总变化次数: {right_changes}")
    print(f"  平均值: {right_gripper.mean():.3f}")
    print(f"  标准差: {right_gripper.std():.3f}")
    print(f"  范围: [{right_gripper.min():.3f}, {right_gripper.max():.3f}]")
    
    # 检测关键帧（显著变化的时刻）
    left_keyframes = np.where(left_diff > threshold)[0]
    right_keyframes = np.where(right_diff > threshold)[0]
    
    if len(left_keyframes) > 0:
        print(f"\n左夹爪关键帧索引 (前10个): {left_keyframes[:10].tolist()}")
    if len(right_keyframes) > 0:
        print(f"右夹爪关键帧索引 (前10个): {right_keyframes[:10].tolist()}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="读取HDF5文件并绘制夹爪开度折线图")
    parser.add_argument("--file", type=str, default="datasets/episode_0.hdf5", help="HDF5文件路径")
    parser.add_argument("--save", type=str, default=None, help="保存图像的路径（可选）")
    parser.add_argument("--threshold", type=float, default=0.05, help="夹爪变化阈值（用于分析）")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"❌ 文件不存在: {args.file}")
        return
    
    # 读取数据
    left_gripper, right_gripper, timestamps = read_gripper_data(args.file)
    
    # 分析数据
    analyze_gripper_changes(left_gripper, right_gripper, args.threshold)
    
    # 自动生成保存路径
    if args.save is None:
        file_stem = Path(args.file).stem  # 获取文件名（不含扩展名）
        args.save = f"gripper_plot_{file_stem}.png"
    
    # 绘制折线图
    plot_gripper_data(left_gripper, right_gripper, timestamps, args.save)


if __name__ == "__main__":
    main()