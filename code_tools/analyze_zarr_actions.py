#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
计算双臂机器人动作执行边界 (Clip Bounds) 并在部署前可视化
用于防抽搐、限制工作空间、分析 Zarr 数据集。

功能:
1. 计算绝对位置界限 (1% 和 99% Percentiles)
2. 计算单步 Delta 界限 (99% Percentile)
3. 绘制高质量的分布和轨迹时序图表
"""

import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ================= 配置 =================
# 美化绘图样式 (Presentation-Ready)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})

# 标签定义
DIM_LABELS_LEFT_ARM = [f"L_Joint_{i}" for i in range(6)]
DIM_LABEL_LEFT_GRIPPER = ["L_Gripper"]
DIM_LABELS_RIGHT_ARM = [f"R_Joint_{i}" for i in range(6)]
DIM_LABEL_RIGHT_GRIPPER = ["R_Gripper"]

ALL_LABELS = DIM_LABELS_LEFT_ARM + DIM_LABEL_LEFT_GRIPPER + DIM_LABELS_RIGHT_ARM + DIM_LABEL_RIGHT_GRIPPER
ARM_LABELS = DIM_LABELS_LEFT_ARM + DIM_LABELS_RIGHT_ARM


# ================= 核心计算逻辑 =================

def load_zarr_actions(zarr_path):
    """
    读取 Zarr 数据集中的 action 数据。
    假设采用标准集中式结构存储：
    data/action: (Total_T, 14)
    meta/episode_ends: (N_episodes,) 记录每个 episode 的结束索引
    """
    root = zarr.open(zarr_path, mode='r')
    
    # 提取完整动作数组和 episode 结束索引
    actions = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]
    
    # 将拉平的数据按照 episode 重组为列表，以便计算确切的单步 delta
    episodes_actions = []
    start_idx = 0
    for end_idx in episode_ends:
        ep_action = actions[start_idx:end_idx]
        if len(ep_action) > 0:
            episodes_actions.append(ep_action)
        start_idx = end_idx
        
    return actions, episodes_actions


def calculate_clip_bounds(actions, episodes_actions, save_dir):
    """
    计算双重限幅:
    1. 绝对位置限幅 (absolute_min, absolute_max): 基于全局历史的 1% 和 99% 分位数
    2. 相对变化率限幅 (max_delta): 基于局部序列的 99% 步长差值分位数
    并将结果保存至对应目录的 txt 文件中。
    """
    print("\n" + "="*50)
    print("📈 开始计算机器人的安全双重限幅参数")
    print("="*50)

    # 1. 绝对位置分位数限幅 (Absolute Boundaries)
    absolute_min = np.percentile(actions, 1, axis=0)
    absolute_max = np.percentile(actions, 99, axis=0)

    # 2. 单步变化率限幅 (Delta Boundaries)
    deltas = []
    for ep in episodes_actions:
        if len(ep) > 1:
            # 计算单条轨迹内的前后帧差异绝对值
            ep_delta = np.abs(ep[1:] - ep[:-1])
            deltas.append(ep_delta)
            
    all_deltas = np.vstack(deltas)
    max_delta = np.percentile(all_deltas, 99, axis=0)

    # 对于夹爪 (Dim 6, 13)，夹爪开关通常是突变/脉冲信号，不应该限制其 delta。
    # 我们将其设为一个极大常数，防止被 Clip
    max_delta[6] = 999.0
    max_delta[13] = 999.0

    print("\n✅ 【绝对位置限幅 (Absolute Clip)】 -- 用于保障工作空间安全")
    print("你可以直接复制下方 NumPy Array 到部署代码的 config 中:")
    print("absolute_min = np.array([")
    print("    " + ", ".join(f"{x:.4f}" for x in absolute_min))
    print("])")
    print("absolute_max = np.array([")
    print("    " + ", ".join(f"{x:.4f}" for x in absolute_max))
    print("])")

    print("\n✅ 【相对单步限幅 (Delta Clip)】 -- 用于防止模型抽风、产生极大加速度")
    print("你可以直接复制下方 NumPy Array 到部署代码的限速算法中:")
    print("max_delta = np.array([")
    print("    " + ", ".join(f"{x:.4f}" for x in max_delta))
    print("])")
    
    # 3. 将结果保存到 txt 文件中
    txt_path = os.path.join(save_dir, "clip_bounds.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("✅ 【绝对位置限幅 (Absolute Clip)】 -- 用于保障工作空间安全\n")
        f.write("absolute_min = np.array([\n    " + ", ".join(f"{x:.4f}" for x in absolute_min) + "\n])\n")
        f.write("absolute_max = np.array([\n    " + ", ".join(f"{x:.4f}" for x in absolute_max) + "\n])\n\n")
        
        f.write("✅ 【相对单步限幅 (Delta Clip)】 -- 用于防止模型抽风、产生极大加速度\n")
        f.write("max_delta = np.array([\n    " + ", ".join(f"{x:.4f}" for x in max_delta) + "\n])\n")
    print(f"\n📂 限幅参数已导出至: {txt_path}")

    return absolute_min, absolute_max, all_deltas


# ================= 绘图汇报生成 =================

def plot_action_distributions(actions, all_deltas, save_dir):
    """
    绘制双子图：绝对位置分布 & 单步变化量 (Delta) 分布 (专门用于汇报展示)
    仅展示 12 个手臂维度，去除夹爪以避免量纲差异过大影响绘图。
    """
    # 提取手臂维度索引 (去除 6 和 13)
    arm_indices = list(range(6)) + list(range(7, 13))
    arm_actions = actions[:, arm_indices]
    arm_deltas = all_deltas[:, arm_indices]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 调色板
    palette = sns.color_palette("muted", len(ARM_LABELS))

    # 子图 1: 绝对位置分布 (Boxplot)
    sns.boxplot(data=arm_actions, ax=axes[0], palette=palette, fliersize=1)
    axes[0].set_xticklabels(ARM_LABELS, rotation=45, ha='right')
    axes[0].set_title("Expert Demonstrations: Absolute Position Distribution", fontsize=16, pad=10)
    axes[0].set_ylabel("Absolute Pos Value")

    # 子图 2: 单步 Delta 分布 (Violin 或 Boxplot)
    # 因为 Delta 数据大头都在 0 附近，使用 violin 效果更好
    sns.violinplot(data=arm_deltas, ax=axes[1], palette=palette, inner="quartile", cut=0)
    axes[1].set_xticklabels(ARM_LABELS, rotation=45, ha='right')
    axes[1].set_title("Expert Demonstrations: 1-Step Delta (Velocity) Distribution", fontsize=16, pad=10)
    axes[1].set_ylabel("Delta Magnitude |$x_t - x_{t-1}$|")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "action_distribution_analysis.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 已生成分布双子图: {save_path}")
    plt.close()


def plot_gripper_states(actions, save_dir):
    """
    绘制双臂夹爪的独立直方图
    """
    left_gripper = actions[:, 6]
    right_gripper = actions[:, 13]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(left_gripper, bins=30, ax=axes[0], color='coral', kde=False)
    axes[0].set_title("Left Gripper State Distribution")
    axes[0].set_xlabel("State Value")

    sns.histplot(right_gripper, bins=30, ax=axes[1], color='teal', kde=False)
    axes[1].set_title("Right Gripper State Distribution")
    axes[1].set_xlabel("State Value")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "gripper_states_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 已生成夹爪直方图: {save_path}")
    plt.close()


def plot_time_series_trajectory(episode_data, save_dir):
    """
    绘制典型轨迹的四联时序折线图（左臂、左夹爪、右臂、右夹爪）
    """
    time_steps = np.arange(len(episode_data))
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 1. 左臂 (0:6)
    for i in range(6):
        axes[0].plot(time_steps, episode_data[:, i], label=f'J{i}', alpha=0.8, linewidth=2)
    axes[0].set_title("Left Arm Trajectory", loc='left')
    axes[0].set_ylabel("Position")
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.08, 1), ncol=1)

    # 2. 左夹爪 (6)
    axes[1].plot(time_steps, episode_data[:, 6], color='red', linewidth=2.5)
    axes[1].set_title("Left Gripper", loc='left')
    axes[1].set_ylabel("State")

    # 3. 右臂 (7:13)
    for i in range(7, 13):
        axes[2].plot(time_steps, episode_data[:, i], label=f'J{i-7}', alpha=0.8, linewidth=2)
    axes[2].set_title("Right Arm Trajectory", loc='left')
    axes[2].set_ylabel("Position")
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.08, 1), ncol=1)

    # 4. 右夹爪 (13)
    axes[3].plot(time_steps, episode_data[:, 13], color='blue', linewidth=2.5)
    axes[3].set_title("Right Gripper", loc='left')
    axes[3].set_ylabel("State")
    axes[3].set_xlabel("Time Step")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "sample_trajectory_timeseries.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 已生成轨迹时序图: {save_path}")
    plt.close()


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="Analyze Zarr Actions and generate clips & charts.")
    parser.add_argument("--zarr_path", type=str, default="datasets_zarr/pick_place_d435_low.zarr", help="Path to your .zarr dataset.")
    parser.add_argument("--output_dir", type=str, default="./analysis_output", help="Directory to save charts.")
    args = parser.parse_args()

    # 获取zarr文件名（不带扩展名）作为专属文件夹名
    zarr_name = os.path.basename(args.zarr_path).replace('.zarr', '')
    actual_output_dir = os.path.join(args.output_dir, zarr_name)

    if not os.path.exists(actual_output_dir):
        os.makedirs(actual_output_dir)

    print(f"Loading Zarr dataset from: {args.zarr_path} ...")
    actions, episodes_actions = load_zarr_actions(args.zarr_path)
    print(f"Total frames: {len(actions)}, Total episodes: {len(episodes_actions)}")
    
    if actions.shape[1] != 14:
        print(f"⚠️ 警告: 动作维度不是 14 而是 {actions.shape[1]}！")
    
    # 任务 1: 计算双重边界并打印 (传入 actual_output_dir 保存 txt)
    _, _, all_deltas = calculate_clip_bounds(actions, episodes_actions, actual_output_dir)

    # 任务 2: 绘制高质图表
    print("\n🖼️  开始绘制分析图表...")
    plot_action_distributions(actions, all_deltas, actual_output_dir)
    plot_gripper_states(actions, actual_output_dir)
    
    # 取一个中等长度的轨迹进行时序可视化展示
    if len(episodes_actions) > 0:
        sample_ep = episodes_actions[len(episodes_actions) // 2]
        plot_time_series_trajectory(sample_ep, actual_output_dir)
    
    print(f"✅ 分析完成！所有图表和配置代码已保存在专属文件夹 {os.path.abspath(actual_output_dir)} 目录下。")


if __name__ == "__main__":
    main()
