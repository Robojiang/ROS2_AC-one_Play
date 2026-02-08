# HDF5 to Zarr 数据转换脚本

## 功能说明

将 ROS2_AC-one_Play 项目的 HDF5 数据集转换为 DP3 训练所需的 Zarr 格式,包含:

- ✅ **三相机点云融合** (Head + Left Wrist + Right Wrist)
- ✅ **自动坐标系转换** (转换到左臂基座坐标系)
- ✅ **工作空间裁剪** (只保留工作台区域点云)
- ✅ **FPS下采样** (固定1024点)
- ✅ **关键帧检测** (基于夹爪开合,不包含暂停)
- ✅ **内存优化** (增量写入Zarr,不占用大量内存)

## 输出格式

生成的 Zarr 文件结构:
```
data/
  ├── state: (N, 14) - 机器人关节状态
  ├── action: (N, 14) - 目标关节位置
  ├── point_cloud: (N, 1024, 6) - 融合点云 [x,y,z,r,g,b]
  ├── images: (N, 4, 240, 320, 3) - 4个相机的RGB图像
  ├── keyframe_mask: (N,) - 关键帧标记
  ├── left_endpose: (N, 7) - 左臂末端位姿 [x,y,z,rx,ry,rz,gripper]
  └── right_endpose: (N, 7) - 右臂末端位姿
meta/
  └── episode_ends: (M,) - 每个episode的结束索引
```

## 使用方法

### 1. Debug模式 (只转换第1个episode)
```bash
conda activate RoboTwin
python convert_hdf5_to_zarr.py --max_episodes 1 --output test_data
```

### 2. 转换前N个episodes
```bash
python convert_hdf5_to_zarr.py --max_episodes 10 --output partial_data
```

### 3. 完整转换 (自动扫描所有HDF5文件)
```bash
python convert_hdf5_to_zarr.py --output full_data
```

### 4. 调试点云和末端位置
```bash
python debug_pointcloud.py --zarr_path test_data.zarr --frame 0
```

## 配置参数

在脚本顶部可以调整以下参数:

### 点云配置
- `MAX_DEPTH_Head`: Head相机最大深度 (默认: 1.0m)
- `MAX_DEPTH_Hand`: 手腕相机最大深度 (默认: 0.6m)
- `FPS_SAMPLE_POINTS`: 点云采样点数 (默认: 1024)

### 工作空间裁剪
```python
USE_WORKSPACE_CROP = True
WORKSPACE_X_RANGE = [-0.4, 0.5]   # x轴范围 (米)
WORKSPACE_Y_RANGE = [-0.5, 3.0]   # y轴范围 (米)
WORKSPACE_Z_RANGE = [-0.2, 1.0]   # z轴范围 (米)
```

### 关键帧检测
- `GRIPPER_DELTA`: 夹爪变化阈值 (默认: 0.05)
- `MIN_INTERVAL`: 最小关键帧间隔 (默认: 5帧)

## 数据说明

### 坐标系
所有点云数据都在**左臂基座坐标系**下:
- X轴: 前后方向 (正向为前)
- Y轴: 左右方向 (正向为右)
- Z轴: 上下方向 (正向为上)

### 关键帧策略
只保留**夹爪开合**触发的关键帧,不包含机器人暂停:
- 第一帧和最后一帧总是关键帧
- 夹爪变化超过阈值时标记为关键帧
- 强制最小间隔避免过密采样

### Action定义
`action[t]` = 下一时刻的目标关节位置 `qpos[t+1]`

## 性能

- 单个episode转换时间: ~2-3分钟 (取决于帧数)
- 内存占用: 峰值 ~2GB (使用Zarr增量写入)
- 关键帧比例: ~2% (根据任务不同有所变化)

## 验证转换结果

```python
import zarr
import numpy as np

# 打开Zarr文件
root = zarr.open('test_data.zarr', 'r')

# 查看结构
print("Keys:", list(root['data'].keys()))
print("Shape:", root['data/point_cloud'].shape)
print("Episodes:", root['meta/episode_ends'][:])

# 检查点云
pc = root['data/point_cloud'][0]
print(f"点云范围: X[{pc[:,0].min():.2f}, {pc[:,0].max():.2f}]")
print(f"非零点数: {(np.abs(pc).sum(axis=1) > 0).sum()}")
```

## 注意事项

1. **自动扫描**: 脚本会自动扫描 `datasets/` 目录下所有HDF5文件并按文件名排序
2. **标定文件**: 需要 `calibration_results/` 下的标定矩阵
3. **内参文件**: 需要 `calibration_results/intrinsics.json`
4. **图像格式**: 自动从BGR转换为RGB,resize到240x320
5. **坐标系**: 所有数据在左臂基座坐标系下

## 最近更新 (2026-02-08)

### ✅ 已修复:
1. **RGB图像颜色错误** - 修复BGR到RGB的转换
2. **文件名硬编码** - 改为自动扫描所有HDF5文件
3. **调试支持** - 添加 `debug_pointcloud.py` 可视化工具

### 🔧 使用建议:
- 首次使用建议 `--max_episodes 1` 快速测试
- 使用 `debug_pointcloud.py` 检查点云和末端位置对齐

## 故障排查

### 找不到标定文件
```
❌ 缺少标定文件: head_base_to_left_refined_icp.txt
```
→ 检查 `calibration_results/` 目录是否包含标定文件

### 内存不足
→ 减少 `--max_episodes` 参数,分批转换

### 点云全为零
→ 检查深度图数据是否有效,调整 `MAX_DEPTH` 参数

## 作者
RoboTwin Team
