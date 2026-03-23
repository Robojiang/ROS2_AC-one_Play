# 更改 RealSense 相机配置及帧率记录

## 问题背景
在更改相机和调整相机帧率（例如尝试设置为 `640x480x60`）时，发现启动配置并没有生效，甚至出现了无效配置导致设备退回默认帧率并报错 `Given value is invalid`。

## 解决过程

遇到类似关于色彩配置(profile)的报错时，关键的参数名定义可以在 ROS2 realsense 驱动的 launch 文件中找到，具体路径为：
`realsense/src/realsense2_camera/launch/rs_launch.py`

查看该源码后确认了以下信息：
1. **D435(I) 彩色相机参数**：名为 `rgb_camera.color_profile`
2. **D405 特有彩色相机参数**：名为 `depth_module.color_profile`
3. **通用深度/红外相机参数**：名为 `depth_module.depth_profile`

由于我们的硬件阵列中既有 D435（头部）又有 D405（手部手臂），所以原有的脚本配置参数存在错位或缺失，导致部分相机并没有按照设定生效。

## 修改方法

在 `realsense/realsense.sh` 脚本中进行两处核心修改：

### 1. 调整配置格式
将分隔符换为与 launch 节点更匹配的**逗号分隔符**：
```bash
# 修改原有的 640x480x90 或 x 分隔
COLOR_PROFILE="640,480,60"
DEPTH_PROFILE="640,480,60"
```

### 2. 补全参数透传
在循环调用 `ros2 launch` 的命令中，将针对不同型号相机的三个核心 profile 参数全部加上，确保无论是启动 D435 还是 D405 都能被独立捕捉：
```bash
  ros2 launch realsense2_camera rs_launch.py \
  camera_name:=${cam_name} \
  rgb_camera.color_profile:=${COLOR_PROFILE} \
  depth_module.color_profile:=${COLOR_PROFILE} \
  depth_module.depth_profile:=${DEPTH_PROFILE} \
  serial_no:=${sn}
```

这样修改后，无需拆分判断每个相机的型号，底层节点会自动读取对应的配置并成功锁定在 60 帧稳定运行。
