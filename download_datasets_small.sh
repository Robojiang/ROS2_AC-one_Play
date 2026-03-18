#!/bin/bash

# 是否开启调试日志
export ALIYUNPAN_VERBOSE=0

# aliyunpan程序所在的绝对路径（请更改成你自己的目录）
export ALIYUNPAN_BIN=/home/tao/my_applications/aliyunpan-v0.3.7-linux-amd64/aliyunpan

# 本地目录（请更改成你自己的目录）
LOCAL_DIR="/media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/arx_data/ROS2_AC-one_Play/datasets"
# 网盘目录（请更改成你自己的目录）
PAN_DIR="/robot_data/datasets_small.zip"

# 执行下载
"$ALIYUNPAN_BIN" download --saveto "$LOCAL_DIR" "$PAN_DIR"