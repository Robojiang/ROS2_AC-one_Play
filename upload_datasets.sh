#!/bin/bash

# 是否开启调试日志
export ALIYUNPAN_VERBOSE=0

# aliyunpan程序所在的绝对路径（请更改成你自己的目录）
export ALIYUNPAN_BIN=/home/arx/haitao_codes/aliyunpan-v0.3.7-linux-amd64/aliyunpan

# 本地目录（请更改成你自己的目录）
LOCAL_DIR="/home/arx/haitao_codes/ROS2_AC-one_Play/act/datasets.zip"
# 网盘目录（请更改成你自己的目录）
PAN_DIR="/robot_data"

# 执行上传
$ALIYUNPAN_BIN upload -exn "^\." -exn "^@eadir$" "$LOCAL_DIR" "$PAN_DIR"