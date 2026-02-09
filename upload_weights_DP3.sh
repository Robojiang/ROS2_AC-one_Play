#!/bin/bash

# 是否开启调试日志
export ALIYUNPAN_VERBOSE=0

# aliyunpan程序所在的绝对路径（请更改成你自己的目录）
export ALIYUNPAN_BIN=/mnt/afs/250010074/tao_files/aliyunpan-v0.3.7-linux-amd64/aliyunpan

# 本地目录（请更改成你自己的目录）
LOCAL_DIR="/mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/policy/DP3/checkpoints/pick_place_d405-dp3-100_w_rgb_42/750.ckpt"
# 网盘目录（请更改成你自己的目录）
PAN_DIR="/weights/pick_place_d405/DP3"

# 执行上传
$ALIYUNPAN_BIN upload -exn "^\." -exn "^@eadir$" "$LOCAL_DIR" "$PAN_DIR"