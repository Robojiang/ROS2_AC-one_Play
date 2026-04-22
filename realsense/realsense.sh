#!/bin/bash

shell_type=${SHELL##*/}
shell_config="source ./install/setup.$shell_type"
shell_exec="exec $shell_type"

declare -A CAMS=(
  # 
  # [camera_h]="135222070706" # D435 new 135122077817 135222070706
  # [camera_h]="409122273228"  # D405
  [camera_l]="260322272375"
  [camera_r]="352122274412"
)

COLOR_PROFILE="640,480,60"
DEPTH_PROFILE="640,480,60"

terminal_cmd() {
  local title="$1"
  local body="$2"

  gnome-terminal --tab --title="$title" -- "$shell_type" -ic "$body"
}

normalize_serial() 
{
  local s="${1//[[:space:]]/}"

  while [[ "$s" == _* ]]; do s="${s#_}"; done
  printf '%s' "$s"
}

serial_is_set() 
{
  local s="$1"
  [[ -n "${s}" && "${s}" != "_" ]]
}

serial_is_online() 
{
  local s="$1"
  [[ ${#ONLINE[@]} -eq 0 ]] && return 0
  [[ -n "${ONLINE[$s]:-}" ]]
}

launch_cam() 
{
  local cam_name="$1" raw_serial="$2"

  if ! serial_is_set "$raw_serial"; then
    return 0
  fi

  digits="$(normalize_serial "$raw_serial")"
  local sn="_${digits}"

  echo "$cam_name -> serial=$sn"
  terminal_cmd "$cam_name" "$shell_config
  ros2 launch realsense2_camera rs_launch.py \
  camera_name:=${cam_name} \
  rgb_camera.color_profile:=${COLOR_PROFILE} \
  depth_module.color_profile:=${COLOR_PROFILE} \
  depth_module.depth_profile:=${DEPTH_PROFILE} \
  align_depth.enable:=true \
  serial_no:=${sn}
  $shell_exec"

  sleep 1
}

for cam in "${!CAMS[@]}"; do
  launch_cam "$cam" "${CAMS[$cam]}"
done
