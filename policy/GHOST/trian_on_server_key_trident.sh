cd cd /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/policy/GHOST

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin

torchrun --nproc_per_node=4 --master_port=29502 train.py --config-name ghost_keyframe_policy task=pick_place_d405 