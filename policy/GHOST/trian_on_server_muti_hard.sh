cd /mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/GHOST

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin

torchrun --nproc_per_node=4 --master_port=29501 train.py task=stack_blocks_two task.dataset_type=hard
torchrun --nproc_per_node=4 --master_port=29500 train.py task=beat_block_hammer task.dataset_type=hard
torchrun --nproc_per_node=4 --master_port=29502 train.py task=stack_blocks_three task.dataset_type=hard