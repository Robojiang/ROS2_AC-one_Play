
cd /mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/DP3

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin


bash train_rgb.sh pick_place_d405 dp3 100 42 0