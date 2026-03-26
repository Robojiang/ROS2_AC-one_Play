cd /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin

python code_tools/convert_hdf5_to_zarr.py

bash /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/policy/GHOST/train_on_server_trident.sh

bash /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/policy/DP3/train_on_server.sh
#11
bash /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/policy/GHOST/train_on_server_key_trident.sh