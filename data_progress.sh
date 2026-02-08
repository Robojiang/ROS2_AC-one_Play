cd /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin

python code_tools/convert_hdf5_to_zarr.py 