# pointnet
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh beat_block_hammer demo_3d_vision_hard  baseline_pn/beat_block_hammer_hard 0 0 
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh stack_blocks_two demo_3d_vision_hard  baseline_pn/stack_blocks_two_hard 0 0 
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh stack_blocks_three demo_3d_vision_hard  baseline_pn/stack_blocks_three_hard 0 0 
# pointnet++
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh beat_block_hammer demo_3d_vision_hard  baseline_pn2/beat_block_hammer_hard 0 0 
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh stack_blocks_two demo_3d_vision_hard  baseline_pn2/stack_blocks_two_hard 0 0 
cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/GHOST
bash eval.sh stack_blocks_three demo_3d_vision_hard  baseline_pn2/stack_blocks_three_hard 0 0 