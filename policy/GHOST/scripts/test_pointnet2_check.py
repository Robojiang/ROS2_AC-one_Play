
import sys
import os
import torch
import torch.nn as nn
from termcolor import cprint

# Add paths
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path) # policy/GHOST/scripts
ghost_dir = os.path.dirname(scripts_dir) # policy/GHOST
policy_dir = os.path.dirname(ghost_dir) # policy

sys.path.append(ghost_dir)
sys.path.append(os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy'))

from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZRGB
from diffusion_policy_3d.model.vision.pointnet2_clean import PointNet2Encoder

def test_encoders():
    print("=== Testing Encoder Compatibility ===")
    
    # 1. Configuration
    B = 4      # Batch size
    N = 1024   # Number of points
    C_in = 6   # Channels (XYZRGB)
    C_out = 256 # Desired output feature dim
    
    dummy_input = torch.randn(B, N, C_in).cuda()
    print(f"Input Shape: {dummy_input.shape} (Batch, Points, Channels)")
    
    # 2. Test Original PointNet
    print("\n--- Testing Original PointNet (PointNetEncoderXYZRGB) ---")
    try:
        net1 = PointNetEncoderXYZRGB(in_channels=C_in, out_channels=C_out).cuda()
        out1 = net1(dummy_input)
        print(f"Output Shape: {out1.shape}")
        if out1.shape == (B, C_out):
            cprint("✅ PointNet Output Shape Correct", "green")
        else:
            cprint(f"❌ PointNet Output Shape Mismatch: {out1.shape}", "red")
    except Exception as e:
        cprint(f"❌ PointNet Failed: {e}", "red")

    # 3. Test New PointNet++
    print("\n--- Testing New PointNet++ (PointNet2Encoder) ---")
    try:
        net2 = PointNet2Encoder(in_channels=C_in, out_channels=C_out).cuda()
        
        # Count parameters to show complexity difference
        params = sum(p.numel() for p in net2.parameters())
        print(f"Model Parameters: {params:,}")
        print("Architecture Check: Has SA1, SA2, SA3 modules? ", 
              all(hasattr(net2, k) for k in ['sa1', 'sa2', 'sa3']))
        
        out2 = net2(dummy_input)
        print(f"Output Shape: {out2.shape}")
        
        if out2.shape == (B, C_out):
            cprint("✅ PointNet++ Output Shape Correct", "green")
            cprint("✅ Seamless Replacement Verified", "green")
        else:
            cprint(f"❌ PointNet++ Output Shape Mismatch: {out2.shape}", "red")
            
    except Exception as e:
        cprint(f"❌ PointNet++ Failed: {e}", "red")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_encoders()
    else:
        print("CUDA not available, skipping test.")
