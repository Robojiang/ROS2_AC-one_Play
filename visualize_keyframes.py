import zarr
import numpy as np
import cv2
import argparse
import os
import time
import transforms3d as t3d
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not found. 3D visualization will be disabled.")

def visualize_keyframes(zarr_path, episode_idx=0, show_3d=False, save_video=False):
    print(f"Opening {zarr_path}...")
    root = zarr.open(zarr_path, mode='r')
    
    # Load data
    point_cloud = root['data/point_cloud']
    keyframe_mask = root['data/keyframe_mask']
    episode_ends = root['meta/episode_ends']
    
    has_endpose = 'data/left_endpose' in root and 'data/right_endpose' in root
    if has_endpose:
        left_endpose = root['data/left_endpose']
        right_endpose = root['data/right_endpose']
    
    has_images = 'data/images' in root
    if has_images:
        images_data = root['data/images']
    
    # Determine range for the requested episode
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0]
    else:
        start_idx = episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx]
        
    print(f"Visualizing Episode {episode_idx} (Frames {start_idx} to {end_idx})")
    
    # Initialize Open3D Visualizer if requested
    vis = None
    pcd_o3d = None
    if show_3d and HAS_OPEN3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Point Cloud", width=800, height=600)
        pcd_o3d = o3d.geometry.PointCloud()
        # Add a dummy point to initialize
        pcd_o3d.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        vis.add_geometry(pcd_o3d)
        
        # Set view control
        ctr = vis.get_view_control()
        ctr.set_front([0, -1, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)

    def render_view(x, y, colors, img_size, label):
        img = np.full((img_size, img_size, 3), 50, dtype=np.uint8)
        scale = img_size / 1.5 # Zoom out a bit to fit more
        offset = img_size / 2
        
        u = (x * scale + offset).astype(int)
        v = (y * scale + offset).astype(int)
        v = img_size - v # Flip Y for image coords
        
        valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u = u[valid]
        v = v[valid]
        c = (colors[valid] * 255).astype(np.uint8)
        
        # Draw points
        for j in range(len(u)):
            # BGR color for OpenCV
            cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
            
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    # Video Writer Initialization
    video_writer_pc = None
    video_writer_rgb = None
    fps = 30
    keyframe_pause_duration = 1.0  # seconds to pause at keyframe
    
    # Pre-calculate next keyframe indices
    next_keyframe_indices = np.full(end_idx, -1, dtype=int)
    last_kf = -1
    # Iterate backwards
    for i in range(end_idx - 1, start_idx - 1, -1):
        if keyframe_mask[i]:
            last_kf = i
        next_keyframe_indices[i] = last_kf

    # Loop through frames
    for i in range(start_idx, end_idx):
        pcd = point_cloud[i]
        is_keyframe = keyframe_mask[i]
        
        # Add Visualization Elements (Laser + Trajectory)
        extra_points_list = []
        if has_endpose:
            for arm_name, endpose_data in [("left", left_endpose), ("right", right_endpose)]:
                pose = endpose_data[i] # [x, y, z, qw, qx, qy, qz]
                pos = pose[:3]
                quat = pose[3:]
                mat = t3d.quaternions.quat2mat(quat)
                direction = mat[:, 0] # X-axis
                
                # 1. Laser (Red)
                dists = np.linspace(0, 0.3, 50)
                pts = pos + direction * dists[:, None]
                cols = np.tile([1.0, 0.0, 0.0], (50, 1)) # Red
                extra_points_list.append(np.hstack([pts, cols]))
                
                # 2. Trajectory Line to Next Keyframe (Green)
                next_kf_idx = next_keyframe_indices[i]
                if next_kf_idx != -1 and next_kf_idx != i:
                    next_pose = endpose_data[next_kf_idx]
                    next_pos = next_pose[:3]
                    
                    # Line points
                    num_line_points = 50
                    alphas = np.linspace(0, 1, num_line_points)[:, None]
                    line_pts = pos * (1 - alphas) + next_pos * alphas
                    line_cols = np.tile([0.0, 1.0, 0.0], (num_line_points, 1)) # Green
                    extra_points_list.append(np.hstack([line_pts, line_cols]))

        if extra_points_list:
            extra_pcd = np.vstack(extra_points_list)
            pcd = np.vstack([pcd, extra_pcd.astype(pcd.dtype)])

        # Extract Point Cloud
        x = pcd[:, 0]
        y = pcd[:, 1]
        z = pcd[:, 2]
        colors = pcd[:, 3:6] 
        
        # Update Open3D
        if vis:
            # Filter for Open3D
            mask_o3d = (z > -0.5) & (z < 2.0)
            points_o3d = pcd[mask_o3d, :3]
            colors_o3d = pcd[mask_o3d, 3:6]
            
            # Debug info for the first frame
            if i == start_idx:
                print(f"Frame {i} Point Cloud Stats:")
                print(f"  Count: {len(points_o3d)}")
                if len(points_o3d) > 0:
                    print(f"  Min: {points_o3d.min(axis=0)}")
                    print(f"  Max: {points_o3d.max(axis=0)}")
            
            vis.clear_geometries()
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(points_o3d)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors_o3d)
            
            # Reset view only on the first frame to fit the point cloud
            reset_bbox = (i == start_idx)
            vis.add_geometry(pcd_o3d, reset_bounding_box=reset_bbox)
            
            # Render options
            opt = vis.get_render_option()
            opt.point_size = 3.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark gray
            
            vis.poll_events()
            vis.update_renderer()
        
        # Filter for 2D views
        mask = (z > -0.5) & (z < 2.0)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        colors = colors[mask]
        
        img_size = 400
        
        # 1. Top View (XY)
        img_xy = render_view(x, y, colors, img_size, "Top View (XY)")
        
        # 2. Front View (XZ) - Z is up
        img_xz = render_view(x, z - 0.5, colors, img_size, "Front View (XZ)")
        
        # 3. Side View (YZ)
        img_yz = render_view(y, z - 0.5, colors, img_size, "Side View (YZ)")
        
        # Combine Point Cloud View
        combined_pc = np.hstack([img_xy, img_xz, img_yz])
        
        # Process RGB Images
        canvas_rgb = None
        if has_images:
            imgs = images_data[i] # (N, H, W, C)
            # Tile
            n = len(imgs)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w, c = imgs[0].shape
            
            canvas_rgb = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for k, img in enumerate(imgs):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为 BGR
                r = k // cols
                c = k % cols
                canvas_rgb[r*h:(r+1)*h, c*w:(c+1)*w] = img

        # Initialize Video Writers (One for PC, One for RGB)
        if save_video:
            if video_writer_pc is None:
                height, width = combined_pc.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename_pc = f"episode_{episode_idx}_keyframes_pc.mp4"
                video_writer_pc = cv2.VideoWriter(filename_pc, fourcc, fps, (width, height))
                print(f"Recording Point Cloud video to {filename_pc}...")
            
            if has_images and video_writer_rgb is None:
                height_rgb, width_rgb = canvas_rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename_rgb = f"episode_{episode_idx}_keyframes_rgb.mp4"
                video_writer_rgb = cv2.VideoWriter(filename_rgb, fourcc, fps, (width_rgb, height_rgb))
                print(f"Recording RGB video to {filename_rgb}...")
        
        # Highlight Keyframe
        if is_keyframe:
            # Draw border on combined PC image
            cv2.rectangle(combined_pc, (0, 0), (combined_pc.shape[1]-1, combined_pc.shape[0]-1), (0, 0, 255), 10)
            cv2.putText(combined_pc, "KEYFRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Draw border on RGB image as well
            if has_images and canvas_rgb is not None:
                cv2.rectangle(canvas_rgb, (0, 0), (canvas_rgb.shape[1]-1, canvas_rgb.shape[0]-1), (0, 0, 255), 10)
                cv2.putText(canvas_rgb, "KEYFRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            print(f"Frame {i}: KEYFRAME")
            
            cv2.imshow("3-View Point Cloud", combined_pc)
            if has_images and canvas_rgb is not None:
                 cv2.imshow("RGB Images", canvas_rgb)

            if save_video:
                # Repeat frames to simulate pause
                for _ in range(int(fps * keyframe_pause_duration)):
                    if video_writer_pc is not None: video_writer_pc.write(combined_pc)
                    if video_writer_rgb is not None: video_writer_rgb.write(canvas_rgb)
            
            cv2.waitKey(int(keyframe_pause_duration * 1000)) # Pause for keyframes
        else:
            cv2.imshow("3-View Point Cloud", combined_pc)
            if has_images and canvas_rgb is not None:
                 cv2.imshow("RGB Images", canvas_rgb)

            if save_video:
                if video_writer_pc is not None: video_writer_pc.write(combined_pc)
                if video_writer_rgb is not None: video_writer_rgb.write(canvas_rgb)

            cv2.waitKey(30) # Play normal frames at ~30fps

    if video_writer_pc is not None:
        video_writer_pc.release()
        print("Point Cloud video saved successfully.")
    
    if video_writer_rgb is not None:
        video_writer_rgb.release()
        print("RGB video saved successfully.")

    if vis:
        vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # stack_blocks_two-demo_3d_vision_easy-100-ppi.zarr beat_block_hammer-demo_3d_vision_easy-100-ppi.zarr
    parser.add_argument("--zarr_path", type=str, default='./test_conversion.zarr', help="Path to the processed Zarr file")
    # parser.add_argument("--zarr_path", type=str, default='policy/VGC/data/beat_block_hammer-demo_3d_vision_easy-100-ppi.zarr', help="Path to the processed Zarr file")
    parser.add_argument("--episode", type=int, default=1, help="Episode index to visualize")
    parser.add_argument("--show_3d", action='store_true', default=True, help="Enable 3D Open3D visualization")
    parser.add_argument("--save_video", action='store_true', help="Save visualization as MP4 video")
    args = parser.parse_args()
    
    visualize_keyframes(args.zarr_path, args.episode, args.show_3d, args.save_video)
