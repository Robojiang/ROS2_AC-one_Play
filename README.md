标定采集前，先开

- 相机的不同帧率分辨率的设置在realsense/install/realsense2_camera/share/realsense2_camera/launch/rs_launch.py文件中的
                 {'name': 'rgb_camera.color_profile',     'default': '640,480,30', 'description': 'color stream profile'},
                {'name': 'depth_module.depth_profile',   'default': '640,480,30', 'description': 'depth stream profile'},
                {'name': 'depth_module.color_profile',   'default': '640,480,30', 'description': 'Depth module color stream profile for d405'},