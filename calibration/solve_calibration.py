#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import glob
import json
import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
DATA_ROOT = "calibration_data_ark"
RESULT_DIR = "calibration_results"
INTRINSICS_FILE = os.path.join(DATA_ROOT, "intrinsics.json")

# 标定板参数 (Eye-in-Hand 用)
CHARUCO_DICT_ID = aruco.DICT_6X6_250
CHARUCO_SQUARES = (11, 8)
CHARUCO_SQUARE_LEN = 0.020
CHARUCO_MARKER_LEN = 0.015

# 单 ID 码参数 (Eye-to-Hand 用，ID 通过参数传入)
SINGLE_MARKER_SIZE = 0.030
# =======================================

def get_aruco_detector(dictionary):
    """
    兼容 OpenCV 4.6+ 的新版 API (ArucoDetector)
    和 旧版 API (detectMarkers)
    """
    params = aruco.DetectorParameters()
    # 如果是 OpenCV 4.7+，推荐使用 ArucoDetector 类
    if hasattr(aruco, 'ArucoDetector'):
        return aruco.ArucoDetector(dictionary, params)
    return None

def get_charuco_detector(board):
    """
    为 OpenCV 4.7+ 创建 CharucoDetector
    """
    if hasattr(aruco, 'CharucoDetector'):
        return aruco.CharucoDetector(board)
    return None

def detect_markers_robust(image, dictionary, detector=None):
    """鲁棒的检测函数，处理不同版本的 OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if detector is not None:
        # 新版 API
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        # 旧版 API (如果 cv2.aruco 没有 detectMarkers，说明没装 opencv-contrib-python)
        if hasattr(aruco, 'detectMarkers'):
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
        else:
            raise AttributeError("❌ 你的 OpenCV 不支持 ArUco，请运行: pip install opencv-contrib-python")
            
    return corners, ids, gray

def load_intrinsics(camera_name):
    """从 json 加载内参并转换为矩阵"""
    if not os.path.exists(INTRINSICS_FILE):
        print(f"⚠️ 找不到 {INTRINSICS_FILE}，请先运行 save_intrinsics.py")
        return None, None
        
    with open(INTRINSICS_FILE, 'r') as f:
        all_data = json.load(f)
    
    if camera_name not in all_data:
        print(f"⚠️ 内参文件中没有 {camera_name} 的数据，使用默认值")
        return None, None
        
    d = all_data[camera_name]
    
    # 构造 K 矩阵
    K = np.array([
        [d['fx'], 0, d['cx']],
        [0, d['fy'], d['cy']],
        [0, 0, 1]
    ])
    
    # 构造 D 向量 (k1, k2, p1, p2, k3)
    D = np.array([d['k1'], d['k2'], d['p1'], d['p2'], d['k3']])
    
    return K, D

def load_pose_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    pose = data['pose'] # [x, y, z, rx, ry, rz]
    
    t = np.array(pose[:3])
    r_mat = R.from_euler('xyz', pose[3:]).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = r_mat
    T[:3, 3] = t
    return T

def run_solver(folder_name, camera_name, mode="eye_in_hand", marker_id=0):
    data_dir = os.path.join(DATA_ROOT, folder_name)
    if not os.path.exists(data_dir):
        print(f"⚠️ 跳过: 找不到数据文件夹 {data_dir}")
        return

    print(f"\n{'='*20} 解算任务: {folder_name} {'='*20}")
    
    # 1. 加载内参
    K, D = load_intrinsics(camera_name)

    # 2. 准备 ArUco
    aruco_dict = aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
    board = aruco.CharucoBoard(CHARUCO_SQUARES, CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, aruco_dict)
    detector = get_aruco_detector(aruco_dict) # 获取兼容的检测器对象
    charuco_detector = get_charuco_detector(board) # CharucoDetector for OpenCV 4.7+

    img_files = sorted(glob.glob(os.path.join(data_dir, "img_*.jpg")))
    json_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.json")))
    
    if len(img_files) == 0:
        print("❌ 文件夹为空，跳过")
        return

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []
    valid_cnt = 0

    for img_path, json_path in zip(img_files, json_files):
        img = cv2.imread(img_path)
        if img is None: continue
        
        # === 视觉检测 (兼容写法) ===
        try:
            corners, ids, gray_img = detect_markers_robust(img, aruco_dict, detector)
        except AttributeError as e:
            print(e)
            return

        rvec, tvec = None, None
        
        if mode == "eye_in_hand":
            # ChArUco 模式 (通常用标定板)
            if ids is not None and len(ids) > 0:
                # 兼容旧版和新版 OpenCV
                if charuco_detector is not None:
                    # OpenCV 4.7+ 新 API
                    c_corners, c_ids, _, _ = charuco_detector.detectBoard(gray_img)
                    if c_ids is not None and len(c_ids) > 4:
                        # 使用 solvePnP 来估计位姿
                        obj_points = board.getChessboardCorners()[c_ids.flatten()]
                        ret, rvec, tvec = cv2.solvePnP(obj_points, c_corners, K, D)
                elif hasattr(aruco, 'interpolateCornersCharuco'):
                    # OpenCV 4.6 及更早版本
                    ret, c_corners, c_ids = aruco.interpolateCornersCharuco(corners, ids, gray_img, board)
                    if ret > 4:
                        valid, rvec, tvec = aruco.estimatePoseCharucoBoard(c_corners, c_ids, board, K, D, None, None)
                else:
                    print("  [Error] OpenCV 版本不支持 ChArUco 检测")
                    return
        else:
            # Single Marker 模式 (固定相机，Eye-to-Hand)
            # 搜索指定的 marker_id
            if ids is not None:
                for i, mid in enumerate(ids):
                    if mid[0] == marker_id:
                        # 兼容新旧 API
                        if hasattr(aruco, 'estimatePoseSingleMarkers'):
                            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], SINGLE_MARKER_SIZE, K, D)
                            rvec, tvec = rvecs[0], tvecs[0]
                        else:
                            # OpenCV 4.7+ 没有 estimatePoseSingleMarkers，使用 solvePnP
                            obj_pts = np.array([
                                [-SINGLE_MARKER_SIZE/2,  SINGLE_MARKER_SIZE/2, 0],
                                [ SINGLE_MARKER_SIZE/2,  SINGLE_MARKER_SIZE/2, 0],
                                [ SINGLE_MARKER_SIZE/2, -SINGLE_MARKER_SIZE/2, 0],
                                [-SINGLE_MARKER_SIZE/2, -SINGLE_MARKER_SIZE/2, 0]
                            ], dtype=np.float32)
                            ret, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], K, D)
                        break
        
        if rvec is None: 
            # print(f"  [Skip] {os.path.basename(img_path)} 未检测到 Marker/Board")
            continue

        # === 机器人位姿处理 ===
        try:
            T_robot = load_pose_json(json_path)
        except Exception as e:
            print(f"  读取 JSON 失败: {e}")
            continue
        
        if mode == "eye_in_hand":
            # Eye-in-Hand (Cam on End)
            # Input: T_Base_End
            R_g2b = T_robot[:3, :3]
            t_g2b = T_robot[:3, 3]
        else:
            # Eye-to-Hand (Cam fixed, Target on End)
            # Standard Practice: Input inv(T_Base_End) = T_End_Base
            # This converts the Eye-to-Hand problem into the AX=XB form.
            R_mat = T_robot[:3, :3]
            t_vec = T_robot[:3, 3]
            
            # --- 尝试两种输入模式 ---
            # Mode A: Inverted (T_End_Base) - 理论上的标准做法
            R_inv = R_mat.T
            t_inv = -R_inv @ t_vec
            
            # Mode B: Non-Inverted (T_Base_End) - 某些库或特定情况下的做法
            # 如果 TSAI 算法对于 Eye-to-Hand 的定义是 A=MotionOfGripperInBase，那么应该是 T_Base_End 之间的差异?
            # OpenCV calibrateHandEye 文档说 "gripper frame to robot base frame" (T_Base_End).
            # 让我们做个简单的 Flag，目前保持 Mode A (Inverted)，
            # 因为 Daniilidis 给出了 61cm 的正确结果，TSAI 给了 28cm 的错误结果。
            # 这说明 Mode A 可能是对的，但 TSAI 在这个数据上失效了。
            
            R_g2b = R_inv
            t_g2b = t_inv

        R_t2c, _ = cv2.Rodrigues(rvec)
        
        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)
        R_target2cam.append(R_t2c)
        # Ensure tvec is flattened to (3,)
        t_target2cam.append(np.array(tvec).flatten())
        valid_cnt += 1
        print(f"  [OK] 帧 {os.path.basename(img_path)}")

    if valid_cnt < 5:
        print(f"❌ 有效帧数不足 5 帧 ({valid_cnt})，无法解算")
        return

    print(f"⏳ 正在计算 ({valid_cnt} 帧)...")
    
    # 对比多种算法
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS"),
    ]
    
    best_T = None
    best_method_name = ""
    # 这里我们只是简单的根据平移量是否离谱来判断，由于无法直接计算残差，
    # 我们将在外部验证脚本中判断好坏。目前优先保存我们认为最稳健的 methods。
    # 通常对于 Eye-to-Hand，Daniilidis 或 Park 表现较好。
    
    # 为了简化，我们运行所有算法并在终端打印结果，最后保存 Daniilidis 的结果（通常对旋转噪声更鲁棒）
    
    print("\n--- 算法对比 ---")
    for meth_enum, meth_name in methods:
        try:
            R_cal, t_cal = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=meth_enum
            )
            print(f"[{meth_name}] Translation: {t_cal.flatten()}")
            
            # 默认选用 DANIILIDIS，因为它在物理高度上验证正确 (0.62m vs TSAI 0.28m)
            # 用户反馈实际高度约 60cm，且 check_data_quality 显示 robot_z(0.25) + cam_dist(0.30) ~ 0.55m
            # 所以 TSAI 的 0.28m 明显是错误收敛 (可能是符号反转或局部极值)
            if meth_name == "DANIILIDIS":
                best_T = np.eye(4)
                best_T[:3, :3] = R_cal
                best_T[:3, 3] = t_cal.flatten()
                best_method_name = meth_name
                
        except cv2.error:
            print(f"[{meth_name}] Failed")

    if best_T is None:
        # Fallback
        return

    print(f"\n✅ 最终选用: {best_method_name}")
    T_final = best_T
        
    # 确保结果文件夹存在
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    save_name = os.path.join(RESULT_DIR, f"{folder_name}.npy")
    txt_name = os.path.join(RESULT_DIR, f"{folder_name}.txt")
    
    np.save(save_name, T_final)
    np.savetxt(txt_name, T_final, fmt="%.6f")
    
    print(f"✅ 成功! 矩阵已保存: {save_name}")
    print("T_calibration:\n", T_final)
        
    # except Exception as e:
    #     print(f"❌ 计算过程出错: {e}")

def main():
    print("开始标定解算...")
    
    # 1. 左臂 Eye-in-Hand
    # 结果: T_left_cam_to_left_end
    run_solver("left_eye_in_hand", "left", mode="eye_in_hand")
    
    # 2. 右臂 Eye-in-Hand
    # 结果: T_right_cam_to_right_end
    run_solver("right_eye_in_hand", "right", mode="eye_in_hand")
    
    # 3. Head -> Left Base (Eye-to-Hand)
    # 结果: T_head_cam_to_left_base (或者 T_left_base_to_head_cam，具体看下面解释)
    # 传入的 gripper2base 是 T_end_to_base
    # 传入的 target2cam 是 T_marker_to_cam
    # 求解的是: T_base_to_cam (Left Base 到 Head Camera 的变换)
    # Marker ID = 1
    run_solver("head_base_to_left", "head", mode="eye_to_hand", marker_id=1)
    
    # 4. Head -> Right Base (Eye-to-Hand)
    # 结果: T_right_base_to_head_cam (Right Base 到 Head Camera 的变换)
    # Marker ID = 0
    run_solver("head_base_to_right", "head", mode="eye_to_hand", marker_id=0)

if __name__ == "__main__":
    main()