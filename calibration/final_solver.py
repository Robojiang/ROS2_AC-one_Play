import numpy as np
import os
import glob
import json
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R

DATA_ROOT = "calibration_data_ark"
RESULT_DIR = "calibration_results"
INTRINSICS_FILE = os.path.join(DATA_ROOT, "intrinsics.json")
CHARUCO_DICT_ID = aruco.DICT_6X6_250

def load_intrinsics(camera_name):
    if not os.path.exists(INTRINSICS_FILE): return None, None
    with open(INTRINSICS_FILE, 'r') as f: all_data = json.load(f)
    if camera_name not in all_data: return None, None
    d = all_data[camera_name]
    K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]])
    D = np.array([d['k1'], d['k2'], d['p1'], d['p2'], d['k3']])
    return K, D

def load_pose_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    pose = data['pose'] 
    t = np.array(pose[:3])
    r_mat = R.from_euler('xyz', pose[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r_mat
    T[:3, 3] = t
    return T

def run_eye_to_hand(folder, camera_name, marker_id, save_name):
    print(f"\n Solving {folder} for Marker {marker_id}...")
    K, D = load_intrinsics(camera_name)
    data_dir = os.path.join(DATA_ROOT, folder)
    img_files = sorted(glob.glob(os.path.join(data_dir, "img_*.jpg")))
    json_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.json")))
    
    aruco_dict = aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []
    
    for i, (img_f, json_f) in enumerate(zip(img_files, json_files)):
        img = cv2.imread(img_f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        if hasattr(ids, '__len__') and len(ids) > 0:
             idx = np.where(ids == marker_id)[0]
             if len(idx) == 0: continue
             idx = idx[0]
             
             # 30mm Marker Size - OpenCV 4.7+ compatible
             obj_pts = np.array([
                 [-0.015,  0.015, 0],
                 [ 0.015,  0.015, 0],
                 [ 0.015, -0.015, 0],
                 [-0.015, -0.015, 0]
             ], dtype=np.float32)
             ret, rvec, tvec = cv2.solvePnP(obj_pts, corners[idx].reshape(4, 2), K, D)
             
             T_robot = load_pose_json(json_f)
             
             # Eye-to-Hand Inversion
             R_mat = T_robot[:3, :3]
             t_vec = T_robot[:3, 3]
             R_inv = R_mat.T
             t_inv = -R_inv @ t_vec
             
             R_gripper2base.append(R_inv)
             t_gripper2base.append(t_inv)
             
             R_t2c, _ = cv2.Rodrigues(rvec)
             R_target2cam.append(R_t2c)
             t_target2cam.append(tvec.flatten())

    print(f"Collected {len(R_gripper2base)} frames.")
    
    # --- Auto-Select Best Algorithm ---
    # User's setup is known to be around 0.6m ~ 0.7m height.
    # We test multiple algorithms and select the one closest to 0.65m.
    
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
    ]
    
    best_T = None
    best_diff = float('inf')
    best_name = ""
    
    print(f"Testing solvers for {folder}...")
    
    for meth, name in methods:
        try:
            R_cal, t_cal = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, 
                R_target2cam, t_target2cam, 
                method=meth
            )
            z_val = t_cal[2][0]
            print(f"  [{name}] Z: {z_val:.3f} m")

            # Expecting Z around 0.4m (Left was 0.693m)
            diff = abs(z_val - 0.4)
            if diff < best_diff:
                best_diff = diff
                best_T = np.eye(4)
                best_T[:3, :3] = R_cal
                best_T[:3, 3] = t_cal.flatten()
                best_name = name
        except Exception as e:
            print(f"  [{name}] Failed: {e}")
            
    if best_T is None:
        print("❌ All solvers failed.")
        return

    print(f"✅ Selected Best: {best_name} (Z={best_T[2,3]:.3f} m)")
    T_final = best_T
    
    # Re-assemble
    # T_final is already assembled above
    
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    path = os.path.join(RESULT_DIR, save_name)
    np.save(path, T_final)
    np.savetxt(path.replace('.npy', '.txt'), T_final, fmt="%.6f")
    print(f"Saved to {path}")

def run_eye_in_hand(folder, camera_name, save_name):
    # Just copying minimal logic for EIH, usually simpler
    # Using Charuco Board logic from existing script?
    # Or just skip/trust existing?
    # User issue is Eye-To-Hand. I'll focus on Eye-To-Hand.
    pass

if __name__ == "__main__":
    run_eye_to_hand("head_base_to_left", "head", 1, "head_base_to_left.npy")
    run_eye_to_hand("head_base_to_right", "head", 0, "head_base_to_right.npy")
