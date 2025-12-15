import numpy as np
import cv2
import json
from pathlib import Path
from collections import deque, defaultdict
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment

class TrackerKalmanFilter:
    def __init__(self, initial_pos):
        # 6 state vars: x, y, z, vx, vy, vz
        # 3 meas vars: x, y, z
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([[1,0,0,1,0,0],
                                             [0,1,0,0,1,0],
                                             [0,0,1,0,0,1],
                                             [0,0,0,1,0,0],
                                             [0,0,0,0,1,0],
                                             [0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0],
                                              [0,1,0,0,0,0],
                                              [0,0,1,0,0,0]], np.float32)
        # Noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        
        # Init state
        self.kf.statePost = np.array([[initial_pos[0]], [initial_pos[1]], [initial_pos[2]], 
                                      [0], [0], [0]], dtype=np.float32)
                                      
    def predict(self):
        return self.kf.predict()
        
    def update(self, measurement):
        m = np.array(measurement, dtype=np.float32).reshape(3,1)
        self.kf.correct(m)
        
    @property
    def pos(self):
        return self.kf.statePost[:3].flatten()

class PairwiseTracker:
    def __init__(self, cameras_file):
        with open(cameras_file, 'r') as f:
            self.cameras = json.load(f)
        
        # Define pairs (Left, Right)
        self.pairs = [
            ('s1-left', 's1-right'),
            ('s2-left', 's2-right'),
            ('s3-left', 's3-right'),
            ('s4-left', 's4-right')
        ]
        
        # Tracking state
        self.next_id = 1
        self.tracks = {}  # id -> list of states
        self.active_tracks = {} # id -> current Kalman filter
        self.max_coast_cycles = 10
        self.min_hits = 5
    
    def triangulate_point(self, cam1_name, cam2_name, pt1, pt2):
        """Triangulate 2D points from two cameras to 3D."""
        cam1 = self.cameras[cam1_name]
        cam2 = self.cameras[cam2_name]
        
        # Projection matrices P = K * [R|t]
        K1 = np.array(cam1['K']).reshape(3,3)
        Rt1 = np.linalg.inv(np.array(cam1['pose_c2w']))[:3, :] # w2c
        P1 = K1 @ Rt1
        
        K2 = np.array(cam2['K']).reshape(3,3)
        Rt2 = np.linalg.inv(np.array(cam2['pose_c2w']))[:3, :] # w2c
        P2 = K2 @ Rt2
        
        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2,1), pt2.reshape(2,1))
        # Handle division by zero
        w = pts4d[3]
        if abs(w) < 1e-6: return np.array([np.nan, np.nan, np.nan])
        pts3d = pts4d[:3] / w
        
        return pts3d.flatten()

    def get_center(self, box):
        # Center of bottom edge is better for ground projection
        return np.array([(box[0]+box[2])/2, box[3]])

    def project_to_ground(self, cam_name, uv):
        """Project pixel to z=0 plane."""
        cam = self.cameras[cam_name]
        K = np.array(cam['K']).reshape(3,3)
        c2w = np.array(cam['pose_c2w'])
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        
        # Ray dir in camera
        ray_c = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1])
        ray_c = ray_c / np.linalg.norm(ray_c)
        
        # Ray in world
        ray_w = R @ ray_c
        
        # Intersect with z=0
        if abs(ray_w[2]) < 1e-4: return None
        alpha = -t[2] / ray_w[2]
        if alpha < 0: return None # Behind camera
        
        return t + alpha * ray_w

    def match_within_pair(self, dets1, dets2, cam1, cam2):
        """Match detections between stereo pair using 3D proximity/Epipolar."""
        matches = []
        unmatched1 = set(range(len(dets1)))
        
        ground_pts1 = []
        for d in dets1:
            c = self.get_center(d['box'])
            gp = self.project_to_ground(cam1, c)
            ground_pts1.append(gp)
            
        ground_pts2 = []
        for d in dets2:
            c = self.get_center(d['box'])
            gp = self.project_to_ground(cam2, c)
            ground_pts2.append(gp)
            
        # Cost matrix
        cost = np.ones((len(dets1), len(dets2))) * 1000.0
        
        for i, gp1 in enumerate(ground_pts1):
            if gp1 is None: continue
            for j, gp2 in enumerate(ground_pts2):
                if gp2 is None: continue
                cost[i,j] = np.linalg.norm(gp1 - gp2)
        
        if len(dets1) > 0 and len(dets2) > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            
            for r, c in zip(row_ind, col_ind):
                if cost[r,c] < 2.0: # 2 meters tolerance
                    # Triangulate
                    c1_uv = np.array([(dets1[r]['box'][0]+dets1[r]['box'][2])/2, (dets1[r]['box'][1]+dets1[r]['box'][3])/2])
                    c2_uv = np.array([(dets2[c]['box'][0]+dets2[c]['box'][2])/2, (dets2[c]['box'][1]+dets2[c]['box'][3])/2])
                    
                    p3d = self.triangulate_point(cam1, cam2, c1_uv, c2_uv)
                    
                    # Sanity checks
                    if np.isnan(p3d).any() or abs(p3d[2]) > 3.0 or np.linalg.norm(p3d[:2]) > 50:
                         p3d = (ground_pts1[r] + ground_pts2[c]) / 2 if (ground_pts1[r] is not None and ground_pts2[c] is not None) else p3d

                    # Keep only if valid 3D
                    if not np.isnan(p3d).any():
                        matches.append({
                            'pos': p3d,
                            'conf': (dets1[r]['conf'] + dets2[c]['conf'])/2,
                            'class': dets1[r]['class']
                        })
                        unmatched1.discard(r)
        
        # Add unmatched from cam1
        for i in unmatched1:
             if ground_pts1[i] is not None:
                 matches.append({
                     'pos': ground_pts1[i],
                     'conf': dets1[i]['conf'],
                     'class': dets1[i]['class']
                 })
                 
        return matches

    def update(self, frame_id, detections_by_camera):
        """Process one frame."""
        # 1. Gather 3D hypotheses from all pairs
        measurements = []
        
        for (c1, c2) in self.pairs:
            dets1 = [d for d in detections_by_camera.get(c1, []) if d['class'] in [2,5,7]] 
            dets2 = [d for d in detections_by_camera.get(c2, []) if d['class'] in [2,5,7]]
            
            pair_measurements = self.match_within_pair(dets1, dets2, c1, c2)
            measurements.extend(pair_measurements)
            
        # 2. Track Association (Hungarian)
        track_ids = list(self.active_tracks.keys())
        matrix = np.ones((len(track_ids), len(measurements))) * 1000.0
        
        for i, tid in enumerate(track_ids):
            kf = self.active_tracks[tid]['kf']
            pred_pos = kf.pos
            for j, meas in enumerate(measurements):
                dist = np.linalg.norm(pred_pos - meas['pos'])
                if dist < 5.0: # Gating
                    matrix[i,j] = dist
        
        assigned_tids = set()
        assigned_meas_idx = set()
        
        if len(track_ids) > 0 and len(measurements) > 0:
            row_ind, col_ind = linear_sum_assignment(matrix)
            for r, c in zip(row_ind, col_ind):
                if matrix[r,c] < 3.0:
                    tid = track_ids[r]
                    meas = measurements[c]
                    
                    kf = self.active_tracks[tid]['kf']
                    kf.update(meas['pos'])
                    self.active_tracks[tid]['hits'] += 1
                    self.active_tracks[tid]['coast'] = 0
                    self.tracks[tid].append({'frame': frame_id, 'pos': kf.pos})
                    
                    assigned_tids.add(tid)
                    assigned_meas_idx.add(c)
            
        # Create new tracks
        for j, meas in enumerate(measurements):
            if j not in assigned_meas_idx:
                self.create_track(frame_id, meas)
                
        # Prune dead tracks
        dead_ids = []
        for tid in self.active_tracks:
            if tid not in assigned_tids:
                self.active_tracks[tid]['coast'] += 1
                self.active_tracks[tid]['kf'].predict()
                if self.active_tracks[tid]['coast'] > self.max_coast_cycles:
                    dead_ids.append(tid)
                    
        for tid in dead_ids:
            del self.active_tracks[tid]
            
    def create_track(self, frame_id, meas):
        kf = TrackerKalmanFilter(meas['pos'])
        
        self.active_tracks[self.next_id] = {
            'kf': kf,
            'hits': 1,
            'coast': 0,
            'class': meas['class']
        }
        self.tracks[self.next_id] = [{'frame': frame_id, 'pos': meas['pos']}]
        self.next_id += 1

    def smooth_tracks(self):
        """Apply post-processing smoothing."""
        smoothed = {}
        for tid, history in self.tracks.items():
            if len(history) < self.min_hits: continue
            
            # Extract arrays
            pos = np.array([h['pos'] for h in history])
            frames = np.array([h['frame'] for h in history])
            
            # SavGol smoothing
            if len(pos) >= 7:
                 try:
                     pos[:,0] = savgol_filter(pos[:,0], 7, 2)
                     pos[:,1] = savgol_filter(pos[:,1], 7, 2)
                     pos[:,2] = savgol_filter(pos[:,2], 7, 2)
                 except: pass
            
            # Solve orientation (yaw) from velocity
            yaws = []
            if len(pos) > 1:
                vel = np.gradient(pos, axis=0)
                yaw = 0.0
                for i in range(len(vel)):
                    v = vel[i]
                    speed = np.linalg.norm(v[:2])
                    if speed > 0.05: # Use threshold
                        yaw = np.arctan2(v[1], v[0])
                    yaws.append(yaw)
            else:
                yaws = [0.0]
                
            # Smooth yaws
            if len(yaws) >= 7:
                 try:
                     yaws_unwrapped = np.unwrap(yaws)
                     yaws_smooth = savgol_filter(yaws_unwrapped, 7, 2)
                     yaws = yaws_smooth
                 except: pass
            
            smoothed[tid] = []
            for i in range(len(history)):
                smoothed[tid].append({
                    'frame': int(frames[i]),
                    'pos': pos[i],
                    'rot': R.from_euler('z', yaws[i]).as_quat().tolist() 
                })
        return smoothed
