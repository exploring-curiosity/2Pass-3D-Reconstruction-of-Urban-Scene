import sys
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO

# Import V4 static logic (using relative import or copying relevant funcs)
# We'll copy the refined static logic here for self-containment
from pass2_dynamic_v5.pairwise_tracker import PairwiseTracker

def matches_pipe_shape(x, y):
    """Refined pipe shape filter."""
    # Bottom edge
    if y < -8 and -6 < x < 16: return True, "bottom"
    # Top edge
    if y > 12 and -10 < x < 14: return True, "top"
    # Left edge
    if x < -8 and -10 < y < 10: return True, "left"
    # Right edge
    if x > 10 and -12 < y < 14: return True, "right"
    return False, None

def get_static_objects(bg_dir, cameras, yolo):
    print("Detecting static objects (V5)...")
    all_dets = []
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    for cam_id in cam_ids:
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists(): continue
        
        bg = cv2.imread(str(bg_path))
        K = np.array(cameras[cam_id]['K']).reshape(3,3)
        pose = np.array(cameras[cam_id]['pose_c2w'])
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        results = yolo.predict(bg, conf=0.45, verbose=False, classes=[2, 3, 5, 7])
        if results[0].boxes is None: continue
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            cx, cy = (bbox[0]+bbox[2])/2, bbox[3]
            
            # Project to ground
            ray_c = np.linalg.inv(K) @ np.array([cx, cy, 1])
            ray_c /= np.linalg.norm(ray_c)
            ray_w = R @ ray_c
            if abs(ray_w[2]) > 1e-4:
                s = -t[2]/ray_w[2]
                if s > 0:
                    pt = t + s*ray_w
                    if np.linalg.norm(pt[:2]) < 30:
                        all_dets.append({'pos': pt[:2], 'conf': conf})

    # Cluster
    from sklearn.cluster import DBSCAN
    if not all_dets: return []
    
    positions = np.array([d['pos'] for d in all_dets])
    db = DBSCAN(eps=2.0, min_samples=1).fit(positions)
    labels = db.labels_
    
    static_objs = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1: continue
        cluster_dets = [all_dets[i] for i in range(len(all_dets)) if labels[i] == label]
        avg_pos = np.mean([d['pos'] for d in cluster_dets], axis=0)
        best_conf = max(d['conf'] for d in cluster_dets)
        
        matches, edge = matches_pipe_shape(avg_pos[0], avg_pos[1])
        if matches:
            # Orientation: Top/Bottom=Horizontal(0), Left/Right=Vertical(90)
            yaw = 0.0 if edge in ['bottom', 'top'] else np.pi/2
            static_objs.append({
                'id': f'static_{label}',
                'class': 'car',
                'width': 1.8, 'length': 4.5, 'height': 1.5,
                'mesh_id': 'car_01',
                'static': True,
                'position': [avg_pos[0], avg_pos[1], 0],
                'yaw': yaw,
                'confidence': best_conf
            })
            
    return static_objs

def main():
    base = Path(__file__).parent.parent
    data_dir = base / "data/processed"
    out_dir = base / "outputs/pass2_dynamic_v5"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Init Tracker
    tracker = PairwiseTracker(base / "outputs/pass1_static/pi3_cameras_corrected.json")
    yolo = YOLO('yolov8x.pt')
    
    # 1. Static Objects
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    static_objs = get_static_objects(base / "data/processed/static_backgrounds", cameras, yolo)
    print(f"Loaded {len(static_objs)} static objects")
    
    # 2. Dynamic Tracking Loop
    print("Running Pairwise Dynamic Tracker...")
    
    # Assume all videos have same length (450 frames)
    # Get list of video files
    video_files = {}
    video_dir = base / "StreetAware-sample"
    
    for cam_id in ['s1-left', 's1-right', 's2-left', 's2-right', 's3-left', 's3-right', 's4-left', 's4-right']:
        p = video_dir / f"{cam_id}.mp4"
        if not p.exists():
             print(f"Warning: {p} does not exist")
        video_files[cam_id] = cv2.VideoCapture(str(p))
        
    num_frames = int(video_files['s1-left'].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {num_frames} frames...")
    
    for frame_idx in tqdm(range(num_frames)):
        # Collect detections from all cameras for this frame
        frame_dets = {} # cam_id -> list of detection dicts
        
        for cam_id, cap in video_files.items():
            ret, frame = cap.read()
            if not ret: continue
            
            # Predict
            results = yolo.predict(frame, conf=0.4, verbose=False, classes=[2, 5, 7])
            dets = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    dets.append({'box': xyxy, 'class': cls, 'conf': conf})
            frame_dets[cam_id] = dets
            
        tracker.update(frame_idx, frame_dets)
        
    # 3. Smoothing and Export
    print("Smoothing tracks...")
    smoothed_tracks = tracker.smooth_tracks()
    
    # 4. Generate Scene JSON
    dynamic_scene_objs = []
    
    for tid, history in smoothed_tracks.items():
        # Get class from active_tracks (or history if stored)
        # Using car default
        obj_entry = {
            'id': f'dyn_{tid}',
            'class': 'car',
            'width': 1.8, 'length': 4.5, 'height': 1.5,
            'mesh_id': 'car_02',
            'static': False,
            'keyframes': []
        }
        
        for state in history:
            obj_entry['keyframes'].append({
                'frame': state['frame'],
                'position': state['pos'].tolist(),
                'rotation': state['rot']
            })
            
        dynamic_scene_objs.append(obj_entry)
        
    print(f"Generated {len(dynamic_scene_objs)} dynamic tracks")
    
    full_scene = static_objs + dynamic_scene_objs
    
    with open(out_dir / "scene_4d.json", 'w') as f:
        json.dump(full_scene, f, indent=2)
        
    print(f"Saved {out_dir}/scene_4d.json")

if __name__ == "__main__":
    main()
