#!/usr/bin/env python3
"""
Single-video motion classification (no masks, single camera).

Goal:
- Take ONE video (e.g. s1-left.mp4)
- Detect people / vehicles with YOLOv8
- Track per-object with ByteTrack
- Classify each track as STATIONARY vs MOVING:
    * If object stays within a small pixel radius
      for >= 2 seconds → STATIONARY
    * Otherwise → MOVING
- Output:
    * Annotated video with colored boxes (green = moving, red = stationary)
    * JSON summary with per-track metadata and counts

No 3D, no cross-camera matching, no masks.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np

# NumPy 2.x compatibility (required by ByteTrack / YOLOX)
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO
from tqdm import tqdm

from utils import setup_logger, load_config
from utils.io_utils import load_ply


@dataclass
class TrackDetection:
    frame_idx: int
    time_sec: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    center: np.ndarray  # [cx, cy]
    position_3d: Optional[np.ndarray] = None  # [X, Y, Z] in world coords


@dataclass
class TrackInfo:
    track_id: int
    class_name: str
    category: str  # "person" or "vehicle" or other
    detections: List[TrackDetection] = field(default_factory=list)
    is_stationary: Optional[bool] = None

    def add_detection(self, det: TrackDetection):
        """Add detection with simple temporal smoothing to reduce jitter.

        We apply an exponential moving average on bbox and center so that
        stationary objects do not pulsate frame-to-frame, while moving
        objects still reflect motion.
        """
        if self.detections:
            prev = self.detections[-1]
            alpha = 0.7  # weight for previous frame (strong smoothing)

            # Smooth center
            det.center = alpha * prev.center + (1.0 - alpha) * det.center

            # Smooth bbox coordinates
            det.bbox = alpha * prev.bbox + (1.0 - alpha) * det.bbox

        self.detections.append(det)

    def motion_state_at_time(
        self,
        current_time: float,
        window_sec: float = 2.0,
        static_radius_m: float = 0.03,
        allow_fallback: bool = False,
    ) -> bool:
        """Return True if the track is stationary around the given time.

        Uses both 3D displacement and 2D image evidence (center motion and
        scale change) over a sliding window.
        """
        times_list: List[float] = []
        pos_list: List[np.ndarray] = []
        center_list: List[np.ndarray] = []
        area_list: List[float] = []

        for d in self.detections:
            if d.position_3d is None:
                continue

            times_list.append(d.time_sec)
            pos_list.append(d.position_3d)
            center_list.append(d.center)
            bx1, by1, bx2, by2 = d.bbox
            area_list.append(float(max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))))

        if len(times_list) < 2:
            if allow_fallback and self.is_stationary:
                return True
            return False

        times = np.array(times_list, dtype=float)
        positions = np.stack(pos_list, axis=0)
        centers = np.stack(center_list, axis=0)
        areas = np.array(area_list, dtype=float)

        mask = (times <= current_time) & (times >= current_time - window_sec)
        if mask.sum() < 2:
            if allow_fallback and self.is_stationary:
                return True
            return False

        window_times = times[mask]
        window_pos = positions[mask]
        window_centers = centers[mask]
        window_areas = areas[mask]

        duration = float(window_times.max() - window_times.min())
        if duration < window_sec * 0.5:
            if allow_fallback and self.is_stationary:
                return True
            return False

        ref_pos = window_pos[0]
        disp3d = np.linalg.norm(window_pos - ref_pos, axis=1)
        max_disp3d = float(disp3d.max())

        ref_center = window_centers[0]
        disp2d = np.linalg.norm(window_centers - ref_center, axis=1)
        max_disp2d = float(disp2d.max())

        area_min = float(window_areas.min())
        area_max = float(window_areas.max())
        area_ratio = area_max / max(area_min, 1e-6)

        # Very strict 2D thresholds so that anything clearly moving in the
        # image (center shift or scale change) is always treated as moving.
        max_center_thresh = 5.0   # pixels
        max_area_ratio_static = 1.05  # <= 5% area change over the window

        # 3D trail length over the window (sum of step-wise displacements).
        traj_len3d = 0.0
        if window_pos.shape[0] >= 2:
            step_disp = np.linalg.norm(window_pos[1:] - window_pos[:-1], axis=1)
            traj_len3d = float(step_disp.sum())

        # Allow a bit more than static_radius_m as total path length for
        # stationary, since there can be noise; scale it from the radius.
        trail_len_thresh = static_radius_m * 5.0  # e.g. 0.03 -> 0.15 m

        cond_trail = traj_len3d <= trail_len_thresh
        cond_2d = max_disp2d <= max_center_thresh
        cond_scale = area_ratio <= max_area_ratio_static

        is_static = cond_trail and cond_2d and cond_scale

        # For tracks that have been globally classified as stationary,
        # optionally fall back to that label even if this short window
        # looks noisy (e.g. brief occlusions or detector glitches).
        if (not is_static) and allow_fallback and self.is_stationary:
            return True

        return is_static

    def classify_motion(self, min_duration_sec: float = 2.0, static_radius_m: float = 0.03):
        """Classify the whole track as stationary or moving.

        A track is considered stationary if:
        - Its total duration is at least ``min_duration_sec`` AND
        - It is stationary (according to ``motion_state_at_time``) for the
          majority of its lifespan.
        """
        if len(self.detections) < 2:
            self.is_stationary = False
            return

        times = np.array([d.time_sec for d in self.detections], dtype=float)
        total_duration = float(times.max() - times.min())
        if total_duration < min_duration_sec:
            self.is_stationary = False
            return

        pos3d_all: List[np.ndarray] = []
        centers_all: List[np.ndarray] = []
        for d in self.detections:
            if d.position_3d is None:
                continue
            pos3d_all.append(d.position_3d)
            centers_all.append(d.center)

        # Global envelope check: if over the whole track the 3D positions and
        # 2D centers stay in a tight ball, treat the object as stationary.
        if len(pos3d_all) >= 5 and total_duration >= 5.0:
            all_pos = np.stack(pos3d_all, axis=0)
            center_pos = all_pos.mean(axis=0)
            radii3d = np.linalg.norm(all_pos - center_pos, axis=1)
            max_radius3d = float(radii3d.max())

            centers_arr = np.stack(centers_all, axis=0)
            mean_center = centers_arr.mean(axis=0)
            radii2d = np.linalg.norm(centers_arr - mean_center, axis=1)
            max_radius2d = float(radii2d.max())

            global_static_radius_3d = static_radius_m * 5.0   # e.g. 0.15 m
            global_static_radius_2d = 30.0                    # pixels

            if max_radius3d <= global_static_radius_3d and max_radius2d <= global_static_radius_2d:
                self.is_stationary = True
                return

        static_flags: List[bool] = []
        for t_val in times:
            is_static_now = self.motion_state_at_time(
                current_time=float(t_val),
                window_sec=min_duration_sec,
                static_radius_m=static_radius_m,
                allow_fallback=False,
            )
            static_flags.append(is_static_now)

        static_fraction = float(np.mean(static_flags)) if static_flags else 0.0
        self.is_stationary = static_fraction >= 0.7


class Camera3D:
    """Simple pinhole camera model with ground-plane projection.

    We intersect camera rays with a fitted plane n·X + d = 0 in DUSt3R world
    coordinates. This plane is estimated once from the pass1 point cloud.
    """

    def __init__(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        plane_normal: np.ndarray,
        plane_d: float,
        calib_width: int,
        calib_height: int,
        video_width: int,
        video_height: int,
    ) -> None:
        self.K = K
        self.R = R
        self.t = t
        self.K_inv = np.linalg.inv(K)

        # Camera center in world coordinates
        self.C = -R.T @ t

        # Ground (or reference) plane parameters
        self.plane_n = plane_normal.astype(float)
        self.plane_d = float(plane_d)

        self.calib_width = float(calib_width)
        self.calib_height = float(calib_height)
        self.video_width = float(video_width)
        self.video_height = float(video_height)

    def project_pixel_to_ground(self, u_img: float, v_img: float) -> Optional[np.ndarray]:
        """Project image pixel to the fitted world plane in DUSt3R coordinates.

        Handles the scale difference between the calibration resolution and the
        actual video resolution by simple linear scaling.
        """
        if self.video_width <= 0 or self.video_height <= 0:
            return None

        # Map pixel coordinates from video resolution to calibration resolution
        u = u_img * (self.calib_width / self.video_width)
        v = v_img * (self.calib_height / self.video_height)

        pix = np.array([u, v, 1.0], dtype=float)
        ray_cam = self.K_inv @ pix
        # Direction of the ray in world coordinates
        ray_world = self.R.T @ ray_cam

        # Intersect with plane n·(C + λD) + d = 0  →  λ = -(n·C + d)/(n·D)
        n = self.plane_n
        d = self.plane_d

        num = -(float(n @ self.C) + d)
        den = float(n @ ray_world)
        if abs(den) < 1e-6:
            return None

        lam = num / den
        if lam <= 0:
            return None

        X = self.C + lam * ray_world
        return X.astype(float)

    def project_world_to_pixel(self, X_world: np.ndarray) -> Optional[np.ndarray]:
        X = np.asarray(X_world, dtype=float).reshape(3,)
        x_cam = self.R @ X + self.t
        if x_cam[2] <= 1e-6:
            return None
        uvw = self.K @ x_cam
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
        u_img = u * (self.video_width / self.calib_width)
        v_img = v * (self.video_height / self.calib_height)
        return np.array([u_img, v_img], dtype=float)


class SingleVideoMotionTracker:
    def __init__(self, config, logger, camera_id: Optional[str] = None):
        self.config = config
        self.logger = logger

        self.video_dir = Path(config["data"]["video_dir"])
        self.output_dir = Path(config["data"]["output_dir"]) / "pass2_dynamic"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pass1 (for camera calibration + point cloud)
        self.pass1_dir = Path(config["data"]["output_dir"]) / "pass1_static"
        self.camera3d: Optional[Camera3D] = None
        # Estimated ground plane n·X + d = 0 in DUSt3R coordinates
        self.plane_normal: Optional[np.ndarray] = None
        self.plane_d: Optional[float] = None
        # World origin at the centroid of all camera centers (for reference)
        self.world_origin: Optional[np.ndarray] = None
        # Memory of known stationary 3D locations (for occlusion handling)
        self.static_memory: List[Dict] = []

        # Which camera/video to use
        self.camera_id = camera_id

        # Classes to track (YOLO class names)
        tracking_cfg = config.get("pass2_dynamic", {}).get("tracking", {})
        self.pedestrian_classes = tracking_cfg.get("pedestrian_classes", ["person"])
        self.vehicle_classes = tracking_cfg.get("vehicle_classes", [
            "car", "truck", "bus", "motorcycle", "bicycle"
        ])
        self.track_classes = self.pedestrian_classes + self.vehicle_classes

        # Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()

        # Tracks: id -> TrackInfo
        self.tracks: Dict[int, TrackInfo] = {}

        # For second pass visualization: frame_idx -> list[(track_id, bbox)]
        self.frame_detections: Dict[int, List[Dict]] = {}

    def _init_models(self):
        self.logger.info("Loading YOLOv8 detector...")
        self.yolo = YOLO("yolov8x.pt")
        self.logger.info("✓ YOLO loaded")

    def _estimate_ground_plane(self) -> None:
        """Estimate ground plane from DUSt3R point cloud using PCA.

        We assume the road is the dominant roughly horizontal surface, so the
        plane normal should have a strong Z component.
        """
        ply_path = self.pass1_dir / "dust3r_pointcloud.ply"
        if not ply_path.exists():
            self.logger.warning(f"No DUSt3R point cloud found at {ply_path}, 3D motion disabled")
            self.plane_normal = None
            self.plane_d = None
            return

        try:
            points, _, _ = load_ply(str(ply_path))
        except Exception as e:
            self.logger.error(f"Failed to load point cloud {ply_path}: {e}")
            self.plane_normal = None
            self.plane_d = None
            return

        if points is None or len(points) < 1000:
            self.logger.warning("Point cloud too small to estimate ground plane")
            self.plane_normal = None
            self.plane_d = None
            return

        # Sub-sample for robustness
        max_pts = 200_000
        if points.shape[0] > max_pts:
            idx = np.random.choice(points.shape[0], max_pts, replace=False)
            pts = points[idx]
        else:
            pts = points

        mean = pts.mean(axis=0)
        X = pts - mean
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
        except Exception as e:
            self.logger.error(f"SVD failed during ground plane estimation: {e}")
            self.plane_normal = None
            self.plane_d = None
            return

        # Candidate plane normal is direction of smallest variance
        candidate = Vt[-1]
        # If that direction is not sufficiently vertical, fall back to the
        # row whose Z component has largest magnitude.
        if abs(candidate[2]) < 0.5:
            k = int(np.argmax(np.abs(Vt[:, 2])))
            candidate = Vt[k]

        normal = candidate.astype(float)
        if normal[2] < 0:
            normal = -normal

        d = float(-normal @ mean)

        self.plane_normal = normal
        self.plane_d = d
        self.logger.info(
            f"Estimated ground plane: n=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], d={d:.3f}"
        )

    def _init_camera3d_for_camera(self, cam_id: str, video_width: int, video_height: int):
        """Load calibration for a given camera from pass1 and build Camera3D."""
        camera_file = self.pass1_dir / "cameras.json"
        if not camera_file.exists():
            self.logger.warning(f"No cameras.json found at {camera_file}, 3D positions disabled")
            self.camera3d = None
            return

        with open(camera_file, "r") as f:
            cam_data = json.load(f)

        if cam_id not in cam_data:
            self.logger.warning(f"Camera {cam_id} not found in cameras.json, 3D positions disabled")
            self.camera3d = None
            return

        # Ensure we have a ground plane
        if self.plane_normal is None or self.plane_d is None:
            self._estimate_ground_plane()
        if self.plane_normal is None or self.plane_d is None:
            self.logger.warning("No valid ground plane; disabling 3D positions")
            self.camera3d = None
            return

        # Compute world origin (centroid of all camera centers) once
        if self.world_origin is None:
            centers = []
            for cid, info_all in cam_data.items():
                R_all = np.array(info_all["R"], dtype=float)
                t_all = np.array(info_all["t"], dtype=float)
                C_all = -R_all.T @ t_all
                centers.append(C_all)

            if centers:
                self.world_origin = np.mean(np.stack(centers, axis=0), axis=0)
                self.logger.info(
                    f"World origin (camera centroid): [{self.world_origin[0]:.3f}, {self.world_origin[1]:.3f}, {self.world_origin[2]:.3f}]"
                )

        info = cam_data[cam_id]
        K = np.array(info["K"], dtype=float)
        R = np.array(info["R"], dtype=float)
        t = np.array(info["t"], dtype=float)
        calib_w = int(info.get("width", video_width))
        calib_h = int(info.get("height", video_height))

        self.camera3d = Camera3D(
            K=K,
            R=R,
            t=t,
            plane_normal=self.plane_normal,
            plane_d=self.plane_d,
            calib_width=calib_w,
            calib_height=calib_h,
            video_width=video_width,
            video_height=video_height,
        )
        self.logger.info(f"✓ Loaded 3D calibration for {cam_id} (calib {calib_w}x{calib_h}, video {video_width}x{video_height})")

    def _resolve_camera_id(self) -> str:
        if self.camera_id is not None:
            return self.camera_id

        # Auto-pick first .mp4 in video_dir
        videos = sorted(self.video_dir.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 videos found in {self.video_dir}")
        return videos[0].stem

    def _class_to_category(self, class_name: str) -> str:
        if class_name in self.pedestrian_classes:
            return "person"
        if class_name in self.vehicle_classes:
            return "vehicle"
        return "other"

    def track_video(self):
        cam_id = self._resolve_camera_id()
        video_path = self.video_dir / f"{cam_id}.mp4"

        self.logger.info(f"Using camera: {cam_id}")
        self.logger.info(f"Video path: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"FPS: {fps:.2f}, Frames: {total_frames}, Size: {width}x{height}")

        # Initialize 3D camera model for this video
        self._init_camera3d_for_camera(cam_id, width, height)

        # ByteTrack
        from yolox.tracker.byte_tracker import BYTETracker

        class Args:
            track_thresh = 0.5
            track_buffer = 30
            match_thresh = 0.8
            mot20 = False

        tracker = BYTETracker(Args(), frame_rate=fps)

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Tracking", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps

            # YOLO detection
            results = self.yolo(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]

                if class_name not in self.track_classes:
                    continue

                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detections.append({
                    "bbox": bbox,
                    "score": conf,
                    "class_name": class_name,
                })

            if detections:
                det_array = np.array(
                    [
                        [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["score"]]
                        for d in detections
                    ],
                    dtype=np.float32,
                )
            else:
                det_array = np.empty((0, 5), dtype=np.float32)

            online_targets = tracker.update(
                det_array,
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]],
            )

            # Map ByteTrack outputs back to detections by IoU
            frame_track_entries: List[Dict] = []

            for target in online_targets:
                track_id = int(target.track_id)
                bbox_tlbr = target.tlbr  # numpy array [x1, y1, x2, y2]

                # Find best matching detection
                best_det = None
                best_iou = 0.0
                for det in detections:
                    iou = self._compute_iou(bbox_tlbr, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                if best_det is None:
                    continue

                x1, y1, x2, y2 = bbox_tlbr
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Approximate ground contact point as bottom center of bbox
                foot_u = (x1 + x2) / 2.0
                foot_v = y2
                pos3d = None
                if self.camera3d is not None:
                    pos3d = self.camera3d.project_pixel_to_ground(foot_u, foot_v)
                    if pos3d is not None and self.world_origin is not None:
                        pos3d = pos3d - self.world_origin

                track_det = TrackDetection(
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    bbox=bbox_tlbr.copy(),
                    center=np.array([cx, cy], dtype=float),
                    position_3d=pos3d,
                )

                if track_id not in self.tracks:
                    category = self._class_to_category(best_det["class_name"])
                    self.tracks[track_id] = TrackInfo(
                        track_id=track_id,
                        class_name=best_det["class_name"],
                        category=category,
                    )

                track_obj = self.tracks[track_id]
                track_obj.add_detection(track_det)
                last_det = track_obj.detections[-1]

                frame_track_entries.append(
                    {
                        "track_id": track_id,
                        "bbox": last_det.bbox.copy(),
                        "position_3d": last_det.position_3d,
                    }
                )

            if frame_track_entries:
                self.frame_detections[frame_idx] = frame_track_entries

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        self.logger.info(f"Total tracks: {len(self.tracks)}")

        # Classify tracks as stationary/moving using 3D motion
        for t in self.tracks.values():
            t.classify_motion(min_duration_sec=2.0, static_radius_m=0.03)

        # Build memory of stationary 3D locations (for occlusion handling)
        self._build_static_memory()

        # Summarize counts
        summary = self._build_summary(cam_id, fps, total_frames)

        # Save JSON
        summary_path = self.output_dir / f"{cam_id}_motion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Saved summary to {summary_path}")

        # Second pass: write annotated video
        output_video_path = self.output_dir / f"{cam_id}_motion_annotated.mp4"
        self._write_annotated_video(video_path, output_video_path, fps, width, height)
        self.logger.info(f"Saved annotated video to {output_video_path}")

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1 = max(float(bbox1[0]), float(bbox2[0]))
        y1 = max(float(bbox1[1]), float(bbox2[1]))
        x2 = min(float(bbox1[2]), float(bbox2[2]))
        y2 = min(float(bbox1[3]), float(bbox2[3]))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _build_static_memory(self) -> None:
        """Cache mean 3D positions of stationary tracks as memory anchors.

        This is used at annotation time so that objects re-detected at the same
        3D location after occlusion are still treated as stationary.
        """
        anchors: List[Dict] = []
        for track in self.tracks.values():
            if not track.is_stationary:
                continue

            pos = [d.position_3d for d in track.detections if d.position_3d is not None]
            if not pos:
                continue

            heights = []
            widths = []
            for d in track.detections:
                if d.position_3d is None:
                    continue
                bx1, by1, bx2, by2 = d.bbox
                widths.append(float(max(0.0, bx2 - bx1)))
                heights.append(float(max(0.0, by2 - by1)))

            if not heights or not widths:
                continue

            mean_pos = np.mean(np.stack(pos, axis=0), axis=0).astype(float)
            mean_h = float(np.mean(heights))
            mean_w = float(np.mean(widths))

            anchors.append(
                {
                    "pos": mean_pos,
                    "height_px": mean_h,
                    "width_px": mean_w,
                    "track_id": int(track.track_id),
                }
            )

        self.static_memory = anchors
        self.logger.info(f"Built static memory with {len(self.static_memory)} anchors")

    def _build_summary(self, cam_id: str, fps: float, total_frames: int) -> Dict:
        moving_counts = {"person": 0, "vehicle": 0, "other": 0}
        stationary_counts = {"person": 0, "vehicle": 0, "other": 0}

        track_summaries = []

        for track in self.tracks.values():
            if not track.detections:
                continue

            times = [d.time_sec for d in track.detections]
            centers = np.stack([d.center for d in track.detections], axis=0)
            avg_center = centers.mean(axis=0)
            duration = max(times) - min(times)

            pos3d_list = [d.position_3d for d in track.detections if d.position_3d is not None]
            avg_pos3d = None
            if pos3d_list:
                avg_pos3d = np.mean(np.stack(pos3d_list, axis=0), axis=0)

            cat = track.category if track.category in moving_counts else "other"
            if track.is_stationary:
                stationary_counts[cat] += 1
            else:
                moving_counts[cat] += 1

            track_summaries.append(
                {
                    "track_id": int(track.track_id),
                    "class_name": track.class_name,
                    "category": track.category,
                    "is_stationary": bool(track.is_stationary),
                    "num_detections": len(track.detections),
                    "duration_sec": float(duration),
                    "avg_center_px": [float(avg_center[0]), float(avg_center[1])],
                    "avg_position_3d": avg_pos3d.tolist() if avg_pos3d is not None else None,
                }
            )

        summary = {
            "camera_id": cam_id,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "num_tracks": len(track_summaries),
            "counts": {
                "moving": moving_counts,
                "stationary": stationary_counts,
            },
            "tracks": track_summaries,
        }
        return summary

    def _write_annotated_video(self, video_path: Path, output_path: Path, fps: float, width: int, height: int):
        self.logger.info("Writing annotated video...")
        cap = cv2.VideoCapture(str(video_path))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

        frame_idx = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="Annotating", unit="frame")

        # Track-level state machine to avoid oscillation between moving/static.
        # For each track we keep:
        #   state: committed stationary flag
        #   inst_prev: previous instantaneous decision
        #   inst_change_time: when inst_prev last changed
        track_state: Dict[int, Dict[str, float]] = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            dets = self.frame_detections.get(frame_idx, [])
            for entry in dets:
                track_id = entry["track_id"]
                bbox = entry["bbox"]
                pos3d = entry.get("position_3d", None)

                track = self.tracks.get(track_id)
                if track is None:
                    continue

                # Instantaneous decision from motion in a 2s window
                inst_static = track.motion_state_at_time(
                    current_time=current_time,
                    window_sec=2.0,
                    static_radius_m=0.03,
                    allow_fallback=True,
                )

                # Retrieve or initialize state machine data
                ts = track_state.get(track_id)
                if ts is None:
                    committed = False  # start as moving
                    inst_prev = inst_static
                    inst_change_time = current_time
                else:
                    committed = bool(ts["state"])
                    inst_prev = bool(ts["inst_prev"])
                    inst_change_time = float(ts["inst_change_time"])

                # Update instantaneous evidence change time
                if inst_static != inst_prev:
                    inst_prev = inst_static
                    inst_change_time = current_time

                # Hysteresis: require longer evidence to switch to static than
                # to switch back to moving.
                enter_static_t = 2.0  # need 2s of static evidence to enter static
                exit_static_t = 0.7   # 0.7s of moving evidence to leave static

                if (not committed) and inst_prev and (current_time - inst_change_time >= enter_static_t):
                    committed = True
                elif committed and (not inst_prev) and (current_time - inst_change_time >= exit_static_t):
                    committed = False

                track_state[track_id] = {
                    "state": float(committed),  # stored as float to keep JSON-like simplicity if needed
                    "inst_prev": float(inst_prev),
                    "inst_change_time": float(inst_change_time),
                }

                is_static_now = committed

                h_det = float(bbox[3] - bbox[1])
                w_det = float(bbox[2] - bbox[0])

                if (not is_static_now) and (pos3d is not None) and self.static_memory and h_det > 0 and w_det > 0:
                    try:
                        for anchor in self.static_memory:
                            # Only allow refinement for the *same* physical
                            # track, never for an unrelated occluding object.
                            if int(anchor.get("track_id", -1)) != int(track_id):
                                continue

                            pos_a = anchor["pos"]
                            h_a = float(anchor["height_px"])
                            if h_a <= 0:
                                continue
                            dist3d = float(np.linalg.norm(pos_a - pos3d))
                            height_ratio = h_det / h_a
                            if dist3d <= 0.015 and 0.9 <= height_ratio <= 1.1:
                                is_static_now = True
                                break
                    except Exception:
                        pass

                color = (0, 0, 255) if is_static_now else (0, 255, 0)  # red=static, green=move

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{track.class_name[:6]}#{track_id}"
                label += "_S" if is_static_now else "_M"

                # If currently moving, draw a 3-second polygon trail on the road.
                if not is_static_now:
                    left_pts: List[Tuple[int, int]] = []
                    right_pts: List[Tuple[int, int]] = []

                    # Prefer 3D ground contact point if available, otherwise
                    # fall back to image bbox bottoms.
                    if self.camera3d is not None and self.world_origin is not None:
                        for det in track.detections:
                            if (
                                current_time - 3.0 <= det.time_sec <= current_time
                                and det.bbox is not None
                                and det.position_3d is not None
                            ):
                                X_world = det.position_3d + self.world_origin
                                uv = self.camera3d.project_world_to_pixel(X_world)
                                if uv is None:
                                    continue

                                bx1, by1, bx2, by2 = det.bbox
                                width = float(max(0.0, bx2 - bx1))
                                half_w = 0.5 * width
                                cx = float(uv[0])
                                cy = float(uv[1])

                                left_pts.append((int(cx - half_w), int(cy)))
                                right_pts.append((int(cx + half_w), int(cy)))
                    else:
                        for det in track.detections:
                            if current_time - 3.0 <= det.time_sec <= current_time and det.bbox is not None:
                                bx1, by1, bx2, by2 = det.bbox
                                left_pts.append((int(bx1), int(by2)))
                                right_pts.append((int(bx2), int(by2)))

                    if len(left_pts) >= 2 and len(right_pts) >= 2:
                        poly_pts = np.array(left_pts + right_pts[::-1], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(frame, [poly_pts], (0, 255, 255))

                # Put label above box
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()


def main():
    config = load_config()
    logger = setup_logger("SingleVideoMotion")

    logger.info("=== Single-Video Motion Classification (pass2) ===\n")

    tracker = SingleVideoMotionTracker(config, logger, camera_id=None)
    tracker.track_video()

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()
