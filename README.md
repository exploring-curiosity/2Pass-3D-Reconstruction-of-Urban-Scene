# 3-Pass 3D Reconstruction of Urban Scenes with Line-of-Sight Audit

A modular pipeline for 4D (3D + time) reconstruction of traffic intersections using multi-camera video footage, with integrated safety analysis. The system performs static scene reconstruction, dynamic object tracking, and Line-of-Sight (LOS) visibility auditing for pedestrian-vehicle safety assessment.

## Overview

This pipeline processes synchronized multi-camera video feeds to create:
1. **Static 3D Scene**: Dense point cloud of the environment (roads, buildings, infrastructure)
2. **Dynamic 4D Tracking**: 3D trajectories of moving objects (vehicles, pedestrians) over time
3. **Line-of-Sight Audit**: Visibility analysis between pedestrians and vehicles for safety assessment
4. **Interactive Visualization**: Web-based 4D viewer with playback controls

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: 8-Camera Synchronized Video               │
│        (s1-left, s1-right, s2-left, s2-right, ... s4-right)        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PASS 1: Static Scene Reconstruction             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Background      │ -> │ Pi3 Multi-View   │ -> │ Orientation    │ │
│  │ Extraction      │    │ Reconstruction   │    │ Correction     │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│                                                                     │
│  Output: pi3_pointcloud_corrected.ply, pi3_cameras_corrected.json  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PASS 2: Dynamic Object Tracking                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ YOLOv8          │ -> │ ByteTrack        │ -> │ Ground Plane   │ │
│  │ Detection       │    │ Multi-Object     │    │ Projection     │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│                                                                     │
│  Output: *_trajectories.json (per-camera 2D+3D tracks)             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        3D Trajectory Reprojection                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Reproject 2D detections to 3D using calibrated cameras      │   │
│  │ and estimated ground plane from Pass 1 point cloud          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output: *_trajectories_pi3.json (3D world coordinates)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PASS 3: Line-of-Sight Audit                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Color-Based     │ -> │ Voxel Grid       │ -> │ Ray Bundle     │ │
│  │ Segmentation    │    │ Occupancy        │    │ Casting        │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│                                                                     │
│  Output: los_audit_report.json, BEV visualizations                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Interactive 4D Web Viewer                      │
│  - Static point cloud with color                                    │
│  - Animated 3D bounding boxes for tracked objects                   │
│  - Time slider for 4D playback                                      │
│  - Orbit/pan/zoom camera controls                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Pi3 Neural Reconstruction**: Uses [Pi3](https://github.com/yyfz233/Pi3) for multi-view 3D reconstruction with automatic camera pose estimation
- **Robust Object Tracking**: YOLOv8x detection + ByteTrack for consistent object tracking across frames
- **Motion Classification**: Distinguishes stationary vs. moving objects using 3D displacement analysis
- **Ground Plane Projection**: Projects 2D detections to 3D using estimated ground plane from point cloud
- **Line-of-Sight Analysis**: Ray-casting based visibility analysis with voxel grid acceleration
- **Safety Auditing**: Identifies critical occlusion zones between pedestrians and vehicles
- **Memory-Efficient Processing**: GPU memory management to prevent OOM errors during long video processing
- **Web-Based Visualization**: Three.js viewer for interactive 4D scene exploration

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (RTX 3080+ recommended, 11+ GB VRAM)
- Linux with CUDA drivers installed
- Conda/Mamba package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/2Pass-3D-Reconstruction-of-Urban-Scene.git
cd 2Pass-3D-Reconstruction-of-Urban-Scene

# Create conda environment
mamba create -n acv2 python=3.10 -y
mamba activate acv2

# Run automated installation
chmod +x install_dependencies.sh
./install_dependencies.sh
```

The installation script handles:
- PyTorch with CUDA support
- YOLOv8 (ultralytics)
- ByteTrack for multi-object tracking
- SAM2 for segmentation
- DUSt3R/Pi3 for 3D reconstruction
- All other dependencies

## Data Preparation

Place your synchronized camera videos in the `StreetAware-sample/` directory:

```
StreetAware-sample/
├── s1-left.mp4
├── s1-right.mp4
├── s2-left.mp4
├── s2-right.mp4
├── s3-left.mp4
├── s3-right.mp4
├── s4-left.mp4
└── s4-right.mp4
```

Static background images should be extracted to:
```
data/processed/static_backgrounds/
├── s1-left_bg.png
├── s1-right_bg.png
└── ...
```

## Usage

### Option 1: Full Pipeline (Recommended)

```bash
python run_pipeline.py
```

This runs the complete pipeline:
1. Pi3 static reconstruction
2. Orientation correction
3. Trajectory reprojection
4. Launches the web viewer

**Command-line options:**
```bash
python run_pipeline.py --skip-pass1    # Skip static reconstruction
python run_pipeline.py --skip-pass2    # Skip dynamic tracking
python run_pipeline.py --viewer-only   # Only launch viewer
python run_pipeline.py --native-viewer # Use Open3D instead of web viewer
python run_pipeline.py --clean         # Clean derived outputs first
python run_pipeline.py --render-video  # Also render annotated MP4
```

### Option 2: Step-by-Step Execution

#### Pass 1: Static Scene Reconstruction

```bash
# Extract static backgrounds (if not already done)
python pass1_static/extract_static_backgrounds.py

# Run Pi3 reconstruction
python pass1_static/test_pi3.py

# Fix orientation and scale
python pass1_static/fix_pi3_orientation.py
```

**Output:**
- `outputs/pass1_static/pi3_pointcloud.ply` - Raw point cloud
- `outputs/pass1_static/pi3_pointcloud_corrected.ply` - Oriented point cloud (Z-up)
- `outputs/pass1_static/pi3_cameras_corrected.json` - Calibrated camera parameters

#### Pass 2: Dynamic Object Tracking

```bash
# Track objects in each camera (run for each camera)
python pass2_dynamic/single_video_motion.py --camera s1-left
python pass2_dynamic/single_video_motion.py --camera s1-right
# ... repeat for all cameras

# Or run all cameras
python pass2_dynamic/run_all_cameras_motion.py

# Reproject trajectories to 3D
python pass2_dynamic/reproject_trajectories.py
```

**Output:**
- `outputs/pass2_dynamic/<camera>_trajectories.json` - Per-camera 2D tracks
- `outputs/pass2_dynamic/<camera>_trajectories_pi3.json` - 3D reprojected tracks
- `outputs/pass2_dynamic/<camera>_motion_annotated.mp4` - Annotated video

#### Pass 3: Line-of-Sight Audit

```bash
# Run LOS audit on scene with objects
python "pass3_los audit/los_audit.py" --scene scene_with_objects.ply --output los_audit_report.json

# Generate Bird's-Eye View visualizations
python "pass3_los audit/visualize_los_audit.py" --report los_audit_report.json
```

**Output:**
- `los_audit_report.json` - Complete visibility analysis
- `los_visualizations/bev_combined.png` - Overview of all LOS rays
- `los_visualizations/bev_<pedestrian_id>.png` - Per-pedestrian visibility
- `los_visualizations/bev_safety_zones.png` - Critical occlusion zones

#### Visualization

```bash
# Web-based 4D viewer
python viewer/viewer_server.py

# Native Open3D viewer
python viewer/native_viewer.py
```

## Output Format

### Trajectory JSON Structure

```json
{
  "camera_id": "s1-left",
  "fps": 30.0,
  "num_tracks": 45,
  "trajectories": [
    {
      "track_id": 1,
      "class_name": "car",
      "category": "vehicle",
      "is_stationary": false,
      "num_frames": 150,
      "frames": [
        {
          "frame_idx": 0,
          "time_sec": 0.0,
          "bbox": [100, 200, 250, 350],
          "center_px": [175, 275],
          "position_3d": [5.2, 3.1, 0.0]
        }
      ]
    }
  ]
}
```

### Camera JSON Structure

```json
{
  "s1-left": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "R": [[...], [...], [...]],
    "t": [x, y, z],
    "pose_c2w": [[...4x4 matrix...]],
    "width": 2592,
    "height": 1944
  }
}
```

### LOS Audit Report Structure

```json
{
  "summary": {
    "total_pairs": 45,
    "visible_pairs": 32,
    "occluded_pairs": 13,
    "avg_visibility_score": 0.72,
    "scale_factor": 1.234,
    "visibility_threshold": 0.60
  },
  "object_positions": {
    "pedestrians": {
      "person_1": {"centroid": [x, y, z], "heading": 0.0}
    },
    "vehicles": {
      "car_1": {"centroid": [x, y, z], "heading": 1.57, "bbox_size": [4.5, 1.8, 1.5]}
    }
  },
  "critical_occlusions": [
    {
      "pedestrian": "person_1",
      "vehicle": "car_2",
      "visibility_score": 0.2,
      "distance": 8.5,
      "primary_occluder": "parked_vehicle"
    }
  ],
  "all_results": [...]
}
```

## Project Structure

```
2Pass-3D-Reconstruction-of-Urban-Scene/
├── config/
│   └── pipeline_config.yaml      # Pipeline configuration
├── data/
│   └── processed/
│       └── static_backgrounds/   # Extracted background images
├── pass1_static/
│   ├── test_pi3.py              # Pi3 reconstruction
│   ├── fix_pi3_orientation.py   # Orientation correction
│   ├── extract_static_backgrounds.py
│   └── ...                      # Alternative reconstruction methods
├── pass2_dynamic/
│   ├── single_video_motion.py   # Per-camera tracking
│   ├── reproject_trajectories.py # 3D reprojection
│   ├── run_all_cameras_motion.py
│   └── ...                      # Multi-camera tracking variants
├── pass3_los audit/
│   ├── los_audit.py             # Line-of-Sight visibility analysis
│   └── visualize_los_audit.py   # BEV visualization generation
├── viewer/
│   ├── viewer_server.py         # Web-based 4D viewer
│   ├── native_viewer.py         # Open3D viewer
│   └── data/                    # Viewer data files
├── utils/
│   ├── camera_utils.py          # Camera projection utilities
│   ├── geometry_utils.py        # 3D geometry functions
│   ├── io_utils.py              # PLY/JSON I/O
│   └── logger.py                # Logging utilities
├── experiments/
│   ├── scripts/                 # Evaluation scripts
│   └── results/                 # Experiment results
├── run_pipeline.py              # Main pipeline runner
├── requirements.txt             # Python dependencies
├── install_dependencies.sh      # Automated setup script
└── README.md
```

## Line-of-Sight (LOS) Audit System

The LOS Audit system (Pass 3) performs pedestrian-vehicle visibility analysis for traffic safety assessment.

### How It Works

1. **Color-Based Segmentation**: Objects are identified by color in the PLY file:
   - 🔵 Blue (RGB: 51, 153, 230) → Pedestrians
   - 🔴 Red (RGB: 230, 51, 51) → Cars
   - 🟢 Green (RGB: 51, 179, 51) → Trucks
   - 🟣 Pink (RGB: 204, 51, 204) → Cycles

2. **Scale Calibration**: Uses biological prior (median pedestrian height = 1.7m) to calibrate scene scale

3. **Voxel Grid Occupancy**: Static scene is voxelized (20cm leaf size) for efficient ray intersection

4. **Ray Bundle Casting**: For each pedestrian-vehicle pair:
   - Rays cast from pedestrian eye level (1.6m) to 5 vehicle keypoints
   - Keypoints: bumper center, headlights (L/R), hood center, roof front
   
5. **Visibility Scoring**: 
   - ≥60% rays clear → **Visible**
   - 40-60% rays clear → **Partial**
   - <40% rays clear → **Occluded**

6. **Occluder Classification**:
   - `static` - Infrastructure (buildings, poles, signs)
   - `parked_vehicle` - Stationary vehicles
   - `moving_vehicle` - Active traffic
   - `pedestrian` - Other pedestrians

### Output Visualizations

The system generates Bird's-Eye View (BEV) diagrams:
- **Combined BEV**: All pedestrian-vehicle rays with visibility coloring
- **Per-Pedestrian BEV**: Individual visibility analysis
- **Safety Zones**: Critical occlusion areas highlighted (distance < 15m, visibility < 40%)

## Configuration

Edit `config/pipeline_config.yaml` to customize:

```yaml
data:
  video_dir: "StreetAware-sample"
  cameras: ["s1-left", "s1-right", ...]
  fps: 30
  frame_sampling: 5  # Process every 5th frame

pass1_static:
  static_gaussians:
    iterations: 30000

pass2_dynamic:
  sample_rate: 5
  tracking:
    detector: "yolov8x"
    conf_threshold: 0.3
    pedestrian_classes: ["person"]
    vehicle_classes: ["car", "bus", "truck", "motorcycle", "bicycle"]

hardware:
  device: "cuda"
  mixed_precision: true
```

## Performance

**Tested on RTX 3080 (10GB VRAM):**

| Stage | Time | Output |
|-------|------|--------|
| Background extraction | ~2 min/camera | 8 PNG images |
| Pi3 reconstruction | ~3-5 min | ~1.6M points |
| Orientation fix | ~30 sec | Corrected PLY + cameras |
| Object tracking | ~5-10 min/camera | Trajectories JSON |
| 3D reprojection | ~1 min | 3D trajectories |
| LOS Audit | ~2-5 min | Visibility report + BEV figures |

## Troubleshooting

### GPU Memory Issues

If you encounter OOM errors:
1. Reduce `frame_sampling` in config (e.g., 5 → 10)
2. The system includes automatic GPU memory management (see `FIXES_APPLIED.md`)
3. Monitor GPU usage: `watch -n 1 nvidia-smi`

### CUDA Not Available

```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Pi3/DUSt3R Import Errors

```bash
# Ensure submodules are initialized
cd Pi3  # or dust3r
git submodule update --init --recursive
pip install -r requirements.txt
```

### ByteTrack Issues

```bash
cd ByteTrack
pip install --no-build-isolation -e .
```

## Dependencies

- **PyTorch** 2.0+ with CUDA
- **ultralytics** (YOLOv8)
- **ByteTrack** (multi-object tracking)
- **SAM2** (segmentation, optional)
- **Pi3** (neural 3D reconstruction)
- **Open3D** (point cloud processing)
- **Three.js** (web visualization)

## References

- [Pi3](https://github.com/yyfz233/Pi3) - Neural multi-view 3D reconstruction
- [DUSt3R](https://github.com/naver/dust3r) - Alternative 3D reconstruction
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Segmentation

## License

This project combines multiple open-source components. Please refer to individual repositories for their respective licenses.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3pass_3d_reconstruction,
  title = {3-Pass 3D Reconstruction of Urban Scenes with Line-of-Sight Audit},
  year = {2024},
  url = {https://github.com/yourusername/2Pass-3D-Reconstruction-of-Urban-Scene}
}
```
