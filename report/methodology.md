# Methodology Report

## Scene & Data
- **Project**: StreetAware-LoS-Audit
- **Scene**: 4-corner intersection with 8 synchronized cameras.
- **Inputs**:
  - Multi-camera videos: `StreetAware-sample/s*-*.mp4`
  - Global configuration: `config/pipeline_config.yaml`

## PASS 1: Static Scene Reconstruction (Done)

- **Goal**: Build a dense static 3D model of the intersection.
- **Method**:
  - Extract static background images per camera.
  - Run **DUSt3R** on backgrounds:
    - Estimate multi-view geometry and camera poses.
    - Produce a global point cloud in a shared world frame.
  - Save outputs in `outputs/pass1_static/`:
    - `dust3r_pointcloud.ply` – dense static 3D scene.
    - `cameras.json` – intrinsics/extrinsics per camera.
- **Status**: Implemented and producing good, consistent static geometry.

## PASS 2: Dynamic Objects (Current Prototype)

### 2.1 Per-Camera Tracking (ByteTrack + YOLO) – Implemented

- **Script**: `pass2_dynamic/single_video_motion.py`
- **Driver**: `pass2_dynamic/run_all_cameras_motion.py`
- **Goal**: Track people and vehicles over time in each camera and classify tracks as stationary vs moving.
- **Method**:
  - Load config and camera list from `pipeline_config.yaml`.
  - For each camera `cam_id`:
    - Open video `StreetAware-sample/{cam_id}.mp4`.
    - Run **YOLOv8x** detections for configured pedestrian/vehicle classes.
    - Run **ByteTrack** for multi-object tracking.
    - For each tracked object:
      - Compute per-frame 2D bbox and center.
      - Project an approximate ground-contact point to 3D using:
        - `outputs/pass1_static/cameras.json` (K, R, t).
        - Ground plane fitted from `dust3r_pointcloud.ply`.
      - Record a 3D trail in the same DUSt3R world frame.
    - Classify each track as **stationary** or **moving** using 3D and 2D motion over time.
  - Outputs per camera in `outputs/pass2_dynamic/`:
    - `{cam_id}_motion_summary.json` – per-track metadata:
      - `track_id`, `class_name`, `category` (person/vehicle),
      - `is_stationary`,
      - `num_detections`,
      - `duration_sec`, `start_time_sec`, `end_time_sec`,
      - `avg_center_px`, `avg_position_3d` (world coords).
    - `{cam_id}_motion_annotated.mp4` – visualization video.
- **Status**: Implemented and run for all 8 cameras; summaries and annotated videos exist.

### 2.2 Multi-Camera Primitive Representation – Implemented (First Pass)

- **Script**: `pass2_dynamic/multi_camera_primitives.py`
- **Goal**: Replace detailed dynamic geometry with simple 3D primitives in the static DUSt3R world frame.
- **Inputs**:
  - All per-camera motion summaries: `outputs/pass2_dynamic/*_motion_summary.json`.
- **Method**:
  1. **Load tracks with 3D info**
     - For each camera summary, collect tracks where `avg_position_3d` is present.
     - Keep only `category` in {`person`, `vehicle`}.
  2. **Multi-camera association (A+2)**
     - Represent each track by:
       - `category`, `class_name`, `avg_position_3d`, `start_time_sec`, `end_time_sec`, `is_stationary`.
     - Union–find clustering based on:
       - Same category.
       - 3D distance:
         - Vehicles: ≤ 2.0 m.
         - Persons:  ≤ 1.0 m.
       - Temporal overlap:
         - Absolute overlap ≥ 0.3 s.
         - Overlap / min(duration_i, duration_j) ≥ 0.3.
     - Result: each cluster ≈ one global object (can have tracks from multiple cameras).
  3. **Assign canonical primitives (Option A)**
     - For each cluster:
       - **Center**: mean of member `avg_position_3d`.
       - **Category**: person or vehicle.
       - **Class label**: most frequent `class_name` in cluster.
       - **Stationary**: majority vote over member `is_stationary`.
       - **Active time**: [min start, max end].
     - Vehicles → **axis-aligned boxes** with class-specific canonical sizes (L×W×H, in “DUSt3R meters”):
       - car:        4.5 × 1.8 × 1.5
       - truck:      8.0 × 2.5 × 3.0
       - bus:       10.0 × 2.5 × 3.2
       - motorcycle: 2.0 × 0.8 × 1.2
       - bicycle:    2.0 × 0.6 × 1.5
       - fallback:   4.0 × 1.7 × 1.5
     - People → **vertical cylinders**:
       - radius 0.4, height 1.7, axis [0, 0, 1].
  4. **Export**
     - `outputs/pass2_dynamic/multi_camera_primitives.json`:
       - One entry per object: primitive type, class, category, center, canonical size, time span, and list of member tracks.
     - `outputs/pass2_dynamic/multi_camera_primitives.ply`:
       - Point samples of all boxes/cylinders (red for vehicles, green for persons) for quick visualization.
- **Status**:
  - Implemented and run once.
  - Centers are expressed in the same world frame as the static DUSt3R point cloud, but the **canonical sizes currently do not match DUSt3R’s arbitrary scale**, leading to poor relative scale and appearance.

## Current Limitations / Issues

- **Scale mismatch** between DUSt3R world and canonical primitive sizes:
  - DUSt3R has an arbitrary metric; current box/cylinder dimensions assume 1 unit ≈ 1 meter.
  - Result: primitives appear too large or too small relative to the static cloud.
- **Primitive-only dynamics**:
  - Dynamic objects are represented by a single primitive per cluster with a time span, not per-frame geometry.
  - No detailed 4D (time-varying volume) modeling yet.
- **Aggressive multi-camera merging**:
  - Clustering can merge nearby vehicles into one global object if they share space and time.
  - No appearance-based separation yet.

## Planned Next Steps

1. **Scale Calibration for Primitives**
   - Estimate an overall or per-class scale factor from DUSt3R:
     - e.g., compare distances between camera centers or known road widths.
   - Apply a global scale to box and cylinder sizes so they visually align with the static point cloud.

2. **Refine Multi-Camera Association**
   - Tighten spatial and temporal thresholds to reduce over-merging.
   - Optionally include simple appearance cues (e.g., average color or bbox aspect ratio).

3. **4D / Per-Frame Primitive Trajectories**
   - For moving objects:
     - Use per-frame 3D positions along each track.
     - Emit a short sequence of primitives (one per time step) instead of a single averaged primitive.
   - For stationary objects:
     - Keep a single primitive with a time span, but validate with 3D motion thresholds.

4. **Integration with Visibility / LoS Analysis**
   - Consume `multi_camera_primitives.json` in the visibility stage to:
     - Treat vehicles/people as analytic occluders (boxes/cylinders) in ray tracing.
     - Combine static DUSt3R geometry + dynamic primitives for more interpretable LoS reports.

5. **Optional Future Enhancements**
   - Replace canonical primitive sizes with scene-calibrated dimensions per cluster using depth and bbox cues.
   - Add semantic tags for special vehicles (e.g., buses, emergency vehicles) and infrastructure occluders.
