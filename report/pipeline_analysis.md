# Pipeline Analysis and Recommendations

**Status: Updated Dec 3, 2025**

## Summary of Changes Made

### 1. Switched from DUSt3R to π³ (Pi3)
- Installed Pi3 from https://github.com/yyfz/Pi3
- Created `pass1_static/test_pi3.py` to run Pi3 on 8 cameras
- Pi3 produces 1.2M points with colors, ~3x larger scene extent than DUSt3R

### 2. Fixed Point Cloud Orientation
- Created `pass1_static/fix_pi3_orientation.py`
- Rotates scene to have ground plane at Z=0
- Scales to approximate metric units (~40m scene extent)
- Exports COLMAP format for potential Gaussian Splatting

### 3. Reprojected Trajectories
- Created `pass2_dynamic/reproject_trajectories.py`
- Uses Pi3 camera calibrations to project 2D detections to 3D
- Ground plane intersection for 3D positioning
- Saved as `*_trajectories_pi3.json`

### 4. New 4D Renderer with Solid Boxes
- Created `pass2_dynamic/render_4d_boxes.py`
- Renders dynamic objects as solid 3D wireframe boxes (not point clouds)
- Proper depth-based rendering
- Color coding: Red=moving vehicle, Orange=stationary vehicle, Green=moving person, Cyan=stationary person

---


## Current Issues Identified

### 1. Rendering Problems (render_4d_scene.py)
- **Colors Lost**: The `project_points_to_image` function filters points but doesn't return indices to filter corresponding colors. The color array becomes misaligned.
- **Blurry Output**: Drawing 1-pixel circles on a 1280x960 canvas with 200K points creates sparse, grainy visualization.
- **Wrong 3D Positions**: Dynamic object positions are in DUSt3R's relative scale (scene extent ~0.77m) which doesn't match real-world meters.

### 2. DUSt3R Limitations
- **Scale Ambiguity**: DUSt3R outputs are in arbitrary units, not metric scale.
- **Small Resolution**: Processing at 512x384 loses detail from 2592x1944 source images.
- **Pair-wise Processing**: 8 cameras = 28 pairs, computationally expensive.
- **Reference View Dependency**: Results vary based on which view is chosen as reference.

### 3. 3D Position Estimation (single_video_motion.py)
- Ground plane projection assumes flat ground and correct camera calibration.
- DUSt3R calibration is at 512x384, but videos are 2592x1944 - scale mismatch in intrinsics.
- World origin is camera centroid, not a meaningful real-world reference.

---

## π³ (Pi3) vs DUSt3R Comparison

| Feature | DUSt3R | π³ (Pi3) |
|---------|--------|----------|
| **Speed** | 1.25 FPS | 57.4 FPS (46x faster) |
| **Reference View** | Required (causes instability) | Not needed (permutation-equivariant) |
| **Multi-view** | Pair-wise then global alignment | Native multi-view support |
| **Output** | Point maps + poses | Point maps + poses + confidence |
| **Dynamic Scenes** | Static only | Supports dynamic scenes |
| **Scale** | Arbitrary | Scale-invariant local maps |
| **Robustness** | Sensitive to view order | Robust to permutations |

### Recommendation: **Use π³ for your 8-camera LOS audit**

**Reasons:**
1. **Native multi-view**: Handles 8 cameras directly without pair-wise processing
2. **Dynamic scene support**: Can handle moving objects in the scene
3. **Much faster**: 46x speedup enables processing more frames
4. **More robust**: No reference view selection issues
5. **Better for LOS**: Confidence scores help identify reliable geometry

---

## Gaussian Splatting: Before or After Pass 2?

### Option A: Gaussian Splatting BEFORE dynamic tracking (Recommended)
**Workflow:**
1. Extract static backgrounds
2. Run 3DGS on static backgrounds → high-quality static scene
3. Track dynamic objects in video
4. Composite dynamic primitives onto 3DGS renders

**Pros:**
- Photorealistic static scene rendering
- Real-time novel view synthesis
- Better depth/geometry for ground plane estimation
- Can use 3DGS depth for better 3D position estimation

**Cons:**
- Requires good camera poses first (use Pi3 or COLMAP)
- Training time for 3DGS (~10-30 min)

### Option B: Gaussian Splatting AFTER dynamic tracking
**Workflow:**
1. Track all objects
2. Remove dynamic objects from frames
3. Run 3DGS on cleaned frames
4. Add dynamic Gaussians for moving objects

**Pros:**
- Cleaner static scene (no ghosting from moving objects)
- Can create dynamic Gaussians for 4D reconstruction

**Cons:**
- More complex pipeline
- Requires good object masks
- Dynamic Gaussian methods are less mature

### **Recommendation: Option A (3DGS before Pass 2)**

For LOS audit, you need accurate static scene geometry. 3DGS provides:
- Dense, accurate depth maps
- Photorealistic rendering for visualization
- Better camera calibration refinement

---

## Proposed New Pipeline

### Phase 1: Camera Calibration & Static Reconstruction
```
1. Extract static backgrounds (existing)
2. Run π³ on 8 background images → camera poses + point maps
3. (Optional) Refine with COLMAP if needed
4. Train 3D Gaussian Splatting on static backgrounds
```

### Phase 2: Dynamic Object Tracking
```
1. Run YOLO + ByteTrack on all videos (existing)
2. Use 3DGS depth maps for better ground plane estimation
3. Project detections to 3D using refined camera calibration
4. Associate tracks across cameras
```

### Phase 3: 4D Visualization
```
1. Render static scene with 3DGS (photorealistic)
2. Overlay dynamic object primitives
3. Export as video
```

---

## Immediate Fixes Needed

### Fix 1: Color preservation in rendering
The projection function must return valid indices to filter colors correctly.

### Fix 2: Use proper intrinsic scaling
When projecting from DUSt3R calibration (512x384) to video resolution (2592x1944):
- Scale focal length by 2592/512 = 5.0625
- Scale principal point similarly

### Fix 3: Better visualization
- Use larger point sizes or splat rendering
- Implement proper depth-based occlusion
- Consider using Open3D or PyVista for better rendering

---

## Next Steps

1. **Install π³**: `git clone https://github.com/yyfz/Pi3.git`
2. **Test π³ on your 8 cameras**: Compare quality with DUSt3R
3. **Fix rendering bugs**: Implement color-preserving projection
4. **Evaluate 3DGS**: Test with nerfstudio or gsplat on static backgrounds
5. **Integrate best approach**: Build unified pipeline

