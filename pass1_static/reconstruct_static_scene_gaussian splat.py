#!/usr/bin/env python3
"""
Static scene 3D reconstruction using Street Gaussians

This script reconstructs the static intersection infrastructure using
the Street Gaussians framework adapted for multi-camera traffic scenes.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, load_config, save_ply, CameraSet, CameraParams


class StaticSceneReconstructor:
    """
    Reconstruct static scene from multiple camera views
    
    Uses Street Gaussians adapted for static infrastructure:
    - Roads
    - Sidewalks
    - Buildings
    - Poles
    - Traffic lights
    - Lane markings
    """
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.device = config['hardware']['device']
        
        # Paths
        self.bg_dir = Path(config['data']['processed_dir']) / "static_backgrounds"
        self.output_dir = Path(config['data']['output_dir']) / "pass1_static"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        self.static_config = config['pass1_static']['static_gaussians']
        self.cameras = config['data']['cameras']
        
        self.logger.info("Static Scene Reconstructor initialized")
    
    def load_background_images(self) -> Dict[str, np.ndarray]:
        """Load all static background images"""
        images = {}
        
        for camera_name in self.cameras:
            img_path = self.bg_dir / f"{camera_name}_bg.png"
            
            if not img_path.exists():
                self.logger.warning(f"Background not found: {img_path}")
                continue
            
            import cv2
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[camera_name] = img
        
        self.logger.info(f"Loaded {len(images)} background images")
        return images
    
    def estimate_camera_poses(self, images: Dict[str, np.ndarray]) -> CameraSet:
        """
        Estimate camera poses from static backgrounds using DUSt3R
        
        DUSt3R predicts 3D point maps for each image and aligns them
        to recover camera poses and a sparse 3D reconstruction.
        
        Returns:
            CameraSet with estimated poses and intrinsics
        """
        self.logger.info("Estimating camera poses with DUSt3R...")
        
        try:
            import sys
            from pathlib import Path
            
            # Add DUSt3R to path
            dust3r_path = Path(__file__).parent.parent / "dust3r"
            sys.path.insert(0, str(dust3r_path))
            
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            
            # Load DUSt3R model
            self.logger.info("Loading DUSt3R model...")
            model_path = dust3r_path / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            
            if not model_path.exists():
                self.logger.error(f"DUSt3R checkpoint not found: {model_path}")
                self.logger.warning("Falling back to placeholder camera poses")
                return self._create_placeholder_cameras(images)
            
            # Patch torch.load to use weights_only=False
            import torch
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
            
            try:
                model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path)).to(self.device)
            finally:
                torch.load = original_torch_load
            
            # Prepare images for DUSt3R
            self.logger.info(f"Processing {len(images)} images...")
            temp_dir = Path("/tmp/dust3r_images")
            temp_dir.mkdir(exist_ok=True)
            
            # Save images temporarily
            image_paths = []
            image_names = []
            for cam_name, img in images.items():
                img_path = temp_dir / f"{cam_name}.jpg"
                import cv2
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_paths.append(str(img_path))
                image_names.append(cam_name)
            
            # Load and preprocess images
            imgs = load_images(image_paths, size=512)
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
            
            # Run inference
            self.logger.info("Running DUSt3R inference...")
            output = inference(pairs, model, self.device, batch_size=1)
            
            # Global alignment to get cameras
            self.logger.info("Performing global alignment...")
            scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
            
            # Extract camera parameters
            self.logger.info("Extracting camera parameters...")
            camera_set = CameraSet()
            
            imgs_dict = {img['idx']: img for img in imgs}
            
            for idx, cam_name in enumerate(image_names):
                img_dict = imgs_dict[idx]
                h, w = img_dict['true_shape'][0]
                
                # Get camera from scene
                focals = scene.get_focals()
                principal_points = scene.get_principal_points()
                im_poses = scene.get_im_poses()
                
                # Extract for this camera
                focal = focals[idx].detach().cpu().numpy()
                pp = principal_points[idx].detach().cpu().numpy()
                pose = im_poses[idx].detach().cpu().numpy()  # 4x4 matrix
                
                # Build intrinsics matrix
                fx = fy = focal[0] if focal.ndim == 1 else focal
                cx, cy = pp[0], pp[1]
                
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Extract rotation and translation from pose
                R = pose[:3, :3]
                t = pose[:3, 3]
                
                # Create camera
                cam = CameraParams(
                    camera_id=cam_name,
                    R=R,
                    t=t,
                    K=K,
                    width=int(w),
                    height=int(h)
                )
                
                camera_set.add_camera(cam)
                self.logger.info(f"  {cam_name}: focal={fx:.1f}px, pos=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
            
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            self.logger.info(f"✓ Estimated poses for {len(camera_set)} cameras")
            return camera_set
            
        except Exception as e:
            self.logger.error(f"DUSt3R pose estimation failed: {e}")
            self.logger.warning("Falling back to placeholder camera poses")
            import traceback
            traceback.print_exc()
            return self._create_placeholder_cameras(images)
    
    def _create_placeholder_cameras(self, images: Dict[str, np.ndarray]) -> CameraSet:
        """Create placeholder camera poses in circular arrangement"""
        self.logger.info("Creating placeholder camera arrangement...")
        
        camera_set = CameraSet()
        
        # Placeholder: Create cameras in a rough circle around origin
        n_cams = len(images)
        for idx, (cam_name, img) in enumerate(images.items()):
            h, w = img.shape[:2]
            
            # Rough estimates
            angle = 2 * np.pi * idx / n_cams
            radius = 20.0  # meters
            
            # Camera position
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_z = 3.0  # Height above ground
            
            # Camera looks toward origin
            look_at = np.array([0.0, 0.0, 0.0])
            cam_pos = np.array([cam_x, cam_y, cam_z])
            
            # Compute rotation (camera looking at origin)
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, np.array([0, 0, 1]))
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            R = np.vstack([right, -up, forward]).T
            
            # Translation
            t = -R @ cam_pos
            
            # Intrinsics (rough estimates)
            focal = max(w, h)
            K = np.array([
                [focal, 0, w/2],
                [0, focal, h/2],
                [0, 0, 1]
            ])
            
            camera = CameraParams(
                K=K,
                R=R,
                t=t,
                width=w,
                height=h,
                camera_id=cam_name
            )
            
            camera_set.add_camera(camera)
        
        self.logger.warning("⚠️  Using PLACEHOLDER camera poses!")
        self.logger.warning("   For production, implement DUSt3R pose estimation")
        
        return camera_set
    
    def create_static_reconstruction(
        self,
        images: Dict[str, np.ndarray],
        cameras: CameraSet
    ):
        """
        Create static scene reconstruction using DUSt3R
        
        Args:
            images: Dictionary of RGB images
            cameras: Estimated camera parameters
        
        Returns:
            Path to output PLY file
        """
        self.logger.info("Creating static scene 3D reconstruction...")
        
        # Generate DUSt3R point cloud
        result = self._create_point_cloud_reconstruction(images, cameras)
        
        if isinstance(result, tuple):
            dust3r_file, _, _ = result
        else:
            dust3r_file = result
        
        return dust3r_file
    
    def _create_point_cloud_reconstruction(
        self,
        images: Dict[str, np.ndarray],
        cameras: CameraSet
    ):
        """
        Create 3D point cloud using DUSt3R's actual point predictions
        
        DUSt3R predicts dense 3D point maps for each image. We extract these
        and merge them into a global point cloud.
        """
        self.logger.info("Creating point cloud from DUSt3R predictions...")
        
        try:
            import sys
            from pathlib import Path
            
            # Add DUSt3R to path
            dust3r_path = Path(__file__).parent.parent / "dust3r"
            sys.path.insert(0, str(dust3r_path))
            
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            
            # Load model
            self.logger.info("Loading DUSt3R model...")
            model_path = dust3r_path / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            
            if not model_path.exists():
                self.logger.error("DUSt3R checkpoint not found")
                return self._create_dummy_output()
            
            # Patch torch.load
            import torch
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
            
            try:
                model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path)).to(self.device)
            finally:
                torch.load = original_torch_load
            
            # Prepare images
            temp_dir = Path("/tmp/dust3r_images")
            temp_dir.mkdir(exist_ok=True)
            
            image_paths = []
            image_names = []
            for cam_name, img in images.items():
                img_path = temp_dir / f"{cam_name}.jpg"
                import cv2
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_paths.append(str(img_path))
                image_names.append(cam_name)
            
            # Run DUSt3R
            self.logger.info("Running DUSt3R inference...")
            imgs = load_images(image_paths, size=512)
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, self.device, batch_size=1)
            
            # Global alignment
            self.logger.info("Performing global alignment...")
            scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
            
            # Extract 3D points from DUSt3R
            self.logger.info("Extracting 3D point cloud from DUSt3R...")
            
            all_points = []
            all_colors = []
            all_confidences = []
            
            # Get predicted pointmaps and colors
            imgs_dict = {img['idx']: img for img in imgs}
            
            for idx, cam_name in enumerate(image_names):
                # Get predicted 3D points in global coordinates
                pts3d = scene.get_pts3d()[idx].detach().cpu().numpy()  # [H, W, 3]
                confidence = scene.get_conf()[idx].detach().cpu().numpy()  # [H, W]
                
                # Get colors from original image
                img_dict = imgs_dict[idx]
                img_rgb = images[cam_name]
                
                # Reshape
                h, w = pts3d.shape[:2]
                pts3d_flat = pts3d.reshape(-1, 3)
                conf_flat = confidence.reshape(-1)
                
                # Resize colors to match DUSt3R resolution
                img_resized = cv2.resize(img_rgb, (w, h))
                colors_flat = img_resized.reshape(-1, 3)
                
                all_points.append(pts3d_flat)
                all_colors.append(colors_flat)
                all_confidences.append(conf_flat)
            
            # Concatenate all points
            points_3d = np.vstack(all_points)
            colors_3d = np.vstack(all_colors)
            confidences = np.concatenate(all_confidences)
            
            self.logger.info(f"Extracted {len(points_3d)} raw points from DUSt3R")
            
            # Filter by confidence
            conf_threshold = np.percentile(confidences, 10)  # Keep top 90%
            mask = confidences > conf_threshold
            points_3d = points_3d[mask]
            colors_3d = colors_3d[mask]
            
            self.logger.info(f"After confidence filter: {len(points_3d)} points")
            
            # Filter outliers by distance
            center = points_3d.mean(axis=0)
            distances = np.linalg.norm(points_3d - center, axis=1)
            max_dist = np.percentile(distances, 95)
            
            mask = distances < max_dist
            points_3d = points_3d[mask]
            colors_3d = colors_3d[mask]
            
            self.logger.info(f"After outlier filter: {len(points_3d)} points")
            
            # Save DUSt3R point cloud
            output_file = self.output_dir / "dust3r_pointcloud.ply"
            save_ply(output_file, points_3d, colors_3d)
            
            # Save cameras
            cameras.save(self.output_dir / "cameras.json")
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            self.logger.info(f"✓ Saved DUSt3R point cloud to {output_file}")
            
            return output_file, points_3d, colors_3d
            
        except Exception as e:
            self.logger.error(f"DUSt3R point cloud extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_dummy_output(), None, None
    
    def _create_dummy_output(self):
        """Create dummy output for testing"""
        output_file = self.output_dir / "dust3r_pointcloud.ply"
        
        points = np.random.randn(1000, 3) * 5
        colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
        
        save_ply(output_file, points, colors)
        
        self.logger.warning("⚠️  Using DUMMY output")
        
        return output_file, None, None
    
    def create_gaussian_splats(self, points_3d, colors_3d, cameras):
        """
        Convert DUSt3R static point cloud into a Gaussian Splat representation.

        This does NOT perform full 3DGS optimization.
        It simply converts sparse DUSt3R points into:
        - mean (x, y, z)
        - spherical covariance (isotropic Gaussian)
        - color (r, g, b)
        - opacity

        Output is compatible with most 3DGS loaders.
        """
        self.logger.info("Converting static scene into Gaussian Splats...")

        if points_3d is None or colors_3d is None:
            self.logger.error("No point cloud available for Gaussian splatting.")
            return None

        # Number of points
        N = points_3d.shape[0]

        # Normalize colors
        colors_norm = colors_3d.astype(np.float32) / 255.0

        # Default gaussian scale (meters)
        base_scale = self.static_config.get("gaussian_base_scale", 0.05)

        # Compute per-point scale using local density
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(points_3d)
        dist, _ = nbrs.kneighbors(points_3d)

        # Mean distance to neighbors → scale
        scales = np.clip(dist.mean(axis=1), 0.01, 0.25)
        scales = scales * base_scale

        # Opacity (default = fully opaque)
        opacity = np.ones((N, 1), dtype=np.float32)

        # Construct Gaussian dictionary
        gaussians = {
            "means": points_3d.astype(np.float32),
            "scales": scales.astype(np.float32).reshape(-1, 1),
            "colors": colors_norm.astype(np.float32),
            "opacity": opacity.astype(np.float32),
        }

        # Save to .npz for compatibility with gsplat/pyrenderers
        output_path = self.output_dir / "static_scene_gaussians.npz"
        np.savez_compressed(output_path, **gaussians)

        self.logger.info(f"✓ Saved Gaussian Splat model → {output_path}")

        return output_path

    
    def run(self):
        """Run complete static scene reconstruction"""
        self.logger.info("=== PASS 1: Static Scene Reconstruction ===\n")
        
        # Load images
        images = self.load_background_images()
        
        if len(images) == 0:
            self.logger.error("No background images found!")
            self.logger.info("Run extract_static_backgrounds.py first")
            return
        
        # Estimate camera poses
        cameras = self.estimate_camera_poses(images)

        # Create static reconstruction
        output_file = self.create_static_reconstruction(images, cameras)
        # _create_point_cloud_reconstruction returns (ply_file, points_3d, colors_3d) on success
        result = self._create_point_cloud_reconstruction(images, cameras)
        ply_file = None
        points = None
        colors = None

        if isinstance(result, tuple):
            ply_file, points, colors = result
        elif isinstance(result, (Path, str)):
            ply_file = result
        else:
            # If create_static_reconstruction() was used earlier and it returned only a path, handle that
            ply_file = result

        # If we have points and colors, create gaussian splats
        if points is not None and colors is not None:
            gs_file = self.create_gaussian_splats(points, colors, cameras)
            if gs_file is not None:
                self.logger.info(f"Gaussian splats saved: {gs_file}")
            else:
                self.logger.warning("Gaussian splat creation failed or returned None.")
        else:
            self.logger.warning("No 3D points/colors available for gaussian splatting. "
                                "Skip gaussian creation.")
        
        self.logger.info("\n✓ Static scene reconstruction complete")
        self.logger.info(f"  Output: {output_file}\n")
        
        return output_file


def main():
    """Main entry point"""
    # Load configuration
    config = load_config()
    
    if not config['pass1_static']['enabled']:
        print("Pass 1 (static reconstruction) is disabled in config")
        return
    
    # Setup logger
    logger = setup_logger(
        name="StaticReconstruction",
        log_dir=config['data']['log_dir'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Run reconstruction
    reconstructor = StaticSceneReconstructor(config, logger)
    reconstructor.run()


if __name__ == "__main__":
    main()
