#!/usr/bin/env python3
"""
Clean Pipeline Runner for 4D Reconstruction.

This script runs the complete pipeline:
1. Pass 1: Static scene reconstruction with Pi3
2. Fix orientation and prepare for Gaussian Splatting
3. Pass 2: Dynamic object tracking
4. Reproject trajectories to 3D
5. Launch interactive viewer

Usage:
    python run_pipeline.py [--skip-pass1] [--skip-pass2] [--viewer-only]
"""

import subprocess
import sys
from pathlib import Path
import argparse
import shutil


def run_command(cmd: list, cwd: str, desc: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=cwd)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {desc}")
        return False
    
    print(f"\n✓ Completed: {desc}")
    return True


def clean_outputs(output_dir: Path):
    """Clean old output files."""
    print("\nCleaning old outputs...")
    
    # Keep only essential files
    pass1_keep = ['pi3_pointcloud.ply', 'pi3_cameras.json']
    pass2_keep = []  # Will regenerate everything
    
    pass1_dir = output_dir / "pass1_static"
    pass2_dir = output_dir / "pass2_dynamic"
    
    # Clean pass1 derived files
    for f in pass1_dir.glob("*"):
        if f.name not in pass1_keep and f.is_file():
            if 'corrected' in f.name or 'transform' in f.name or 'colmap' in f.name:
                f.unlink()
                print(f"  Removed {f.name}")
    
    # Clean colmap_sparse directory
    colmap_dir = pass1_dir / "colmap_sparse"
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
        print("  Removed colmap_sparse/")
    
    # Clean pass2 derived files (keep original trajectories and videos)
    for f in pass2_dir.glob("*_pi3.json"):
        f.unlink()
        print(f"  Removed {f.name}")
    
    for f in pass2_dir.glob("4d_*.mp4"):
        f.unlink()
        print(f"  Removed {f.name}")


def main():
    parser = argparse.ArgumentParser(description="4D Reconstruction Pipeline")
    parser.add_argument('--skip-pass1', action='store_true', 
                        help='Skip Pi3 reconstruction (use existing)')
    parser.add_argument('--skip-pass2', action='store_true',
                        help='Skip dynamic tracking (use existing)')
    parser.add_argument('--viewer-only', action='store_true',
                        help='Only launch the viewer')
    parser.add_argument('--native-viewer', action='store_true',
                        help='Use native Open3D viewer instead of web viewer')
    parser.add_argument('--clean', action='store_true',
                        help='Clean derived outputs before running')
    parser.add_argument('--render-video', action='store_true',
                        help='Also render MP4 video')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs"
    
    # Use current python executable
    python_exe = sys.executable
    
    if args.clean:
        clean_outputs(output_dir)
    
    if args.viewer_only:
        # Just launch viewer
        if args.native_viewer:
            cmd = [python_exe, 'viewer/native_viewer.py']
        else:
            cmd = [python_exe, 'viewer/viewer_server.py']
        subprocess.run(cmd, cwd=str(base_dir))
        return
    
    success = True
    
    # Step 1: Pi3 Static Reconstruction
    if not args.skip_pass1:
        pi3_ply = output_dir / "pass1_static" / "pi3_pointcloud.ply"
        if not pi3_ply.exists():
            cmd = [python_exe, 'pass1_static/test_pi3.py']
            success = subprocess.run(cmd, cwd=str(base_dir)).returncode == 0
            if not success:
                print("❌ Pi3 reconstruction failed")
                return
    
    # Step 2: Fix Orientation
    cmd = [python_exe, 'pass1_static/fix_pi3_orientation.py']
    success = subprocess.run(cmd, cwd=str(base_dir)).returncode == 0
    if not success:
        print("❌ Orientation fix failed")
        return
    
    # Step 3: Dynamic Tracking (if not skipping)
    if not args.skip_pass2:
        # Check if trajectories exist
        traj_files = list((output_dir / "pass2_dynamic").glob("*_trajectories.json"))
        if not traj_files:
            print("\n⚠️  No trajectory files found. Run dynamic tracking first:")
            print("    python pass2_dynamic/single_video_motion.py --camera <cam_id>")
            print("    for each camera: s1-left, s1-right, s2-left, s2-right, etc.")
    
    # Step 4: Reproject Trajectories
    cmd = [python_exe, 'pass2_dynamic/reproject_trajectories.py']
    success = subprocess.run(cmd, cwd=str(base_dir)).returncode == 0
    if not success:
        print("❌ Trajectory reprojection failed")
        return
    
    # Step 5: Render Video (optional)
    if args.render_video:
        cmd = [python_exe, 'pass2_dynamic/render_4d_boxes.py', '--camera', 's1-right']
        subprocess.run(cmd, cwd=str(base_dir))
    
    # Step 6: Launch Viewer
    print("\n" + "="*60)
    print("  Pipeline Complete! Launching Viewer...")
    print("="*60 + "\n")
    
    if args.native_viewer:
        cmd = [python_exe, 'viewer/native_viewer.py']
    else:
        cmd = [python_exe, 'viewer/viewer_server.py']
    subprocess.run(cmd, cwd=str(base_dir))


if __name__ == "__main__":
    main()
