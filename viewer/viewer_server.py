#!/usr/bin/env python3
"""
Interactive 4D Scene Viewer Server.

Serves a web-based 3D viewer for the reconstructed scene with:
- Static point cloud visualization
- Dynamic object boxes with animation
- Camera controls (orbit, pan, zoom)
- Time slider for 4D playback
"""

import sys
from pathlib import Path
import json
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_ply


def prepare_viewer_data(output_dir: Path):
    """Prepare data files for the web viewer."""
    
    pass1_dir = output_dir / "pass1_static"
    pass2_dir = output_dir / "pass2_dynamic"
    viewer_dir = output_dir.parent / "viewer" / "data"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    
    # Load point cloud with intelligent subsampling
    print("Loading point cloud...")
    ply_path = pass1_dir / "pi3_pointcloud_corrected.ply"
    points, colors, _ = load_ply(str(ply_path))
    
    print(f"  Loaded {len(points)} points")
    
    # Subsample for web performance (300k points is reasonable)
    max_points = 300000
    if len(points) > max_points:
        # Use voxel-based subsampling for better coverage
        voxel_size = 0.05  # 5cm voxels
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        
        if len(unique_idx) > max_points:
            # Still too many, random sample from unique
            sample_idx = np.random.choice(unique_idx, max_points, replace=False)
        else:
            sample_idx = unique_idx
        
        points = points[sample_idx]
        colors = colors[sample_idx]
        print(f"  Subsampled to {len(points)} points (voxel-based)")
    
    # Convert from Z-up to Y-up for Three.js
    # Three.js: X=right, Y=up, Z=forward
    # Our data: X=east, Y=north, Z=up
    # Swap Y and Z
    points_threejs = points.copy()
    points_threejs[:, 1] = points[:, 2]  # Y = Z (height)
    points_threejs[:, 2] = points[:, 1]  # Z = Y (north)
    points = points_threejs
    
    # Save as JSON for web
    point_data = {
        'positions': points.tolist(),
        'colors': (colors / 255.0).tolist()
    }
    
    with open(viewer_dir / "pointcloud.json", 'w') as f:
        json.dump(point_data, f)
    print(f"  Saved point cloud to {viewer_dir / 'pointcloud.json'}")
    
    # Load trajectories with STRICT filtering
    print("Loading trajectories...")
    all_tracks = []
    min_track_length = 30  # Require at least 30 frames (2 seconds at 15fps)
    scene_bounds = (-25, 25)  # Tighter bounds - intersection is ~40m
    min_movement = 2.0  # Minimum movement in meters to be considered "moving"
    
    traj_files = list(pass2_dir.glob("*_trajectories_pi3.json"))
    for traj_file in traj_files:
        with open(traj_file) as f:
            data = json.load(f)
        
        for traj in data['trajectories']:
            # Skip short tracks (likely noise)
            if traj['num_frames'] < min_track_length:
                continue
            
            # Skip stationary objects (parked cars, etc.)
            if traj['is_stationary']:
                continue
                
            track = {
                'track_id': traj['track_id'],
                'class_name': traj['class_name'],
                'category': traj['category'],
                'is_stationary': traj['is_stationary'],
                'frames': {}
            }
            
            positions = []
            for frame in traj['frames']:
                pos = frame.get('position_3d')
                if pos:
                    # Filter out-of-bounds
                    if (scene_bounds[0] <= pos[0] <= scene_bounds[1] and
                        scene_bounds[0] <= pos[1] <= scene_bounds[1]):
                        # Convert to Three.js coords (swap Y and Z)
                        pos_threejs = [pos[0], pos[2], pos[1]]
                        track['frames'][frame['frame_idx']] = pos_threejs
                        positions.append(pos[:2])  # XY for movement calc
            
            # Check if track has enough movement
            if len(positions) >= 2:
                positions = np.array(positions)
                total_movement = np.linalg.norm(positions[-1] - positions[0])
                
                # Only add if significant movement
                if total_movement >= min_movement and len(track['frames']) >= min_track_length:
                    # Compute heading for each frame based on movement direction
                    frame_keys = sorted(track['frames'].keys(), key=int)
                    for i, fk in enumerate(frame_keys):
                        if i < len(frame_keys) - 1:
                            # Use next position to compute heading
                            curr_pos = track['frames'][fk]
                            next_pos = track['frames'][frame_keys[i+1]]
                            dx = next_pos[0] - curr_pos[0]
                            dz = next_pos[2] - curr_pos[2]
                            heading = float(np.arctan2(dx, dz))  # Heading in radians
                        else:
                            # Use previous heading for last frame
                            heading = track['frames'][frame_keys[i-1]][3] if len(track['frames'][frame_keys[i-1]]) > 3 else 0
                        
                        # Add heading to position [x, y, z, heading]
                        track['frames'][fk] = track['frames'][fk] + [heading]
                    
                    all_tracks.append(track)
    
    # Deduplicate tracks by ID
    tracks_by_id = {}
    for track in all_tracks:
        tid = track['track_id']
        if tid not in tracks_by_id:
            tracks_by_id[tid] = track
        else:
            # Merge frames
            tracks_by_id[tid]['frames'].update(track['frames'])
    
    tracks_list = list(tracks_by_id.values())
    print(f"  Loaded {len(tracks_list)} unique tracks")
    
    # Get frame range
    all_frames = set()
    for track in tracks_list:
        all_frames.update(track['frames'].keys())
    
    min_frame = min(all_frames) if all_frames else 0
    max_frame = max(all_frames) if all_frames else 0
    
    trajectory_data = {
        'tracks': tracks_list,
        'min_frame': min_frame,
        'max_frame': max_frame,
        'fps': 15.0
    }
    
    with open(viewer_dir / "trajectories.json", 'w') as f:
        json.dump(trajectory_data, f)
    print(f"  Saved trajectories to {viewer_dir / 'trajectories.json'}")
    
    # Load cameras
    print("Loading cameras...")
    with open(pass1_dir / "pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    camera_data = {}
    for cam_name, cam_info in cameras.items():
        camera_data[cam_name] = {
            'position': cam_info['t'],
            'K': cam_info['K'],
            'R': cam_info['R']
        }
    
    with open(viewer_dir / "cameras.json", 'w') as f:
        json.dump(camera_data, f)
    print(f"  Saved cameras to {viewer_dir / 'cameras.json'}")
    
    return viewer_dir


def create_viewer_html(viewer_dir: Path):
    """Create the HTML viewer file."""
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>4D Scene Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white;
            overflow: hidden;
        }
        #container { width: 100vw; height: 100vh; }
        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
            min-width: 250px;
        }
        #controls h3 { margin-bottom: 10px; color: #4ecdc4; }
        #controls label { display: block; margin: 8px 0 4px; font-size: 12px; }
        #controls input[type="range"] { width: 100%; }
        #controls button {
            margin: 5px 5px 5px 0;
            padding: 8px 15px;
            background: #4ecdc4;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            color: #1a1a2e;
            font-weight: bold;
        }
        #controls button:hover { background: #45b7aa; }
        #controls .checkbox-group { margin: 10px 0; }
        #controls .checkbox-group label { display: inline; margin-left: 5px; }
        #info {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 8px;
            font-size: 12px;
        }
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
        }
        #legend h4 { margin-bottom: 10px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; margin-right: 10px; border-radius: 3px; }
        #frame-display { font-size: 18px; font-weight: bold; color: #4ecdc4; }
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="controls">
        <h3>4D Scene Viewer</h3>
        <div id="frame-display">Frame: 0 / 0</div>
        
        <label>Time: <span id="time-value">0.00s</span></label>
        <input type="range" id="time-slider" min="0" max="100" value="0">
        
        <div>
            <button id="play-btn">▶ Play</button>
            <button id="reset-btn">⟲ Reset</button>
        </div>
        
        <label>Playback Speed: <span id="speed-value">1.0x</span></label>
        <input type="range" id="speed-slider" min="0.1" max="3" step="0.1" value="1">
        
        <label>Point Size: <span id="point-size-value">2</span></label>
        <input type="range" id="point-size-slider" min="1" max="10" value="2">
        
        <div class="checkbox-group">
            <input type="checkbox" id="show-static" checked>
            <label for="show-static">Show Static Objects</label>
        </div>
        <div class="checkbox-group">
            <input type="checkbox" id="show-moving" checked>
            <label for="show-moving">Show Moving Objects</label>
        </div>
        <div class="checkbox-group">
            <input type="checkbox" id="show-pointcloud" checked>
            <label for="show-pointcloud">Show Point Cloud</label>
        </div>
    </div>
    
    <div id="legend">
        <h4>Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff4444;"></div>
            <span>Moving Vehicle</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffa500;"></div>
            <span>Stationary Vehicle</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #44ff44;"></div>
            <span>Moving Person</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #44ffff;"></div>
            <span>Stationary Person</span>
        </div>
    </div>
    
    <div id="info">
        <div>Controls: Left-click drag to rotate, Right-click drag to pan, Scroll to zoom</div>
        <div id="stats"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(30, 30, 30);
        camera.lookAt(0, 0, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(80, 40, 0x444444, 0x333333);
        scene.add(gridHelper);
        
        // Axes helper
        const axesHelper = new THREE.AxesHelper(10);
        scene.add(axesHelper);
        
        // Data containers
        let pointCloud = null;
        let trajectoryData = null;
        let boxMeshes = [];
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        
        // Box sizes
        const VEHICLE_SIZE = [4.5, 1.8, 1.5];
        const PERSON_SIZE = [0.5, 0.5, 1.7];
        
        // Colors
        const COLORS = {
            movingVehicle: 0xff4444,
            stationaryVehicle: 0xffa500,
            movingPerson: 0x44ff44,
            stationaryPerson: 0x44ffff
        };
        
        function getBoxColor(category, isStationary) {
            if (category === 'vehicle') {
                return isStationary ? COLORS.stationaryVehicle : COLORS.movingVehicle;
            }
            return isStationary ? COLORS.stationaryPerson : COLORS.movingPerson;
        }
        
        function getBoxSize(category) {
            return category === 'vehicle' ? VEHICLE_SIZE : PERSON_SIZE;
        }
        
        // Load point cloud
        async function loadPointCloud() {
            const response = await fetch('data/pointcloud.json');
            const data = await response.json();
            
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(data.positions.flat());
            const colors = new Float32Array(data.colors.flat());
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({
                size: 0.1,
                vertexColors: true,
                sizeAttenuation: true
            });
            
            pointCloud = new THREE.Points(geometry, material);
            scene.add(pointCloud);
            
            document.getElementById('stats').textContent = 
                `Points: ${data.positions.length.toLocaleString()}`;
        }
        
        // Load trajectories
        async function loadTrajectories() {
            const response = await fetch('data/trajectories.json');
            trajectoryData = await response.json();
            
            document.getElementById('time-slider').max = trajectoryData.max_frame;
            updateFrameDisplay();
            
            // Create box meshes for each track
            for (const track of trajectoryData.tracks) {
                const size = getBoxSize(track.category);
                const color = getBoxColor(track.category, track.is_stationary);
                
                const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
                const material = new THREE.MeshBasicMaterial({
                    color: color,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.8
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.visible = false;
                mesh.userData = {
                    trackId: track.track_id,
                    category: track.category,
                    isStationary: track.is_stationary,
                    frames: track.frames
                };
                
                scene.add(mesh);
                boxMeshes.push(mesh);
            }
            
            updateBoxes(0);
        }
        
        function updateBoxes(frame) {
            const showStatic = document.getElementById('show-static').checked;
            const showMoving = document.getElementById('show-moving').checked;
            
            for (const mesh of boxMeshes) {
                const frameStr = frame.toString();
                const pos = mesh.userData.frames[frameStr];
                
                if (pos) {
                    const shouldShow = mesh.userData.isStationary ? showStatic : showMoving;
                    mesh.visible = shouldShow;
                    
                    if (shouldShow) {
                        // pos is [X, Y(height), Z, heading]
                        const boxHeight = getBoxSize(mesh.userData.category)[2];
                        mesh.position.set(pos[0], pos[1] + boxHeight/2, pos[2]);
                        
                        // Apply heading rotation (around Y axis in Three.js)
                        if (pos.length > 3) {
                            mesh.rotation.y = pos[3];
                        }
                    }
                } else {
                    mesh.visible = false;
                }
            }
        }
        
        function updateFrameDisplay() {
            const maxFrame = trajectoryData ? trajectoryData.max_frame : 0;
            document.getElementById('frame-display').textContent = 
                `Frame: ${currentFrame} / ${maxFrame}`;
            document.getElementById('time-value').textContent = 
                `${(currentFrame / 15).toFixed(2)}s`;
        }
        
        // Controls
        document.getElementById('time-slider').addEventListener('input', (e) => {
            currentFrame = parseInt(e.target.value);
            updateBoxes(currentFrame);
            updateFrameDisplay();
        });
        
        document.getElementById('play-btn').addEventListener('click', () => {
            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            currentFrame = 0;
            document.getElementById('time-slider').value = 0;
            updateBoxes(0);
            updateFrameDisplay();
        });
        
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = `${playbackSpeed.toFixed(1)}x`;
        });
        
        document.getElementById('point-size-slider').addEventListener('input', (e) => {
            const size = parseFloat(e.target.value);
            document.getElementById('point-size-value').textContent = size;
            if (pointCloud) {
                pointCloud.material.size = size * 0.05;
            }
        });
        
        document.getElementById('show-pointcloud').addEventListener('change', (e) => {
            if (pointCloud) pointCloud.visible = e.target.checked;
        });
        
        document.getElementById('show-static').addEventListener('change', () => updateBoxes(currentFrame));
        document.getElementById('show-moving').addEventListener('change', () => updateBoxes(currentFrame));
        
        // Animation loop
        let lastTime = 0;
        function animate(time) {
            requestAnimationFrame(animate);
            
            if (isPlaying && trajectoryData) {
                const deltaTime = (time - lastTime) / 1000;
                if (deltaTime > 1 / (15 * playbackSpeed)) {
                    currentFrame++;
                    if (currentFrame > trajectoryData.max_frame) {
                        currentFrame = 0;
                    }
                    document.getElementById('time-slider').value = currentFrame;
                    updateBoxes(currentFrame);
                    updateFrameDisplay();
                    lastTime = time;
                }
            }
            
            controls.update();
            renderer.render(scene, camera);
        }
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Initialize
        Promise.all([loadPointCloud(), loadTrajectories()]).then(() => {
            console.log('Data loaded');
            animate(0);
        });
    </script>
</body>
</html>'''
    
    html_path = viewer_dir / "index.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created viewer at {html_path}")
    return html_path


class ViewerHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve from viewer directory."""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        path = super().translate_path(path)
        rel_path = Path(path).relative_to(Path.cwd())
        return str(self.directory / rel_path)


def start_server(viewer_dir: Path, port: int = 8080):
    """Start HTTP server for the viewer."""
    
    import functools
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(viewer_dir))
    
    server = HTTPServer(('localhost', port), handler)
    
    print(f"\n=== 4D Scene Viewer ===")
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="4D Scene Viewer")
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs"
    viewer_dir = base_dir / "viewer"
    
    # Prepare data
    prepare_viewer_data(output_dir)
    
    # Create HTML
    create_viewer_html(viewer_dir)
    
    # Start server
    if not args.no_browser:
        start_server(viewer_dir, args.port)
    else:
        print(f"Viewer ready at {viewer_dir / 'index.html'}")
        print(f"Run: python -m http.server {args.port} --directory {viewer_dir}")


if __name__ == "__main__":
    main()
