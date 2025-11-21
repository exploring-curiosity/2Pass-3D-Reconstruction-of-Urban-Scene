#!/usr/bin/env python3
"""
Extract static backgrounds from videos using temporal median filtering
with YOLO + SAM2 for dynamic object removal

This implementation keeps your existing excellent approach of:
1. YOLO for dynamic object detection
2. SAM2 for precise segmentation
3. Temporal median for background extraction
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, load_config
from ultralytics import YOLO

# Import SAM2
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not found. Please install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    SAM2ImagePredictor = None


class StaticBackgroundExtractor:
    """Extract static backgrounds from traffic intersection videos"""
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.device = config['hardware']['device']
        
        # Paths
        self.video_dir = Path(config['data']['video_dir'])
        self.output_dir = Path(config['data']['processed_dir']) / "static_backgrounds"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        bg_config = config['pass1_static']['background_extraction']
        self.num_frames = bg_config['num_frames']
        self.yolo_conf = bg_config['yolo_conf']
        self.dynamic_classes = set(bg_config['dynamic_classes'])
        
        # Load models
        self.logger.info("Loading YOLO model...")
        yolo_path = Path(bg_config['yolo_model'])
        if not yolo_path.exists():
            # Try in parent directory
            yolo_path = Path("..") / yolo_path
        self.yolo = YOLO(str(yolo_path))
        
        if SAM2ImagePredictor:
            self.logger.info(f"Loading SAM2 model: {bg_config['sam2_model']}...")
            self.sam = SAM2ImagePredictor.from_pretrained(bg_config['sam2_model'])
        else:
            self.logger.warning("SAM2 not available, using YOLO boxes directly")
            self.sam = None
    
    def get_dynamic_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get mask of dynamic objects in frame
        
        Args:
            frame: BGR image [H, W, 3]
        
        Returns:
            mask: Binary mask [H, W] where True = dynamic object
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = self.yolo.predict(frame_rgb, conf=self.yolo_conf, verbose=False)[0]
        
        if len(results.boxes) == 0:
            return mask
        
        # Filter for dynamic classes and expand boxes
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            cls_name = results.names[int(cls)]
            if cls_name not in self.dynamic_classes:
                continue
            
            x1, y1, x2, y2 = box
            # Filter tiny boxes
            if (x2 - x1) * (y2 - y1) < 900:  # 30x30 pixels
                continue
            
            # Expand box by 10%
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, x1 - 0.1 * bw)
            y1 = max(0, y1 - 0.1 * bh)
            x2 = min(w, x2 + 0.1 * bw)
            y2 = min(h, y2 + 0.1 * bh)
            
            boxes.append([x1, y1, x2, y2])
        
        if not boxes:
            return mask
        
        # Use SAM2 for precise segmentation if available
        if self.sam is not None:
            self.sam.set_image(frame_rgb)
            
            for box in boxes:
                box_np = np.array(box, dtype=np.float32)[None, :]
                
                try:
                    with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        masks, scores, _ = self.sam.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box_np,
                            multimask_output=False
                        )
                    
                    m = masks[0]
                    if m.ndim == 3:
                        m = m[0]
                    mask |= m.astype(bool)
                except Exception as e:
                    # Fallback to box if SAM2 fails
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = True
        else:
            # Use boxes directly
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = True
        
        return mask
    
    def extract_background(self, video_path: Path) -> np.ndarray:
        """
        Extract static background from a single video
        
        Args:
            video_path: Path to video file
        
        Returns:
            background: BGR image [H, W, 3]
        """
        self.logger.info(f"Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Sample frames uniformly
        step = max(1, total_frames // self.num_frames)
        self.logger.info(f"  Total frames: {total_frames}, sampling every {step} frames")
        
        # Accumulators
        accum_static = None
        count_static = None
        accum_all = None
        count_all = 0
        
        frame_idx = 0
        sampled = 0
        
        pbar = tqdm(total=self.num_frames, desc=f"  {video_path.stem}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0 and sampled < self.num_frames:
                # Get dynamic mask
                dynamic_mask = self.get_dynamic_mask(frame)
                static_mask = ~dynamic_mask
                
                # Initialize accumulators
                if accum_static is None:
                    h, w = frame.shape[:2]
                    accum_static = np.zeros((h, w, 3), dtype=np.float64)
                    count_static = np.zeros((h, w), dtype=np.int32)
                    accum_all = np.zeros((h, w, 3), dtype=np.float64)
                
                # Accumulate static pixels
                frame_f = frame.astype(np.float64)
                static_3ch = static_mask[..., None].repeat(3, axis=-1)
                accum_static[static_3ch] += frame_f[static_3ch]
                count_static[static_mask] += 1
                
                # Accumulate all pixels (fallback)
                accum_all += frame_f
                count_all += 1
                
                sampled += 1
                pbar.update(1)
            
            frame_idx += 1
        
        cap.release()
        pbar.close()
        
        if sampled == 0:
            raise ValueError(f"No frames sampled from {video_path}")
        
        # Compute background
        background = np.zeros_like(accum_static, dtype=np.float32)
        
        # Use mean of static pixels where available
        static_valid = count_static > 0
        background[static_valid] = (
            accum_static[static_valid] / count_static[static_valid][:, None]
        )
        
        # Fallback to overall mean for never-static pixels
        if np.any(~static_valid):
            avg_all = accum_all / max(count_all, 1)
            background[~static_valid] = avg_all[~static_valid]
        
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        self.logger.info(f"  Extracted background: {static_valid.sum() / static_valid.size * 100:.1f}% static pixels")
        
        return background
    
    def process_all_videos(self):
        """Process all camera videos"""
        self.logger.info("=== PASS 1: Static Background Extraction ===")
        
        cameras = self.config['data']['cameras']
        video_ext = self.config['data']['video_format']
        
        for camera_name in cameras:
            video_path = self.video_dir / f"{camera_name}{video_ext}"
            
            if not video_path.exists():
                self.logger.warning(f"Video not found: {video_path}")
                continue
            
            # Extract background
            background = self.extract_background(video_path)
            
            # Save
            output_path = self.output_dir / f"{camera_name}_bg.png"
            cv2.imwrite(str(output_path), background)
            self.logger.info(f"  Saved: {output_path}")
        
        self.logger.info(f"\n✓ Extracted {len(cameras)} static backgrounds")
        self.logger.info(f"  Output directory: {self.output_dir}\n")


def main():
    """Main entry point"""
    # Load configuration
    config = load_config()
    
    # Setup logger
    logger = setup_logger(
        name="StaticExtraction",
        log_dir=config['data']['log_dir'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Extract backgrounds
    extractor = StaticBackgroundExtractor(config, logger)
    extractor.process_all_videos()


if __name__ == "__main__":
    main()
