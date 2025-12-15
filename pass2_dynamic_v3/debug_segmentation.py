#!/usr/bin/env python3
"""
Debug Road Segmentation - Create 2D Sample Overlays
====================================================
Show road vs non-road detection on actual camera images
to help debug the segmentation model.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try different models
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class MultiModelSegmenter:
    """Try multiple segmentation approaches."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        print("Loading segmentation models...")
        
        # Model 1: DeepLabV3 (COCO)
        weights1 = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model1 = deeplabv3_resnet50(weights=weights1).to(device).eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("  Models loaded!")
    
    def segment_deeplabv3(self, img_rgb: np.ndarray) -> np.ndarray:
        """DeepLabV3 - returns class IDs per pixel."""
        h, w = img_rgb.shape[:2]
        input_tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model1(input_tensor)['out']
        
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        return output.argmax(1)[0].cpu().numpy()
    
    def create_road_mask_simple(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Simple color-based road detection for asphalt.
        Roads are typically:
        - Gray/dark gray
        - Low saturation
        - Lower part of image
        """
        h, w = img_rgb.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Road is gray - low saturation
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Gray mask: low saturation, moderate value
        gray_mask = (saturation < 50) & (value > 40) & (value < 200)
        
        # Weight by vertical position (lower = more likely road)
        y_weights = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        weighted = gray_mask.astype(float) * y_weights
        
        return weighted > 0.5
    
    def create_overlay(self, img_rgb: np.ndarray, mask: np.ndarray, 
                       color_road=(100, 100, 100), color_curb=(34, 139, 34)) -> np.ndarray:
        """Create overlay visualization."""
        overlay = img_rgb.copy()
        
        # Road = gray overlay
        road_overlay = np.zeros_like(overlay)
        road_overlay[mask] = color_road
        
        # Curb = green overlay
        curb_overlay = np.zeros_like(overlay)
        curb_overlay[~mask] = color_curb
        
        # Blend
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1-alpha, road_overlay, alpha, 0)
        overlay = cv2.addWeighted(overlay, 1-alpha/2, curb_overlay, alpha/2, 0)
        
        return overlay

def generate_samples():
    base_dir = Path(__file__).parent.parent
    
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    video_dir = base_dir / "StreetAware-sample"
    out_dir = base_dir / "outputs" / "pass2_dynamic_v3" / "segmentation_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    segmenter = MultiModelSegmenter()
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right']
    
    print("\n=== GENERATING SEGMENTATION SAMPLES ===\n")
    
    for cam_id in cam_ids:
        print(f"{cam_id}:")
        
        # Static background
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if bg_path.exists():
            print(f"  Background image...")
            bg_img = cv2.imread(str(bg_path))
            bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
            # DeepLabV3
            seg = segmenter.segment_deeplabv3(bg_rgb)
            bg_road_dl = (seg == 0)  # Background class
            
            # Simple gray detection
            road_simple = segmenter.create_road_mask_simple(bg_rgb)
            
            # Create overlays
            overlay_dl = segmenter.create_overlay(bg_rgb, bg_road_dl)
            overlay_simple = segmenter.create_overlay(bg_rgb, road_simple)
            
            # Save
            cv2.imwrite(str(out_dir / f"{cam_id}_bg_original.png"), bg_img)
            cv2.imwrite(str(out_dir / f"{cam_id}_bg_deeplabv3.png"), 
                       cv2.cvtColor(overlay_dl, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{cam_id}_bg_simple.png"),
                       cv2.cvtColor(overlay_simple, cv2.COLOR_RGB2BGR))
            
            print(f"    DeepLabV3 road: {bg_road_dl.sum()/bg_road_dl.size*100:.1f}%")
            print(f"    Simple road: {road_simple.sum()/road_simple.size*100:.1f}%")
        
        # Video frame
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            print(f"  Video frame 100...")
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # DeepLabV3
                seg = segmenter.segment_deeplabv3(frame_rgb)
                road_dl = (seg == 0)
                
                # Simple
                road_simple = segmenter.create_road_mask_simple(frame_rgb)
                
                # Overlays
                overlay_dl = segmenter.create_overlay(frame_rgb, road_dl)
                overlay_simple = segmenter.create_overlay(frame_rgb, road_simple)
                
                cv2.imwrite(str(out_dir / f"{cam_id}_frame100_original.png"), frame)
                cv2.imwrite(str(out_dir / f"{cam_id}_frame100_deeplabv3.png"),
                           cv2.cvtColor(overlay_dl, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{cam_id}_frame100_simple.png"),
                           cv2.cvtColor(overlay_simple, cv2.COLOR_RGB2BGR))
                
                print(f"    DeepLabV3 road: {road_dl.sum()/road_dl.size*100:.1f}%")
                print(f"    Simple road: {road_simple.sum()/road_simple.size*100:.1f}%")
    
    print(f"\nSamples saved to: {out_dir}")
    print("\nFiles created:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")

if __name__ == "__main__":
    generate_samples()
