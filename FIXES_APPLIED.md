# Fixes Applied to Prevent System Hangs

## Problem Analysis

The system was hanging at s3-left camera due to **GPU memory accumulation** from:
1. SAM2 loading full images to GPU without cleanup
2. CLIP feature extraction accumulating tensors
3. No periodic memory management
4. No error recovery on GPU OOM

## Fixes Implemented

### 1. GPU Memory Monitoring
```python
# Check GPU memory before SAM2 calls
mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
if mem_allocated > 10.0:
    # Skip SAM2, use bbox mask instead
    torch.cuda.empty_cache()
    return None
```

### 2. Periodic Cleanup
```python
# Every 10 processed frames
if processed_count % 10 == 0:
    torch.cuda.empty_cache()

# Every 50 SAM2 calls
if self.sam2_call_count % 50 == 0:
    torch.cuda.empty_cache()
```

### 3. Explicit Tensor Cleanup
```python
# In CLIP feature extraction
features_np = features.cpu().numpy()[0]
del crop_tensor, features
torch.cuda.empty_cache()
```

### 4. Error Handling
```python
try:
    results = self.detector(frame_rgb, ...)
except Exception as e:
    logger.error(f"Detection failed: {e}")
    torch.cuda.empty_cache()
    continue  # Skip frame instead of crashing
```

### 5. Progress Logging
```python
# Every 20 frames, log GPU usage
if processed_count % 20 == 0:
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"{camera_id}: Frame {frame_idx}, GPU: {mem_gb:.2f}GB")
```

### 6. Garbage Collection
```python
# After each camera
import gc
gc.collect()
torch.cuda.empty_cache()
```

## How to Run (Fixed Version)

```bash
# Activate environment
conda activate acv2

# Run with monitoring
python pass2_dynamic/create_tracking_mosaic.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

## Expected Behavior

1. **No more hangs** - System will skip SAM2 if GPU memory is high
2. **Graceful degradation** - Falls back to bbox masks if SAM2 fails
3. **Progress visibility** - Logs GPU memory every 20 frames
4. **Error recovery** - Continues processing even if individual frames fail
5. **Memory cleanup** - Periodic GPU cache clearing prevents accumulation

## Performance Impact

- **Slightly slower** due to periodic cleanup (every 10 frames)
- **More robust** - Won't crash or hang
- **Adaptive** - Skips SAM2 when memory is tight
- **Logged** - Can see exactly where issues occur

## If It Still Hangs

1. **Reduce sample_rate** in config (e.g., from 5 to 10)
2. **Disable SAM2** temporarily:
   ```python
   self.sam2 = None  # In _init_segmentor()
   ```
3. **Disable CLIP**:
   ```python
   self.use_clip = False  # In _init_clip()
   ```
4. **Monitor with**:
   ```bash
   nvidia-smi dmon -s u -d 1
   ```

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| SAM2 calls | No cleanup | Cleanup every 50 calls |
| CLIP calls | Tensors accumulate | Explicit deletion + cleanup |
| Frame processing | No cleanup | Cleanup every 10 frames |
| Error handling | Crash on error | Skip frame, continue |
| GPU monitoring | None | Log every 20 frames |
| Memory threshold | None | Skip SAM2 if >10GB used |

## Testing Checklist

- [x] Added GPU memory monitoring
- [x] Added periodic cleanup (every 10 frames)
- [x] Added SAM2 call counting and cleanup
- [x] Added explicit tensor deletion in CLIP
- [x] Added error handling for detection
- [x] Added error handling for segmentation
- [x] Added error handling for feature extraction
- [x] Added progress logging
- [x] Added garbage collection
- [x] Added memory threshold checks

All fixes are **backward compatible** and will gracefully degrade if models fail.
