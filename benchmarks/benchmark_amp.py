#!/usr/bin/env python3
"""
Benchmark script for E2FGVI_HQ cleaner with and without AMP (Automatic Mixed Precision).

Compares inference performance with fp32 vs fp16 (AMP).
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.utils.video_utils import VideoLoader


def prepare_test_data(
    video_path: Path, max_frames: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Load video frames and create dummy masks for testing."""
    print(f"Loading video frames from {video_path}...")
    loader = VideoLoader(video_path)
    
    frames = []
    for i, frame in enumerate(tqdm(loader, total=min(max_frames, len(loader)), desc="Loading frames")):
        # Convert BGR to RGB
        frames.append(frame[:, :, ::-1].copy())
        if len(frames) >= max_frames:
            break
    
    frames_np = np.array(frames)
    
    # Create dummy watermark masks (small rectangle in corner)
    h, w = frames_np.shape[1:3]
    masks_np = np.zeros((len(frames_np), h, w), dtype=np.uint8)
    
    # Simulate watermark region (e.g., bottom-right corner)
    mask_h, mask_w = 60, 140
    y1, x1 = h - mask_h - 20, w - mask_w - 20
    masks_np[:, y1:y1+mask_h, x1:x1+mask_w] = 255
    
    print(f"Loaded {len(frames_np)} frames with shape {frames_np.shape}")
    print(f"Mask region: ({x1}, {y1}) to ({x1+mask_w}, {y1+mask_h})")
    
    return frames_np, masks_np


def benchmark_cleaner(
    cleaner: E2FGVIHDCleaner,
    frames: np.ndarray,
    masks: np.ndarray,
    warmup: int = 1,
    label: str = "Cleaner",
) -> Tuple[float, int]:
    """
    Benchmark the cleaner.
    
    Returns:
        Tuple of (total_time_seconds, num_frames_processed)
    """
    # Warmup
    if warmup > 0:
        print(f"  Warmup ({warmup} run)...")
        warmup_frames = frames[:min(10, len(frames))]
        warmup_masks = masks[:min(10, len(masks))]
        cleaner.clean(warmup_frames, warmup_masks)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    print(f"  Running benchmark...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    result = cleaner.clean(frames, masks)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start_time
    
    return total_time, len(result)


def run_benchmark(
    video_path: Path,
    max_frames: int = 50,
):
    """
    Run benchmark comparing FP32 vs AMP (FP16) inference.
    """
    print(f"\n{'='*60}")
    print(f"E2FGVI_HQ AMP Benchmark")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames}")
    print()
    
    # Prepare test data
    frames, masks = prepare_test_data(video_path, max_frames)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. AMP benchmark requires GPU.")
        return None
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    results = {}
    
    # Benchmark without AMP (FP32)
    print("=" * 60)
    print("Testing WITHOUT AMP (FP32)...")
    print("=" * 60)
    config_fp32 = E2FGVIHDConfig(enable_amp=False, enable_torch_compile=False)
    cleaner_fp32 = E2FGVIHDCleaner(config=config_fp32)
    
    time_fp32, num_frames = benchmark_cleaner(
        cleaner_fp32, frames, masks, warmup=1, label="FP32"
    )
    fps_fp32 = num_frames / time_fp32
    print(f"  Time: {time_fp32:.2f}s")
    print(f"  FPS: {fps_fp32:.2f}")
    results["fp32"] = {"time": time_fp32, "fps": fps_fp32}
    
    # Clean up
    del cleaner_fp32
    torch.cuda.empty_cache()
    print()
    
    # Benchmark with AMP (FP16)
    print("=" * 60)
    print("Testing WITH AMP (FP16)...")
    print("=" * 60)
    config_amp = E2FGVIHDConfig(enable_amp=True, enable_torch_compile=False)
    cleaner_amp = E2FGVIHDCleaner(config=config_amp)
    
    time_amp, num_frames = benchmark_cleaner(
        cleaner_amp, frames, masks, warmup=1, label="AMP"
    )
    fps_amp = num_frames / time_amp
    print(f"  Time: {time_amp:.2f}s")
    print(f"  FPS: {fps_amp:.2f}")
    results["amp"] = {"time": time_amp, "fps": fps_amp}
    
    # Clean up
    del cleaner_amp
    torch.cuda.empty_cache()
    print()
    
    # Summary
    speedup = time_fp32 / time_amp
    improvement = (speedup - 1) * 100
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Mode':<15} {'Time (s)':<12} {'FPS':<12}")
    print("-" * 40)
    print(f"{'FP32':<15} {time_fp32:<12.2f} {fps_fp32:<12.2f}")
    print(f"{'AMP (FP16)':<15} {time_amp:<12.2f} {fps_amp:<12.2f}")
    print("-" * 40)
    print(f"Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")
    print("=" * 60)
    
    return {
        "fp32_fps": fps_fp32,
        "amp_fps": fps_amp,
        "speedup": speedup,
        "improvement_percent": improvement,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark E2FGVI_HQ AMP performance")
    parser.add_argument(
        "--video",
        type=str,
        default="resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4",
        help="Path to test video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames to test",
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        # Try to find a video in resources
        resources_dir = Path("resources")
        if resources_dir.exists():
            videos = list(resources_dir.glob("*.mp4"))
            if videos:
                video_path = videos[0]
                print(f"Using video: {video_path}")
            else:
                print(f"Error: No video found in {resources_dir}")
                exit(1)
        else:
            print(f"Error: Video not found: {video_path}")
            exit(1)
    
    run_benchmark(
        video_path=video_path,
        max_frames=args.max_frames,
    )

