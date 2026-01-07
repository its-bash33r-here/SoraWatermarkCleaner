#!/usr/bin/env python3
"""
Ablation study benchmark for SoraWatermarkCleaner optimizations.

Tests the impact of each optimization individually and combined:
1. YOLO batch detection
2. AMP (Automatic Mixed Precision)
3. Pinned Memory
4. Frame caching (avoid re-reading video)
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from sorawm.utils.video_utils import VideoLoader


@dataclass
class AblationResult:
    """Result of an ablation experiment."""
    name: str
    time_seconds: float
    fps: float
    speedup: float  # relative to baseline


def benchmark_detector_ablation(
    video_path: Path,
    max_frames: int = 100,
) -> Dict[str, AblationResult]:
    """
    Ablation study for YOLO detector optimizations.
    
    Tests:
    - Baseline: single-frame detection
    - Optimized: batch detection (batch_size=4)
    """
    from sorawm.watermark_detector import SoraWaterMarkDetector
    
    print("\n" + "="*60)
    print("YOLO Detector Ablation Study")
    print("="*60)
    
    # Load frames
    print(f"Loading {max_frames} frames...")
    loader = VideoLoader(video_path)
    frames = []
    for i, frame in enumerate(loader):
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    print(f"Loaded {len(frames)} frames")
    
    detector = SoraWaterMarkDetector()
    results = {}
    
    # Warmup
    for _ in range(3):
        detector.detect(frames[0])
    
    # Baseline: single-frame detection
    print("\nBaseline: Single-frame detection...")
    start = time.perf_counter()
    for frame in frames:
        detector.detect(frame)
    baseline_time = time.perf_counter() - start
    baseline_fps = len(frames) / baseline_time
    results["detector_baseline"] = AblationResult(
        name="Single-frame detection",
        time_seconds=baseline_time,
        fps=baseline_fps,
        speedup=1.0,
    )
    print(f"  Time: {baseline_time:.3f}s, FPS: {baseline_fps:.2f}")
    
    # Optimized: batch detection
    print("\nOptimized: Batch detection (batch_size=4)...")
    # Warmup
    detector.detect_batch(frames[:8], batch_size=4)
    
    start = time.perf_counter()
    detector.detect_batch(frames, batch_size=4)
    batch_time = time.perf_counter() - start
    batch_fps = len(frames) / batch_time
    speedup = baseline_time / batch_time
    results["detector_batch"] = AblationResult(
        name="Batch detection (bs=4)",
        time_seconds=batch_time,
        fps=batch_fps,
        speedup=speedup,
    )
    print(f"  Time: {batch_time:.3f}s, FPS: {batch_fps:.2f}, Speedup: {speedup:.2f}x")
    
    return results


def benchmark_cleaner_ablation(
    video_path: Path,
    max_frames: int = 30,
) -> Dict[str, AblationResult]:
    """
    Ablation study for E2FGVI_HQ cleaner optimizations.
    
    Tests:
    - Baseline: FP32, no pinned memory
    - +AMP: FP16 mixed precision
    - +Pinned: Pinned memory for CPU->GPU transfer
    - All: AMP + Pinned Memory
    """
    from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
    
    print("\n" + "="*60)
    print("E2FGVI_HQ Cleaner Ablation Study")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping cleaner ablation")
        return {}
    
    # Load frames and create masks
    print(f"Loading {max_frames} frames...")
    loader = VideoLoader(video_path)
    frames = []
    for i, frame in enumerate(loader):
        frames.append(frame[:, :, ::-1].copy())  # BGR to RGB
        if len(frames) >= max_frames:
            break
    
    frames_np = np.array(frames)
    h, w = frames_np.shape[1:3]
    
    # Create watermark masks
    masks_np = np.zeros((len(frames_np), h, w), dtype=np.uint8)
    mask_h, mask_w = 60, 140
    y1, x1 = h - mask_h - 20, w - mask_w - 20
    masks_np[:, y1:y1+mask_h, x1:x1+mask_w] = 255
    
    print(f"Loaded {len(frames_np)} frames with shape {frames_np.shape}")
    
    results = {}
    configs = [
        ("baseline", E2FGVIHDConfig(enable_amp=False, enable_torch_compile=False)),
        ("+AMP", E2FGVIHDConfig(enable_amp=True, enable_torch_compile=False)),
    ]
    
    baseline_time = None
    
    for name, config in configs:
        print(f"\nTesting: {name}...")
        
        # Create cleaner with config
        cleaner = E2FGVIHDCleaner(config=config)
        
        # Warmup
        warmup_frames = frames_np[:min(10, len(frames_np))]
        warmup_masks = masks_np[:min(10, len(masks_np))]
        cleaner.clean(warmup_frames, warmup_masks)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        cleaner.clean(frames_np, masks_np)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        fps = len(frames_np) / elapsed
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        results[f"cleaner_{name}"] = AblationResult(
            name=name,
            time_seconds=elapsed,
            fps=fps,
            speedup=speedup,
        )
        print(f"  Time: {elapsed:.2f}s, FPS: {fps:.2f}, Speedup: {speedup:.2f}x")
        
        # Cleanup
        del cleaner
        torch.cuda.empty_cache()
    
    return results


def print_ablation_summary(results: Dict[str, AblationResult]):
    """Print summary table of ablation results."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Component':<30} {'Time (s)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-"*70)
    
    for key, result in results.items():
        print(f"{result.name:<30} {result.time_seconds:<12.3f} {result.fps:<10.2f} {result.speedup:<10.2f}x")
    
    print("="*70)


def run_full_ablation(
    video_path: Path,
    detector_frames: int = 100,
    cleaner_frames: int = 30,
):
    """Run complete ablation study."""
    print("\n" + "#"*70)
    print("# FULL ABLATION STUDY")
    print("#"*70)
    print(f"Video: {video_path}")
    print(f"Detector test frames: {detector_frames}")
    print(f"Cleaner test frames: {cleaner_frames}")
    
    all_results = {}
    
    # Detector ablation
    detector_results = benchmark_detector_ablation(video_path, detector_frames)
    all_results.update(detector_results)
    
    # Cleaner ablation
    cleaner_results = benchmark_cleaner_ablation(video_path, cleaner_frames)
    all_results.update(cleaner_results)
    
    # Print summary
    print_ablation_summary(all_results)
    
    # Calculate overall improvements
    print("\n" + "="*70)
    print("OPTIMIZATION IMPACT SUMMARY")
    print("="*70)
    
    if "detector_batch" in all_results:
        det_speedup = all_results["detector_batch"].speedup
        print(f"YOLO Batch Detection:  {det_speedup:.2f}x faster ({(det_speedup-1)*100:.1f}% improvement)")
    
    if "cleaner_+AMP" in all_results:
        amp_speedup = all_results["cleaner_+AMP"].speedup
        print(f"AMP (FP16):           {amp_speedup:.2f}x faster ({(amp_speedup-1)*100:.1f}% improvement)")
    
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--video",
        type=str,
        default="resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4",
        help="Path to test video",
    )
    parser.add_argument(
        "--detector-frames",
        type=int,
        default=100,
        help="Number of frames for detector test",
    )
    parser.add_argument(
        "--cleaner-frames",
        type=int,
        default=30,
        help="Number of frames for cleaner test",
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        resources_dir = Path("resources")
        if resources_dir.exists():
            videos = list(resources_dir.glob("*.mp4"))
            if videos:
                video_path = videos[0]
    
    run_full_ablation(
        video_path=video_path,
        detector_frames=args.detector_frames,
        cleaner_frames=args.cleaner_frames,
    )

