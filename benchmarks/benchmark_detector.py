#!/usr/bin/env python3
"""
Benchmark script for YOLO watermark detector.

Compares single-frame detection vs batch detection performance.
"""

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.utils.video_utils import VideoLoader


def benchmark_single_detection(
    detector: SoraWaterMarkDetector, frames: List[np.ndarray], warmup: int = 5
) -> Tuple[float, List[dict]]:
    """
    Benchmark single-frame detection (original method).
    
    Returns:
        Tuple of (total_time_seconds, detection_results)
    """
    # Warmup
    for i in range(min(warmup, len(frames))):
        detector.detect(frames[i])
    
    # Benchmark
    results = []
    start_time = time.perf_counter()
    
    for frame in frames:
        result = detector.detect(frame)
        results.append(result)
    
    total_time = time.perf_counter() - start_time
    return total_time, results


def benchmark_batch_detection(
    detector: SoraWaterMarkDetector,
    frames: List[np.ndarray],
    batch_size: int = 8,
    warmup: int = 1,
) -> Tuple[float, List[dict]]:
    """
    Benchmark batch detection (optimized method).
    
    Returns:
        Tuple of (total_time_seconds, detection_results)
    """
    # Warmup
    warmup_frames = frames[: min(batch_size * warmup, len(frames))]
    detector.detect_batch(warmup_frames, batch_size=batch_size)
    
    # Benchmark
    start_time = time.perf_counter()
    results = detector.detect_batch(frames, batch_size=batch_size)
    total_time = time.perf_counter() - start_time
    
    return total_time, results


def verify_results_match(
    single_results: List[dict], batch_results: List[dict]
) -> bool:
    """Verify that single and batch detection results match."""
    if len(single_results) != len(batch_results):
        return False
    
    for i, (single, batch) in enumerate(zip(single_results, batch_results)):
        if single["detected"] != batch["detected"]:
            logger.warning(f"Frame {i}: detected mismatch - single={single['detected']}, batch={batch['detected']}")
            return False
        if single["detected"] and batch["detected"]:
            # Allow small numerical differences in bbox
            if single["bbox"] != batch["bbox"]:
                s_bbox = single["bbox"]
                b_bbox = batch["bbox"]
                diff = max(abs(s_bbox[j] - b_bbox[j]) for j in range(4))
                if diff > 2:  # Allow up to 2 pixel difference
                    logger.warning(f"Frame {i}: bbox mismatch - single={s_bbox}, batch={b_bbox}")
                    return False
    return True


def run_benchmark(
    video_path: Path,
    max_frames: int = 100,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
):
    """
    Run benchmark comparing single vs batch detection.
    
    Args:
        video_path: Path to test video
        max_frames: Maximum number of frames to test
        batch_sizes: List of batch sizes to test
    """
    print(f"\n{'='*60}")
    print(f"YOLO Watermark Detector Benchmark")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames}")
    print()
    
    # Load video frames
    print("Loading video frames...")
    loader = VideoLoader(video_path)
    frames = []
    for i, frame in enumerate(tqdm(loader, total=min(max_frames, len(loader)), desc="Loading frames")):
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    
    num_frames = len(frames)
    print(f"Loaded {num_frames} frames")
    print(f"Frame shape: {frames[0].shape}")
    print()
    
    # Initialize detector
    print("Initializing detector...")
    detector = SoraWaterMarkDetector()
    print()
    
    # Benchmark single detection (baseline)
    print("Running single-frame detection (baseline)...")
    single_time, single_results = benchmark_single_detection(detector, frames)
    single_fps = num_frames / single_time
    print(f"  Time: {single_time:.3f}s")
    print(f"  FPS: {single_fps:.2f}")
    print(f"  Detected: {sum(1 for r in single_results if r['detected'])}/{num_frames} frames")
    print()
    
    # Benchmark batch detection with different batch sizes
    print("Running batch detection with different batch sizes...")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'FPS':<12} {'Speedup':<12} {'Match'}")
    print("-" * 60)
    
    best_speedup = 1.0
    best_batch_size = 1
    
    for batch_size in batch_sizes:
        batch_time, batch_results = benchmark_batch_detection(
            detector, frames, batch_size=batch_size
        )
        batch_fps = num_frames / batch_time
        speedup = single_time / batch_time
        match = verify_results_match(single_results, batch_results)
        
        print(f"{batch_size:<12} {batch_time:<12.3f} {batch_fps:<12.2f} {speedup:<12.2f}x {'✓' if match else '✗'}")
        
        if speedup > best_speedup and match:
            best_speedup = speedup
            best_batch_size = batch_size
    
    print("-" * 60)
    print()
    print(f"Best batch size: {best_batch_size} (speedup: {best_speedup:.2f}x)")
    print()
    
    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Original (single):     {single_fps:.2f} FPS")
    print(f"Optimized (batch={best_batch_size}):  {single_fps * best_speedup:.2f} FPS")
    print(f"Improvement:           {(best_speedup - 1) * 100:.1f}%")
    print("="*60)
    
    return {
        "single_fps": single_fps,
        "best_batch_fps": single_fps * best_speedup,
        "best_batch_size": best_batch_size,
        "best_speedup": best_speedup,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark YOLO watermark detector")
    parser.add_argument(
        "--video",
        type=str,
        default="resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4",
        help="Path to test video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of batch sizes to test",
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
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    run_benchmark(
        video_path=video_path,
        max_frames=args.max_frames,
        batch_sizes=batch_sizes,
    )

