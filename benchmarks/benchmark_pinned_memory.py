#!/usr/bin/env python3
"""
Benchmark script for Pinned Memory optimization in E2FGVI_HQ cleaner.

Compares CPU->GPU data transfer performance with and without pinned memory.
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def benchmark_data_transfer(
    frames_np: np.ndarray,
    masks_np: np.ndarray,
    use_pinned_memory: bool,
    num_iterations: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark data transfer from CPU to GPU.
    
    Returns:
        Tuple of (total_time_seconds, avg_time_per_iteration)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare tensors
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).float()
    frames_tensor = frames_tensor / 255.0 * 2 - 1
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(1).unsqueeze(0).float()
    masks_tensor = masks_tensor / 255.0
    
    if use_pinned_memory:
        frames_tensor = frames_tensor.pin_memory()
        masks_tensor = masks_tensor.pin_memory()
    
    # Warmup
    _ = frames_tensor.to(device, non_blocking=use_pinned_memory)
    _ = masks_tensor.to(device, non_blocking=use_pinned_memory)
    torch.cuda.synchronize()
    
    # Benchmark
    total_time = 0
    for _ in range(num_iterations):
        start = time.perf_counter()
        frames_gpu = frames_tensor.to(device, non_blocking=use_pinned_memory)
        masks_gpu = masks_tensor.to(device, non_blocking=use_pinned_memory)
        torch.cuda.synchronize()  # Wait for transfer to complete
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        del frames_gpu, masks_gpu
        torch.cuda.empty_cache()
    
    avg_time = total_time / num_iterations
    return total_time, avg_time


def prepare_test_data(num_frames: int = 30, height: int = 1280, width: int = 704) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data."""
    print(f"Creating test data: {num_frames} frames at {width}x{height}...")
    
    # Create random frames
    frames_np = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
    
    # Create masks
    masks_np = np.zeros((num_frames, height, width), dtype=np.uint8)
    masks_np[:, -80:-20, -160:-20] = 255  # Watermark region
    
    data_size_mb = (frames_np.nbytes + masks_np.nbytes) / (1024 * 1024)
    print(f"Data size: {data_size_mb:.2f} MB")
    
    return frames_np, masks_np


def run_benchmark(
    num_frames: int = 30,
    height: int = 1280,
    width: int = 704,
    num_iterations: int = 10,
):
    """
    Run benchmark comparing pinned vs non-pinned memory transfer.
    """
    print(f"\n{'='*60}")
    print(f"Pinned Memory Benchmark")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires GPU.")
        return None
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Test: {num_frames} frames at {width}x{height}")
    print(f"Iterations: {num_iterations}")
    print()
    
    # Prepare test data
    frames_np, masks_np = prepare_test_data(num_frames, height, width)
    print()
    
    # Benchmark without pinned memory
    print("Testing WITHOUT pinned memory...")
    total_normal, avg_normal = benchmark_data_transfer(
        frames_np, masks_np, use_pinned_memory=False, num_iterations=num_iterations
    )
    print(f"  Total time: {total_normal:.4f}s")
    print(f"  Avg per iteration: {avg_normal*1000:.2f}ms")
    print()
    
    # Benchmark with pinned memory
    print("Testing WITH pinned memory...")
    total_pinned, avg_pinned = benchmark_data_transfer(
        frames_np, masks_np, use_pinned_memory=True, num_iterations=num_iterations
    )
    print(f"  Total time: {total_pinned:.4f}s")
    print(f"  Avg per iteration: {avg_pinned*1000:.2f}ms")
    print()
    
    # Summary
    speedup = avg_normal / avg_pinned
    improvement = (speedup - 1) * 100
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Mode':<20} {'Avg Time (ms)':<15} {'Throughput (GB/s)':<15}")
    print("-" * 50)
    
    data_size_gb = (frames_np.nbytes + masks_np.nbytes) / (1024**3)
    throughput_normal = data_size_gb / avg_normal
    throughput_pinned = data_size_gb / avg_pinned
    
    print(f"{'Normal':<20} {avg_normal*1000:<15.2f} {throughput_normal:<15.2f}")
    print(f"{'Pinned Memory':<20} {avg_pinned*1000:<15.2f} {throughput_pinned:<15.2f}")
    print("-" * 50)
    print(f"Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")
    print("=" * 60)
    
    return {
        "normal_ms": avg_normal * 1000,
        "pinned_ms": avg_pinned * 1000,
        "speedup": speedup,
        "improvement_percent": improvement,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark pinned memory performance")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames to test",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1280,
        help="Frame height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Frame width",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for averaging",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_iterations=args.iterations,
    )

