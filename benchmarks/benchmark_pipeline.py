#!/usr/bin/env python3
"""
Benchmark script for the complete SoraWM pipeline.

Compares performance before and after optimizations.
"""

import time
from pathlib import Path
from typing import Optional

from loguru import logger

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


def benchmark_pipeline(
    input_video_path: Path,
    output_video_path: Path,
    cleaner_type: CleanerType = CleanerType.E2FGVI_HQ,
    warmup: bool = True,
) -> dict:
    """
    Benchmark the complete SoraWM pipeline.
    
    Returns:
        Dict with timing information
    """
    print(f"\n{'='*60}")
    print(f"SoraWM Pipeline Benchmark")
    print(f"{'='*60}")
    print(f"Input:  {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Cleaner: {cleaner_type.value}")
    print()
    
    # Initialize SoraWM
    print("Initializing SoraWM...")
    init_start = time.perf_counter()
    sora_wm = SoraWM(cleaner_type=cleaner_type)
    init_time = time.perf_counter() - init_start
    print(f"  Initialization time: {init_time:.2f}s")
    print()
    
    # Warmup run (optional, for more stable measurements)
    if warmup:
        print("Warmup run (first run may be slower due to JIT compilation)...")
        warmup_output = output_video_path.parent / f"warmup_{output_video_path.name}"
        sora_wm.run(input_video_path, warmup_output, quiet=True)
        if warmup_output.exists():
            warmup_output.unlink()
        print("  Warmup complete")
        print()
    
    # Benchmark run
    print("Running benchmark...")
    
    progress_times = []
    last_progress = 0
    last_time = time.perf_counter()
    
    def progress_callback(progress: int):
        nonlocal last_progress, last_time
        current_time = time.perf_counter()
        if progress > last_progress:
            progress_times.append({
                "progress": progress,
                "elapsed": current_time - last_time,
            })
            last_progress = progress
            last_time = current_time
    
    start_time = time.perf_counter()
    sora_wm.run(input_video_path, output_video_path, progress_callback=progress_callback, quiet=False)
    total_time = time.perf_counter() - start_time
    
    print()
    print(f"{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    
    # Check output
    if output_video_path.exists():
        output_size = output_video_path.stat().st_size / (1024 * 1024)  # MB
        print(f"Output size: {output_size:.2f} MB")
    
    print("="*60)
    
    return {
        "init_time": init_time,
        "total_time": total_time,
        "cleaner_type": cleaner_type.value,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark SoraWM pipeline")
    parser.add_argument(
        "--video",
        type=str,
        default="resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4",
        help="Path to test video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmark_output.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--cleaner",
        type=str,
        choices=["lama", "e2fgvi_hq"],
        default="e2fgvi_hq",
        help="Cleaner type to use",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run",
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cleaner_type = CleanerType.LAMA if args.cleaner == "lama" else CleanerType.E2FGVI_HQ
    
    results = benchmark_pipeline(
        input_video_path=video_path,
        output_video_path=output_path,
        cleaner_type=cleaner_type,
        warmup=not args.no_warmup,
    )
    
    print(f"\nBenchmark results: {results}")

