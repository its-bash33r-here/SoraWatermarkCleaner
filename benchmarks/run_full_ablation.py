#!/usr/bin/env python3
"""
Complete Ablation Study Runner

Runs all ablation experiments and generates a comprehensive report.
Results are saved to JSON and a Markdown report is generated.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    time_seconds: float
    fps: float
    frames: int
    speedup: float = 1.0
    baseline_name: Optional[str] = None


@dataclass 
class AblationResults:
    """All ablation study results."""
    timestamp: str
    gpu_name: str
    video_path: str
    video_info: Dict
    
    # Detector results
    detector_single_frame: BenchmarkResult
    detector_batch: BenchmarkResult
    
    # Cleaner results
    cleaner_baseline: BenchmarkResult
    cleaner_amp: BenchmarkResult
    
    # Pinned memory results
    pinned_memory_normal: BenchmarkResult
    pinned_memory_pinned: BenchmarkResult
    
    # End-to-end results
    e2e_optimized: BenchmarkResult


def get_gpu_name() -> str:
    """Get GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def benchmark_detector(video_path: str, num_frames: int = 100) -> tuple:
    """Benchmark YOLO detector with single-frame vs batch detection."""
    print("\n" + "="*60)
    print("YOLO Detector Benchmark")
    print("="*60)
    
    # Load frames
    loader = VideoLoader(video_path)
    frames = []
    for i, frame in enumerate(tqdm(loader, desc="Loading frames", total=num_frames)):
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    
    print(f"Loaded {len(frames)} frames")
    
    # Initialize detector
    detector = SoraWaterMarkDetector()
    
    # Warmup
    _ = detector.detect(frames[0])
    
    # Single-frame detection
    print("\nSingle-frame detection...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for frame in frames:
        _ = detector.detect(frame)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    single_time = time.perf_counter() - start
    single_fps = len(frames) / single_time
    print(f"  Time: {single_time:.3f}s, FPS: {single_fps:.2f}")
    
    # Batch detection (batch_size=4)
    print("\nBatch detection (batch_size=4)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    _ = detector.detect_batch(frames, batch_size=4)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    batch_time = time.perf_counter() - start
    batch_fps = len(frames) / batch_time
    speedup = single_time / batch_time
    print(f"  Time: {batch_time:.3f}s, FPS: {batch_fps:.2f}, Speedup: {speedup:.2f}x")
    
    single_result = BenchmarkResult(
        name="Single-frame Detection",
        time_seconds=single_time,
        fps=single_fps,
        frames=len(frames),
        speedup=1.0,
        baseline_name=None
    )
    
    batch_result = BenchmarkResult(
        name="Batch Detection (bs=4)",
        time_seconds=batch_time,
        fps=batch_fps,
        frames=len(frames),
        speedup=speedup,
        baseline_name="Single-frame Detection"
    )
    
    return single_result, batch_result


def benchmark_cleaner(video_path: str, num_frames: int = 30) -> tuple:
    """Benchmark E2FGVI_HQ cleaner with/without AMP."""
    print("\n" + "="*60)
    print("E2FGVI_HQ Cleaner Benchmark")
    print("="*60)
    
    # Load frames
    loader = VideoLoader(video_path)
    frames = []
    for i, frame in enumerate(tqdm(loader, desc="Loading frames", total=num_frames)):
        frames.append(frame[:, :, ::-1].copy())  # BGR to RGB
        if len(frames) >= num_frames:
            break
    
    frames_np = np.array(frames)
    h, w = frames_np.shape[1:3]
    print(f"Loaded {len(frames_np)} frames with shape {frames_np.shape}")
    
    # Create mask (bottom-right corner)
    masks_np = np.zeros((len(frames_np), h, w), dtype=np.uint8)
    masks_np[:, -80:-20, -160:-20] = 255
    
    results = []
    
    # Test configurations (skip torch.compile in component tests, only test in e2e)
    configs = [
        ("Baseline (FP32)", E2FGVIHDConfig(enable_amp=False, enable_torch_compile=False)),
        ("+AMP (FP16)", E2FGVIHDConfig(enable_amp=True, enable_torch_compile=False)),
    ]
    
    baseline_time = None
    
    for name, config in configs:
        print(f"\nTesting: {name}...")
        
        # Create cleaner
        cleaner = E2FGVIHDCleaner(config=config)
        
        # Warmup
        warmup_frames = frames_np[:10]
        warmup_masks = masks_np[:10]
        _ = cleaner.clean(warmup_frames, warmup_masks)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cleaner.clean(frames_np, masks_np)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        fps = len(frames_np) / elapsed
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        print(f"  Time: {elapsed:.3f}s, FPS: {fps:.2f}, Speedup: {speedup:.2f}x")
        
        results.append(BenchmarkResult(
            name=name,
            time_seconds=elapsed,
            fps=fps,
            frames=len(frames_np),
            speedup=speedup,
            baseline_name="Baseline (FP32)" if name != "Baseline (FP32)" else None
        ))
        
        # Clean up
        del cleaner
        torch.cuda.empty_cache()
    
    return results[0], results[1]


def benchmark_pinned_memory(num_frames: int = 30, width: int = 704, height: int = 1280) -> tuple:
    """Benchmark pinned memory transfer."""
    print("\n" + "="*60)
    print("Pinned Memory Benchmark")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterations = 10
    
    # Create test data
    frames_np = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
    masks_np = np.random.randint(0, 255, (num_frames, height, width), dtype=np.uint8)
    
    data_size_mb = (frames_np.nbytes + masks_np.nbytes) / (1024 * 1024)
    print(f"Data size: {data_size_mb:.2f} MB")
    
    # Normal transfer
    print("\nNormal transfer...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        frames_tensor = torch.from_numpy(frames_np).to(device).float()
        masks_tensor = torch.from_numpy(masks_np).to(device).float()
        torch.cuda.synchronize()
    normal_time = time.perf_counter() - start
    normal_avg = normal_time / iterations * 1000  # ms
    print(f"  Avg time: {normal_avg:.2f}ms")
    
    # Pinned memory transfer
    print("\nPinned memory transfer...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        frames_tensor = torch.from_numpy(frames_np).pin_memory().to(device, non_blocking=True).float()
        masks_tensor = torch.from_numpy(masks_np).pin_memory().to(device, non_blocking=True).float()
        torch.cuda.synchronize()
    pinned_time = time.perf_counter() - start
    pinned_avg = pinned_time / iterations * 1000  # ms
    speedup = normal_time / pinned_time
    print(f"  Avg time: {pinned_avg:.2f}ms, Speedup: {speedup:.2f}x")
    
    normal_result = BenchmarkResult(
        name="Normal Transfer",
        time_seconds=normal_time / iterations,
        fps=num_frames / (normal_time / iterations),
        frames=num_frames,
        speedup=1.0
    )
    
    pinned_result = BenchmarkResult(
        name="Pinned Memory Transfer",
        time_seconds=pinned_time / iterations,
        fps=num_frames / (pinned_time / iterations),
        frames=num_frames,
        speedup=speedup,
        baseline_name="Normal Transfer"
    )
    
    return normal_result, pinned_result


def benchmark_e2e(video_path: str, output_path: str) -> BenchmarkResult:
    """Benchmark end-to-end pipeline with all optimizations."""
    print("\n" + "="*60)
    print("End-to-End Pipeline Benchmark (All Optimizations)")
    print("="*60)
    
    # Get video info
    loader = VideoLoader(video_path)
    total_frames = loader.total_frames
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}, Resolution: {loader.width}x{loader.height}")
    
    # Initialize SoraWM with all optimizations (including torch.compile)
    print("\nInitializing SoraWM with all optimizations...")
    sora_wm = SoraWM(cleaner_type=CleanerType.E2FGVI_HQ, enable_torch_compile=True)
    
    # Run benchmark
    print("Processing video...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    sora_wm.run(
        input_video_path=Path(video_path),
        output_video_path=Path(output_path),
        quiet=True,
    )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    
    fps = total_frames / elapsed
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Processing speed: {fps:.2f} FPS")
    
    return BenchmarkResult(
        name="E2E Pipeline (All Optimizations)",
        time_seconds=elapsed,
        fps=fps,
        frames=total_frames,
        speedup=1.0  # Will be compared manually
    )


def generate_markdown_report(results: AblationResults, output_path: str):
    """Generate Markdown report from results."""
    
    md = []
    md.append("# SoraWatermarkCleaner 优化消融实验报告\n")
    md.append(f"**生成时间:** {results.timestamp}\n")
    md.append(f"**GPU:** {results.gpu_name}\n")
    md.append(f"**测试视频:** {results.video_path}\n")
    md.append(f"**视频信息:** {results.video_info['frames']} 帧, {results.video_info['width']}x{results.video_info['height']}, {results.video_info['fps']} FPS\n")
    
    md.append("\n---\n")
    
    # Summary table
    md.append("## 📊 优化效果总结\n")
    md.append("| 优化项 | 基准耗时 | 优化后耗时 | 提升倍数 | 提升百分比 |")
    md.append("|--------|----------|------------|----------|------------|")
    
    # Detector
    det_baseline = results.detector_single_frame.time_seconds
    det_opt = results.detector_batch.time_seconds
    det_speedup = det_baseline / det_opt
    det_improve = (det_speedup - 1) * 100
    md.append(f"| YOLO 批处理检测 | {det_baseline:.3f}s | {det_opt:.3f}s | **{det_speedup:.2f}x** | +{det_improve:.1f}% |")
    
    # Cleaner AMP
    clean_baseline = results.cleaner_baseline.time_seconds
    clean_opt = results.cleaner_amp.time_seconds
    clean_speedup = clean_baseline / clean_opt
    clean_improve = (clean_speedup - 1) * 100
    md.append(f"| AMP 混合精度 (FP16) | {clean_baseline:.3f}s | {clean_opt:.3f}s | **{clean_speedup:.2f}x** | +{clean_improve:.1f}% |")
    
    # Pinned Memory
    pin_baseline = results.pinned_memory_normal.time_seconds
    pin_opt = results.pinned_memory_pinned.time_seconds
    pin_speedup = pin_baseline / pin_opt
    pin_improve = (pin_speedup - 1) * 100
    md.append(f"| Pinned Memory | {pin_baseline*1000:.2f}ms | {pin_opt*1000:.2f}ms | **{pin_speedup:.2f}x** | +{pin_improve:.1f}% |")
    
    md.append("\n")
    
    # Detailed results
    md.append("## 🔍 详细测试结果\n")
    
    # Detector
    md.append("### 1. YOLO 水印检测器\n")
    md.append(f"测试帧数: {results.detector_single_frame.frames}\n")
    md.append("| 方法 | 耗时 (s) | FPS | 加速比 |")
    md.append("|------|----------|-----|--------|")
    md.append(f"| 单帧检测 (基准) | {results.detector_single_frame.time_seconds:.3f} | {results.detector_single_frame.fps:.2f} | 1.00x |")
    md.append(f"| 批量检测 (bs=4) | {results.detector_batch.time_seconds:.3f} | {results.detector_batch.fps:.2f} | **{det_speedup:.2f}x** |")
    md.append("\n")
    
    # Cleaner
    md.append("### 2. E2FGVI_HQ 视频修复模型\n")
    md.append(f"测试帧数: {results.cleaner_baseline.frames}\n")
    md.append("| 配置 | 耗时 (s) | FPS | 加速比 |")
    md.append("|------|----------|-----|--------|")
    md.append(f"| FP32 (基准) | {results.cleaner_baseline.time_seconds:.3f} | {results.cleaner_baseline.fps:.2f} | 1.00x |")
    md.append(f"| AMP FP16 | {results.cleaner_amp.time_seconds:.3f} | {results.cleaner_amp.fps:.2f} | **{clean_speedup:.2f}x** |")
    md.append("\n")
    
    # Pinned Memory
    md.append("### 3. 数据传输优化 (Pinned Memory)\n")
    md.append(f"测试帧数: {results.pinned_memory_normal.frames}\n")
    md.append("| 方法 | 平均耗时 (ms) | 吞吐量 (帧/s) | 加速比 |")
    md.append("|------|---------------|---------------|--------|")
    md.append(f"| 普通传输 (基准) | {results.pinned_memory_normal.time_seconds*1000:.2f} | {results.pinned_memory_normal.fps:.2f} | 1.00x |")
    md.append(f"| Pinned Memory | {results.pinned_memory_pinned.time_seconds*1000:.2f} | {results.pinned_memory_pinned.fps:.2f} | **{pin_speedup:.2f}x** |")
    md.append("\n")
    
    # E2E
    md.append("### 4. 端到端流水线性能\n")
    md.append(f"测试帧数: {results.e2e_optimized.frames}\n")
    md.append("| 指标 | 值 |")
    md.append("|------|-----|")
    md.append(f"| 总耗时 | **{results.e2e_optimized.time_seconds:.2f}s** |")
    md.append(f"| 处理速度 | **{results.e2e_optimized.fps:.2f} FPS** |")
    md.append(f"| 每帧耗时 | {results.e2e_optimized.time_seconds/results.e2e_optimized.frames*1000:.2f}ms |")
    md.append("\n")
    
    # Combined improvement estimate
    md.append("## 📈 综合优化效果估算\n")
    md.append("| 组件 | 优化前占比 | 加速比 | 优化后占比 |")
    md.append("|------|------------|--------|------------|")
    md.append(f"| YOLO 检测 | ~5% | {det_speedup:.2f}x | ~{5/det_speedup:.1f}% |")
    md.append(f"| E2FGVI 推理 | ~90% | {clean_speedup:.2f}x | ~{90/clean_speedup:.1f}% |")
    md.append(f"| 数据传输 | ~5% | {pin_speedup:.2f}x | ~{5/pin_speedup:.1f}% |")
    md.append("\n")
    
    # Estimated total speedup (weighted)
    # Assuming E2FGVI takes 90% of time, detector 5%, transfer 5%
    estimated_speedup = 1 / (0.05/det_speedup + 0.90/clean_speedup + 0.05/pin_speedup)
    md.append(f"**综合加速比估算:** ~**{estimated_speedup:.2f}x**\n")
    
    md.append("\n---\n")
    md.append("## 🛠️ 实现的优化技术\n")
    md.append("1. **YOLO 批处理检测**: 将多帧打包成 batch 一次性推理，减少 GPU kernel launch 开销\n")
    md.append("2. **AMP 混合精度**: 使用 FP16 进行推理，减少显存带宽需求，提高计算吞吐\n")
    md.append("3. **Pinned Memory**: 使用页锁定内存加速 CPU→GPU 数据传输\n")
    md.append("4. **视频帧缓存**: 一次性读取所有帧，避免重复 I/O\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    print(f"\nMarkdown report saved to: {output_path}")
    return '\n'.join(md)


def main():
    parser = argparse.ArgumentParser(description="Run complete ablation study")
    parser.add_argument("--video", type=str, 
                       default="resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4",
                       help="Input video path")
    parser.add_argument("--detector-frames", type=int, default=100,
                       help="Number of frames for detector benchmark")
    parser.add_argument("--cleaner-frames", type=int, default=30,
                       help="Number of frames for cleaner benchmark")
    parser.add_argument("--output-json", type=str, default="outputs/ablation_results.json",
                       help="Output JSON path")
    parser.add_argument("--output-md", type=str, default="outputs/ablation_report.md",
                       help="Output Markdown path")
    parser.add_argument("--skip-e2e", action="store_true",
                       help="Skip end-to-end benchmark (faster)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    print("="*70)
    print("COMPLETE ABLATION STUDY")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Detector frames: {args.detector_frames}")
    print(f"Cleaner frames: {args.cleaner_frames}")
    print(f"GPU: {get_gpu_name()}")
    
    # Get video info
    loader = VideoLoader(args.video)
    video_info = {
        "frames": loader.total_frames,
        "width": loader.width,
        "height": loader.height,
        "fps": loader.fps
    }
    
    # Run benchmarks
    det_single, det_batch = benchmark_detector(args.video, args.detector_frames)
    clean_baseline, clean_amp = benchmark_cleaner(args.video, args.cleaner_frames)
    pin_normal, pin_pinned = benchmark_pinned_memory(args.cleaner_frames, loader.width, loader.height)
    
    if not args.skip_e2e:
        e2e_result = benchmark_e2e(args.video, "outputs/ablation_e2e_output.mp4")
    else:
        # Placeholder if skipped
        e2e_result = BenchmarkResult(
            name="E2E Pipeline (Skipped)",
            time_seconds=0,
            fps=0,
            frames=loader.total_frames,
            speedup=1.0
        )
    
    # Create results object
    results = AblationResults(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gpu_name=get_gpu_name(),
        video_path=args.video,
        video_info=video_info,
        detector_single_frame=det_single,
        detector_batch=det_batch,
        cleaner_baseline=clean_baseline,
        cleaner_amp=clean_amp,
        pinned_memory_normal=pin_normal,
        pinned_memory_pinned=pin_pinned,
        e2e_optimized=e2e_result
    )
    
    # Save to JSON
    results_dict = {
        "timestamp": results.timestamp,
        "gpu_name": results.gpu_name,
        "video_path": results.video_path,
        "video_info": results.video_info,
        "detector_single_frame": asdict(results.detector_single_frame),
        "detector_batch": asdict(results.detector_batch),
        "cleaner_baseline": asdict(results.cleaner_baseline),
        "cleaner_amp": asdict(results.cleaner_amp),
        "pinned_memory_normal": asdict(results.pinned_memory_normal),
        "pinned_memory_pinned": asdict(results.pinned_memory_pinned),
        "e2e_optimized": asdict(results.e2e_optimized)
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nJSON results saved to: {args.output_json}")
    
    # Generate Markdown report
    report = generate_markdown_report(results, args.output_md)
    
    # Print report to console
    print("\n" + "="*70)
    print("MARKDOWN REPORT")
    print("="*70)
    print(report)


if __name__ == "__main__":
    main()

