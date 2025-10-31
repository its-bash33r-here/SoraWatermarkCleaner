"""
sorawm/pipeline_manager.py - 真正的检测-清理并行流水线
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import multiprocessing as mp
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import ffmpeg
import threading
from collections import defaultdict

from sorawm.schemas import FrameData, BBoxData
from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.utils.video_utils import VideoLoader
from sorawm.utils.imputation_utils import (
    find_2d_data_bkps,
    get_interval_average_bbox,
    find_idxs_interval,
)


class BaseWorker(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


class DetectorWorker(BaseWorker):
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector = None

    def process(self, data: FrameData) -> tuple[int, BBoxData | None]:
        detection_result = self.detector.detect(data.frame)
        if detection_result["detected"]:
            x1, y1, x2, y2 = detection_result["bbox"]
            x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
            bbox_data = BBoxData(
                x1=x1, y1=y1, x2=x2, y2=y2, x_center=x_center, y_center=y_center
            )
            return data.idx, bbox_data
        else:
            return data.idx, None

    def run(self):
        self.detector = SoraWaterMarkDetector()
        logger.debug(f"DetectorWorker started (PID: {mp.current_process().pid})")

        while True:
            data = self.input_queue.get()
            if data is None:
                self.output_queue.put(None)
                logger.debug("DetectorWorker received stop signal")
                break

            try:
                result = self.process(data)
                self.output_queue.put(result)
            except Exception as e:
                logger.error(f"Error in DetectorWorker: {e}")
                self.output_queue.put((data.idx, None))


class CleanerWorker(BaseWorker):
    def __init__(
        self, input_queue: mp.Queue, output_queue: mp.Queue, width: int, height: int
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cleaner = None
        self.width = width
        self.height = height

    def process(self, frame: np.ndarray, bbox_data: BBoxData | None) -> np.ndarray:
        if bbox_data is None:
            return frame

        x1, y1, x2, y2 = bbox_data.x1, bbox_data.y1, bbox_data.x2, bbox_data.y2
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        cleaned_frame = self.cleaner.clean(frame, mask)
        return cleaned_frame

    def run(self):
        self.cleaner = WaterMarkCleaner()
        logger.debug(f"CleanerWorker started (PID: {mp.current_process().pid})")

        while True:
            data = self.input_queue.get()
            if data is None:
                self.output_queue.put(None)
                logger.debug("CleanerWorker received stop signal")
                break

            try:
                idx, frame, bbox_data = data
                cleaned_frame = self.process(frame, bbox_data)
                self.output_queue.put((idx, cleaned_frame))
            except Exception as e:
                logger.error(f"Error in CleanerWorker: {e}")
                idx, frame, _ = data
                self.output_queue.put((idx, frame))


class PipelineManager:
    """Manages the parallel pipeline for watermark detection and removal"""

    def __init__(
        self,
        num_detector_workers: int = 1,
        num_cleaner_workers: int = 1,
        buffer_size: int = 30,
    ):
        self.num_detector_workers = num_detector_workers
        self.num_cleaner_workers = num_cleaner_workers
        self.buffer_size = buffer_size

        self.detector_input_queue = mp.Queue(maxsize=buffer_size)
        self.detector_output_queue = mp.Queue(maxsize=buffer_size)
        self.cleaner_input_queue = mp.Queue(maxsize=buffer_size)
        self.cleaner_output_queue = mp.Queue(maxsize=buffer_size)

        self.detector_workers = []
        self.cleaner_workers = []

    def start_workers(self, width: int, height: int):
        """Start all worker processes"""
        # Start detector workers
        for _ in range(self.num_detector_workers):
            worker = DetectorWorker(
                self.detector_input_queue, self.detector_output_queue
            )
            process = mp.Process(target=worker.run)
            process.start()
            self.detector_workers.append(process)

        # Start cleaner workers
        for _ in range(self.num_cleaner_workers):
            worker = CleanerWorker(
                self.cleaner_input_queue, self.cleaner_output_queue, width, height
            )
            process = mp.Process(target=worker.run)
            process.start()
            self.cleaner_workers.append(process)

        logger.info(
            f"Started {self.num_detector_workers} detector workers and "
            f"{self.num_cleaner_workers} cleaner workers"
        )

    def stop_workers(self):
        """Stop all worker processes gracefully"""
        for worker in self.detector_workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Force terminating DetectorWorker {worker.pid}")
                worker.terminate()
                worker.join()

        for worker in self.cleaner_workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Force terminating CleanerWorker {worker.pid}")
                worker.terminate()
                worker.join()

        logger.info("All workers stopped")

    def run_pipeline(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        """Run with true detect-clean parallelism"""

        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        # Start all workers
        self.start_workers(width, height)

        # Setup output video writer
        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",
        }

        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(
                int(int(input_video_loader.original_bitrate) * 1.2)
            )
        else:
            output_options["crf"] = "18"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        # Storage
        frame_bboxes = {}  # 所有帧的bbox结果
        bbox_centers = []  # 用于插值
        bboxes = []  # 用于插值
        detect_missed = []  # 未检测到的帧索引
        missed_frames_cache = {}  # 缓存未检测到的帧
        frames_cache = {}  # 缓存用于加载
        
        detection_complete = False
        frames_sent_to_detector = 0
        frames_received_from_detector = 0
        detector_workers_finished = 0
        
        frames_sent_to_cleaner = 0
        frames_received_from_cleaner = 0
        cleaner_workers_finished = 0
        
        next_frame_to_write = 0
        cleaned_frames_buffer = {}

        # ========== 线程1: 加载帧并发送到检测队列 ==========
        def load_and_send_to_detector():
            nonlocal frames_sent_to_detector
            for idx, frame in enumerate(
                tqdm(input_video_loader, total=total_frames, desc="Loading frames")
            ):
                frame_data = FrameData(idx=idx, frame=frame)
                self.detector_input_queue.put(frame_data)
                frames_cache[idx] = frame  # 缓存所有帧
                frames_sent_to_detector += 1

                if progress_callback and idx % 10 == 0:
                    progress = 5 + int((idx / total_frames) * 15)
                    progress_callback(progress)

            # 发送停止信号
            for _ in range(self.num_detector_workers):
                self.detector_input_queue.put(None)
            logger.debug("All frames sent to detector")

        # ========== 线程2: 接收检测结果并调度清理 ==========
        def receive_detection_and_dispatch_cleaning():
            nonlocal frames_received_from_detector, detector_workers_finished
            nonlocal frames_sent_to_cleaner, detection_complete

            with tqdm(total=total_frames, desc="Detecting & Dispatching") as pbar:
                while detector_workers_finished < self.num_detector_workers:
                    result = self.detector_output_queue.get()

                    if result is None:
                        detector_workers_finished += 1
                        continue

                    idx, bbox_data = result
                    frames_received_from_detector += 1
                    pbar.update(1)

                    if bbox_data is not None:
                        # 检测到bbox，立即发送到清理队列
                        frame_bboxes[idx] = {
                            "bbox": (bbox_data.x1, bbox_data.y1, bbox_data.x2, bbox_data.y2)
                        }
                        bbox_centers.append((bbox_data.x_center, bbox_data.y_center))
                        bboxes.append((bbox_data.x1, bbox_data.y1, bbox_data.x2, bbox_data.y2))
                        
                        # 立即发送到清理队列（检测和清理并行！）
                        frame = frames_cache[idx]
                        self.cleaner_input_queue.put((idx, frame, bbox_data))
                        frames_sent_to_cleaner += 1
                    else:
                        # 未检测到，暂存
                        frame_bboxes[idx] = {"bbox": None}
                        detect_missed.append(idx)
                        bbox_centers.append(None)
                        bboxes.append(None)
                        missed_frames_cache[idx] = frames_cache[idx]

                    if progress_callback and frames_received_from_detector % 10 == 0:
                        progress = 20 + int((frames_received_from_detector / total_frames) * 30)
                        progress_callback(progress)

            detection_complete = True
            logger.info(f"Detection complete. Missed frames: {len(detect_missed)}")

            # ========== 插值处理 ==========
            if detect_missed:
                logger.info(f"Imputing {len(detect_missed)} missed detections...")
                bkps = find_2d_data_bkps(bbox_centers)
                bkps_full = [0] + bkps + [total_frames]
                interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
                missed_intervals = find_idxs_interval(detect_missed, bkps_full)

                for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
                    if (
                        interval_idx < len(interval_bboxes)
                        and interval_bboxes[interval_idx] is not None
                    ):
                        frame_bboxes[missed_idx]["bbox"] = interval_bboxes[interval_idx]
                    else:
                        before = max(missed_idx - 1, 0)
                        after = min(missed_idx + 1, total_frames - 1)
                        before_box = frame_bboxes.get(before, {}).get("bbox")
                        after_box = frame_bboxes.get(after, {}).get("bbox")
                        if before_box:
                            frame_bboxes[missed_idx]["bbox"] = before_box
                        elif after_box:
                            frame_bboxes[missed_idx]["bbox"] = after_box

                # 发送之前暂存的帧到清理队列
                logger.info(f"Sending {len(detect_missed)} imputed frames to cleaner...")
                for missed_idx in detect_missed:
                    frame = missed_frames_cache[missed_idx]
                    bbox = frame_bboxes[missed_idx]["bbox"]
                    bbox_data = None
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        bbox_data = BBoxData(
                            x1=x1, y1=y1, x2=x2, y2=y2, x_center=0, y_center=0
                        )
                    self.cleaner_input_queue.put((missed_idx, frame, bbox_data))
                    frames_sent_to_cleaner += 1

            # 发送停止信号给清理workers
            for _ in range(self.num_cleaner_workers):
                self.cleaner_input_queue.put(None)
            logger.debug("All frames sent to cleaner")

        # ========== 线程3: 接收清理结果并按顺序写入 ==========
        def receive_and_write_cleaned_frames():
            nonlocal frames_received_from_cleaner, cleaner_workers_finished
            nonlocal next_frame_to_write

            with tqdm(total=total_frames, desc="Cleaning & Writing") as pbar:
                while (
                    cleaner_workers_finished < self.num_cleaner_workers
                    or next_frame_to_write < total_frames
                ):
                    result = self.cleaner_output_queue.get()

                    if result is None:
                        cleaner_workers_finished += 1
                        continue

                    idx, cleaned_frame = result
                    cleaned_frames_buffer[idx] = cleaned_frame
                    frames_received_from_cleaner += 1
                    pbar.update(1)

                    # 按顺序写入
                    while next_frame_to_write in cleaned_frames_buffer:
                        process_out.stdin.write(
                            cleaned_frames_buffer[next_frame_to_write].tobytes()
                        )
                        del cleaned_frames_buffer[next_frame_to_write]
                        # 清理缓存
                        if next_frame_to_write in frames_cache:
                            del frames_cache[next_frame_to_write]
                        if next_frame_to_write in missed_frames_cache:
                            del missed_frames_cache[next_frame_to_write]
                        next_frame_to_write += 1

                        if progress_callback and next_frame_to_write % 10 == 0:
                            progress = 50 + int((next_frame_to_write / total_frames) * 45)
                            progress_callback(progress)

        # 启动所有线程
        logger.info("Starting parallel pipeline...")
        load_thread = threading.Thread(target=load_and_send_to_detector)
        dispatch_thread = threading.Thread(target=receive_detection_and_dispatch_cleaning)
        write_thread = threading.Thread(target=receive_and_write_cleaned_frames)

        load_thread.start()
        dispatch_thread.start()
        write_thread.start()

        # 等待所有线程完成
        load_thread.join()
        logger.debug("Load thread finished")
        dispatch_thread.join()
        logger.debug("Dispatch thread finished")
        write_thread.join()
        logger.debug("Write thread finished")

        # 关闭输出
        process_out.stdin.close()
        process_out.wait()

        # 停止workers
        self.stop_workers()

        if progress_callback:
            progress_callback(100)

        logger.info(f"Pipeline complete! Output saved to: {temp_output_path}")
        return temp_output_path