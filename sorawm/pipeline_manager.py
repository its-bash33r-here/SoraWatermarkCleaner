"""
sorawm/pipeline_manager.py
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
        self.detector = None  # Initialize in run() to avoid pickling issues

    def process(self, data: FrameData) -> tuple[int, BBoxData | None]:
        detection_result = self.detector.detect(data.frame)
        if detection_result["detected"]:
            x1, y1, x2, y2 = detection_result["bbox"]
            x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
            bbox_data = BBoxData(
                x1=x1, y1=y1, x2=x2, y2=y2, 
                x_center=x_center, y_center=y_center
            )
            return data.idx, bbox_data
        else:
            return data.idx, None
    
    def run(self):
        """Main worker loop for detection"""
        # Initialize detector in the worker process to avoid serialization issues
        self.detector = SoraWaterMarkDetector()
        logger.debug(f"DetectorWorker started (PID: {mp.current_process().pid})")
        
        while True:
            data = self.input_queue.get()
            if data is None:  # Poison pill to stop the worker
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
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, width: int, height: int):
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
        """Main worker loop for cleaning"""
        self.cleaner = WaterMarkCleaner()
        logger.debug(f"CleanerWorker started (PID: {mp.current_process().pid})")
        
        while True:
            data = self.input_queue.get()
            if data is None:  # Poison pill
                self.output_queue.put(None)
                logger.debug("CleanerWorker received stop signal")
                break
            
            try:
                idx, frame, bbox_data = data
                cleaned_frame = self.process(frame, bbox_data)
                self.output_queue.put((idx, cleaned_frame))
            except Exception as e:
                logger.error(f"Error in CleanerWorker: {e}")
                # Return original frame on error
                idx, frame, _ = data
                self.output_queue.put((idx, frame))


class PipelineManager:
    """Manages the parallel pipeline for watermark detection and removal"""
    
    def __init__(
        self, 
        num_detector_workers: int = 1,
        num_cleaner_workers: int = 1,
        buffer_size: int = 30
    ):
        self.num_detector_workers = num_detector_workers
        self.num_cleaner_workers = num_cleaner_workers
        self.buffer_size = buffer_size
        
        # Queues for pipeline stages
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
            worker = DetectorWorker(self.detector_input_queue, self.detector_output_queue)
            process = mp.Process(target=worker.run)
            process.start()
            self.detector_workers.append(process)
        
        # Start cleaner workers
        for _ in range(self.num_cleaner_workers):
            worker = CleanerWorker(
                self.cleaner_input_queue, 
                self.cleaner_output_queue,
                width, 
                height
            )
            process = mp.Process(target=worker.run)
            process.start()
            self.cleaner_workers.append(process)
        
        logger.info(f"Started {self.num_detector_workers} detector workers and "
                   f"{self.num_cleaner_workers} cleaner workers")
    
    def stop_workers(self):
        """Stop all worker processes gracefully"""
        # Send poison pills to detector workers
        for _ in range(self.num_detector_workers):
            self.detector_input_queue.put(None)
        
        # Wait for detector workers to finish
        for worker in self.detector_workers:
            worker.join()
        
        # Send poison pills to cleaner workers
        for _ in range(self.num_cleaner_workers):
            self.cleaner_input_queue.put(None)
        
        # Wait for cleaner workers to finish
        for worker in self.cleaner_workers:
            worker.join()
        
        logger.info("All workers stopped")
    
    def run_pipeline(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        """Run the complete pipeline with overlapping detection and cleaning"""
        
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames
        
        # Start worker processes
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
        
        # Storage for detection results
        frame_bboxes = {}
        bbox_centers = []
        bboxes = []
        detect_missed = []
        frames_cache = {}  # Cache frames for cleaning stage
        
        # Phase 1: Detection with overlapping frame loading
        logger.info("Phase 1: Detecting watermarks...")
        detection_done = False
        frames_sent = 0
        frames_received = 0
        
        def send_frames():
            nonlocal frames_sent
            for idx, frame in enumerate(tqdm(input_video_loader, total=total_frames, desc="Loading frames")):
                frame_data = FrameData(idx=idx, frame=frame)
                self.detector_input_queue.put(frame_data)
                frames_cache[idx] = frame  # Cache for later cleaning
                frames_sent += 1
                
                if progress_callback and idx % 10 == 0:
                    progress = 5 + int((idx / total_frames) * 20)
                    progress_callback(progress)
            
            # Signal end of frames
            for _ in range(self.num_detector_workers):
                self.detector_input_queue.put(None)
        
        # Start frame sending in a separate thread
        send_thread = threading.Thread(target=send_frames)
        send_thread.start()
        
        # Collect detection results
        workers_finished = 0
        with tqdm(total=total_frames, desc="Detecting watermarks") as pbar:
            while workers_finished < self.num_detector_workers:
                result = self.detector_output_queue.get()
                
                if result is None:
                    workers_finished += 1
                    continue
                
                idx, bbox_data = result
                if bbox_data is not None:
                    frame_bboxes[idx] = {"bbox": (bbox_data.x1, bbox_data.y1, bbox_data.x2, bbox_data.y2)}
                    bbox_centers.append((bbox_data.x_center, bbox_data.y_center))
                    bboxes.append((bbox_data.x1, bbox_data.y1, bbox_data.x2, bbox_data.y2))
                else:
                    frame_bboxes[idx] = {"bbox": None}
                    detect_missed.append(idx)
                    bbox_centers.append(None)
                    bboxes.append(None)
                
                frames_received += 1
                pbar.update(1)
                
                if progress_callback and frames_received % 10 == 0:
                    progress = 25 + int((frames_received / total_frames) * 20)
                    progress_callback(progress)
        
        send_thread.join()
        
        # Phase 2: Imputation for missed detections
        logger.info(f"Phase 2: Imputing {len(detect_missed)} missed detections...")
        if detect_missed:
            bkps = find_2d_data_bkps(bbox_centers)
            bkps_full = [0] + bkps + [total_frames]
            interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
            missed_intervals = find_idxs_interval(detect_missed, bkps_full)
            
            for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
                if (interval_idx < len(interval_bboxes) and 
                    interval_bboxes[interval_idx] is not None):
                    frame_bboxes[missed_idx]["bbox"] = interval_bboxes[interval_idx]
                else:
                    # Fallback: use neighboring frames
                    before = max(missed_idx - 1, 0)
                    after = min(missed_idx + 1, total_frames - 1)
                    before_box = frame_bboxes.get(before, {}).get("bbox")
                    after_box = frame_bboxes.get(after, {}).get("bbox")
                    if before_box:
                        frame_bboxes[missed_idx]["bbox"] = before_box
                    elif after_box:
                        frame_bboxes[missed_idx]["bbox"] = after_box
        
        if progress_callback:
            progress_callback(50)
        
        # Phase 3: Cleaning
        logger.info("Phase 3: Removing watermarks...")
        
        # Send frames to cleaner workers
        for idx in range(total_frames):
            frame = frames_cache[idx]
            bbox = frame_bboxes[idx]["bbox"]
            bbox_data = None
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_data = BBoxData(x1=x1, y1=y1, x2=x2, y2=y2, x_center=0, y_center=0)
            
            self.cleaner_input_queue.put((idx, frame, bbox_data))
        
        # Signal end of cleaning tasks
        for _ in range(self.num_cleaner_workers):
            self.cleaner_input_queue.put(None)
        
        # Collect cleaned frames and write in order
        cleaned_frames = {}
        workers_finished = 0
        next_frame_to_write = 0
        
        with tqdm(total=total_frames, desc="Cleaning frames") as pbar:
            while workers_finished < self.num_cleaner_workers or next_frame_to_write < total_frames:
                result = self.cleaner_output_queue.get()
                
                if result is None:
                    workers_finished += 1
                    continue
                
                idx, cleaned_frame = result
                cleaned_frames[idx] = cleaned_frame
                pbar.update(1)
                
                # Write frames in order
                while next_frame_to_write in cleaned_frames:
                    process_out.stdin.write(cleaned_frames[next_frame_to_write].tobytes())
                    del cleaned_frames[next_frame_to_write]
                    del frames_cache[next_frame_to_write]  # Free memory
                    next_frame_to_write += 1
                    
                    if progress_callback and next_frame_to_write % 10 == 0:
                        progress = 50 + int((next_frame_to_write / total_frames) * 40)
                        progress_callback(progress)
        
        process_out.stdin.close()
        process_out.wait()
        
        # Stop all workers
        self.stop_workers()
        
        if progress_callback:
            progress_callback(95)
        
        # # Phase 4: Merge audio
        # logger.info("Phase 4: Merging audio track...")
        # self.merge_audio_track(input_video_path, temp_output_path, output_video_path)
        
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Pipeline complete! Output saved to: {output_video_path}")
        return temp_output_path
  