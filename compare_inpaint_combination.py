

from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector
import cv2
from pathlib import Path
import numpy as np
from typing import List
from sorawm.iopaint.schema import InpaintRequest
from sorawm.iopaint.schema import HDStrategy
from tqdm import tqdm

cleaner = WaterMarkCleaner()
detector = SoraWaterMarkDetector()

def put_text_with_background(image: np.ndarray, text_lines: List[str], position: str = 'top_right'):
    """
    Add text with semi-transparent background for better readability
    """
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_height = 30
    padding = 10
    
    # Calculate text dimensions
    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in text_lines]
    max_text_width = max(size[0] for size in text_sizes)
    total_text_height = len(text_lines) * line_height
    
    # Calculate background rectangle
    bg_width = max_text_width + padding * 2
    bg_height = total_text_height + padding * 2
    
    if position == 'top_right':
        x_start = width - bg_width - 10
        y_start = 10
    elif position == 'top_left':
        x_start = 10
        y_start = 10
    else:
        x_start = 10
        y_start = 10
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, 
                  (x_start, y_start), 
                  (x_start + bg_width, y_start + bg_height), 
                  (0, 0, 0), 
                  -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Draw text
    for i, line in enumerate(text_lines):
        y_pos = y_start + padding + (i + 1) * line_height - 5
        cv2.putText(image, line, 
                   (x_start + padding, y_pos), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return image

def auto_stack_images(images: List[np.ndarray]) -> np.ndarray:
    """
    stack the image in the original shape with rows and cols adpataivie
    """
    if not images:
        raise ValueError("No images to stack")
    
    if len(images) == 1:
        return images[0]
    
    # Calculate optimal grid dimensions
    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    # Get the maximum dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Ensure all images have the same number of channels
    num_channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create padded images to have uniform size
    padded_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Pad the image to max dimensions
        padded = np.zeros((max_height, max_width, 3), dtype=img.dtype)
        padded[:img.shape[0], :img.shape[1]] = img
        padded_images.append(padded)
    
    # Fill remaining slots with black images if needed
    total_slots = rows * cols
    while len(padded_images) < total_slots:
        padded_images.append(np.zeros((max_height, max_width, 3), dtype=images[0].dtype))
    
    # Stack images in rows
    row_images = []
    for i in range(rows):
        row_start = i * cols
        row_end = min(row_start + cols, len(padded_images))
        row = np.hstack(padded_images[row_start:row_end])
        row_images.append(row)
    
    # Stack rows vertically
    result = np.vstack(row_images)
    return result

def compare_inpaint_combination(original_image: np.ndarray, watermark_mask: np.ndarray, combinations: List[InpaintRequest]) \
    -> np.array:

    
    original_copy = original_image.copy()
    height, width = original_image.shape[:2]
    
    # Add label to original image
    put_text_with_background(original_copy, ["Original Image"], position='top_right')
    res = [original_copy]
    
    for comb in combinations:
        inpaint_result = cleaner.clean(original_image, watermark_mask, comb)
        # Prepare text lines for this combination
        text_lines = [f"HD: {comb.hd_strategy}", f"Margin: {comb.hd_strategy_crop_margin}"]
        if comb.hd_strategy_resize_limit:
            text_lines.append(f"Limit: {comb.hd_strategy_resize_limit}")
        put_text_with_background(inpaint_result, text_lines, position='top_right')
        res.append(inpaint_result)
    res = auto_stack_images(res)
    return res

def show_image(image: np.ndarray):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_dir = Path("videos")
    video_paths = list(video_dir.rglob("*.mp4"))    
    combinations = [    
        InpaintRequest(hd_strategy=HDStrategy.RESIZE, hd_strategy_crop_margin=128, hd_strategy_resize_limit=1280),
        InpaintRequest(hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=128),
        InpaintRequest(hd_strategy=HDStrategy.RESIZE, hd_strategy_crop_margin=196, hd_strategy_resize_limit=2048),
        InpaintRequest(hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=196),
        InpaintRequest(hd_strategy=HDStrategy.RESIZE, hd_strategy_crop_margin=256, hd_strategy_resize_limit=2048),
        InpaintRequest(hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=256),
        ]
    for video_path in tqdm(video_paths, desc="Processing videos"):
        input_video_name = video_path.name
        output_dir = Path("outputs/compare_inpaint_combination") / input_video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        sampled_frames = [1, 10, 100, 200]
        cap = cv2.VideoCapture(str(video_path))

        for frame_idx in tqdm(sampled_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                pass
            else:
                output_path = output_dir / f"{input_video_name}_frame_{frame_idx}.png"
                if output_path.exists():
                    continue
                else:
                    detector_result = detector.detect(frame)
                    if detector_result["detected"]:
                        watermark_mask = detector_result["bbox"]
                        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        mask[watermark_mask[1]:watermark_mask[3], watermark_mask[0]:watermark_mask[2]] = 255
                        result = compare_inpaint_combination(frame, mask, combinations)
                        cv2.imwrite(output_path, result)
                    # show_image(result)

    cap.release()

    
    

    