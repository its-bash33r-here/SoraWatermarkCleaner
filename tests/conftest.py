import tempfile
from pathlib import Path

import numpy as np
import pytest

from sorawm.schemas import CleanerType


@pytest.fixture
def temp_dir():
    """
    Provide a temporary filesystem directory for tests.
    
    Returns:
        path (pathlib.Path): Path to a temporary directory. The directory is removed automatically after the fixture finishes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """
    Generate a 100x100 RGB test image with random pixel values.
    
    Returns:
        image (numpy.ndarray): Array of shape (100, 100, 3) and dtype `uint8` containing random RGB pixel values in the range [0, 255].
    """
    # Create a simple RGB image (100x100)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_video_path(temp_dir):
    """
    Return a Path to a test video file named "test_video.mp4" inside the provided temporary directory.
    
    Parameters:
        temp_dir (Path): Directory to contain the test video path.
    
    Returns:
        Path: Path to "test_video.mp4" located within `temp_dir`.
    """
    return temp_dir / "test_video.mp4"


@pytest.fixture
def sample_mask():
    """
    Generate a 100x100 binary mask with a central 60x60 white square.
    
    Returns:
        mask (np.ndarray): 100x100 uint8 array where background pixels are 0 and the central 60x60 region is 255.
    """
    # Create a binary mask (100x100)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # White square in center
    return mask


@pytest.fixture
def mock_video_frames():
    """
    Generate a list of mock video frames.
    
    Returns:
        frames (list[numpy.ndarray]): A list of 10 NumPy arrays, each with shape (100, 100, 3), dtype `uint8`, and pixel values in the range 0–255.
    """
    # Create 10 frames of 100x100 RGB images
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def mock_video_masks():
    """
    Generate a list of synthetic 100x100 binary masks with a moving square region.
    
    Each returned mask is a NumPy uint8 array where a 40×40 square region is set to 255 and the rest is 0.
    For mask index i (0–9) the square is positioned with its top-left corner at (10+i, 10+i).
    
    Returns:
        list[numpy.ndarray]: Ten 100×100 uint8 masks; 255 indicates the masked area, 0 indicates background.
    """
    # Create 10 masks of 100x100
    masks = []
    for i in range(10):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Add some variation to masks
        mask[10 + i : 50 + i, 10 + i : 50 + i] = 255
        masks.append(mask)
    return masks


@pytest.fixture
def cleaner_types():
    """
    Return the list of cleaner types available for tests.
    
    Returns:
        list[CleanerType]: A list containing available CleanerType members.
    """
    return [CleanerType.LAMA, CleanerType.E2FGVI_HQ]


@pytest.fixture
def detection_result_with_watermark():
    """
    Provide a mock detection result indicating a detected watermark.
    
    Returns:
        result (dict): Mock detection result with keys:
            detected (bool): True when a watermark is detected.
            bbox (tuple[int, int, int, int] | None): Bounding box as (x1, y1, x2, y2) or None.
            confidence (float | None): Detection confidence score between 0 and 1.
            center (tuple[int, int] | None): Center coordinates (x, y) of the detection or None.
    """
    return {
        "detected": True,
        "bbox": (10, 10, 50, 50),
        "confidence": 0.95,
        "center": (30, 30),
    }


@pytest.fixture
def detection_result_no_watermark():
    """
    Provide a mock detection result representing the absence of a watermark.
    
    Returns:
        result (dict): Detection result with the following keys:
            - "detected" (bool): `False` indicating no watermark was found.
            - "bbox" (None): No bounding box is available.
            - "confidence" (None): No confidence score is available.
            - "center" (None): No center coordinate is available.
    """
    return {
        "detected": False,
        "bbox": None,
        "confidence": None,
        "center": None,
    }