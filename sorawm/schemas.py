from pydantic import BaseModel, ConfigDict
import numpy as np


class FrameData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx: int
    frame: np.ndarray
    # bbox: Optional[tuple] = None  # (x1, y1, x2, y2) or None


class BBoxData(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    x_center: int
    y_center: int
