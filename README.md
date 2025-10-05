# SoraWatermarkCleaner

This project provides a elegant way to remove the sora watermark in the sora2 generated videos.

- Watermark removed


https://github.com/user-attachments/assets/8cdc075e-7d15-4d04-8fa2-53dd287e5f4c



- Original





https://github.com/user-attachments/assets/3c850ff1-b8e3-41af-a46f-2c734406e77d









## 1. Method

The SoraWatermarkCleaner(we call it `SoraWm` later) is composed of two parst:

- SoraWaterMarkDetector: We trained a yolov11s version to detect the sora watermark. (Thank you yolo!)

- WaterMarkCleaner: We refer iopaint's implementation for watermark removal using the lama model.

  (This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!)

Our SoraWm is purely deeplearning driven and yields good results in many generated videos.



## 2. Installation

We highly recommend use the `uv` to install the the envs:

1. installation:

```bash
uv sync
```

> now the envs will be installed at the `.ven`, you can actiavte the env using:
>
> ```bash
> source .venv/bin/activate
> ```

2. Downloaded the pretrained models:

The trained yolo weights will be stored in the `resources` dir as the `best.pt`.  And it will be automatically downloaded from https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/best.pt . The `Lama` model is downloaded from https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt, and will be stored in the torch cache dir. Both downloads are automatic, if you fail, please check your internet status.

## 3.  Demo

To have a basically usage, just try the `example.py`:

```python

from pathlib import Path
from sorawm.core import SoraWM


if __name__ == "__main__":
    input_video_path = Path(
        "resources/dog_vs_sam.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)

```

We also provide you with a `streamlit` based interactive web page, try it with:

```bash
python app.py
```



