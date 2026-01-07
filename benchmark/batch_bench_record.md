# Batch Processing Benchmark Results

## Ablation Study

| Detector | Batch | Cleaner | TorchCompile | Time (s) | Speedup |
|:--------:|:-----:|:-------:|:------------:|:--------:|:-------:|
| YOLO     | ×     | LAMA    | ×            | 44.33    | -       |
| YOLO     | ×     | E2FGVI  | ×            | 142.42   | 1.00×   |
| YOLO     | ×     | E2FGVI  | ✓            | 117.19   | 1.22×   |
| YOLO     | 4     | E2FGVI  | ✓            | 82.63    | 1.72×   |

> **Note**: Speedup is calculated relative to the E2FGVI baseline (142.42s).
> LAMA is a different cleaner and not directly comparable.
