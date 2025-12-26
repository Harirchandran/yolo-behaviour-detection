# 🤖 Models

This directory contains trained model weights.

## Available Models

### best_colab.pt ⭐ (Recommended)

**Primary model** trained on Google Colab with YOLOv8 Medium architecture.

| Property | Value |
|----------|-------|
| Architecture | **YOLOv8m (Medium)** |
| Size | ~155MB |
| Epochs | 100 |
| Input Size | 640x640 |
| Classes | 4 |
| Training | Google Colab (GPU) |

> This is the more accurate model - use this for production!

---

### best.pt (Lightweight)

Smaller model trained locally with YOLOv8 Nano architecture.

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n (Nano) |
| Size | ~18MB |
| Input Size | 640x640 |
| Classes | 4 |

> Use this for faster inference on resource-limited devices.

## Usage

### Python

```python
from ultralytics import YOLO

# Load the model (choose one)
model = YOLO('models/best.pt')        # Local model
# model = YOLO('models/best_colab.pt')  # Colab model

# Run inference
results = model('path/to/image.jpg')
results[0].show()
```

### Command Line

```bash
# Using local model
yolo detect predict model=models/best.pt source=path/to/image.jpg

# Using Colab model
yolo detect predict model=models/best_colab.pt source=path/to/image.jpg
```

## Classes

| ID | Name |
|----|------|
| 0 | helmet |
| 1 | without_helmet |
| 2 | alcohol |
| 3 | ciggaret |


| Metric | Value |
|--------|-------|
| mAP@50 | TBD |
| mAP@50-95 | TBD |
| Precision | TBD |
| Recall | TBD |

> Update these metrics after training!

## Training Your Own

See `src/train.py` or the training documentation in `docs/TRAINING.md`.

## Notes

- Model was trained on NVIDIA GPU with CUDA
- For CPU inference, model will automatically use CPU
- Inference speed varies based on hardware
