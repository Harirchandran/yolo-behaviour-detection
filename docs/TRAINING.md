# 🏋️ Training Guide

Complete guide for training the YOLO Behaviour Detection model.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)
- At least 8GB GPU memory
- Properly formatted dataset

## Quick Start

```bash
# 1. Verify your dataset
python tools/dataset_analyzer.py dataset

# 2. Start training
python src/train.py
```

## Training Configuration

Edit `src/train.py` to customize training parameters:

```python
model.train(
    data='dataset/data.yaml',
    epochs=100,           # Number of training epochs
    imgsz=640,            # Input image size
    batch=16,             # Batch size (reduce if OOM error)
    patience=10,          # Early stopping patience
    workers=0,            # Windows requires 0
    mosaic=1.0,           # Mosaic augmentation
)
```

## Parameter Guide

### Essential Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `epochs` | 100-300 | More epochs = better (with early stopping) |
| `imgsz` | 640 | Use 1280 for small objects |
| `batch` | 16 | Reduce to 8 if OOM error |
| `patience` | 10-20 | Epochs to wait before early stop |

### For Small Objects

If your objects are small (like cigarettes far away):

```python
model.train(
    imgsz=1280,           # Higher resolution
    batch=8,              # Smaller batch for memory
)
```

### For Limited GPU Memory

```python
model.train(
    batch=8,              # Smaller batch
    imgsz=416,            # Smaller images
)
```

## Monitoring Training

Training outputs are saved to `runs/detect/`:

```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model (use this!)
│   └── last.pt          # Last checkpoint
├── results.csv          # Training metrics
├── confusion_matrix.png
├── results.png          # Loss/metrics graphs
└── ...
```

### TensorBoard (Optional)

```bash
tensorboard --logdir runs/detect/train
```

## After Training

1. **Copy best weights:**
   ```bash
   cp runs/detect/train/weights/best.pt models/best.pt
   ```

2. **Test the model:**
   ```bash
   yolo detect predict model=models/best.pt source=dataset/test/images/
   ```

3. **Run the demo:**
   ```bash
   streamlit run src/demo_app.py
   ```

## Troubleshooting

### CUDA Out of Memory

```python
batch=8,    # Reduce batch size
imgsz=416,  # Or reduce image size
```

### Training Too Slow (CPU)

Verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Poor Results

1. Check dataset quality with `dataset_analyzer.py`
2. Ensure balanced class distribution
3. Train for more epochs
4. Use a larger model (yolov8s.pt instead of yolov8n.pt)

## Resume Training

If training was interrupted:

```python
model = YOLO('runs/detect/train/weights/last.pt')
model.train(resume=True)
```

## Export Model

### ONNX (for deployment)
```bash
yolo export model=models/best.pt format=onnx
```

### TensorRT (for NVIDIA edge devices)
```bash
yolo export model=models/best.pt format=engine
```
