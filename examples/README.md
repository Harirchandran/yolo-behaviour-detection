# 📸 Examples

This directory contains sample outputs and demonstration files.

## Contents

### training_batch_sample.jpg

Sample training batch showing augmented images with annotations during training.

## Adding Examples

You can add:
- Sample detection results
- Before/after comparisons
- Demo GIFs
- Screenshots of the Streamlit app

## Generating Detection Samples

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/best.pt')

# Run detection
results = model('path/to/test/image.jpg')

# Save annotated image
results[0].save('examples/detection_result.jpg')
```
