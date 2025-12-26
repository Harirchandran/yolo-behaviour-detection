# 📦 Dataset

This directory contains the training dataset in YOLO format.

> ⚠️ **Dataset not included in repository!**  
> Download from Kaggle using the instructions below.

## 📥 Download Dataset

### Option 1: Using Kaggle CLI
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset (requires kaggle.json API key)
kaggle datasets download -d <dataset-name>

# Unzip to this folder
unzip <dataset>.zip -d dataset/
```

### Option 2: Using Google Colab
See `notebooks/training_colab.ipynb` - it automatically downloads the dataset from Kaggle.
│   ├── images/         # Training images
│   └── labels/         # YOLO format annotations
├── valid/              # Validation set (~10% of data)
│   ├── images/
│   └── labels/
└── test/               # Test set (~10% of data)
    ├── images/
    └── labels/
```

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | helmet | Person wearing helmet |
| 1 | without_helmet | Person not wearing helmet |
| 2 | alcohol | Alcohol bottle/can |
| 3 | ciggaret | Cigarette |

## Annotation Format

Each image has a corresponding `.txt` file with the same name in the `labels/` folder.

**Format:** One object per line
```
class_id center_x center_y width height
```

**Example:**
```
0 0.456 0.312 0.124 0.089
1 0.723 0.567 0.098 0.134
```

All coordinates are **normalized** (0.0 to 1.0) relative to image dimensions.

## Statistics

Run the analyzer to get current statistics:
```bash
python tools/dataset_analyzer.py dataset
```

## data.yaml

The `data.yaml` file configures the dataset for YOLO training:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  - helmet
  - without_helmet
  - alcohol
  - ciggaret
```

## Adding New Images

1. Place images in the appropriate `images/` folder
2. Create corresponding labels in the `labels/` folder
3. Run `python tools/dataset_analyzer.py dataset` to verify

## Notes

- Images should be `.jpg` or `.png` format
- Labels must have the exact same filename (minus extension)
- Empty label files indicate images with no objects (background images)
