# YOLO Behaviour Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**A real-time object detection system for safety and behaviour monitoring using YOLOv8**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Training](#-training) • [Tools](#-tools)

</div>

---

## 🎯 Overview

This project implements a YOLO-based detection system designed to identify safety violations and risky behaviours in real-time. The model is trained to detect:

| Class | Description | Use Case |
|-------|-------------|----------|
| 🪖 **Helmet** | Person wearing a helmet | Safety compliance |
| ⛑️ **No Helmet** | Person without a helmet | Safety violation detection |
| 🍺 **Alcohol** | Alcohol bottles/cans | Substance monitoring |
| 🚬 **Cigarette** | Cigarettes/smoking | Smoking detection |

### Use Cases
- 🏗️ Construction site safety monitoring
- 🏭 Factory floor compliance
- 🚦 Traffic safety enforcement
- 🏢 Workplace policy enforcement

---

## ✨ Features

- **Real-time Detection**: Process images, videos, or live webcam feeds
- **High Accuracy**: Trained on curated dataset with data augmentation
- **Easy Integration**: Streamlit-based demo app for quick testing
- **Complete Pipeline**: From data annotation to model deployment
- **Utility Tools**: Dataset cleaning, annotation editing, and verification tools

---

## 📁 Project Structure

```
yolo-behaviour-detection/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
│
├── src/                     # Core source code
│   ├── train.py             # Model training script
│   └── demo_app.py          # Streamlit detection app
│
├── notebooks/               # Jupyter notebooks
│   └── training_colab.ipynb # Google Colab training notebook
│
├── tools/                   # Utility tools
│   ├── dataset_analyzer.py  # Dataset verification & analysis
│   ├── annotation_editor.py # GUI annotation tool
│   ├── dataset_cleaner.py   # GUI dataset cleaner
│   ├── create_visualizations.py
│   └── batch_rename.py      # Smart file renamer
│
├── dataset/                 # Training dataset (YOLO format)
│   ├── data.yaml           # Dataset configuration
│   ├── train/              # Training images & labels
│   ├── valid/              # Validation images & labels
│   └── test/               # Test images & labels
│
├── models/                  # Trained model weights
│   ├── best.pt             # Locally trained model
│   └── best_colab.pt       # Colab trained model
│
├── examples/                # Sample outputs
│   └── training_batch_sample.jpg
│
└── docs/                    # Additional documentation
    └── TRAINING.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolo-behaviour-detection.git
   cd yolo-behaviour-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Quick Start

### Run Detection Demo

Launch the Streamlit app for real-time detection:

```bash
streamlit run src/demo_app.py
```

Then:
1. Upload your trained model (`models/best_colab.pt` for best results)
2. Select input source (Image/Video/Webcam)
3. Adjust confidence threshold as needed
4. View detection results and violation logs!

### Command Line Inference

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('models/best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

---

## 🏋️ Training

### Train Your Own Model

1. **Prepare your dataset** in YOLO format (see `dataset/` structure)

2. **Verify dataset integrity**
   ```bash
   python tools/dataset_analyzer.py dataset
   ```

3. **Start training**
   ```bash
   python src/train.py
   ```

### Training Configuration

Edit `src/train.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `imgsz` | 640 | Image size |
| `batch` | 16 | Batch size |
| `patience` | 10 | Early stopping patience |

> 💡 **Tip**: For small objects, increase `imgsz` to 1280

---

## 🛠️ Tools

### Dataset Analyzer
Comprehensive dataset verification tool:
```bash
python tools/dataset_analyzer.py ../dataset
```

Features:
- Folder structure analysis
- Image-label pair verification
- Class distribution statistics
- Object size analysis

### Annotation Editor
GUI tool for editing YOLO annotations:
```bash
python tools/annotation_editor.py
```

**Hotkeys:**
- `Q/W/E/R`: Select class (Helmet/No Helmet/Alcohol/Cigarette)
- `Left Drag`: Draw new bounding box
- `Right Click`: Delete box
- `←/→`: Navigate images
- `Del`: Delete current image

### Dataset Cleaner
Review and clean your dataset:
```bash
python tools/dataset_cleaner.py
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | TBD |
| mAP@50-95 | TBD |
| Inference Speed | ~15ms (GPU) |

> Add your actual metrics after training!

---

## 📜 Dataset Format

This project uses the standard YOLO annotation format:

```
dataset/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images (.jpg, .png)
│   └── labels/        # YOLO format annotations (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Label Format** (one line per object):
```
class_id center_x center_y width height
```
All values are normalized (0-1) relative to image dimensions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Core detection framework
- [Streamlit](https://streamlit.io/) - Demo application framework
- [OpenCV](https://opencv.org/) - Image processing

---

<div align="center">

**Made with ❤️ for Safety**

⭐ Star this repo if you find it useful!

</div>
