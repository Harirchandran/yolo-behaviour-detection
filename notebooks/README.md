# 📓 Notebooks

This directory contains Jupyter notebooks used for training and experimentation.

## Available Notebooks

### training_colab.ipynb

Google Colab notebook used for training the **production model** with GPU acceleration.

**Training Details:**
| Parameter | Value |
|-----------|-------|
| Model | YOLOv8m (Medium) |
| Epochs | 100 |
| Dataset | 4 classes (helmet, no_helmet, alcohol, ciggaret) |
| Output | `models/best_colab.pt` |

**Features:**
- Cloud-based training with free GPU (T4/V100)
- Automatic checkpointing to Google Drive
- Resume training from last checkpoint
- Visualization of results (confusion matrix, loss curves)

**Usage:**
1. Upload to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Mount Google Drive for dataset and backups
4. Run all cells to train or resume training

## Output

The notebook produces:
- `best.pt` - Best model weights (saved as `models/best_colab.pt`)
- Confusion matrix visualization
- Training loss/accuracy curves
- Batch prediction samples
