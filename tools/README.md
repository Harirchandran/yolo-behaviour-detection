# 🛠️ Tools

This directory contains utility tools for dataset management and analysis.

## Available Tools

### 1. Dataset Analyzer (`dataset_analyzer.py`)

Comprehensive dataset verification and analysis tool.

**Features:**
- Folder structure analysis
- Image-label pair verification
- Class distribution statistics
- Object size analysis with training recommendations

**Usage:**
```bash
# Analyze the dataset
python dataset_analyzer.py ../dataset

# Or specify any path
python dataset_analyzer.py /path/to/your/dataset
```

---

### 2. Annotation Editor (`annotation_editor.py`)

GUI tool for creating and editing YOLO format annotations.

**Usage:**
```bash
python annotation_editor.py
```

**Hotkeys:**
| Key | Action |
|-----|--------|
| `Q` | Select Helmet class |
| `W` | Select No Helmet class |
| `E` | Select Alcohol class |
| `R` | Select Cigarette class |
| `Left Drag` | Draw new bounding box |
| `Right Click` | Delete nearest box |
| `←` / `→` | Navigate between images |
| `Delete` | Delete current image |

---

### 3. Dataset Cleaner (`dataset_cleaner.py`)

GUI tool for reviewing and cleaning dataset images.

**Usage:**
```bash
python dataset_cleaner.py
```

**Features:**
- Visual review of annotated images
- Quick navigation with keyboard
- Delete problematic images with one key
- Auto-save progress

**Hotkeys:**
| Key | Action |
|-----|--------|
| `1` | Switch to Train split |
| `2` | Switch to Valid split |
| `3` | Switch to Test split |
| `←` / `→` | Navigate images |
| `Delete` | Delete current image |

---

### 4. Create Visualizations (`create_visualizations.py`)

Generate visualized versions of all dataset images with bounding boxes drawn.

**Usage:**
```bash
python create_visualizations.py
```

This creates a `dataset_visualized/` folder with all images annotated.

---

### 5. Batch Rename (`batch_rename.py`)

Smart file renaming utility that names files based on their content.

**Usage:**
```bash
# Preview changes (dry run)
python batch_rename.py

# Edit the file and set DRY_RUN = False to apply
```

**Naming Format:** `{split}_{number}_{class}.jpg`

Example: `train_0001_helmet.jpg`

---

## Dependencies

All tools require the packages in `requirements.txt`. Additionally:

- **GUI Tools** (annotation_editor, dataset_cleaner): Require `tkinter` (usually included with Python)
- **Visualization**: Requires `opencv-python`

---

## Tips

1. **Always run from the tools directory** or adjust the paths in the configuration section of each script.

2. **Backup your data** before running batch operations like rename or delete.

3. **Run analyzer first** before training to catch any dataset issues.
