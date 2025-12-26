"""
YOLO Dataset Analyzer - Comprehensive Dataset Verification Tool

This tool combines multiple analysis functions:
- Structure analysis (folder hierarchy)
- Class distribution analysis
- Image-Label pair verification
- Object size analysis for training optimization

Usage:
    python dataset_analyzer.py [dataset_path]
    
Example:
    python dataset_analyzer.py ../dataset
"""

import os
import sys
import numpy as np
from collections import Counter
from pathlib import Path

# --- CONFIGURATION ---
DEFAULT_ROOT = '../dataset'
CLASS_NAMES = {0: 'Helmet', 1: 'No Helmet', 2: 'Alcohol', 3: 'Ciggaret'}
# ---------------------


def analyze_structure(root_folder):
    """Analyzes and displays the folder structure of the dataset."""
    print(f"\n{'='*60}")
    print(f"📂 FOLDER STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    
    if not os.path.exists(root_folder):
        print(f"❌ Error: The folder '{root_folder}' was not found.")
        return False

    for root, dirs, files in os.walk(root_folder):
        # Skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        level = root.replace(root_folder, '').count(os.sep)
        indent = '    ' * level
        folder_name = os.path.basename(root)
        
        print(f"{indent}📁 {folder_name}/  [{len(files)} files]")
        
        if files:
            extensions = set(os.path.splitext(f)[1].lower() for f in files)
            print(f"{indent}    Extensions: {', '.join(extensions)}")
    
    return True


def check_pairs(root_folder):
    """Verifies that each image has a corresponding label file."""
    print(f"\n{'='*60}")
    print(f"🔗 IMAGE-LABEL PAIR VERIFICATION")
    print(f"{'='*60}")
    
    splits = ['train', 'valid', 'test']
    all_good = True
    
    for split in splits:
        img_path = os.path.join(root_folder, split, 'images')
        lbl_path = os.path.join(root_folder, split, 'labels')
        
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"⚠️  Skipping {split} (folder missing)")
            continue
            
        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_path) 
                     if f.endswith('.txt')}
        
        matched = img_files.intersection(lbl_files)
        orphan_images = img_files - lbl_files
        orphan_labels = lbl_files - img_files
        
        print(f"\n📂 {split.upper()} SET:")
        print(f"   ✅ Matched Pairs: {len(matched)}")
        
        if orphan_images:
            print(f"   ❌ Images without Labels: {len(orphan_images)}")
            print(f"      Examples: {list(orphan_images)[:3]}")
            all_good = False
        else:
            print("   ✨ No orphan images.")

        if orphan_labels:
            print(f"   ❌ Labels without Images: {len(orphan_labels)}")
            print(f"      Examples: {list(orphan_labels)[:3]}")
            all_good = False
        else:
            print("   ✨ No orphan labels.")
    
    return all_good


def analyze_classes(root_folder):
    """Analyzes class distribution across the dataset."""
    print(f"\n{'='*60}")
    print(f"📊 CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    splits = ['train', 'valid', 'test']
    total_class_counts = Counter()
    box_sizes = []
    
    for split in splits:
        lbl_path = os.path.join(root_folder, split, 'labels')
        if not os.path.exists(lbl_path):
            continue
            
        split_counts = Counter()
        
        for filename in os.listdir(lbl_path):
            if not filename.endswith('.txt'):
                continue
                
            filepath = os.path.join(lbl_path, filename)
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            w, h = float(parts[3]), float(parts[4])
                            split_counts[class_id] += 1
                            total_class_counts[class_id] += 1
                            box_sizes.append(w * h)
            except Exception as e:
                print(f"   ⚠️ Error reading {filename}: {e}")
        
        print(f"\n📂 {split.upper()} Distribution:")
        if not split_counts:
            print("   No labels found.")
        else:
            for cls_id in sorted(split_counts.keys()):
                name = CLASS_NAMES.get(cls_id, f"Unknown({cls_id})")
                print(f"   {name:<12}: {split_counts[cls_id]:5d} instances")
    
    # Summary
    print(f"\n{'─'*40}")
    print(f"📊 TOTAL ACROSS ALL SPLITS:")
    total = sum(total_class_counts.values())
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = total_class_counts[cls_id]
        pct = (count / total * 100) if total > 0 else 0
        name = CLASS_NAMES[cls_id]
        print(f"   {name:<12}: {count:5d} instances ({pct:.1f}%)")
    
    # Object size analysis
    if box_sizes:
        print(f"\n{'─'*40}")
        print(f"📏 OBJECT SIZE ANALYSIS:")
        avg_area = np.mean(box_sizes)
        small_objs = sum(1 for x in box_sizes if x < 0.005)
        large_objs = sum(1 for x in box_sizes if x > 0.1)
        
        print(f"   Average Box Area : {avg_area:.4f} (relative to image)")
        print(f"   Small Objects    : {small_objs} ({small_objs/len(box_sizes)*100:.1f}%)")
        print(f"   Large Objects    : {large_objs} ({large_objs/len(box_sizes)*100:.1f}%)")
        
        if avg_area < 0.01:
            print("   💡 Recommendation: Objects are small. Consider 'imgsz=1280' for training.")
        elif avg_area > 0.3:
            print("   💡 Recommendation: Objects are large. 'imgsz=640' or even 416 is ideal.")
        else:
            print("   💡 Recommendation: Standard size objects. 'imgsz=640' is optimal.")
    
    return total_class_counts


def run_full_analysis(root_folder):
    """Runs all analysis functions on the dataset."""
    print(f"\n{'#'*60}")
    print(f"#  YOLO DATASET ANALYZER")
    print(f"#  Dataset: {root_folder}")
    print(f"{'#'*60}")
    
    if not os.path.exists(root_folder):
        print(f"\n❌ Error: Dataset folder '{root_folder}' not found!")
        return
    
    analyze_structure(root_folder)
    pairs_ok = check_pairs(root_folder)
    analyze_classes(root_folder)
    
    # Final verdict
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    if pairs_ok:
        print("🎉 Dataset is ready for training!")
    else:
        print("⚠️ Some issues found. Please review orphan files above.")


if __name__ == "__main__":
    # Get dataset path from command line or use default
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = DEFAULT_ROOT
    
    run_full_analysis(dataset_path)
