import os
import glob

# --- CONFIGURATION ---
ROOT_FOLDER = '../dataset'
# Set this to False when you are ready to actually rename files!
DRY_RUN = False 

# Your class names from the YAML
CLASS_NAMES = {
    0: 'helmet',
    1: 'no_helmet',
    2: 'alcohol',
    3: 'ciggaret'
}

def get_class_string(label_path):
    """Reads the label file and returns a string describing its content."""
    if not os.path.exists(label_path):
        return "empty"

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        unique_classes = sorted(list(set([int(line.split()[0]) for line in lines if line.strip()])))
        
        if not unique_classes:
            return "empty"
        
        # Map IDs to names
        names = [CLASS_NAMES.get(cls, 'unknown') for cls in unique_classes]
        
        # Construct the suffix
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]}-{names[1]}"
        else:
            return "multi" # Too many names, keep it short
            
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return "error"

def rename_dataset():
    splits = ['train', 'valid', 'test']
    
    print(f"--- Starting Smart Rename (Dry Run: {DRY_RUN}) ---")
    
    for split in splits:
        img_dir = os.path.join(ROOT_FOLDER, split, 'images')
        lbl_dir = os.path.join(ROOT_FOLDER, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        # Get all images (sort them to ensure consistent ordering)
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nProcessing {split.upper()} ({len(images)} files)...")
        
        for idx, filename in enumerate(images):
            # Get paths
            old_img_path = os.path.join(img_dir, filename)
            
            # Find corresponding label
            basename = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            old_lbl_name = basename + ".txt"
            old_lbl_path = os.path.join(lbl_dir, old_lbl_name)
            
            # Determine content string
            content_str = get_class_string(old_lbl_path)
            
            # Create NEW filenames
            # Format: split_number_content.jpg (e.g., train_001_helmet.jpg)
            new_basename = f"{split}_{idx+1:04d}_{content_str}"
            
            new_img_name = new_basename + ext
            new_lbl_name = new_basename + ".txt"
            
            new_img_path = os.path.join(img_dir, new_img_name)
            new_lbl_path = os.path.join(lbl_dir, new_lbl_name)
            
            # Execution
            if DRY_RUN:
                # Print just the first 3 to show example
                if idx < 3: 
                    print(f"   [Preview] {filename}  -->  {new_img_name}")
            else:
                # Actual Rename
                os.rename(old_img_path, new_img_path)
                if os.path.exists(old_lbl_path):
                    os.rename(old_lbl_path, new_lbl_path)

    if DRY_RUN:
        print("\n✅ Dry run finished. Check the examples above.")
        print(">> Change 'DRY_RUN = False' in the script and run again to apply changes.")
    else:
        print("\n✅ All files renamed successfully!")

if __name__ == "__main__":
    rename_dataset()