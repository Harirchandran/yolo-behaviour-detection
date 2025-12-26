import cv2
import os
import shutil
import numpy as np

# --- CONFIGURATION ---
INPUT_ROOT = '../dataset'
OUTPUT_ROOT = '../dataset_visualized'

# Define your class names exactly as they are in your data.yaml
CLASS_NAMES = ['helmet', 'no_helmet', 'alcohol', 'ciggaret']

# Define distinct colors for each class (B, G, R format)
COLORS = [
    (0, 255, 0),   # Green (helmet)
    (0, 0, 255),   # Red (no_helmet)
    (255, 0, 0),   # Blue (alcohol)
    (0, 255, 255)  # Yellow (ciggaret)
]
# ---------------------

def draw_yolo_boxes(img, label_path, class_names, colors):
    """Reads YOLO label file and draws boxes on the image."""
    h_img, w_img, _ = img.shape
    
    if not os.path.exists(label_path):
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue

        cls_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        # --- Convert YOLO normalized coordinates to Pixels ---
        x1 = int((x_center - width / 2) * w_img)
        y1 = int((y_center - height / 2) * h_img)
        x2 = int((x_center + width / 2) * w_img)
        y2 = int((y_center + height / 2) * h_img)

        # Get color and name
        color = colors[cls_id % len(colors)]
        label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw Text with background for readability
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1) # Filled rectangle behind text
        # Use white or black text color depending on background brightness for contrast
        text_color = (0,0,0) if np.mean(color) > 127 else (255,255,255) 
        cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img

def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input folder '{INPUT_ROOT}' does not exist.")
        return

    print(f"--- Starting Visualization Genertion ---")
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

    splits = ['train', 'valid', 'test']

    for split in splits:
        src_img_dir = os.path.join(INPUT_ROOT, split, 'images')
        src_lbl_dir = os.path.join(INPUT_ROOT, split, 'labels')
        
        dst_img_dir = os.path.join(OUTPUT_ROOT, split, 'images')

        if not os.path.exists(src_img_dir):
            print(f"Skipping {split} (source images not found)")
            continue

        # Create destination folders (we don't need labels folder in output, just images)
        os.makedirs(dst_img_dir, exist_ok=True)

        image_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_images = len(image_files)
        print(f"\nProcessing {split.upper()} split: {total_images} images...")

        for i, filename in enumerate(image_files):
            # 1. Read Image
            img_path = os.path.join(src_img_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            # 2. Find Label
            basename = os.path.splitext(filename)[0]
            label_path = os.path.join(src_lbl_dir, basename + ".txt")

            # 3. Draw boxes
            visualized_img = draw_yolo_boxes(img, label_path, CLASS_NAMES, COLORS)

            # 4. Save to destination
            dst_path = os.path.join(dst_img_dir, filename)
            cv2.imwrite(dst_path, visualized_img)

            # Print progress every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == total_images:
                 print(f"  [Progress] {i+1}/{total_images} done")

    print(f"\n✅ Finished! Output saved to root folder: '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    # Ensure opencv is installed: pip install opencv-python
    main()