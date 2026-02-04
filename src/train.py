from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # 1. Initialize the Model
    # We use 'yolov8n.pt' (Nano) for speed. 
    # If you have a powerful GPU (RTX 3060 or better), you can change this to 'yolov8s.pt' (Small) for better accuracy.
    print("--- Initializing Model ---")
    model = YOLO('models/yolov8n.pt') 

    # 2. Check Device (Optional visual check)
    device_name = 'GPU (Cuda)' if torch.cuda.is_available() else 'CPU'
    print(f"--- Training on: {device_name} ---")

    # 3. Start Training
    # These parameters are tuned based on your 'final_verification.py' output.
    model.train(
        data='dataset/data.yaml', 
        epochs=100,                        # 100 is ideal. If it stops improving, it will stop early automatically (patience).
        imgsz=640,                        # Confirmed by your object size analysis.
        batch=16,                         # Safe batch size to avoid "Out of Memory" errors.
        name='final_helmet_run',          # The name of the folder where results will be saved.
        workers=0,                        # MANDATORY for Windows to prevent crashing.
        patience=10,                      # If accuracy doesn't improve for 10 epochs, stop early to save time.
        exist_ok=True,                    # Overwrite existing folder if you run it twice.
        
        # Augmentation settings (Standard YOLOv8 defaults are usually best, but we ensure they are active)
        mosaic=1.0,                       # Helps learn small objects like 'ciggaret' by stitching images together.
        device=0,                         # Force GPU usage
    )

    print("--- Training Complete! ---")