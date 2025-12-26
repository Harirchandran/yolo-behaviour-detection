import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import json

# --- CONFIGURATION ---
REAL_DATASET_DIR = "../dataset"
VISUAL_DATASET_DIR = "../dataset_visualized"
APP_TITLE = "YOLO Dataset Cleaner V2 (Auto-Save Enabled)"
PROGRESS_FILE = "cleaner_progress.json"

CLASS_NAMES = {0: 'Helmet', 1: 'No Helmet', 2: 'Alcohol', 3: 'Ciggaret'}
# ---------------------

class DatasetCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x800")
        
        # State variables
        self.current_split = "train" 
        self.image_list = []
        self.current_index = 0
        self.deleted_count = 0
        
        # Cache for stats
        self.stats = {"train": 0, "valid": 0, "test": 0}
        
        # Load progress if exists
        self.saved_progress = self.load_progress()

        # --- GUI LAYOUT ---
        # 1. Top Control Bar
        self.top_frame = tk.Frame(root, bg="#333", height=50)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_stats = tk.Label(self.top_frame, text="Loading...", fg="white", bg="#333", font=("Arial", 12, "bold"))
        self.lbl_stats.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.btn_frame = tk.Frame(self.top_frame, bg="#333")
        self.btn_frame.pack(side=tk.RIGHT, padx=20)
        
        self.btn_train = tk.Button(self.btn_frame, text="Train (1)", command=lambda: self.change_split("train"))
        self.btn_train.pack(side=tk.LEFT, padx=5)
        self.btn_valid = tk.Button(self.btn_frame, text="Valid (2)", command=lambda: self.change_split("valid"))
        self.btn_valid.pack(side=tk.LEFT, padx=5)
        self.btn_test = tk.Button(self.btn_frame, text="Test (3)", command=lambda: self.change_split("test"))
        self.btn_test.pack(side=tk.LEFT, padx=5)

        # 2. Main Content Area
        self.content_frame = tk.Frame(root, bg="#222")
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(self.content_frame, bg="#000")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.canvas_frame, bg="#000", text="No Image", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(self.content_frame, bg="#444", width=300)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=0)
        self.info_frame.pack_propagate(False)

        self.lbl_filename = tk.Label(self.info_frame, text="File: -", fg="white", bg="#444", font=("Arial", 10, "bold"), wraplength=280)
        self.lbl_filename.pack(pady=20, padx=10, anchor="w")
        
        self.lbl_classes = tk.Label(self.info_frame, text="Classes: -", fg="lightgreen", bg="#444", justify=tk.LEFT, wraplength=280)
        self.lbl_classes.pack(pady=10, padx=10, anchor="w")
        
        self.lbl_path = tk.Label(self.info_frame, text="Path: -", fg="#aaa", bg="#444", font=("Arial", 8), wraplength=280)
        self.lbl_path.pack(pady=10, padx=10, anchor="w")

        self.lbl_status = tk.Label(self.info_frame, text="Ready", fg="cyan", bg="#444", font=("Arial", 14, "bold"))
        self.lbl_status.pack(side=tk.BOTTOM, pady=30)

        # --- BINDINGS ---
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Delete>", self.delete_current_image)
        self.root.bind("1", lambda e: self.change_split("train"))
        self.root.bind("2", lambda e: self.change_split("valid"))
        self.root.bind("3", lambda e: self.change_split("test"))
        
        # Save on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- INITIALIZATION ---
        self.scan_datasets()
        
        # Restore last used split or default to train
        last_split = self.saved_progress.get("last_active_split", "train")
        self.change_split(last_split)

    def load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_progress(self):
        # Save the filename of the current image for the current split
        if self.image_list and self.current_index < len(self.image_list):
            current_file = self.image_list[self.current_index]
            self.saved_progress[self.current_split] = current_file
        
        self.saved_progress["last_active_split"] = self.current_split
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.saved_progress, f)

    def on_close(self):
        self.save_progress()
        self.root.destroy()

    def scan_datasets(self):
        self.data_map = {"train": [], "valid": [], "test": []}
        
        if not os.path.exists(VISUAL_DATASET_DIR):
            messagebox.showerror("Error", f"Folder '{VISUAL_DATASET_DIR}' not found.")
            self.root.destroy()
            return

        for split in ["train", "valid", "test"]:
            path = os.path.join(VISUAL_DATASET_DIR, split, "images")
            if os.path.exists(path):
                files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                self.data_map[split] = files
                self.stats[split] = len(files)
        
        self.update_stats_label()

    def update_stats_label(self):
        text = (f"Train: {len(self.data_map['train'])} | "
                f"Valid: {len(self.data_map['valid'])} | "
                f"Test: {len(self.data_map['test'])} | "
                f"DELETED SESSION: {self.deleted_count}")
        self.lbl_stats.config(text=text)

    def change_split(self, split_name):
        # Save progress of old split before switching
        self.save_progress()
        
        self.current_split = split_name
        self.image_list = self.data_map[split_name]
        
        # Try to restore index for this split
        last_file = self.saved_progress.get(split_name)
        if last_file and last_file in self.image_list:
            self.current_index = self.image_list.index(last_file)
            print(f"Resuming {split_name} at {last_file}")
        else:
            self.current_index = 0
        
        # Update buttons UI
        for btn, name in [(self.btn_train, "train"), (self.btn_valid, "valid"), (self.btn_test, "test")]:
            if name == split_name:
                btn.config(bg="lightblue", fg="black")
            else:
                btn.config(bg="#555", fg="white")
        
        self.load_current_image()

    def load_current_image(self):
        if not self.image_list:
            self.image_label.config(image="", text="No images in this split.")
            self.lbl_filename.config(text="File: None")
            self.lbl_classes.config(text="Classes: None")
            return

        if self.current_index >= len(self.image_list):
            self.current_index = len(self.image_list) - 1
        
        filename = self.image_list[self.current_index]
        
        # Paths
        vis_img_path = os.path.join(VISUAL_DATASET_DIR, self.current_split, "images", filename)
        
        try:
            pil_img = Image.open(vis_img_path)
            w_canvas = self.canvas_frame.winfo_width()
            h_canvas = self.canvas_frame.winfo_height()
            
            if w_canvas < 10: w_canvas, h_canvas = 800, 600
                
            img_w, img_h = pil_img.size
            ratio = min(w_canvas/img_w, h_canvas/img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.tk_img, text="")
        except Exception as e:
            self.image_label.config(text=f"Error: {e}", image="")

        self.lbl_filename.config(text=f"File: {filename}\n({self.current_index + 1}/{len(self.image_list)})")
        
        real_label_path = os.path.join(REAL_DATASET_DIR, self.current_split, "labels", os.path.splitext(filename)[0] + ".txt")
        class_info = "No Label File"
        if os.path.exists(real_label_path):
            try:
                with open(real_label_path, 'r') as f:
                    lines = f.readlines()
                cls_ids = [int(line.split()[0]) for line in lines if line.strip()]
                cls_names = [CLASS_NAMES.get(c, str(c)) for c in cls_ids]
                class_info = "\n".join(cls_names)
            except: pass
        
        self.lbl_classes.config(text=f"Classes Found:\n{class_info}")
        self.lbl_path.config(text=real_label_path)
        self.lbl_status.config(text="Viewing", fg="cyan")

    def next_image(self, event=None):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def delete_current_image(self, event=None):
        if not self.image_list: return
        filename = self.image_list[self.current_index]
        basename = os.path.splitext(filename)[0]
        
        paths = [
            os.path.join(VISUAL_DATASET_DIR, self.current_split, "images", filename),
            os.path.join(REAL_DATASET_DIR, self.current_split, "images", filename),
            os.path.join(REAL_DATASET_DIR, self.current_split, "labels", basename + ".txt")
        ]
        
        for p in paths:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        
        self.deleted_count += 1
        self.lbl_status.config(text="DELETED!", fg="red")
        del self.image_list[self.current_index]
        self.data_map[self.current_split] = self.image_list
        self.update_stats_label()
        
        if self.current_index >= len(self.image_list) and self.current_index > 0:
            self.current_index -= 1
        self.load_current_image()
        self.root.focus_set()
        
        # Auto-save after delete so you don't lose progress if crash
        self.save_progress()

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCleanerApp(root)
    root.mainloop()