import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import glob
import json

# --- CONFIGURATION ---
DATASET_DIR = "../dataset"
PROGRESS_FILE = "editor_progress.json"

# Define your classes and their specific colors/Hotkeys
# Format: ID: (Name, Color, Hotkey_Char)
CLASSES = {
    0: ("Helmet", "lime", "q"),
    1: ("No Helmet", "red", "w"),
    2: ("Alcohol", "cyan", "e"),
    3: ("Ciggaret", "yellow", "r")
}
# ---------------------

class YOLOLabelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Ultimate Editor (Draw: Left Drag | Delete Box: Right Click | Del Img: Del)")
        self.root.geometry("1400x900")
        
        # State
        self.current_split = "train"
        self.image_list = []
        self.current_index = 0
        self.current_class_id = 0 # Default to Helmet
        self.rectangles = [] # Store canvas object IDs
        self.scale_factor = 1.0
        
        # Load Progress
        self.saved_progress = self.load_progress()
        
        # --- UI LAYOUT ---
        # Top Bar
        self.top_frame = tk.Frame(root, bg="#222", height=60)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Split Buttons
        self.btn_train = tk.Button(self.top_frame, text="Train", bg="lightblue", command=lambda: self.change_split("train"))
        self.btn_train.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_valid = tk.Button(self.top_frame, text="Valid", bg="#555", fg="white", command=lambda: self.change_split("valid"))
        self.btn_valid.pack(side=tk.LEFT, padx=5, pady=10)
        self.btn_test = tk.Button(self.top_frame, text="Test", bg="#555", fg="white", command=lambda: self.change_split("test"))
        self.btn_test.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Class Selector Display
        self.lbl_active_class = tk.Label(self.top_frame, text=f"ACTIVE: {CLASSES[0][0]}", bg="lime", fg="black", font=("Arial", 12, "bold"), width=25)
        self.lbl_active_class.pack(side=tk.LEFT, padx=30, pady=10)
        
        self.lbl_info = tk.Label(self.top_frame, text="Hotkeys: Q=Helmet, W=NoHelmet, E=Alcohol, R=Ciggaret", bg="#222", fg="#aaa")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        # Main Canvas
        self.canvas_frame = tk.Frame(root, bg="#333")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # --- BINDINGS ---
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Delete>", self.delete_entire_image)
        
        # Class Hotkeys
        for cls_id, data in CLASSES.items():
            self.root.bind(data[2], lambda event, c=cls_id: self.set_class(c))
            
        # Mouse Interactions
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click) # Right click to delete box
        self.root.bind("<Configure>", self.on_resize) # Window resize

        # Start
        self.scan_dataset()
        last_split = self.saved_progress.get("last_split", "train")
        self.change_split(last_split)

    # --- CORE LOGIC ---
    
    def scan_dataset(self):
        self.data_map = {"train": [], "valid": [], "test": []}
        for split in ["train", "valid", "test"]:
            path = os.path.join(DATASET_DIR, split, "images")
            if os.path.exists(path):
                self.data_map[split] = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])

    def change_split(self, split):
        self.save_current_labels() # Auto-save before switch
        self.save_progress()
        
        self.current_split = split
        self.image_list = self.data_map[split]
        
        # Restore index
        last_file = self.saved_progress.get(split)
        if last_file and last_file in self.image_list:
            self.current_index = self.image_list.index(last_file)
        else:
            self.current_index = 0
            
        # Update UI colors
        for btn, s in [(self.btn_train, "train"), (self.btn_valid, "valid"), (self.btn_test, "test")]:
            btn.config(bg="lightblue" if s == split else "#555", fg="black" if s == split else "white")
            
        self.load_image()

    def load_image(self):
        self.canvas.delete("all")
        self.rectangles = []
        
        if not self.image_list:
            self.canvas.create_text(400, 300, text="No Images Found", fill="white")
            return

        filename = self.image_list[self.current_index]
        img_path = os.path.join(DATASET_DIR, self.current_split, "images", filename)
        lbl_path = os.path.join(DATASET_DIR, self.current_split, "labels", os.path.splitext(filename)[0] + ".txt")
        
        self.root.title(f"Editing: {filename} ({self.current_index+1}/{len(self.image_list)})")

        # Load and Resize Image
        try:
            pil_img = Image.open(img_path)
            self.orig_w, self.orig_h = pil_img.size
            
            # Calculate Scale
            canv_w = self.canvas.winfo_width()
            canv_h = self.canvas.winfo_height()
            if canv_w < 10: canv_w, canv_h = 1000, 700
            
            ratio = min(canv_w / self.orig_w, canv_h / self.orig_h)
            self.scale_factor = ratio
            
            new_w = int(self.orig_w * ratio)
            new_h = int(self.orig_h * ratio)
            
            self.tk_img = ImageTk.PhotoImage(pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS))
            
            # Draw Image centered
            self.off_x = (canv_w - new_w) // 2
            self.off_y = (canv_h - new_h) // 2
            
            self.canvas.create_image(self.off_x, self.off_y, anchor=tk.NW, image=self.tk_img)
            
            # Load Labels
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        self.draw_yolo_box(cls, cx, cy, w, h)
                        
        except Exception as e:
            print(f"Error: {e}")

    def draw_yolo_box(self, cls, cx, cy, w, h):
        # Convert YOLO to Canvas Coordinates
        pixel_w = w * self.orig_w * self.scale_factor
        pixel_h = h * self.orig_h * self.scale_factor
        center_x = cx * self.orig_w * self.scale_factor + self.off_x
        center_y = cy * self.orig_h * self.scale_factor + self.off_y
        
        x1 = center_x - pixel_w/2
        y1 = center_y - pixel_h/2
        x2 = center_x + pixel_w/2
        y2 = center_y + pixel_h/2
        
        color = CLASSES.get(cls, ("Unknown", "white", ""))[1]
        
        # Create Rectangle Object
        rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=("box", str(cls)))
        
        # Store metadata in tags or separate list? separate list is easier for saving
        self.rectangles.append({"id": rect_id, "cls": cls})

    def save_current_labels(self):
        if not self.image_list: return
        
        filename = self.image_list[self.current_index]
        lbl_path = os.path.join(DATASET_DIR, self.current_split, "labels", os.path.splitext(filename)[0] + ".txt")
        
        lines = []
        for rect in self.rectangles:
            coords = self.canvas.coords(rect["id"])
            x1, y1, x2, y2 = coords
            
            # Convert back to YOLO
            # Remove offset
            x1 -= self.off_x
            x2 -= self.off_x
            y1 -= self.off_y
            y2 -= self.off_y
            
            # Scale down to original image size
            x1 /= self.scale_factor
            x2 /= self.scale_factor
            y1 /= self.scale_factor
            y2 /= self.scale_factor
            
            # Calculate Center and Width/Height
            w = (x2 - x1)
            h = (y2 - y1)
            cx = x1 + w/2
            cy = y1 + h/2
            
            # Normalize (0-1)
            norm_cx = cx / self.orig_w
            norm_cy = cy / self.orig_h
            norm_w = w / self.orig_w
            norm_h = h / self.orig_h
            
            # Constraint check
            norm_cx = max(0, min(1, norm_cx))
            norm_cy = max(0, min(1, norm_cy))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            lines.append(f"{rect['cls']} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
        with open(lbl_path, 'w') as f:
            f.writelines(lines)
            
    # --- INTERACTIONS ---
    
    def set_class(self, cls_id):
        self.current_class_id = cls_id
        name, color, _ = CLASSES[cls_id]
        self.lbl_active_class.config(text=f"ACTIVE: {name}", bg=color)

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # Create temporary box
        color = CLASSES[self.current_class_id][1]
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=color, width=2, dash=(2,2))

    def on_mouse_drag(self, event):
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        # Finalize the box
        self.canvas.delete(self.current_rect)
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        
        # Ignore tiny clicks
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return
            
        # Draw permanent box
        color = CLASSES[self.current_class_id][1]
        rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=("box", str(self.current_class_id)))
        
        self.rectangles.append({"id": rect_id, "cls": self.current_class_id})

    def on_right_click(self, event):
        # Find closest object
        item = self.canvas.find_closest(event.x, event.y)
        if not item: return
        
        # Check if it's one of our boxes
        tags = self.canvas.gettags(item)
        if "box" in tags:
            self.canvas.delete(item)
            # Remove from logic list
            self.rectangles = [r for r in self.rectangles if r["id"] != item[0]]

    def delete_entire_image(self, event):
        if not self.image_list: return
        filename = self.image_list[self.current_index]
        basename = os.path.splitext(filename)[0]
        
        # Delete files
        try:
            os.remove(os.path.join(DATASET_DIR, self.current_split, "images", filename))
            lbl = os.path.join(DATASET_DIR, self.current_split, "labels", basename + ".txt")
            if os.path.exists(lbl): os.remove(lbl)
            
            # UI Update
            del self.image_list[self.current_index]
            if self.current_index >= len(self.image_list):
                self.current_index -= 1
            self.load_image()
            self.save_progress()
        except Exception as e:
            print(e)

    def next_image(self, event=None):
        self.save_current_labels() # SAVE BEFORE MOVING
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self, event=None):
        self.save_current_labels() # SAVE BEFORE MOVING
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
            
    def on_resize(self, event):
        # Debounce or just reload current image to fit new size
        if event.widget == self.root:
            pass # Too heavy to re-render on every pixel drag, let's keep it simple

    def save_progress(self):
        if self.image_list:
            self.saved_progress[self.current_split] = self.image_list[self.current_index]
        self.saved_progress["last_split"] = self.current_split
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.saved_progress, f)

    def load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    return json.load(f)
            except: pass
        return {}

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOLabelEditor(root)
    root.mainloop()