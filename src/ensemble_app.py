import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import time
import tempfile
from datetime import datetime
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION & STYLING
# ---------------------------
st.set_page_config(page_title="AI Enforcement Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Directories
LOG_DIR = "detections"
IMG_DIR = os.path.join(LOG_DIR, "thumbnails")
os.makedirs(IMG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "log.csv")

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Timestamp", "Event", "Confidence", "Image_Path"]).to_csv(LOG_FILE, index=False)

if 'run_active' not in st.session_state:
    st.session_state.run_active = False

# Rate Limiting for Logging (Prevent spamming CSV)
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = 0

# ---------------------------
# UTILITIES
# ---------------------------
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def validate_event_anatomy(person_box, object_box, event_name):
    """
    Checks if the object is:
    1. Overlapping with the person (IoU > 0.05)
    2. In the correct anatomical region (Head vs Upper Body)
    """
    # 1. Base Overlap Check
    if calculate_iou(person_box, object_box) < 0.01: # Looser IoU, relies more on position
        # Check containment or significant intersection
        px1, py1, px2, py2 = person_box
        ox1, oy1, ox2, oy2 = object_box
        
        # Check if object center is inside person box horizontally
        o_center_x = (ox1 + ox2) / 2
        if not (px1 < o_center_x < px2):
            return False
            
    # 2. Geometric / Anatomy Check
    px1, py1, px2, py2 = person_box
    p_height = py2 - py1
    
    ox1, oy1, ox2, oy2 = object_box
    o_center_y = (oy1 + oy2) / 2
    
    # Relative position from top of person (0.0 = Top, 1.0 = Bottom)
    rel_pos = (o_center_y - py1) / p_height
    
    if event_name in ['helmet', 'without_helmet']:
        # Head Region: Should be in top 33% 
        # Extending slightly to 40% to account for neck/posture
        return 0.0 <= rel_pos <= 0.40
        
    elif event_name in ['ciggaret', 'alcohol']:
        # Upper Body Region: Head to Chest (Top 50%)
        # Cigarettes/Drinks held near mouth/chest
        return 0.0 <= rel_pos <= 0.55
        
    return True

def save_violation(frame, event_name, conf):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"{timestamp}_{event_name.replace(' ', '_')}.jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    
    # Save Image
    cv2.imwrite(img_path, frame)
    
    # Log to CSV
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event_name,
        "Confidence": f"{conf:.2f}",
        "Image_Path": img_path
    }])
    # Append without index
    new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    return True

# ---------------------------
# MODEL LOADER
# ---------------------------
@st.cache_resource
def load_models(use_gpu=True):
    device = 0 if use_gpu and torch.cuda.is_available() else 'cpu'
    models = {}
    try:
        model_path = 'models/yolov8n.pt'
        if not os.path.exists(model_path):
            model_path = 'yolov8n.pt'
        models['standard'] = YOLO(model_path)
        
        if os.path.exists('models/best.pt'):
            models['custom_fast'] = YOLO('models/best.pt')
        if os.path.exists('models/best_colab.pt'):
            models['custom_accurate'] = YOLO('models/best_colab.pt')
    except Exception as e:
        st.error(f"Error: {e}")
        return None, device
    return models, device

# ---------------------------
# CORE PROCESSING LOGIC
# ---------------------------
def analyze_frame(frame, models, device, conf_person, conf_obj, active_events=None):
    """
    Analyzes a single frame and returns the annotated frame, detailed event list, and threat count.
    """
    events = [] # List of dicts: {'name': str, 'conf': float}
    
    # 1. Persons
    results_std = models['standard'](frame, classes=[0], conf=conf_person, device=device, verbose=False)[0]
    persons = [box.xyxy[0].cpu().numpy() for box in results_std.boxes]

    # 2. Violations
    custom_dets = []
    for model_key in ['custom_fast', 'custom_accurate']:
        if model_key in models:
            res = models[model_key](frame, conf=conf_obj, device=device, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                custom_dets.append({
                    'cls': res.names[cls_id], 
                    'box': box.xyxy[0].cpu().numpy(), 
                    'conf': float(box.conf[0])
                })

    # 3. Correlation (Ensemble)
    annotated_frame = frame.copy()
    active_threat_count = 0
    
    for p_box in persons:
        x1, y1, x2, y2 = map(int, p_box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for det in custom_dets:
        obj_box = det['box']
        name = det['cls']
        conf = det['conf']
        
        is_relevant = False
        for p_box in persons:
            # UPDATED: Use Anatomy Logic
            if validate_event_anatomy(p_box, obj_box, name):
                is_relevant = True
                break
        
        if is_relevant:
            label = ""
            color = (0, 0, 255)
            event_entry = None
            
            if name == 'ciggaret':
                if 'Smoking' not in active_events: continue
                label = f"🚭 SMOKING {conf:.2f}"
                event_entry = {"name": "Smoking", "conf": conf}
                active_threat_count += 1
            elif name == 'alcohol':
                if 'Alcohol' not in active_events: continue
                label = f"🍺 ALCOHOL {conf:.2f}"
                event_entry = {"name": "Alcohol", "conf": conf}
                active_threat_count +=1
            elif name == 'without_helmet':
                if 'No Helmet' not in active_events: continue
                label = f"⛑️ NO HELMET {conf:.2f}"
                event_entry = {"name": "No Helmet", "conf": conf}
                active_threat_count +=1
            elif name == 'helmet':
                if 'Helmet' not in active_events: continue
                color = (255, 128, 0)
                label = f"✅ HELMET {conf:.2f}"
            
            if event_entry:
                events.append(event_entry)

            x1, y1, x2, y2 = map(int, obj_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    return annotated_frame, events, active_threat_count

# ---------------------------
# LOGGING HANDLER
# ---------------------------
def handle_logging(frame, events, log_placeholder, img_placeholder=None):
    current_time = time.time()
    
    # 3-Second Cooldown to prevent spam
    if events and (current_time - st.session_state.last_log_time > 3.0):
        # Log the highest confidence event
        top_event = max(events, key=lambda x: x['conf'])
        saved = save_violation(frame, top_event['name'], top_event['conf'])
        
        if saved:
            st.session_state.last_log_time = current_time
            # Toast notification
            st.toast(f"🚨 Logged: {top_event['name']}", icon="📸")
            
            # Show Latest Evidence
            if img_placeholder:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_placeholder.image(img_rgb, caption=f"Last Alert: {top_event['name']}", use_container_width=True)
            
    # Always update the table view
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                log_placeholder.dataframe(
                    df.tail(10).iloc[::-1][['Timestamp', 'Event', 'Confidence']], 
                    use_container_width=True, 
                    hide_index=True
                )
    except Exception as e:
        pass # Handle empty CSV read errors gracefully

# ---------------------------
# LAYOUT & CONTROLS
# ---------------------------
st.markdown("## 🛡️ Sentinel AI: Enforcement Dashboard")

# Top Control Bar
with st.expander("⚙️ System Config", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        use_gpu = st.checkbox("🔥 Enable GPU", value=torch.cuda.is_available())
    with c2:
        conf_person = st.slider("Person Sensitivity", 0.0, 1.0, 0.45)
    with c3:
        conf_obj = st.slider("Event Sensitivity", 0.0, 1.0, 0.30)
    
    frame_skip = st.slider("Frame Splitting (Process every Nth frame)", 0, 25, 0)
        
    st.markdown("##### 🎯 Active Detection Filters")
    active_events = st.multiselect(
        "Select Events to Detect:",
        options=['Smoking', 'Alcohol', 'No Helmet', 'Helmet'],
        default=['Smoking', 'Alcohol', 'No Helmet', 'Helmet']
    )

models, device = load_models(use_gpu)

# Sidebar Input Selector
st.sidebar.header("Source")
input_source = st.sidebar.radio("Input Mode", ["Webcam", "Video File", "Image File"])

# Main Dashboard Grid
col_video, col_stats = st.columns([3, 1])

with col_stats:
    st.markdown("### 📊 Metrics")
    metric_threats = st.empty()
    metric_fps = st.empty()
    
    st.markdown("### 🎮 Control")
    if input_source == "Webcam":
        if not st.session_state.run_active:
            if st.button("▶️ START LIVE", type="primary", use_container_width=True):
                st.session_state.run_active = True
                st.rerun()
        else:
            if st.button("⏹️ STOP", type="secondary", use_container_width=True):
                st.session_state.run_active = False
                st.rerun()
    else:
        st.info(f"Mode: {input_source}")

    st.markdown("### 📝 Alerts")
    
    # Latest Evidence Wdiget
    latest_evidence_placeholder = st.empty()
    
    log_container = st.empty()

with col_video:
    video_placeholder = st.empty()

# ---------------------------
# MAIN LOOP
# ---------------------------
if models:
    # --- WEBCAM MODE ---
    if input_source == "Webcam":
        if st.session_state.run_active:
            cap = cv2.VideoCapture(0)
            prev_time = time.time()
            
            frame_count = 0
            while st.session_state.run_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cam Disconnected")
                    break
                
                frame_count += 1
                if frame_count % (frame_skip + 1) != 0:
                    continue
                
                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Analyze
                final_img, events, threats = analyze_frame(frame, models, device, conf_person, conf_obj, active_events)
                
                # Update UI
                metric_threats.metric("Active Threats", threats)
                metric_fps.metric("FPS", f"{int(fps)}")
                video_placeholder.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Logging
                handle_logging(frame, events, log_container, latest_evidence_placeholder)

            cap.release()
        else:
            video_placeholder.image("https://placehold.co/800x600?text=Live+Feed+Inactive", use_container_width=True)

    # --- IMAGE FILE MODE ---
    elif input_source == "Image File":
        img_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            final_img, events, threats = analyze_frame(frame, models, device, conf_person, conf_obj, active_events)
            
            metric_threats.metric("Detected Threats", threats)
            metric_fps.metric("Mode", "Static")
            video_placeholder.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            if events:
                handle_logging(frame, events, log_container, latest_evidence_placeholder)
                st.sidebar.error("Threats Detected & Logged!")

    # --- VIDEO FILE MODE ---
    elif input_source == "Video File":
        vid_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if vid_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stop_vid = st.sidebar.button("Stop Video")
            
            prev_time = time.time()
            
            frame_count = 0
            while cap.isOpened() and not stop_vid:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % (frame_skip + 1) != 0:
                    continue
                    
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                final_img, events, threats = analyze_frame(frame, models, device, conf_person, conf_obj, active_events)
                
                metric_threats.metric("Active Threats", threats)
                metric_fps.metric("FPS", f"{int(fps)}")
                video_placeholder.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Logging
                handle_logging(frame, events, log_container, latest_evidence_placeholder)

            cap.release()
