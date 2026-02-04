import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="YOLO Behaviour Detection App", layout="wide")

st.title("🚀 YOLO Behaviour Detection Tester")
st.markdown("Test helmet, smoking, and alcohol detection models.")

# Initialize session state for logs and webcam
if 'event_log' not in st.session_state:
    st.session_state.event_log = []
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

# ---------------------------
# CACHED MODEL LOADER
# ---------------------------
@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLO model and caches it to prevent reloading 
    every time a slider changes.
    """
    return YOLO(model_path)

# ---------------------------
# SIDEBAR: SETUP
# ---------------------------
st.sidebar.header("🛠️ Settings")
model_file = st.sidebar.file_uploader("Upload Trained Model (.pt)", type=["pt"])

if model_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    try:
        model = load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()
else:
    st.sidebar.warning("⚠️ Please upload a YOLO model to begin.")
    st.stop()

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Higher values reduce false positives but might miss some detections."
)

# ---------------------------
# CORE LOGIC FUNCTIONS
# ---------------------------
def analyze_events(detections):
    """
    Custom business logic to detect specific events based on 
    your trained classes: ['helmet', 'without_helmet', 'alcohol', 'ciggaret']
    """
    classes = [d['class_name'] for d in detections]
    event = None

    # Logic based on your specific Confusion Matrix
    if "alcohol" in classes:
        event = "⚠️ Alcohol Detected"
    
    # Note: 'ciggaret' spelling matches your dataset labels
    elif "ciggaret" in classes:
        event = "🚭 Smoking Detected"
        
    elif "without_helmet" in classes:
        event = "⛑️ Helmet Violation"
    
    return event

def run_detection(img_bgr, conf_threshold):
    """
    Runs inference on a single frame and returns annotated image + metadata.
    """
    # Run YOLO inference
    device = 0 if torch.cuda.is_available() else 'cpu'
    results = model(img_bgr, conf=conf_threshold, device=device)[0]

    detections = []
    # Plot predictions on the image
    annotated = results.plot()

    # Extract class names and confidence scores
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results.names[cls]
        
        detections.append({
            "class_id": cls,
            "class_name": name,
            "confidence": round(conf, 2)
        })

    # Check for events
    event = analyze_events(detections)
    
    # Log event if found
    if event:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.event_log.append({
            "time": timestamp,
            "event": event,
            "detections": ", ".join([d['class_name'] for d in detections])
        })

    return annotated, detections, event


# ---------------------------
# MAIN INTERFACE
# ---------------------------
input_type = st.sidebar.radio("Select Input Source", ["Image", "Video", "Webcam"])

# --- IMAGE MODE ---
if input_type == "Image":
    st.header("🖼️ Image Analysis")
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, dets, event = run_detection(img, conf_threshold)

        # Convert BGR to RGB for Streamlit display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

        st.subheader("📊 Detection Data")
        st.write(dets)

        if event:
            st.error(f"🚨 ALERT: **{event}**")

# --- VIDEO MODE ---
elif input_type == "Video":
    st.header("🎥 Video Analysis")
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # UI for stopping video
        stop_button = st.button("Stop Processing")

        frame_count = 0
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # OPTIMIZATION: Process only every 3rd frame to reduce lag
            if frame_count % 3 != 0:
                continue

            annotated, dets, event = run_detection(frame, conf_threshold)
            
            # Convert to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, use_container_width=True)

        cap.release()
        st.success("Video processing complete.")

# --- WEBCAM MODE ---
elif input_type == "Webcam":
    st.header("📷 Live Webcam Feed")
    st.caption("Note: Performance depends on your laptop's CPU speed.")

    if st.button("Start/Stop Webcam"):
        st.session_state.run_webcam = not st.session_state.run_webcam

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            
            annotated, _, event = run_detection(frame, conf_threshold)
            
            # Convert to RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, use_container_width=True)
            
        cap.release()


# ---------------------------
# LOGGING SECTION
# ---------------------------
st.markdown("---")
st.subheader("📜 Violation Log")

if st.session_state.event_log:
    # Convert log to DataFrame and show newest first
    df = pd.DataFrame(st.session_state.event_log).iloc[::-1]
    st.dataframe(df, use_container_width=True)

    # CSV Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Report (CSV)",
        csv,
        "violation_report.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.info("No violations detected in this session yet.")