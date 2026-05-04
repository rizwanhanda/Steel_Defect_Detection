import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="SteelSight AI: Industrial Scanner", layout="wide")

# Initialize Gemini 1.5 Flash
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load the Corrected Model File
@st.cache_resource
def load_u_net():
    # Corrected filename per your latest update
    return tf.keras.models.load_model('steel_model_best.keras', compile=False)

model = load_u_net()

# --- 2. IMAGE PROCESSING ENGINE ---

def apply_industrial_overlay(raw_image, mask):
    """Blends defect masks with the raw steel strip for a high-tech glow."""
    overlay = np.zeros_like(raw_image)
    colors = {
        1: [0, 255, 255],   # Cyan: Pitting
        2: [255, 255, 0],   # Yellow: Inclusion
        3: [255, 0, 0],     # Red: Scratches (CRITICAL)
        4: [255, 0, 255]    # Magenta: Patches
    }
    for class_id, color in colors.items():
        overlay[mask == class_id] = color
    
    # 70/30 Blend for visibility of underlying steel texture
    return cv2.addWeighted(raw_image, 0.7, overlay, 0.3, 0)

def get_gemini_analysis(img_path, defect_type):
    """Sends the frozen frame to Gemini for Root Cause Analysis (RCA)."""
    raw_img = PIL.Image.open(img_path)
    prompt = f"""
    Act as a Senior Metallurgical Engineer. A {defect_type} was detected on the production line.
    1. Identify the visual characteristics of this defect.
    2. Suggest a likely machine failure (e.g., roller tension, cooling spray).
    3. Provide one immediate corrective action.
    """
    response = gemini_model.generate_content([prompt, raw_img])
    return response.text

# --- 3. DASHBOARD UI ---

st.title("🏗️ SteelSight AI: Industrial Surface Inspection")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🕹️ Line Controls")
run_belt = st.sidebar.toggle("Activate Conveyor Belt")
speed = st.sidebar.slider("Line Speed (s)", 0.01, 0.2, 0.05)

if "defect_logs" not in st.session_state:
    st.session_state.defect_logs = []

# Main Layout
col1, col2 = st.columns([3, 1])
with col1:
    view_placeholder = st.empty()
    st.caption("Monitoring active steel strips ($256 \times 1600$ resolution)")

with col2:
    st.subheader("📋 Detection Logs")
    log_display = st.empty()
    analyze_btn = st.button("✨ Get AI Expert RCA")

# --- 4. SIMULATION LOOP ---

SAMPLE_DIR = "test_samples"
# Automatically detect images in your new folder
sample_files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('.jpg', '.png'))]

if run_belt:
    for img_path in sample_files:
        if not run_belt: break
        
        # 1. Load Strip
        raw = cv2.imread(img_path)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        
        # 2. Predict (Full Strip at once for Efficiency)
        input_data = cv2.resize(raw, (1600, 256)) / 255.0
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        pred_mask = model.predict(input_data, verbose=0)
        pred_mask = np.argmax(pred_mask[0], axis=-1)
        
        # 3. Apply Visual Overlay
        full_glow_strip = apply_industrial_overlay(raw, pred_mask)
        
        # 4. Rolling Simulation (Sliding Window)
        window_w = 400
        for x in range(0, 1600 - window_w, 20):
            current_view = full_glow_strip[:, x : x + window_w]
            view_placeholder.image(current_view, use_column_width=True)
            
            # Detect Critical Defects (Class 3)
            if np.any(pred_mask[:, x : x + window_w] == 3):
                new_entry = {"Time": time.strftime("%H:%M:%S"), "Type": "SCRATCH", "ID": os.path.basename(img_path)}
                if not any(d['ID'] == new_entry['ID'] for d in st.session_state.defect_logs[-3:]):
                    st.session_state.defect_logs.append(new_entry)
                
            log_display.table(st.session_state.defect_logs[-5:])
            time.sleep(speed)

# 5. Gemini Expert Analysis
if analyze_btn and st.session_state.defect_logs:
    with st.spinner("Analyzing root cause..."):
        last_defect = st.session_state.defect_logs[-1]
        analysis = get_gemini_analysis(os.path.join(SAMPLE_DIR, last_defect["ID"]), last_defect["Type"])
        st.write("### 🤖 Senior Engineer Analysis")
        st.info(analysis)