import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. CORE SYSTEM SETUP ---
st.set_page_config(page_title="SteelSight AI | Industrial Control Room", layout="wide")

# Custom CSS for a professional TIET lab appearance
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .main-container { border: 2px solid #343a40; padding: 20px; border-radius: 10px; background-color: #161b22; }
    .stButton>button { height: 3em; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# State Management
if 'op_mode' not in st.session_state: st.session_state.op_mode = 'IDLE' # IDLE, RUNNING, PAUSED
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []

# Gemini AI Initialization
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. AI MODEL LOADER (With Keras 3 Bridge) ---

@st.cache_resource
def load_industrial_model():
    # Mapping custom objects ensures the .keras file loads without TypeErrors
    custom_objects = {
        'dice_coef': lambda y_t, y_p: 1.0, 
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_industrial_model()

# --- 3. LAYOUT STRUCTURE ---

st.title("🏗️ SteelSight AI: Industrial Control Room")
st.caption("Advanced Surface Inspection System | Thapar Institute of Engineering & Technology")
st.markdown("---")

# Sidebar - Operational Controls
with st.sidebar:
    st.header("🎮 Line Command")
    col_start, col_halt = st.columns(2)
    with col_start:
        if st.button("🚀 ENGAGE", use_container_width=True, type="primary"): 
            st.session_state.op_mode = 'RUNNING'
    with col_halt:
        if st.button("⏸️ HALT", use_container_width=True): 
            st.session_state.op_mode = 'PAUSED'
    
    if st.button("🔄 SYSTEM RESET", use_container_width=True):
        st.session_state.op_mode = 'IDLE'
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    belt_speed = st.select_slider("Conveyor Speed (px/step)", options=[10, 20, 30, 40, 50], value=30)
    st.write(f"Current Status: **{st.session_state.op_mode}**")

# Main Dashboard: Dual-Pane Viewing
col_raw, col_mask = st.columns(2)
with col_raw:
    st.info("📷 Optical Sensor: Raw Surface")
    raw_viewport = st.empty()

with col_mask:
    st.info("🔬 AI Vision: Defect Segmentation")
    mask_viewport = st.empty()

# Bottom Section: Diagnostics
st.markdown("---")
col_log, col_expert = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection History")
    log_table = st.empty()

with col_expert:
    st.subheader("🤖 Gemini Root Cause Analysis")
    if st.button("✨ ANALYZE CURRENT FRAME", use_container_width=True):
        if st.session_state.logs:
            st.session_state.op_mode = 'PAUSED'
            with st.spinner("Consulting AI Expert..."):
                last_event = st.session_state.logs[-1]
                analysis_img = PIL.Image.open(last_event['Path'])
                prompt = "Act as a Senior Quality Engineer. Analyze this steel surface defect. What is the likely machinery failure and required fix?"
                response = gemini_model.generate_content([prompt, analysis_img])
                st.success("Analysis Delivered")
                st.write(response.text)
        else:
            st.warning("No defects logged yet. Start the conveyor to begin scanning.")

# --- 4. ENGINE & SIMULATION ---

SAMPLE_PATH = "test_samples"
sample_files = sorted([os.path.join(SAMPLE_PATH, f) for f in os.listdir(SAMPLE_PATH) if f.endswith(('.jpg', '.png'))])

def build_thermal_mask(pred_data):
    """
    Builds a high-contrast segmentation heatmap.
    Uses a 0.5 threshold to keep background clean.
    """
    # pred_data is (256, 1600, 4)
    h, w, c = pred_data.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Colors: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    palette = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    
    # Check each pixel for the highest probability > 0.5
    for i in range(c):
        heatmap[pred_data[:, :, i] > 0.5] = palette[i]
        
    return heatmap

# THE ACTIVE LOOP
if st.session_state.op_mode == 'RUNNING' and sample_files:
    while st.session_state.ptr_idx < len(sample_files):
        img_path = sample_files[st.session_state.ptr_idx]
        
        # Load Image
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # FIX: Remove manual division by 255.0 to prevent double-scaling
        input_tensor = cv2.resize(raw_rgb, (1600, 256))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        
        # AI Inference
        raw_preds = model.predict(input_tensor, verbose=0)[0]
        full_mask_rgb = build_thermal_mask(raw_preds)
        
        # Simulation Slicing
        view_w = 450
        for x in range(st.session_state.ptr_x, 1600 - view_w, belt_speed):
            if st.session_state.op_mode != 'RUNNING':
                st.session_state.ptr_x = x # Save position for Resume
                st.rerun()

            # High-speed display update
            raw_viewport.image(raw_rgb[:, x : x + view_w], use_container_width=True)
            mask_viewport.image(full_mask_rgb[:, x : x + view_w], use_container_width=True)
            
            # Logging Logic
            if np.any(raw_preds[:, x : x + view_w, 2] > 0.5): # Detect Class 3 (Red)
                new_log = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path), "Path": img_path}
                if not any(l["ID"] == new_log["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(new_log)
                log_table.table(st.session_state.logs[-5:])

            time.sleep(0.01) # Small delay for smoother rendering
            
        # Reset and Move to next strip
        st.session_state.ptr_idx += 1
        st.session_state.ptr_x = 0
        if st.session_state.ptr_idx >= len(sample_files):
            st.session_state.ptr_idx = 0 # Loop the conveyor
            
    st.rerun()