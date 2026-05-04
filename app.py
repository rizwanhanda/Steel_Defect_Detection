import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. DASHBOARD CONFIG ---
st.set_page_config(page_title="SteelSight AI | Industrial Control Room", layout="wide")

# State Management
if 'op_mode' not in st.session_state: st.session_state.op_mode = 'IDLE'
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []

# Gemini AI Setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. THE AI BACKBONE ---

@st.cache_resource
def load_steel_model():
    # Mapping custom objects ensures the .keras file loads correctly
    custom_objects = {
        'dice_coef': lambda y_t, y_p: 1.0, 
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_steel_model()

# --- 3. LAYOUT & UI ---

st.title("🏗️ SteelSight AI: Industrial Control Room")
st.markdown("---")

with st.sidebar:
    st.header("🎮 operational Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ENGAGE", use_container_width=True, type="primary"): st.session_state.op_mode = 'RUNNING'
    with c2:
        if st.button("⏸️ HALT", use_container_width=True): st.session_state.op_mode = 'PAUSED'
    
    if st.button("🔄 SYSTEM RESET", use_container_width=True):
        st.session_state.op_mode = 'IDLE'
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    belt_speed = st.select_slider("Line Speed (Pixels/Step)", options=[10, 20, 30, 40, 50], value=30)
    st.info(f"System: **{st.session_state.op_mode}**")

# Dual Viewports
col_raw, col_vision = st.columns(2)
with col_raw:
    st.caption("📷 Optical Sensor: Raw Surface")
    raw_view = st.empty()

with col_vision:
    st.caption("🔬 AI Vision: Image + Mask Overlay")
    vision_view = st.empty()

st.markdown("---")
col_log, col_rca = st.columns([1, 2])
with col_log:
    st.subheader("📋 Detection History")
    log_area = st.empty()

with col_rca:
    st.subheader("🤖 Gemini Root Cause Analysis")
    if st.button("✨ ANALYZE CURRENT DEFECT", use_container_width=True):
        if st.session_state.logs:
            st.session_state.op_mode = 'PAUSED'
            with st.spinner("Analyzing steel surface topology..."):
                last_img = st.session_state.logs[-1]['Path']
                raw_img = PIL.Image.open(last_img)
                response = gemini_model.generate_content(["Act as a metallurgical expert. Analyze this industrial steel strip for defects.", raw_img])
                st.info(response.text)

# --- 4. ENGINE & SIMULATION ---

SAMPLE_PATH = "test_samples"
files = sorted([os.path.join(SAMPLE_PATH, f) for f in os.listdir(SAMPLE_PATH) if f.endswith(('.jpg', '.png'))])

def create_blended_overlay(image, pred_probs):
    """Blends the mask directly onto the image for the right-side view."""
    mask = np.argmax(pred_probs, axis=-1)
    overlay = np.zeros_like(image)
    
    # Industrial Palette: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    colors = {1: [0, 255, 255], 2: [255, 255, 0], 3: [255, 0, 0], 4: [255, 0, 255]}
    
    for cid, color in colors.items():
        overlay[mask == cid] = color
    
    # 70% Raw Image + 30% Thermal Mask
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

if st.session_state.op_mode == 'RUNNING' and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        
        # Load
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Predict once for the strip
        # Resizing to 1600x256 without 255 division as scaling is in model
        input_data = cv2.resize(raw_rgb, (1600, 256))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        raw_preds = model.predict(input_data, verbose=0)[0]
        # Pre-calculate the entire blended vision strip
        vision_strip = create_blended_overlay(raw_rgb, raw_preds)
        
        # Sliding Window View
        window_w = 450
        for x in range(st.session_state.ptr_x, 1600 - window_w, belt_speed):
            if st.session_state.op_mode != 'RUNNING':
                st.session_state.ptr_x = x
                st.rerun()

            # Update side-by-side feeds
            raw_view.image(raw_rgb[:, x : x + window_w], use_container_width=True)
            vision_view.image(vision_strip[:, x : x + window_w], use_container_width=True)
            
            # Log Class 3 (Scratches)
            if np.any(np.argmax(raw_preds[:, x : x + window_w], axis=-1) == 3):
                new_log = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path), "Path": img_path}
                if not any(l["ID"] == new_log["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(new_log)
                log_area.table(st.session_state.logs[-5:])

            time.sleep(0.01) # Ultra-low sleep for maximum smoothness
            
        st.session_state.ptr_idx += 1
        st.session_state.ptr_x = 0
        if st.session_state.ptr_idx >= len(files): st.session_state.ptr_idx = 0
    st.rerun()