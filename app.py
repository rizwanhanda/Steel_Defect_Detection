import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Persistent Session States
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'freeze_raw' not in st.session_state: st.session_state.freeze_raw = None
if 'freeze_viz' not in st.session_state: st.session_state.freeze_viz = None

# Gemini Fix: Reverting to the standard model identifier
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.sidebar.error("Gemini API Key missing or invalid.")

# --- 2. AI BACKBONE ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@st.cache_resource
def load_mill_model():
    # Bridge for custom training metrics and Keras 3 versioning
    custom_objects = {
        'dice_coef': dice_coef,
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_mill_model()

# --- 3. UI LAYOUT & SIDEBAR ---

st.title("🏗️ SteelSight AI: Industrial Control Room")
st.caption("Precision Surface Monitoring System | Mill Line 01")
st.markdown("---")

with st.sidebar:
    st.header("🎮 Line Control")
    # Using a toggle for "sticky" state - avoids the button highlighting issue
    engage_system = st.toggle("🚀 ENGAGE ROLLING MILL", value=False)
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    belt_speed = st.select_slider("Mill Speed (Step Size)", options=[5, 10, 20, 30, 40], value=20)
    
    status_color = "green" if engage_system else "red"
    st.markdown(f"Current Status: <b style='color:{status_color};'>{'RUNNING' if engage_system else 'HALTED'}</b>", unsafe_allow_html=True)

# Main Feed Containers (Synced side-by-side)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 📷 Optical Sensor")
    raw_feed = st.empty()
with col_b:
    st.markdown("#### 🔬 AI Analytics")
    viz_feed = st.empty()

# Restore freeze frame if halted
if not engage_system and st.session_state.freeze_raw is not None:
    raw_feed.image(st.session_state.freeze_raw, use_container_width=True)
    viz_feed.image(st.session_state.freeze_viz, use_container_width=True)

st.markdown("---")
col_log, col_gemini = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection Log")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with col_gemini:
    st.subheader("🤖 AI Expert RCA")
    if st.button("✨ ANALYZE CURRENT DEFECT", use_container_width=True):
        if st.session_state.freeze_raw is not None:
            with st.spinner("Processing metallurgical data..."):
                try:
                    analysis_pil = PIL.Image.fromarray(st.session_state.freeze_raw)
                    response = gemini_model.generate_content([
                        "Act as a metallurgy expert. This is a slice of a steel strip from a production line. "
                        "Analyze any visible defects and suggest a likely mechanical cause.", 
                        analysis_pil
                    ])
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini Error: {str(e)}")
        else:
            st.warning("Engage the mill and halt at a defect to perform analysis.")

# --- 4. THE PROCESSING ENGINE ---

SAMPLE_DIR = "test_samples"
files = sorted([os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_FOLDER) if f.endswith(('.jpg', '.png'))]) if os.path.exists(SAMPLE_DIR) else []

def generate_full_overlay(image, probs):
    """Syncs mask mapping with industrial colors: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta."""
    # probs is (256, 1600, 4)
    mask = np.argmax(probs, axis=-1)
    # Only show colors if confidence is high, else black background
    max_prob = np.max(probs, axis=-1)
    
    overlay = np.zeros_like(image)
    colors = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    
    for cid, color in colors.items():
        # Mask indices are 0-3 for the 4 defect classes
        overlay[(mask == cid) & (max_prob > 0.5)] = color
        
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# Main Animation Loop
if engage_system and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        
        # Load and Inference once per strip
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocessing (Model has its own rescaling)
        input_data = cv2.resize(raw_rgb, (1600, 256))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        preds = model.predict(input_data, verbose=0)[0]
        full_viz_strip = generate_full_overlay(raw_rgb, preds)
        
        # Slicing Window
        win_w = 450
        for x in range(st.session_state.ptr_x, 1600 - win_w, belt_speed):
            if not engage_system:
                st.session_state.ptr_x = x # Save progress
                st.rerun()

            # Capture current slice
            st.session_state.freeze_raw = raw_rgb[:, x : x + win_w]
            st.session_state.freeze_viz = full_viz_strip[:, x : x + win_w]
            
            # Sync Render
            raw_feed.image(st.session_state.freeze_raw, use_container_width=True)
            viz_feed.image(st.session_state.freeze_viz, use_container_width=True)
            
            # Auto-log Critical (Class 3: Red)
            if np.any((np.argmax(preds[:, x : x + win_w], axis=-1) == 2) & (np.max(preds[:, x : x + win_w], axis=-1) > 0.6)):
                entry = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path)}
                if not any(l["ID"] == entry["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(entry)
                log_area.table(st.session_state.logs[-5:])

            time.sleep(0.01)
            
        st.session_state.ptr_idx += 1
        st.session_state.ptr_x = 0
        if st.session_state.ptr_idx >= len(files): st.session_state.ptr_idx = 0
    st.rerun()