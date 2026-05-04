import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Persistent State Management
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'freeze_raw' not in st.session_state: st.session_state.freeze_raw = None
if 'freeze_viz' not in st.session_state: st.session_state.freeze_viz = None

# Gemini Setup - Standard Model ID
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.sidebar.error("Gemini API Configuration Error.")

# --- 2. AI BACKBONE ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@st.cache_resource
def load_mill_model():
    custom_objects = {
        'dice_coef': dice_coef,
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_mill_model()

# --- 3. UI LAYOUT: SIDEBAR & HEADERS ---

st.title("🏗️ SteelSight AI: Industrial Control Room")
st.caption("Advanced Surface Inspection System | Thapar Institute of Engineering & Technology")
st.markdown("---")

with st.sidebar:
    st.header("🎮 Line Control")
    # Toggle switch provides a cleaner "Engaged" state than buttons
    engage_system = st.toggle("🚀 ENGAGE ROLLING MILL", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.freeze_raw = None
        st.session_state.freeze_viz = None
        st.rerun()

    st.markdown("---")
    belt_speed = st.select_slider("Mill Speed (Pixels/Step)", options=[5, 10, 15, 20, 25, 30], value=15)
    
    status_text = "RUNNING" if engage_system else "HALTED"
    status_color = "#28a745" if engage_system else "#dc3545"
    st.markdown(f"Status: <span style='color:{status_color}; font-weight:bold;'>{status_text}</span>", unsafe_allow_html=True)

# Main Viewports
col_raw, col_viz = st.columns(2)
with col_raw:
    st.markdown("#### 📷 Optical Sensor: Raw Surface")
    raw_view = st.empty()
with col_viz:
    st.markdown("#### 🔬 AI Vision: Defect Overlay")
    viz_view = st.empty()

# Persistence: Show the last frame even when halted
if not engage_system and st.session_state.freeze_raw is not None:
    raw_view.image(st.session_state.freeze_raw, use_container_width=True)
    viz_view.image(st.session_state.freeze_viz, use_container_width=True)

# Lower Dashboard
st.markdown("---")
col_log, col_expert = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection History")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with col_expert:
    st.subheader("🤖 AI Expert Analysis")
    if st.button("✨ ANALYZE FROZEN FRAME", use_container_width=True):
        if st.session_state.freeze_raw is not None:
            with st.spinner("Consulting Metallurgical Expert..."):
                try:
                    analysis_pil = PIL.Image.fromarray(st.session_state.freeze_raw)
                    response = gemini_model.generate_content([
                        "Act as a Senior Metallurgy Engineer. This is a steel surface image from the Patiala mill line. "
                        "Identify the defect present and provide the likely root cause and machine fix.", 
                        analysis_pil
                    ])
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")
        else:
            st.warning("Engage the mill and pause on a defect to perform analysis.")

# --- 4. THE PROCESSING ENGINE ---

SAMPLE_DIR = "test_samples"
# Fixed the variable name typo here:
files = sorted([os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(SAMPLE_DIR) else []

def build_vision_overlay(image, probs):
    """Blends mask onto image: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta."""
    mask = np.argmax(probs, axis=-1)
    max_prob = np.max(probs, axis=-1)
    
    overlay = np.zeros_like(image)
    # Mapping probabilities to high-contrast colors
    palette = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    
    for cid, color in palette.items():
        # Confidence thresholding keeps the "Yellow Flood" away
        overlay[(mask == cid) & (max_prob > 0.5)] = color
        
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# The Main Animation Loop
if engage_system and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        
        # Load once per strip
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference (passing raw 0-255 values to internal Rescaling layer)
        input_data = cv2.resize(raw_rgb, (1600, 256))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        preds = model.predict(input_data, verbose=0)[0]
        full_viz_strip = build_vision_overlay(raw_rgb, preds)
        
        # Sliding Window
        win_w = 450
        for x in range(st.session_state.ptr_x, 1600 - win_w, belt_speed):
            # Break if user toggles off
            if not engage_system:
                st.session_state.ptr_x = x # Save position
                st.rerun()

            # Store current slice for persistence
            st.session_state.freeze_raw = raw_rgb[:, x : x + win_w]
            st.session_state.freeze_viz = full_viz_strip[:, x : x + win_w]
            
            # Sync Render
            raw_view.image(st.session_state.freeze_raw, use_container_width=True)
            viz_view.image(st.session_state.freeze_viz, use_container_width=True)
            
            # Log Critical Defect (Class 3: Scratches)
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