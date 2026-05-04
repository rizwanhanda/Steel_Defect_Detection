import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. INDUSTRIAL THEME & CONFIG ---
st.set_page_config(page_title="SteelSight AI | TIET Control Room", layout="wide")

# Persistent State Management
if 'op_mode' not in st.session_state: st.session_state.op_mode = 'IDLE'
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'last_frame_raw' not in st.session_state: st.session_state.last_frame_raw = None
if 'last_frame_viz' not in st.session_state: st.session_state.last_frame_viz = None

# Gemini Initialization (Pinned to latest stable flash)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- 2. AI MODEL LOADER (Keras 3 Bridge) ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@st.cache_resource
def load_industrial_model():
    custom_objects = {
        'dice_coef': dice_coef,
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_industrial_model()

# --- 3. UI LAYOUT ---

st.title("🏗️ SteelSight AI: Industrial Control Room")
st.caption("Advanced Surface Inspection System | Thapar Institute of Engineering & Technology")
st.markdown("---")

with st.sidebar:
    st.header("🎮 Operational Command")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 ENGAGE", type="primary", use_container_width=True): 
            st.session_state.op_mode = 'RUNNING'
    with col_b:
        if st.button("⏸️ HALT", use_container_width=True): 
            st.session_state.op_mode = 'PAUSED'
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.op_mode = 'IDLE'
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    line_speed = st.select_slider("Line Speed (Step Size)", options=[5, 10, 20, 30, 40, 50], value=20)
    st.info(f"System Status: **{st.session_state.op_mode}**")

# Dual Viewports
col_raw, col_viz = st.columns(2)
with col_raw:
    st.markdown("#### 📷 Optical Sensor: Raw Surface")
    raw_viewport = st.empty()

with col_viz:
    st.markdown("#### 🔬 AI Vision: Image + Mask Overlay")
    viz_viewport = st.empty()

# Persistent display logic: If paused or idle, show the last captured frames
if st.session_state.last_frame_raw is not None:
    raw_viewport.image(st.session_state.last_frame_raw, use_container_width=True)
    viz_viewport.image(st.session_state.last_frame_viz, use_container_width=True)

st.markdown("---")
col_log, col_rca = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection History")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with col_rca:
    st.subheader("🤖 Gemini Root Cause Analysis")
    if st.button("✨ ANALYZE FROZEN FRAME", use_container_width=True):
        if st.session_state.last_frame_raw is not None:
            with st.spinner("Consulting AI Expert..."):
                try:
                    # Convert the current raw frame slice to a PIL Image for Gemini
                    analysis_pil = PIL.Image.fromarray(st.session_state.last_frame_raw)
                    prompt = "Act as a Senior Quality Engineer. Analyze this industrial steel strip for defects. Identify the class and suggest machinery adjustment."
                    response = gemini_model.generate_content([prompt, analysis_pil])
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")
        else:
            st.warning("No frame captured. Run the conveyor to identify defects.")

# --- 4. THE ENGINE ---

SAMPLE_PATH = "test_samples"
files = sorted([os.path.join(SAMPLE_PATH, f) for f in os.listdir(SAMPLE_PATH) if f.endswith(('.jpg', '.png'))])

def generate_blended_view(image, pred_probs):
    """Blends mask onto image with high contrast."""
    mask = np.argmax(pred_probs, axis=-1)
    overlay = np.zeros_like(image)
    # 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    colors = {1: [0, 255, 255], 2: [255, 255, 0], 3: [255, 0, 0], 4: [255, 0, 255]}
    for cid, color in colors.items():
        overlay[mask == cid] = color
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

if st.session_state.op_mode == 'RUNNING' and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        
        # Load and Prepare
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Model Inference
        input_data = cv2.resize(raw_rgb, (1600, 256))
        # Note: No division by 255 because of the model's internal scaling layer
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        raw_preds = model.predict(input_data, verbose=0)[0]
        blended_full = generate_blended_view(raw_rgb, raw_preds)
        
        # Sliding Window View
        window_w = 450
        for x in range(st.session_state.ptr_x, 1600 - window_w, line_speed):
            if st.session_state.op_mode != 'RUNNING':
                st.session_state.ptr_x = x # Save current position
                st.rerun()

            # Slice and Store in session state for persistence
            st.session_state.last_frame_raw = raw_rgb[:, x : x + window_w]
            st.session_state.last_frame_viz = blended_full[:, x : x + window_w]
            
            # Render
            raw_viewport.image(st.session_state.last_frame_raw, use_container_width=True)
            viz_viewport.image(st.session_state.last_frame_viz, use_container_width=True)
            
            # Log Scratches (Class 3)
            if np.any(np.argmax(raw_preds[:, x : x + window_w], axis=-1) == 3):
                new_entry = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path)}
                if not any(l["ID"] == new_entry["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(new_entry)
                log_area.table(st.session_state.logs[-5:])

            time.sleep(0.01)
            
        st.session_state.ptr_idx += 1
        st.session_state.ptr_x = 0
        if st.session_state.ptr_idx >= len(files): st.session_state.ptr_idx = 0
    st.rerun()