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

# CSS to fix the "small toggle" and add industrial styling
st.markdown("""
    <style>
    /* Scale up the toggle switch */
    [data-testid="stCheckbox"] { transform: scale(1.5); padding-left: 20px; }
    .stToggle { margin-top: 10px; margin-bottom: 20px; }
    
    /* Industrial containers */
    .viewport-box { border: 2px solid #343a40; border-radius: 10px; padding: 15px; background-color: #1a1c23; }
    .stButton>button { height: 4em; font-size: 18px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# State Persistence
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'freeze_raw' not in st.session_state: st.session_state.freeze_raw = None
if 'freeze_viz' not in st.session_state: st.session_state.freeze_viz = None

# Gemini Initialization - Explicit Model Prefix
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Using 'models/' prefix to resolve v1beta 404 errors
    gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
except Exception:
    st.sidebar.error("Gemini API connection failed.")

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

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Mill Line 01 | Thapar Institute of Engineering & Technology")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    # Big toggle for Engaging the system
    is_running = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.freeze_raw = None
        st.session_state.freeze_viz = None
        st.rerun()

    st.markdown("---")
    step_size = st.select_slider("Conveyor Step Size (px)", options=[5, 10, 20, 30, 40], value=15)
    
    st.markdown(f"**Current Status:** {'🟢 RUNNING' if is_running else '🔴 HALTED'}")

# Dual Visualizer Section
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Feed")
    raw_viewport = st.empty()
with col_right:
    st.markdown("#### 🔬 AI Vision")
    viz_viewport = st.empty()

# Persistent Render for Halt/Idle
if not is_running and st.session_state.freeze_raw is not None:
    raw_viewport.image(st.session_state.freeze_raw, use_container_width=True)
    viz_viewport.image(st.session_state.freeze_viz, use_container_width=True)

st.markdown("---")
col_log, col_expert = st.columns([1, 2])

with col_log:
    st.subheader("📋 System Logs")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with col_expert:
    st.subheader("🤖 Root Cause Analysis")
    if st.button("✨ ANALYZE FROZEN FRAME WITH GEMINI", use_container_width=True):
        if st.session_state.freeze_raw is not None:
            with st.spinner("Analyzing surface topology..."):
                try:
                    # Convert frozen array to PIL for Gemini
                    pil_img = PIL.Image.fromarray(st.session_state.freeze_raw)
                    prompt = "Act as an industrial metallurgy expert. Identify any defects in this steel image and suggest the machinery fix."
                    response = gemini_model.generate_content([prompt, pil_img])
                    st.success("Analysis Complete")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")
        else:
            st.warning("Stop the conveyor on a defect to run AI analysis.")

# --- 4. ENGINE & ANIMATION ---

SAMPLE_DIR = "test_samples"
files = sorted([os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('.jpg', '.png'))])

def generate_overlay(image, preds):
    """Syncs mask mapping to industrial standards."""
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(image)
    # Colors: 1:Cyan, 2:Yellow, 3:Red (Critical), 4:Magenta
    palette = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    for cid, color in palette.items():
        overlay[(mask == cid) & (conf > 0.5)] = color
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# The Main Processing Loop
if is_running and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        
        # Strip Processing
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference
        input_data = cv2.resize(raw_rgb, (1600, 256))
        # Note: Model includes internal rescaling logic
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        pred_probs = model.predict(input_data, verbose=0)[0]
        full_viz_strip = generate_overlay(raw_rgb, pred_probs)
        
        # Sliding Window Animation
        win_w = 450
        for x in range(st.session_state.ptr_x, 1600 - win_w, step_size):
            if not is_running:
                st.session_state.ptr_x = x # Save position for Resume
                st.rerun()

            # Update the freeze-frame state immediately
            st.session_state.freeze_raw = raw_rgb[:, x : x + win_w]
            st.session_state.freeze_viz = full_viz_strip[:, x : x + win_w]
            
            # Synchronized UI Update
            raw_viewport.image(st.session_state.freeze_raw, use_container_width=True)
            viz_viewport.image(st.session_state.freeze_viz, use_container_width=True)
            
            # Log Detection (Class 3: Scratches)
            if np.any((np.argmax(pred_probs[:, x : x + win_w], axis=-1) == 2) & (np.max(pred_probs[:, x : x + win_w], axis=-1) > 0.6)):
                entry = {"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(img_path)}
                if not any(l["File"] == entry["File"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(entry)
                log_area.table(st.session_state.logs[-5:])

            time.sleep(0.01)
            
        # Reset for next image strip
        st.session_state.ptr_idx += 1
        st.session_state.ptr_x = 0
        if st.session_state.ptr_idx >= len(files): st.session_state.ptr_idx = 0
    st.rerun()