import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="SteelSight AI | Industrial Control Room", layout="wide")

# Custom CSS for an industrial "Dark Mode" look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; }
    .status-box { padding: 10px; border-radius: 5px; border: 1px solid #4B4B4B; background-color: #1A1C23; }
    </style>
    """, unsafe_allow_html=True)

if 'run_state' not in st.session_state: st.session_state.run_state = 'IDLE'
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'current_x' not in st.session_state: st.session_state.current_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []

# API Setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. THE AI BACKBONE ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@st.cache_resource
def load_unet():
    # Bridge for custom training functions and Keras 3 versioning
    custom_objects = {
        'dice_coef': dice_coef,
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_unet()

# --- 3. THE LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.markdown("---")

# Sidebar - Controls
with st.sidebar:
    st.header("🎮 Operational Controls")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🚀 ENGAGE", type="primary"): st.session_state.run_state = 'RUNNING'
    with btn_col2:
        if st.button("⏸️ HALT"): st.session_state.run_state = 'PAUSED'
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.run_state = 'IDLE'
        st.session_state.current_idx = 0
        st.session_state.current_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    line_speed = st.slider("Mill Feed Rate (Step Size)", 10, 80, 30)
    st.info(f"System Status: **{st.session_state.run_state}**")

# Top Row: The Dual Feed
col_feed, col_mask = st.columns(2)
with col_feed:
    st.markdown("<div class='status-box'><b>LIVE OPTICAL FEED: LINE 01 (PATIALA)</b></div>", unsafe_allow_html=True)
    raw_placeholder = st.empty()

with col_mask:
    st.markdown("<div class='status-box'><b>AI ANALYTICS: THERMAL DEFECT MASK</b></div>", unsafe_allow_html=True)
    mask_placeholder = st.empty()

# Bottom Row: Logs and AI RCA
st.markdown("---")
col_log, col_gemini = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection Events")
    log_display = st.empty()

with col_gemini:
    st.subheader("🤖 AI Expert (Root Cause Analysis)")
    if st.button("✨ CONSULT AI EXPERT ON CURRENT DEFECT"):
        if st.session_state.logs:
            st.session_state.run_state = 'PAUSED'
            with st.spinner("Analyzing steel surface topology..."):
                # Analysis Logic
                last_img = st.session_state.logs[-1]['Path']
                raw_img = PIL.Image.open(last_img)
                response = gemini_model.generate_content(["Analyze this industrial steel strip. Identify the defect and suggest a machinery adjustment.", raw_img])
                st.success("Analysis Complete")
                st.markdown(response.text)
        else:
            st.warning("No defects captured in the current session.")

# --- 4. THE CONVEYOR LOGIC ---

SAMPLE_FOLDER = "test_samples"
files = sorted([os.path.join(SAMPLE_FOLDER, f) for f in os.listdir(SAMPLE_FOLDER) if f.endswith(('.jpg', '.png'))])

def get_mask_overlay(mask):
    """Generates the high-contrast heatmap for the separate mask view."""
    h, w = mask.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {1: [0, 255, 255], 2: [255, 255, 0], 3: [255, 0, 0], 4: [255, 0, 255]}
    for cid, color in colors.items():
        heatmap[mask == cid] = color
    return heatmap

# Simulation Execution
if st.session_state.run_state == 'RUNNING' and files:
    while st.session_state.current_idx < len(files):
        img_path = files[st.session_state.current_idx]
        
        # 1. Processing once per strip for speed
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(img, (1600, 256)) / 255.0
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        pred = model.predict(input_data, verbose=0)
        mask = np.argmax(pred[0], axis=-1)
        heatmap_full = get_mask_overlay(mask)
        
        # 2. Smooth Sliding Window Loop
        window_w = 450
        for x in range(st.session_state.current_x, 1600 - window_w, line_speed):
            if st.session_state.run_state != 'RUNNING':
                st.session_state.current_x = x
                st.rerun()

            # Update Placeholders
            raw_placeholder.image(img[:, x:x+window_w], use_column_width=True)
            mask_placeholder.image(heatmap_full[:, x:x+window_w], use_column_width=True)
            
            # Logging Critical Defects (Class 3: Scratches)
            if np.any(mask[:, x:x+window_w] == 3):
                new_entry = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path), "Path": img_path}
                if not any(l["ID"] == new_entry["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(new_entry)
                log_display.table(st.session_state.logs[-5:])

            time.sleep(0.02) # Stable framerate
        
        # Advance to next image
        st.session_state.current_idx += 1
        st.session_state.current_x = 0
        if st.session_state.current_idx >= len(files):
            st.session_state.current_idx = 0 
    
    st.rerun()