import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. SYSTEM CONFIG & CSS ---
st.set_page_config(page_title="SteelSight AI | Industrial Hub", layout="wide")

# Industrial styling and UI polish
st.markdown("""
    <style>
    /* Bigger, readable toggle */
    [data-testid="stCheckbox"] { font-size: 20px; font-weight: bold; margin-bottom: 20px; }
    .stApp { background-color: #0E1117; }
    .log-box { font-family: monospace; font-size: 14px; border: 1px solid #333; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# State Management
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None

# Gemini Config - Fixed model string
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.sidebar.warning("⚠️ Gemini API not ready")

# --- 2. AI LOADER ---

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
st.markdown("---")

with st.sidebar:
    st.header("🎮 Line Command")
    # Big Switch
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    # FIXED: Value 10 now matches an option in the list
    step = st.select_slider("Step Size (Speed)", options=[5, 10, 20, 30, 40], value=10)
    
    status = "🟢 ACTIVE" if engage else "🔴 HALTED"
    st.write(f"Line Status: **{status}**")

# The Viewport
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 📷 Optical Sensor")
    v_raw = st.empty()
with col_b:
    st.markdown("#### 🔬 AI Vision")
    v_viz = st.empty()

# Restore frames when halted
if not engage and st.session_state.f_raw is not None:
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

st.markdown("---")
c_log, c_expert = st.columns([1, 2])

with c_log:
    st.subheader("📋 Production Logs")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with c_expert:
    st.subheader("🤖 Gemini RCA")
    if st.button("✨ ANALYZE DEFECT AT HALT", use_container_width=True):
        if st.session_state.f_raw is not None:
            with st.spinner("Consulting Specialist..."):
                try:
                    img_pill = PIL.Image.fromarray(st.session_state.f_raw)
                    resp = gemini_model.generate_content([
                        "Industrial analysis: Inspect this steel surface slice. Identify defects and machinery fixes.", 
                        img_pill
                    ])
                    st.info(resp.text)
                except Exception as e:
                    st.error(f"Gemini API: {str(e)}")
        else:
            st.warning("Halt the mill on a defect first.")

# --- 4. ENGINE ---

DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

def get_overlay(img, preds):
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(img)
    # Palette: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    for cid, color in pal.items():
        overlay[(mask == cid) & (conf > 0.5)] = color
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

if engage and files:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        raw_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # Process once
        inp = cv2.resize(raw_rgb, (1600, 256))
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        preds = model.predict(inp, verbose=0)[0]
        full_viz = get_overlay(raw_rgb, preds)
        
        # Sliding
        for x in range(st.session_state.ptr_x, 1200, step):
            if not engage:
                st.session_state.ptr_x = x
                st.rerun()

            st.session_state.f_raw = raw_rgb[:, x : x + 400]
            st.session_state.f_viz = full_viz[:, x : x + 400]
            
            # Atomic update to prevent lag
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Detect Red Defects (Class 3)
            if np.any((np.argmax(preds[:, x:x+400], axis=-1) == 2) & (np.max(preds[:, x:x+400], axis=-1) > 0.6)):
                log_entry = {"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(path)}
                if not any(l["File"] == log_entry["File"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(log_entry)
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()