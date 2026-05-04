import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
from google.api_core import exceptions
import tensorflow as tf
import keras

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="SteelSight AI | Demo Stable", layout="wide")

# Persistent State
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None
if 'last_ai_time' not in st.session_state: st.session_state.last_ai_time = 0

# --- 2. THE AI BACKBONE (REDUCED QUOTA USAGE) ---

@st.cache_resource
def get_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # HARD-PINNED: We remove the list_models() loop to save quota
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return None

gemini_model = get_gemini_model()

@st.cache_resource
def load_mill_model():
    custom_objects = {
        'dice_coef': lambda y_t, y_p: 1.0, 
        'Functional': keras.models.Model,
        'silu': tf.nn.silu,
        'focal_loss_fixed': lambda y_t, y_p: 0.0
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_mill_model()

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    step = st.select_slider("Speed", options=[5, 10, 20, 30], value=10)

col_left, col_right = st.columns(2)
with col_left:
    v_raw = st.empty()
with col_right:
    v_viz = st.empty()

# Persistent Render
if not engage and st.session_state.f_raw is not None:
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

st.markdown("---")
c_log, c_expert = st.columns([1, 2])

with c_log:
    st.subheader("📋 Detection Log")
    if st.session_state.logs:
        st.table(st.session_state.logs[-5:])

with c_expert:
    st.subheader("🤖 AI Expert Analysis")
    
    # RATE LIMIT UI: 60-second lockout
    time_since_last = time.time() - st.session_state.last_ai_time
    wait_time = 60 - time_since_last
    
    if engage:
        st.info("⏸️ Halt the conveyor to enable AI Expert analysis.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    elif wait_time > 0:
        st.warning(f"Quota cooling down... {int(wait_time)}s remaining.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    else:
        if st.button("✨ ANALYZE FRAME", use_container_width=True):
            if st.session_state.f_raw is not None:
                st.session_state.last_ai_time = time.time()
                with st.spinner("Consulting Specialist..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        response = gemini_model.generate_content([
                            "Act as a metallurgy expert. Identify defects in this steel strip and suggest a machinery fix.", 
                            img_pil
                        ])
                        st.success("Analysis Delivered")
                        st.write(response.text)
                    except exceptions.ResourceExhausted:
                        st.error("Google Quota Hit. Please wait 60s.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# --- 4. ENGINE ---
DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

if engage and files:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        raw_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # Inference
        inp = cv2.resize(raw_rgb, (1600, 256))
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        preds = model.predict(inp, verbose=0)[0]
        
        # Overlay Logic
        mask = np.argmax(preds, axis=-1)
        conf = np.max(preds, axis=-1)
        overlay = np.zeros_like(raw_rgb)
        pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
        for cid, color in pal.items():
            overlay[(mask == cid) & (conf > 0.5)] = color
        full_viz = cv2.addWeighted(raw_rgb, 0.7, overlay, 0.3, 0)
        
        # Sliding
        for x in range(st.session_state.ptr_x, 1150, step):
            if not engage:
                st.session_state.ptr_x = x
                st.rerun()

            st.session_state.f_raw = raw_rgb[:, x : x + 450]
            st.session_state.f_viz = full_viz[:, x : x + 450]
            
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Log Red Scratches
            if np.any((np.argmax(preds[:, x:x+450], axis=-1) == 2) & (np.max(preds[:, x:x+450], axis=-1) > 0.6)):
                if not any(l["File"] == os.path.basename(path) for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append({"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(path)})
            
            time.sleep(0.01)

        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()