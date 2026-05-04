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

# --- 1. SYSTEM CONFIG & INDUSTRIAL THEME ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Custom CSS for Large Controls and Professional Appearance
st.markdown("""
    <style>
    /* Make Toggle Switch Huge */
    .stCheckbox { transform: scale(1.6); margin-left: 30px; margin-top: 15px; }
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* Industrial Viewport Containers */
    .viewport-box { 
        border: 2px solid #30363d; 
        border-radius: 12px; 
        background-color: #161b22; 
        padding: 10px;
    }
    
    /* Expert Analysis Box */
    .expert-box {
        background-color: #1c2128;
        border-left: 6px solid #238636;
        padding: 20px;
        border-radius: 4px;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .stButton>button { height: 4em; font-weight: bold; font-size: 1.1em; }
    </style>
    """, unsafe_allow_html=True)

# Persistent State Management
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None
if 'last_call' not in st.session_state: st.session_state.last_call = 0

# --- 2. FOOLPROOF GEMINI DISCOVERY ---

@st.cache_resource
def init_gemini():
    """Dynamically finds the correct model ID to solve 404 errors."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Fetch all models supported for generation
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority: standard flash -> latest flash -> any flash -> first available
        target = "gemini-1.5-flash"
        # Check if the API wants the 'models/' prefix or not based on discovery
        best_match = next((m for m in available_models if target in m), available_models[0])
        
        return genai.GenerativeModel(best_match)
    except Exception as e:
        st.sidebar.error(f"Gemini Discovery Error: {str(e)}")
        return None

gemini_model = init_gemini()

# --- 3. AI BACKBONE (U-Net) ---

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

# --- 4. MAIN INTERFACE ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Mill Line 01 | Precision Surface Inspection | TIET Patiala")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    # Big Sticky Toggle
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.f_raw = None
        st.session_state.f_viz = None
        st.rerun()

    st.markdown("---")
    step = st.select_slider("Mill Speed (Step Size)", options=[5, 10, 15, 20, 30, 40], value=10)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"**Line Status:** <span style='color:{status_clr};'>{ 'RUNNING' if engage else 'HALTED'}</span>", unsafe_allow_html=True)

# Layout: Synced Viewports
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Sensor: Raw Surface")
    v_raw = st.empty()
with col_right:
    st.markdown("#### 🔬 AI Vision: Defect Segmentation")
    v_viz = st.empty()

# Persistent Frame Display: Ensures images stay during HALT
if not engage and st.session_state.f_raw is not None:
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

# Diagnostics Row
st.markdown("---")
c_log, c_expert = st.columns([1, 2])

with c_log:
    st.subheader("📋 Detection Events")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with c_expert:
    st.subheader("🤖 AI Expert: Root Cause Analysis")
    
    # Cooldown Logic to prevent 429 Quota Exceeded errors
    cooldown = 60 - (time.time() - st.session_state.last_call)
    if cooldown > 0:
        st.warning(f"System cooling down. Please wait {int(cooldown)}s before next AI analysis.")
        st.button("✨ ANALYZE CURRENT DEFECT", disabled=True)
    else:
        if st.button("✨ ANALYZE CURRENT DEFECT", use_container_width=True):
            if st.session_state.f_raw is not None and gemini_model:
                st.session_state.last_call = time.time()
                with st.spinner("Consulting Metallurgical Specialist..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        prompt = "Act as a Senior Metallurgy Engineer. Analyze this steel surface for defects. Identify class and machine fix."
                        response = gemini_model.generate_content([prompt, img_pil])
                        st.markdown(f"<div class='expert-box'>{response.text}</div>", unsafe_allow_html=True)
                    except exceptions.ResourceExhausted:
                        st.error("Quota reached. Wait 60s.")
                    except Exception as e:
                        st.error(f"Analysis Failed: {str(e)}")
            else:
                st.warning("Stop the conveyor on a defect to perform analysis.")

# --- 5. THE SCANNING ENGINE ---

DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

def get_overlay(img, preds):
    """Syncs segmentation with industrial colors."""
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(img)
    # 0:Cyan, 1:Yellow, 2:Red, 3:Magenta
    palette = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    for cid, color in palette.items():
        overlay[(mask == cid) & (conf > 0.5)] = color
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

if engage and files:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        raw_bgr = cv2.imread(path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference (Model handles internal rescaling)
        inp = cv2.resize(raw_rgb, (1600, 256))
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        preds = model.predict(inp, verbose=0)[0]
        full_viz = get_overlay(raw_rgb, preds)
        
        # Slicing Window Loop
        for x in range(st.session_state.ptr_x, 1150, step):
            if not engage:
                st.session_state.ptr_x = x # Save precise stop point
                st.rerun()

            st.session_state.f_raw = raw_rgb[:, x : x + 450]
            st.session_state.f_viz = full_viz[:, x : x + 450]
            
            # Atomic UI update: Updates both views in one browser repaint
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Logging Logic for Class 3 (Red Scratches)
            if np.any((np.argmax(preds[:, x:x+450], axis=-1) == 2) & (np.max(preds[:, x:x+450], axis=-1) > 0.6)):
                entry = {"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(path)}
                if not any(l["File"] == entry["File"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(entry)
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        # Iterate to next strip
        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()