import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
from google import genai 
import tensorflow as tf
import keras

# --- 1. SYSTEM CONFIG & INDUSTRIAL THEME ---
st.set_page_config(page_title="SteelSight AI | Industrial Hub", layout="wide")

st.markdown("""
    <style>
    /* Large, high-visibility toggle for industrial use */
    [data-testid="stCheckbox"] { transform: scale(1.6); margin-left: 20px; margin-top: 10px; }
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Industrial Container Styling */
    .expert-response { 
        background-color: #161b22; 
        border-left: 5px solid #238636; 
        padding: 20px; 
        border-radius: 4px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stButton>button { 
        height: 3.8em; 
        font-weight: 700; 
        background-color: #238636; 
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Persistent State Management
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None
if 'last_ai_time' not in st.session_state: st.session_state.last_ai_time = 0

# --- 2. AI BACKBONE & CLIENTS ---

@st.cache_resource
def get_genai_client():
    try:
        # Initializing Gemini 3 Flash Client
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

client = get_genai_client()
MODEL_ID = "gemini-3-flash-preview" 

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

def create_solid_overlay(img, preds):
    """Fixes 'weird masks' using smoothing and morphological closing."""
    # 1. Smooth the raw predictions to eliminate confidence speckling
    preds_smoothed = cv2.GaussianBlur(preds, (5, 5), 0)
    
    mask = np.argmax(preds_smoothed, axis=-1)
    conf = np.max(preds_smoothed, axis=-1)
    
    overlay = np.zeros_like(img)
    # Palette: 0:Cyan, 1:Yellow, 2:Red, 3:Magenta
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    
    kernel = np.ones((5, 5), np.uint8)
    
    for cid, color in pal.items():
        # Create binary class mask
        class_binary = ((mask == cid) & (conf > 0.5)).astype(np.uint8)
        # Morphological Closing: Glues the 'mesh' into a solid blob
        class_binary = cv2.morphologyEx(class_binary, cv2.MORPH_CLOSE, kernel)
        overlay[class_binary == 1] = color
        
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.f_raw = None
        st.session_state.f_viz = None
        st.rerun()

    st.markdown("---")
    step = st.select_slider("Mill Speed (Step Size)", options=[5, 10, 20, 30], value=10)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"Status: <b style='color:{status_clr};'>{'RUNNING' if engage else 'HALTED'}</b>", unsafe_allow_html=True)

# Main Viewports
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Sensor")
    v_raw = st.empty()
with col_right:
    st.markdown("#### 🔬 AI Vision (Solid Masking)")
    v_viz = st.empty()

# Halt Persistence: Display the last frames when conveyor stops
if not engage and st.session_state.f_raw is not None:
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

st.markdown("---")
c_log, c_expert = st.columns([1, 2])

with c_log:
    st.subheader("📋 Detection Log")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with c_expert:
    st.subheader("🤖 AI Expert Root Cause Analysis")
    
    # Cooldown Logic (60s)
    time_since_last = time.time() - st.session_state.last_ai_time
    wait_time = 60 - time_since_last
    
    if engage:
        st.info("⏸️ Halt the conveyor to enable Gemini 3 analysis.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    elif wait_time > 0:
        st.warning(f"Expert is preparing. Please wait {int(wait_time)}s.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    else:
        if st.button("✨ ANALYZE FRAME", use_container_width=True):
            if st.session_state.f_raw is not None and client:
                st.session_state.last_ai_time = time.time()
                with st.spinner("Gemini 3 Flash is performing surface diagnostics..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[
                                "Act as a senior metallurgy engineer. Analyze this steel surface image for defects. Identify class and suggest machinery fix.", 
                                img_pil
                            ]
                        )
                        st.success("Analysis Delivered")
                        st.markdown(f"<div class='expert-response'>{response.text}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini 3 Error: {str(e)}")
            else:
                st.warning("Stop the mill on a defect to perform analysis.")

# --- 4. SCANNING ENGINE ---
DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

if engage and files:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        raw_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # 1. Prediction Block
        inp = cv2.resize(raw_rgb, (1600, 256))
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        preds = model.predict(inp, verbose=0)[0]
        
        # 2. Build Solid Heatmap
        full_viz = create_solid_overlay(raw_rgb, preds)
        
        # 3. Sliding Viewport Loop
        for x in range(st.session_state.ptr_x, 1150, step):
            if not engage:
                st.session_state.ptr_x = x # Save precise stop coordinate
                st.rerun()

            # Store and display current slices
            st.session_state.f_raw = raw_rgb[:, x : x + 450]
            st.session_state.f_viz = full_viz[:, x : x + 450]
            
            # Atomic render to eliminate UI lag
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Real-time Log: Class 3 (Red) Defects
            if np.any((np.argmax(preds[:, x:x+450], axis=-1) == 2) & (np.max(preds[:, x:x+450], axis=-1) > 0.6)):
                if not any(l["File"] == os.path.basename(path) for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append({"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(path)})
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        # Iterate to next strip
        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()