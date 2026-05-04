import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf
import keras

# --- 1. INDUSTRIAL THEME & UI POLISH ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Custom CSS to fix the "small toggle" and stabilize the layout
st.markdown("""
    <style>
    /* Make the Toggle Switch Huge */
    .stCheckbox { transform: scale(1.5); margin-left: 20px; }
    div[data-testid="stSidebarUserContent"] { padding-top: 2rem; }
    
    /* Center and Style Viewports */
    .viewport-box { 
        border: 2px solid #3e444d; 
        border-radius: 10px; 
        padding: 5px; 
        background-color: #0d1117; 
    }
    
    /* Button Styling */
    .stButton>button { 
        height: 3.5em; 
        font-weight: bold; 
        background-color: #238636; 
        color: white; 
        border: none;
    }
    .stButton>button:hover { background-color: #2ea043; border: none; }
    </style>
    """, unsafe_allow_html=True)

# State Persistence
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None

# --- 2. FOOLPROOF GEMINI INITIALIZATION ---

@st.cache_resource
def get_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Find the correct model ID dynamically to avoid 404 v1beta errors
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Look for the best available flash model
        target = next((m for m in models if "gemini-1.5-flash" in m), models[0])
        return genai.GenerativeModel(target)
    except Exception as e:
        st.sidebar.error(f"Gemini Init Error: {str(e)}")
        return None

gemini_model = get_gemini_model()

# --- 3. AI MODEL LOADER (Keras 3 Bridge) ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@st.cache_resource
def load_mill_model():
    # Placeholder lambda for focal loss since we only need to load, not train
    custom_objects = {
        'dice_coef': dice_coef,
        'Functional': keras.models.Model,
        'silu': tf.nn.silu,
        'focal_loss_fixed': lambda y_true, y_pred: 0.0
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_mill_model()

# --- 4. CONTROL PANEL ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Mill Line 01 | Precision Surface Inspection | TIET Patiala")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ operational Controls")
    engage = st.toggle("🚀 ENGAGE MILL LINE", value=False)
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    # Corrected the default value to 10 to match an option
    line_step = st.select_slider("Conveyor Speed (Step)", options=[5, 10, 20, 30, 40], value=10)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"Status: <b style='color:{status_clr};'>{'RUNNING' if engage else 'HALTED'}</b>", unsafe_allow_html=True)

# Layout Containers
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 📷 optical Sensor (Raw)")
    v_raw = st.empty()
with col_b:
    st.markdown("#### 🔬 AI Vision (Heatmap)")
    v_viz = st.empty()

# PERSISTENT DISPLAY: Keep images visible during Halt
if not engage and st.session_state.f_raw is not None:
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

st.markdown("---")
col_log, col_expert = st.columns([1, 2])

with col_log:
    st.subheader("📋 Detection Events")
    log_box = st.empty()
    if st.session_state.logs:
        log_box.table(st.session_state.logs[-5:])

with col_expert:
    st.subheader("🤖 AI Expert Root Cause Analysis")
    if st.button("✨ ANALYZE CURRENT DEFECT", use_container_width=True):
        if st.session_state.f_raw is not None and gemini_model:
            with st.spinner("Analyzing steel surface topology..."):
                try:
                    img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                    prompt = "Act as an industrial metallurgy expert. Identify defects in this steel slice and suggest a machinery fix."
                    response = gemini_model.generate_content([prompt, img_pil])
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini Analysis Failed: {str(e)}")
        elif not gemini_model:
            st.error("Gemini is not configured properly.")
        else:
            st.warning("Stop the conveyor on a defect to perform analysis.")

# --- 5. THE SCANNING ENGINE ---

DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

def build_overlay(img, preds):
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(img)
    # Mapping probabilities to high-contrast colors (1:Cyan, 2:Yellow, 3:Red, 4:Magenta)
    palette = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    for cid, color in palette.items():
        overlay[(mask == cid) & (conf > 0.5)] = color
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

if engage and files:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        raw_bgr = cv2.imread(path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Predict once per strip
        input_tensor = cv2.resize(raw_rgb, (1600, 256))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        preds = model.predict(input_tensor, verbose=0)[0]
        full_viz = build_overlay(raw_rgb, preds)
        
        # Sliding Window Loop
        for x in range(st.session_state.ptr_x, 1200, line_step):
            if not engage:
                st.session_state.ptr_x = x # Save precise coordinate
                st.rerun()

            # Store and display current frame
            st.session_state.f_raw = raw_rgb[:, x : x + 400]
            st.session_state.f_viz = full_viz[:, x : x + 400]
            
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Log Scratches (Class 3: Red)
            if np.any((np.argmax(preds[:, x:x+400], axis=-1) == 2) & (np.max(preds[:, x:x+400], axis=-1) > 0.6)):
                entry = {"Time": time.strftime("%H:%M:%S"), "File": os.path.basename(path)}
                if not any(l["File"] == entry["File"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(entry)
                log_box.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        # Move to next strip
        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()