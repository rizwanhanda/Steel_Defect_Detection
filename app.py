import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
from google import genai # Using the new recommended SDK
import tensorflow as tf
import keras

# --- 1. INDUSTRIAL UI SETUP ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Polishing the UI: Large Toggle, Clean Containers, Dark Theme
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    /* Make toggle switch large and readable */
    [data-testid="stCheckbox"] { transform: scale(1.2); font-weight: bold; }
    .viewport-container { border: 1px solid #30363d; border-radius: 8px; padding: 10px; background-color: #161b22; }
    .stButton>button { height: 3.5em; border-radius: 8px; font-weight: 700; background-color: #238636; color: white; }
    </style>
    """, unsafe_allow_html=True)

# State Management for your XPS 15 performance
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None

# New Gemini SDK Initialization
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.sidebar.error("API Key not found in Streamlit Secrets.")

# --- 2. AI BACKBONE ---

@st.cache_resource
def load_mill_model():
    # Mapping 'Functional' handles the Keras 3 version mismatch from your logs
    custom_objects = {
        'dice_coef': lambda y_t, y_p: 1.0, 
        'Functional': keras.models.Model,
        'silu': tf.nn.silu 
    }
    return keras.models.load_model('steel_model_best.keras', custom_objects=custom_objects, compile=False)

model = load_mill_model()

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Precision Surface Inspection | Mill Line 01 (Patiala)")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET"):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    # FIXED: Value 10 now exists in the options list
    step = st.select_slider("Mill Speed (Step Size)", options=[5, 10, 20, 30, 40], value=10)
    
    status_text = "🟢 RUNNING" if engage else "🔴 HALTED"
    st.markdown(f"Status: **{status_text}**")

# Viewport Logic
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Sensor")
    v_raw = st.empty()
with col_right:
    st.markdown("#### 🔬 AI Vision")
    v_viz = st.empty()

# PERSISTENCE: Keep images visible on Halt
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
    st.subheader("🤖 AI Expert RCA")
    if st.button("✨ ANALYZE FROZEN FRAME", use_container_width=True):
        if st.session_state.f_raw is not None:
            with st.spinner("Consulting Specialist..."):
                try:
                    # PIL conversion for the new SDK
                    img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=["As an industrial engineer, analyze this steel defect and suggest a machinery fix.", img_pil]
                    )
                    st.success("Analysis Complete")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")
        else:
            st.warning("Stop the conveyor on a defect to inspect.")

# --- 4. ENGINE ---

SAMPLE_DIR = "test_samples"
files = sorted([os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('.jpg', '.png'))])

def get_overlay(img, preds):
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(img)
    # 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    for cid, color in pal.items():
        overlay[(mask == cid) & (conf > 0.5)] = color
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

if engage and files:
    while st.session_state.ptr_idx < len(files):
        img_path = files[st.session_state.ptr_idx]
        raw_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Inference (No extra scaling needed as model has a Rescaling layer)
        inp = cv2.resize(raw_rgb, (1600, 256))
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        preds = model.predict(inp, verbose=0)[0]
        full_viz = get_overlay(raw_rgb, preds)
        
        # Sliding Loop
        for x in range(st.session_state.ptr_x, 1200, step):
            if not engage:
                st.session_state.ptr_x = x # Lock current X position
                st.rerun()

            # Synchronized update: f_raw and f_viz update in the same millisecond
            st.session_state.f_raw = raw_rgb[:, x : x + 400]
            st.session_state.f_viz = full_viz[:, x : x + 400]
            
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Log Class 3 (Red) Defects
            if np.any((np.argmax(preds[:, x:x+400], axis=-1) == 2) & (np.max(preds[:, x:x+400], axis=-1) > 0.6)):
                entry = {"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path)}
                if not any(l["ID"] == entry["ID"] for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append(entry)
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()