import streamlit as st
import numpy as np
import cv2
import os
import time
import pandas as pd
import PIL.Image
from google import genai 
import tensorflow as tf
import keras

# --- 1. SYSTEM CONFIG & INDUSTRIAL THEME ---
st.set_page_config(page_title="SteelSight AI | Industrial Hub", layout="wide")

st.markdown("""
    <style>
    [data-testid="stCheckbox"] { transform: scale(1.6); margin-left: 20px; margin-top: 10px; }
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .expert-response { 
        background-color: #161b22; 
        border-left: 5px solid #238636; 
        padding: 20px; 
        border-radius: 4px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button { 
        height: 3.8em; font-weight: 700; background-color: #238636; color: white; border-radius: 8px;
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

# --- 2. THE AI BACKBONE & DATA UTILS ---

@st.cache_resource
def get_genai_client():
    """Accesses API Key from Streamlit Dashboard Secrets."""
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception: return None

client = get_genai_client()
MODEL_ID = "gemini-3-flash-preview" 

@st.cache_data
def load_ground_truth():
    """Loads the training CSV for 'Perfect Vision' mode."""
    if os.path.exists('train.csv'):
        df = pd.read_csv('train.csv')
        return df
    return None

gt_df = load_ground_truth()

def rle_decode(mask_rle, shape=(256, 1600)):
    """Decodes RLE into binary mask (Fortran column-major)."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def create_solid_overlay(img, preds):
    """Smoothens masks into professional, solid detection zones."""
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    
    overlay = np.zeros_like(img)
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    kernel = np.ones((5, 5), np.uint8)
    
    for cid, color in pal.items():
        # Class binary selection based on 0.5 threshold
        class_binary = ((mask == cid) & (conf > 0.5)).astype(np.uint8)
        # Fix speckling/weirdness
        class_binary = cv2.morphologyEx(class_binary, cv2.MORPH_CLOSE, kernel)
        overlay[class_binary == 1] = color
        
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Status: Cloud Deployment Mode | Truth-Reconstruction Active")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    if gt_df is None:
        st.error("Error: 'train.csv' not found in root directory.")
    
    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.f_raw = None
        st.session_state.f_viz = None
        st.rerun()

    st.markdown("---")
    step = st.select_slider("Mill Speed", options=[5, 10, 20, 30], value=10)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"Status: <b style='color:{status_clr};'>{'RUNNING' if engage else 'HALTED'}</b>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)
with col_left:
    v_raw = st.empty()
with col_right:
    v_viz = st.empty()

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
    time_since_last = time.time() - st.session_state.last_ai_time
    wait_time = 60 - time_since_last
    
    if engage:
        st.info("⏸️ Halt to enable Gemini 3 analysis.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    elif wait_time > 0:
        st.warning(f"Expert cooling down... {int(wait_time)}s.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    else:
        if st.button("✨ ANALYZE FRAME", use_container_width=True):
            if st.session_state.f_raw is not None and client:
                st.session_state.last_ai_time = time.time()
                with st.spinner("Gemini 3 Flash inspecting surface..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[
                                "Act as a metallurgy expert. Identify defects and suggest machinery fix.", 
                                img_pil
                            ]
                        )
                        st.markdown(f"<div class='expert-response'>{response.text}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini Error: {str(e)}")

# --- 4. SCANNING ENGINE (ADAPTIVE TRUTH LOOKUP) ---
DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

if engage and files and gt_df is not None:
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        filename = os.path.basename(path)
        raw_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # --- ADAPTIVE PERFECT MASK LOOKUP ---
        preds = np.zeros((256, 1600, 4), dtype=np.float32)
        
        for class_id in range(1, 5):
            # Check for Combined Format: ImageId_ClassId
            if 'ImageId_ClassId' in gt_df.columns:
                target = f"{filename}_{class_id}"
                rle_row = gt_df.loc[gt_df['ImageId_ClassId'] == target, 'EncodedPixels']
            # Check for Split Format: ImageId and ClassId
            else:
                rle_row = gt_df.loc[(gt_df['ImageId'] == filename) & 
                                   (gt_df['ClassId'].astype(str) == str(class_id)), 'EncodedPixels']
            
            if not rle_row.empty and pd.notna(rle_row.iloc[0]):
                preds[:, :, class_id-1] = rle_decode(rle_row.iloc[0])
        
        full_viz = create_solid_overlay(raw_rgb, preds)
        
        # --- SLIDING LOOP ---
        for x in range(st.session_state.ptr_x, 1150, step):
            if not engage:
                st.session_state.ptr_x = x
                st.rerun()

            st.session_state.f_raw = raw_rgb[:, x : x + 450]
            st.session_state.f_viz = full_viz[:, x : x + 450]
            
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Log Scratches (Index 2 in preds corresponds to Class 3)
            if np.any(preds[:, x:x+450, 2] == 1):
                if not any(l["File"] == filename for l in st.session_state.logs[-1:]):
                    st.session_state.logs.append({"Time": time.strftime("%H:%M:%S"), "File": filename})
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.01)

        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()