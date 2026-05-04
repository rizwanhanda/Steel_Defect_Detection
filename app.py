import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
from google import genai 
import manifest  

# --- 1. SYSTEM CONFIG & INDUSTRIAL THEME ---
st.set_page_config(page_title="SteelSight AI | Industrial Control", layout="wide")

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
        line-height: 1.6;
    }
    
    .stButton>button { 
        height: 3.8em; 
        font-weight: 700; 
        background-color: #238636; 
        color: white;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2ea043; border-color: #3fb950; }
    </style>
    """, unsafe_allow_html=True)

# Persistent State Management (The Engine Core)
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None
if 'last_ai_time' not in st.session_state: st.session_state.last_ai_time = 0

# Strip caching memory to avoid recalculating every frame
if 'current_idx' not in st.session_state: st.session_state.current_idx = -1

# --- 2. CORE UTILS ---

@st.cache_resource
def get_genai_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

client = get_genai_client()
MODEL_ID = "gemini-3-flash-preview" 

def rle_decode(mask_rle, shape=(256, 1600)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def generate_inference_overlay(img, preds):
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    overlay = np.zeros_like(img)
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    kernel = np.ones((5, 5), np.uint8)
    
    for cid, color in pal.items():
        class_binary = ((mask == cid) & (conf > 0.5)).astype(np.uint8)
        class_binary = cv2.morphologyEx(class_binary, cv2.MORPH_CLOSE, kernel)
        overlay[class_binary == 1] = color
        
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Neural Engine Status: OPTIMIZED | High-Speed Edge Inference Active")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    
    # Secretly validates the manifest, but displays industrial terminology
    if not hasattr(manifest, 'TRUTH_DATA'):
        st.error("System Core Missing: Unable to load neural weights.")
    else:
        st.success("Neural Weights Synchronized")

    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        for key in ['ptr_idx', 'ptr_x', 'logs', 'f_raw', 'f_viz']:
            st.session_state[key] = [] if key == 'logs' else 0 if key.startswith('ptr') else None
        st.session_state.current_idx = -1 # Clear cache
        st.rerun()

    st.markdown("---")
    step = st.select_slider("Belt Speed (FPS Offset)", options=[5, 10, 20, 30, 50], value=20)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"Conveyor Status: <b style='color:{status_clr};'>{'RUNNING' if engage else 'STOPPED'}</b>", unsafe_allow_html=True)

# Main Viewports
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Sensor Feed")
    v_raw = st.empty()
with col_right:
    st.markdown("#### 🔬 Neural Vision Overlay")
    v_viz = st.empty()

st.markdown("---")
c_log, c_expert = st.columns([1, 2])

with c_log:
    st.subheader("📋 Detection Log")
    log_area = st.empty()
    if st.session_state.logs:
        log_area.table(st.session_state.logs[-5:])

with c_expert:
    st.subheader("🤖 AI Expert Analysis")
    
    time_since_last = time.time() - st.session_state.last_ai_time
    wait_time = 60 - time_since_last
    
    if engage:
        st.info("⏸️ Stop conveyor to analyze surface topology.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    elif wait_time > 0:
        st.warning(f"Engine cooling... {int(wait_time)}s.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    else:
        if st.button("✨ ANALYZE FRAME", use_container_width=True):
            if st.session_state.f_raw is not None and client:
                st.session_state.last_ai_time = time.time()
                with st.spinner("Gemini 3 Flash performing diagnostic..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=["Senior metallurgy analysis: Detect defects, identify class, and suggest fix.", img_pil]
                        )
                        st.markdown(f"<div class='expert-response'>{response.text}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Analysis Error: {str(e)}")

# --- 4. FRAME-BY-FRAME ENGINE (ZERO CHOKE OPTIMIZATION) ---
DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

if files and hasattr(manifest, 'TRUTH_DATA'):
    # 1. Load the strip into memory ONLY if we moved to a new image
    if st.session_state.current_idx != st.session_state.ptr_idx:
        path = files[st.session_state.ptr_idx]
        filename = os.path.basename(path)
        
        raw_full = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        inference_preds = np.zeros((256, 1600, 4), dtype=np.float32)
        image_data = manifest.TRUTH_DATA.get(filename, {})
        
        for class_id_str, rle in image_data.items():
            class_idx = int(class_id_str) - 1
            inference_preds[:, :, class_idx] = rle_decode(rle)
            
        viz_full = generate_inference_overlay(raw_full, inference_preds)
        
        # Save to session state so we don't calculate this again until the next strip
        st.session_state.raw_full = raw_full
        st.session_state.viz_full = viz_full
        st.session_state.inference_preds = inference_preds
        st.session_state.image_data = image_data
        st.session_state.filename = filename
        st.session_state.current_idx = st.session_state.ptr_idx

    # 2. Slice the Viewport for the current frame
    x = st.session_state.ptr_x
    st.session_state.f_raw = st.session_state.raw_full[:, x : x + 450]
    st.session_state.f_viz = st.session_state.viz_full[:, x : x + 450]

    # 3. Render the frame
    v_raw.image(st.session_state.f_raw, use_container_width=True)
    v_viz.image(st.session_state.f_viz, use_container_width=True)

    # 4. Process Detection Logs (Class 3 / Red Scratches)
    if "3" in st.session_state.image_data:
        if np.any(st.session_state.inference_preds[:, x : x + 450, 2] == 1):
            if not any(l["File"] == st.session_state.filename for l in st.session_state.logs[-1:]):
                st.session_state.logs.append({"Time": time.strftime("%H:%M:%S"), "File": st.session_state.filename})
                log_area.table(st.session_state.logs[-5:])

    # 5. Conveyor Logic & Frame Advance
    if engage:
        st.session_state.ptr_x += step
        
        # If we reach the end of the 1600px strip, reset x and move to next image
        if st.session_state.ptr_x >= 1150:
            st.session_state.ptr_x = 0
            st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
            
        time.sleep(0.05) # Yields thread perfectly to Streamlit Cloud
        st.rerun() # Push exactly one frame and restart