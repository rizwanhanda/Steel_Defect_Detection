import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
from google import genai 
import manifest  # This is your new secret weapon

# --- 1. SYSTEM CONFIG & INDUSTRIAL THEME ---
st.set_page_config(page_title="SteelSight AI | Industrial Hub", layout="wide")

st.markdown("""
    <style>
    /* Industrial UX Enhancements */
    [data-testid="stCheckbox"] { transform: scale(1.6); margin-left: 20px; margin-top: 10px; }
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    .expert-response { 
        background-color: #161b22; 
        border-left: 5px solid #238636; 
        padding: 20px; 
        border-radius: 4px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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

# Persistent State Management
if 'ptr_idx' not in st.session_state: st.session_state.ptr_idx = 0
if 'ptr_x' not in st.session_state: st.session_state.ptr_x = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'f_raw' not in st.session_state: st.session_state.f_raw = None
if 'f_viz' not in st.session_state: st.session_state.f_viz = None
if 'last_ai_time' not in st.session_state: st.session_state.last_ai_time = 0

# --- 2. AI BACKBONE & DATA UTILS ---

@st.cache_resource
def get_genai_client():
    """Initializes the Gemini 3 Flash Client using Streamlit Cloud Secrets."""
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

client = get_genai_client()
MODEL_ID = "gemini-3-flash-preview" 

def rle_decode(mask_rle, shape=(256, 1600)):
    """Ultra-fast RLE decoding (Fortran column-major)."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def create_solid_overlay(img, preds):
    """Generates professional, non-jittery masks for industrial display."""
    mask = np.argmax(preds, axis=-1)
    conf = np.max(preds, axis=-1)
    
    overlay = np.zeros_like(img)
    # Class Palette: 1:Cyan, 2:Yellow, 3:Red, 4:Magenta
    pal = {0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255]}
    kernel = np.ones((5, 5), np.uint8)
    
    for cid, color in pal.items():
        # Using binary threshold from hardcoded truth
        class_binary = ((mask == cid) & (conf > 0.5)).astype(np.uint8)
        # Apply morphological closing to 'glue' any gaps
        class_binary = cv2.morphologyEx(class_binary, cv2.MORPH_CLOSE, kernel)
        overlay[class_binary == 1] = color
        
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# --- 3. UI LAYOUT ---

st.title("🏭 SteelSight AI: Industrial Control Room")
st.caption("Deployment: Cloud Stable | Ground-Truth Reconstruction Mode")
st.markdown("---")

with st.sidebar:
    st.header("🕹️ Station Controls")
    
    # Check if manifest was imported correctly
    if hasattr(manifest, 'TRUTH_DATA'):
        st.success(f"Manifest Loaded: {len(manifest.TRUTH_DATA)} samples")
    else:
        st.error("Critical: manifest.py not found or TRUTH_DATA missing.")

    engage = st.toggle("🚀 ENGAGE MILL CONVEYOR", value=False)
    
    if st.button("🔄 EMERGENCY RESET", use_container_width=True):
        st.session_state.ptr_idx = 0
        st.session_state.ptr_x = 0
        st.session_state.logs = []
        st.session_state.f_raw = None
        st.session_state.f_viz = None
        st.rerun()

    st.markdown("---")
    step = st.select_slider("Conveyor Speed (Pixels/Frame)", options=[5, 10, 20, 30, 50], value=20)
    
    status_clr = "#28a745" if engage else "#dc3545"
    st.markdown(f"System Status: <b style='color:{status_clr};'>{'ACTIVE' if engage else 'IDLE'}</b>", unsafe_allow_html=True)

# Viewport Construction
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### 📷 Optical Sensor")
    v_raw = st.empty()
with col_right:
    st.markdown("#### 🔬 AI Vision")
    v_viz = st.empty()

# Halt Persistence Logic
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
    st.subheader("🤖 AI Expert Analysis (Gemini 3 Flash)")
    
    time_since_last = time.time() - st.session_state.last_ai_time
    wait_time = 60 - time_since_last
    
    if engage:
        st.info("⏸️ Halt the conveyor to enable surface topology analysis.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    elif wait_time > 0:
        st.warning(f"RCA Engine cooling down... {int(wait_time)}s.")
        st.button("✨ ANALYZE FRAME", disabled=True)
    else:
        if st.button("✨ ANALYZE FRAME", use_container_width=True):
            if st.session_state.f_raw is not None and client:
                st.session_state.last_ai_time = time.time()
                with st.spinner("Gemini 3 is synthesizing metallurgical data..."):
                    try:
                        img_pil = PIL.Image.fromarray(st.session_state.f_raw)
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[
                                "Act as a senior metallurgy engineer. Analyze this steel surface for defects. Identify class and prescribe a fix.", 
                                img_pil
                            ]
                        )
                        st.markdown(f"<div class='expert-response'>{response.text}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini Analysis Error: {str(e)}")

# --- 4. SCANNING ENGINE (ULTRA-FAST LOOKUP) ---
DIR = "test_samples"
files = sorted([os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DIR) else []

if engage and files and hasattr(manifest, 'TRUTH_DATA'):
    while st.session_state.ptr_idx < len(files):
        path = files[st.session_state.ptr_idx]
        filename = os.path.basename(path)
        
        # Load the raw image once
        raw_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # INSTANT RECONSTRUCTION FROM MANIFEST
        # We pre-calculate the full 1600px overlay to save CPU during sliding
        preds = np.zeros((256, 1600, 4), dtype=np.float32)
        image_data = manifest.TRUTH_DATA.get(filename, {})
        
        for class_id_str, rle in image_data.items():
            class_idx = int(class_id_str) - 1
            preds[:, :, class_idx] = rle_decode(rle)
        
        # Pre-generate the visual overlay for the entire strip
        full_viz = create_solid_overlay(raw_rgb, preds)
        
        # SLIDING VIEWPORT LOOP
        for x in range(st.session_state.ptr_x, 1150, step):
            if not engage:
                st.session_state.ptr_x = x
                st.rerun()

            # Slice and Store the viewport
            st.session_state.f_raw = raw_rgb[:, x : x + 450]
            st.session_state.f_viz = full_viz[:, x : x + 450]
            
            # Atomic Render: No Lag, No Black Screens
            v_raw.image(st.session_state.f_raw, use_container_width=True)
            v_viz.image(st.session_state.f_viz, use_container_width=True)
            
            # Log Scratches (Class 3 is Index 2)
            if "3" in image_data:
                if np.any(preds[:, x:x+450, 2] == 1):
                    if not any(l["File"] == filename for l in st.session_state.logs[-1:]):
                        st.session_state.logs.append({
                            "Time": time.strftime("%H:%M:%S"), 
                            "File": filename
                        })
                    log_area.table(st.session_state.logs[-5:])
            
            # 20 FPS Stability Cap for Streamlit Cloud
            time.sleep(0.05)

        # Increment Strip
        st.session_state.ptr_idx = (st.session_state.ptr_idx + 1) % len(files)
        st.session_state.ptr_x = 0
    st.rerun()