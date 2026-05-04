import streamlit as st
import numpy as np
import cv2
import os
import time
import PIL.Image
import google.generativeai as genai
import tensorflow as tf

# --- 1. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="SteelSight AI | Industrial Dashboard", layout="wide")

# Connect to Gemini 1.5 Flash
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CUSTOM OBJECTS (Fixes the Loading Error) ---

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def focal_loss(gamma=2., alpha=4.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        loss = -y_true * tf.keras.backend.pow(1.0 - y_pred, gamma) * tf.keras.backend.log(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
    return focal_loss_fixed

@st.cache_resource
def load_u_net():
    # We define the keys to match common names used in Severstal training scripts
    custom_objects = {
        'dice_coef': dice_coef,
        'focal_loss_fixed': focal_loss(),
        'focal_loss': focal_loss()
    }
    return tf.keras.models.load_model(
        'steel_model_best.keras', 
        custom_objects=custom_objects, 
        compile=False
    )

model = load_u_net()

# --- 3. PROCESSING TOOLS ---

def apply_glow_overlay(raw_image, mask):
    overlay = np.zeros_like(raw_image)
    colors = {
        1: [0, 255, 255],   # Cyan
        2: [255, 255, 0],   # Yellow
        3: [255, 0, 0],     # Red (Critical Scratch)
        4: [255, 0, 255]    # Magenta
    }
    for class_id, color in colors.items():
        overlay[mask == class_id] = color
    return cv2.addWeighted(raw_image, 0.7, overlay, 0.3, 0)

# --- 4. USER INTERFACE ---

st.title("🏭 SteelSight AI: Real-Time Quality Control")
st.markdown("---")

col_main, col_log = st.columns([3, 1])

with col_main:
    st.write("### LIVE FEED: Line 01 (Patiala Mill)")
    view_port = st.empty()
    run_conveyor = st.toggle("Engage Rolling Mill", value=True)

with col_log:
    st.write("### Detection Events")
    log_area = st.empty()
    if st.button("Consult AI Expert"):
        if "last_defect_img" in st.session_state:
            with st.spinner("Gemini is analyzing the surface..."):
                raw_img = PIL.Image.open(st.session_state.last_defect_img)
                response = gemini_model.generate_content([
                    "As a metallurgical expert, look at this steel strip. A defect was detected. What is the root cause?",
                    raw_img
                ])
                st.info(response.text)
        else:
            st.warning("No defects captured for analysis yet.")

# --- 5. SIMULATION LOOP ---

SAMPLE_FOLDER = "test_samples"
files = [os.path.join(SAMPLE_FOLDER, f) for f in os.listdir(SAMPLE_FOLDER) if f.endswith(('.jpg', '.png'))]

if "logs" not in st.session_state:
    st.session_state.logs = []

if run_conveyor:
    for img_path in files:
        if not run_conveyor: break
        
        # Pre-process
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(img, (1600, 256)) / 255.0
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        # Predict & Overlay
        pred = model.predict(input_data, verbose=0)
        mask = np.argmax(pred[0], axis=-1)
        full_result = apply_glow_overlay(img, mask)
        
        # Slide Window (The Conveyor Effect)
        for x in range(0, 1200, 30):
            view_port.image(full_result[:, x : x + 400], use_column_width=True)
            
            # Check for Class 3 (Scratches)
            if np.any(mask[:, x : x + 400] == 3):
                st.session_state.last_defect_img = img_path
                if not any(d["ID"] == os.path.basename(img_path) for d in st.session_state.logs):
                    st.session_state.logs.append({"Time": time.strftime("%H:%M:%S"), "ID": os.path.basename(img_path)})
                log_area.table(st.session_state.logs[-5:])
            
            time.sleep(0.05)