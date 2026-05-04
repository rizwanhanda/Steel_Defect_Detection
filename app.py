import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Custom Logic to load the model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@st.cache_resource
def load_my_model():
    # We use a dummy function for focal_loss since we only need the model for prediction
    return tf.keras.models.load_model('steel_model_best.keras', 
                                      custom_objects={'dice_coef': dice_coef, 'focal_loss_fixed': lambda x, y: x})

# 2. UI Configuration
st.set_page_config(page_title="Steel Sight AI", layout="wide")
st.title("🏗️ Steel Surface Defect Detector")
st.write("Upload a surface image to detect industrial defects in real-time.")

model = load_my_model()

# 3. File Uploader
uploaded_file = st.file_uploader("Choose a steel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(opencv_image, caption="Original Surface", use_container_width=True)
    
    if st.button("🔍 Run Analysis"):
        # Preprocess
        input_img = cv2.resize(opencv_image, (1600, 256))
        input_img = input_img / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        # Predict
        with st.spinner("Processing deep features..."):
            prediction = model.predict(input_img)[0]
        
        with col2:
            # Create a combined heatmap of all 4 defect classes
            heatmap = np.max(prediction, axis=-1)
            st.image(heatmap, caption="Defect Heatmap", use_container_width=True)
            
            # Logic check
            if np.max(heatmap) > 0.5:
                st.error(f"⚠️ DEFECT DETECTED (Confidence: {np.max(heatmap):.2%})")
            else:
                st.success("✅ SURFACE CLEAR")