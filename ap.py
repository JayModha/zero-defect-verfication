import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Custom F1Score metric class (must match your training code exactly)
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_pred.shape[-1] == 1: 
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        else:  
            y_pred_binary = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        else:
            y_true = tf.cast(y_true, tf.float32)
        
        self.precision.update_state(y_true, y_pred_binary, sample_weight)
        self.recall.update_state(y_true, y_pred_binary, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

st.set_page_config(page_title="Zero-Defect Verifier", layout="centered")
st.title("🔍 Zero-Defect Packaging Verifier")

# Detect available models
model_root = "models"
categories = sorted([d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))])
if not categories:
    st.error("No trained models found. Please train first using `train_model.py`.")
    st.stop()

category = st.selectbox("Choose Product Type", categories)
model_path = os.path.join(model_root, category, "best_anomaly_model.keras")

@st.cache_resource
def load_model_cached(path):
    return load_model(path, custom_objects={'F1Score': F1Score})

model = load_model_cached(model_path)

# Upload image
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype("float32") / 255.0

    prediction = model.predict(np.expand_dims(image_normalized, axis=0))[0][0]
    label = "✅ Non-defective" if prediction < 0.5 else "❌ Defective"

    st.image(image, caption=f"{label} (Confidence: {prediction:.2f})", use_container_width=True)