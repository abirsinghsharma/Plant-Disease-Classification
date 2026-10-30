
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_mobilenetv2.keras",
        compile=False
    )

@st.cache_data
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)

st.title("🌿 Plant Disease Classification")
st.write("Upload a leaf image to predict its disease.")

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Could not load model or class names: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", width="stretch")

    if st.button("Predict Disease", type="primary"):
        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)[None, ...]

        with st.spinner("Analyzing image..."):
            scores = model.predict(array, verbose=0)[0]

        index = int(np.argmax(scores))

        st.success(f"Prediction: {class_names[index]}")
        st.metric(
            "Model confidence",
            f"{float(scores[index]) * 100:.2f}%"
        )

        st.caption(
            "This is a model estimate, not a confirmed diagnosis."
        )
