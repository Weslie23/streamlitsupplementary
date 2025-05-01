import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_concrete_model():
    return load_model('concrete_mixture_classifier.h5')

model = load_concrete_model()

st.write("""# Concrete Mixture Classification System""")
file = st.file_uploader("Upload a photo of concrete mixture", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert("RGB")
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['high_strength', 'medium_strength', 'low_strength']
    result = class_names[np.argmax(prediction)]
    st.success(f"Prediction: **{result.replace('_', ' ').title()}**")
