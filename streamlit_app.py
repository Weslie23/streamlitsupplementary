import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_water_model():
    return load_model('water_classifier.h5')

model = load_water_model()

st.title("💧 Water Filtration Classification System")
st.write("Upload an image of water and find out if it's clean or not clean.")

file = st.file_uploader("Upload a water image", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert("RGB")
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.info("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Water Sample", use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Not Clean', 'Clean']
    result = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
