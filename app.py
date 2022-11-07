import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from PIL import Image
import numpy as np

st.markdown("<h1 style='text-align: center; font-size: 100px; padding-bottom: 10px;'>R7 v/s M10</h1>",
            unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 20px; padding-top: 0px;'>Ronaldo and Messi image classifier made by Arjunan K.</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 0px; padding-bottom: 10px;'></h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload the Image", type=["jpg", "png"])

if st.button("Predict"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        
        img = np.asarray(image)
        resize = tf.image.resize(img, (256, 256))
        model = tf.keras.models.load_model('models/imageclassifier.h5')
        yhat = model.predict(np.expand_dims(resize/255, 0))
        message = ""
        if yhat >= 0.5:
            message = "Predicted as Ronaldo"
        else:
            message = "Predicted as Messi"
        result = round(yhat[0][0], 4)
        st.write("")
        st.write(message, result)