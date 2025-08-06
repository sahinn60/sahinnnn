import streamlit as st
import cv2
import numpy as np

st.title("✅ OpenCV Test App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode image from numpy array
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show image
    st.image(image, channels="BGR", caption="Uploaded Image")
