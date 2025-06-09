import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit App
st.set_page_config(page_title="Charcoal Sketch Generator", layout="centered")
st.title("🖌️ Charcoal Sketch Generator")
st.write("Upload an image to generate a charcoal-style pencil sketch.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define dodge function
def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

# Main logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert to RGB again for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Original Image", use_container_width=True)

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Invert grayscale image
    img_invert = cv2.bitwise_not(img_gray)

    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img_invert, (25, 25), sigmaX=0, sigmaY=0)

    # Create pencil sketch
    pencil_sketch = dodgeV2(img_gray, img_blur)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(pencil_sketch)

    # Morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    charcoal_effect = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Show result
    st.image(charcoal_effect, caption="Charcoal Sketch", use_container_width=True, channels="GRAY")

    # Option to download
    result = Image.fromarray(charcoal_effect)
    st.download_button(
        label="Download Sketch",
        data=cv2.imencode('.png', charcoal_effect)[1].tobytes(),
        file_name='charcoal_sketch.png',
        mime='image/png'
    )
else:
    st.info("Please upload an image to get started.")
