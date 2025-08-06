import streamlit as st
import torch
from PIL import Image
import shutil
from pathlib import Path
import uuid
import cv2
import numpy as np

# Load model only once
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom',
                          path='C:/code/satellite-image-detection/yolov5/runs/train/cyclone_detector/weights/best.pt',
                          force_reload=False)

model = load_model()
model.conf = 0.25  # initial confidence threshold for YOLO model (not final filter)

# Final filter threshold to show results
CONFIDENCE_THRESHOLD = 0.70

st.title("🌪️ Cyclone Detection from Satellite Images")
st.write("Upload a satellite image to detect cyclone presence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)
    
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    image_path = temp_dir / unique_filename

    # Save uploaded file
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display input image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Cyclone"):
        st.info("Running detection...")

        # Run detection
        results = model(str(image_path))
        results.render()  # Updates results.ims with boxes and labels

        detections = results.xyxy[0]  # Tensor with detections

        # Filter based on confidence threshold
        if detections.shape[0] > 0 and any(det[4].item() >= CONFIDENCE_THRESHOLD for det in detections):
            # Cyclone confidently detected
            result_img = Image.fromarray(results.ims[0])
            st.success("✅ Cyclone detected with high confidence!")
            st.image(result_img, caption="Detection Result", use_container_width=True)
        else:
            st.warning("⚠️ No cyclone detected with confidence above 70%.")

    # Cleanup temp upload directory
    shutil.rmtree(temp_dir)
