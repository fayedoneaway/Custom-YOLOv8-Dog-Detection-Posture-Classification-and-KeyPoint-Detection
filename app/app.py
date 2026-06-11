import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Custom YOLOv8 Model: \n"
         "Dog Detection with Posture Classification\n"
         "and Keypoint Detection")

@st.cache_resource
def load_model1():
    return YOLO("models/detect_dog_posture.pt")

@st.cache_resource
def load_model2():
    return YOLO("models/keypoint_detection.pt")

model_1 = load_model1()
model_2 = load_model2()

image_choice = st.selectbox("Choose an image",["demo/Image1.jpg", "demo/Image2.jpg", "demo/Image3.jpg", "demo/Image4.jpg", "demo/Image5.jpg",
                                               "demo/Image6.jpg",])

image = Image.open(image_choice)

# st.image(image, caption="Uploaded Image", use_column_width=True)

img_array = np.array(image)
results_1 = model_1(img_array)
results_2 = model_2(img_array)

annotated_img = results_1[0].plot(conf=False)
st.image(annotated_img, caption="Posture Predictions", use_container_width=True)

second_annotated_img = results_2[0].plot(conf=False)
st.image(second_annotated_img, caption="Pose Results", use_container_width=True)
