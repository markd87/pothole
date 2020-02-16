import streamlit as st
from PIL import Image as img
from fastai.vision import *

st.title("Pothole identifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","jpg"])
if uploaded_file is not None:
    image = img.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    learn = load_learner(".", file="model.pkl")
    label = learn.predict(open_image(uploaded_file))
    st.write(f"{str(label[0])}, Probability: {label[2].max():.3f}")