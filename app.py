import streamlit as st
from PIL import Image as img
from fastai.vision import open_image, load_learner, Learner
# from fastai import defaults
import torch
import fastai

fastai.torch_core.defaults.device = torch.device('cpu')

st.title("Pothole identifier")
st.markdown("by Mark Danovich, mark.danovich@gmail.com")

def get_model() -> Learner:
    return  load_learner(".", file="model.pkl")

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","jpg", "png"])
if uploaded_file is not None:
    image = img.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    learn = get_model()
    label = learn.predict(open_image(uploaded_file))
    if str(label[0]) == 'pothole':
        output = 'Pothole'
    else:
        output = 'Not a Pothole'
    st.write(f"{output}, Probability: {label[2].max():.3f}")

