import streamlit as st
from PIL import Image
from fastai.vision import load_learner, open_image

st.title("Pothole identifier")
st.markdown("by Mark Danovich, mark.danovich@gmail.com")

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    learn = load_learner(".", file="model.pkl")
    label = learn.predict(open_image(uploaded_file))
    if str(label[0]) == 'pothole':
        output = 'Pothole'
    else:
        output = 'Not a Pothole'
    st.write(f"{output}, Probability: {label[2].max():.3f}")
