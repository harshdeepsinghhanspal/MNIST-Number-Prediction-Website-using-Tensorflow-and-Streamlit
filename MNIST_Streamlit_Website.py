import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the model
model = load_model('mnist_model.h5')

# Title
st.title("MNIST Digit Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert('L')

    # Preprocess the image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0

    # Display the uploaded image
    st.image(image.reshape(28, 28), caption='Uploaded Image', use_column_width=True)

    # Predict the digit
    pred = model.predict(image)
    digit = np.argmax(pred, axis=1)[0]

    # Display the prediction
    st.write(f"Predicted Digit: {digit}")
