import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
import gdown

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Define Google Drive file ID and local model path
file_id = '11GC4iF6J7krI6ooUhctsl9fkTmeY234N'  # Replace with your file ID
local_model_path = 'Flower_Recog_Model.h5'

# Download model from Google Drive if it doesn't exist locally
if not os.path.exists(local_model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, local_model_path, quiet=False)

# Custom object handling (if any)
# custom_objects = {'CustomLayer': CustomLayer}  # Example for custom layer
custom_objects = {}  # Update this if you have custom layers/objects

try:
    model = load_model(local_model_path, custom_objects=custom_objects)
except TypeError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f'The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%'
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Create an 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, width=200)
    st.markdown(classify_images(file_path))
