import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
import boto3

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Define S3 parameters
bucket_name = 'your-bucket-name'
model_file_key = 'Flower_Recog_Model.h5'
local_model_path = 'Flower_Recog_Model.h5'

# Download model from S3 if it doesn't exist locally
if not os.path.exists(local_model_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_file_key, local_model_path)

model = load_model(local_model_path)

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
