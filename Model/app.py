from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import pickle as pkl
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from flask_cors import CORS
import tensorflow as tf
import base64

app = Flask(__name__)
allowed_origins = [
    'http://127.0.0.1:5500',
    'https://fit-buddy-app.vercel.app/' 
]

CORS(app, origins=allowed_origins)

# Load Pneumonia prediction model
pneumonia_model = load_model('../Model/PNEUMONIA/Model/Pneumonia.h5')

# Load Osteoporosis prediction model
osteoporosis_model = load_model('../Model/OSTEOPOROSIS/Model/Osteoporosis.h5')

# Load brain tumor prediction model
brain_tumor_model = load_model('../Model/BRAIN TUMOR/Model/model.h5')

image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

@app.route('/predict/pneumonia', methods=['POST'])
def predict_pneumonia():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'})

    file = request.files['image']
    img_stream = io.BytesIO(file.read())
    img = cv2.imdecode(np.frombuffer(img_stream.getvalue(), np.uint8), 1)
    img = cv2.resize(img, (224, 224))
    
    prediction = pneumonia_model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction)
    classes = ['NORMAL', 'PNEUMONIA']

    return jsonify({'prediction': classes[predicted_class]})


@app.route('/predict/brain_tumor', methods=['POST'])
def predict_brain_tumor():
    file = request.files['image']
    img_stream = io.BytesIO(file.read())
    img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    img_array = preprocess_image(img)

    predictions = brain_tumor_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    tumor_types = ['Giloma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
    predicted_tumor_type = tumor_types[predicted_class]

    return jsonify({'prediction': predicted_tumor_type})

@app.route('/predict/osteoporosis', methods=['POST'])
def predict_osteoporosis():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(
        image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image_gen.standardize(image)
    image = np.expand_dims(image, axis=0)
    pred = osteoporosis_model.predict(image)
    class_labels = ['Healthy', 'Osteoporosis']
    pred_class = class_labels[np.argmax(pred)]
    return jsonify({'prediction': pred_class})

