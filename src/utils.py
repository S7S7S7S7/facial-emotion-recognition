from tkinter import Image
from tensorflow.keras.models import load_model
import numpy as np 
MODEL_PATH = "models/cnn_model.h5"
model = load_model(MODEL_PATH)
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


def preprocess_face(face_array):
    # Convert NumPy array to PIL, grayscale, resize to 48x48
    face_pil = Image.fromarray(face_array).convert('L')
    face_resized = face_pil.resize((48,48))
    # Convert to array, normalize, add batch dimension
    face_array = np.expand_dims(np.array(face_resized)/255.0, axis=(0,-1))
    return face_array
