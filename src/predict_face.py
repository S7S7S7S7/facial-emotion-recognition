from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from src.train_model import model

def preprocess_face(face_array):
    # Convert NumPy array to PIL Image
    face_pil = Image.fromarray(face_array).convert('L')  # Grayscale
    face_resized = face_pil.resize((48,48))
    face_array = img_to_array(face_resized)/255.0  # Normalize
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    return face_array
