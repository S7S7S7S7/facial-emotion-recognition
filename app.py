import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = "models/cnn_model.h5"
model = load_model(MODEL_PATH)
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preprocess_face(face_array):
    face_pil = Image.fromarray(face_array).convert('L')
    face_resized = face_pil.resize((48,48))
    face_array = np.expand_dims(np.array(face_resized)/255.0, axis=(0,-1))
    return face_array

# Streamlit UI
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = img_array[y:y+h, x:x+w]
        st.image(face_crop, caption="Detected Face", use_column_width=True)

        # Preprocess and predict
        # Assume face_crop is the detected face (NumPy array)
        face_input = preprocess_face(face_crop)  # shape (1,48,48,1)
        prediction = model.predict(face_input)   # shape (1,7)
        class_idx = np.argmax(prediction)        # index of max probability
        confidence = float(np.max(prediction))   # confidence score
        predicted_emotion = classes[class_idx]

        # Display result in Streamlit
        st.success(f"Predicted Emotion: {predicted_emotion} ({confidence*100:.2f}%)")

    else:
        st.warning("No face detected!")
