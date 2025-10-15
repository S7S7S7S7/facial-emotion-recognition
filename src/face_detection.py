# OpenCV library for computer vision tasks like image processing and object detection.
import cv2
import numpy as np

def detect_face(image_path):
    # cv2.data.haarcascades points to the folder where OpenCV stores its pre-trained XML files.
    # contains the trained weights to detect faces.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Loads the image from disk into a NumPy array (img).
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Converts the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detects faces in the image.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # (x, y): top-left corner of face
    # (w, h): width and height of face rectangle
    for (x, y, w, h) in faces:
        #img[y:y+h, x:x+w]: crops the face from the original image.
        return img[y:y+h, x:x+w]
    return None
