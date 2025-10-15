Project Overview

This project is a web-based Facial Emotion Recognition (FER) application built using Convolutional Neural Networks (CNN).
The app detects faces in uploaded images and predicts the corresponding human emotion.

Predicted Emotions:
[Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral]

What I Did (Project Contributions):- 

1.Dataset Handling and Preprocessing:

Loaded images from train/ and test/ folders.
Normalized pixel values to [0,1].
Converted images to grayscale and resized to 48×48 pixels for CNN input.
Split dataset into training, validation, and testing sets.
Implemented one-hot encoding for categorical labels.
Visualized sample images for EDA.

2.CNN Model Development

Built a custom CNN architecture with multiple Conv2D, MaxPooling, BatchNormalization, and Dropout layers.
Trained the model on the FER dataset using Adam optimizer and categorical cross-entropy loss.
Added EarlyStopping and ModelCheckpoint callbacks to prevent overfitting and save the best model.
Achieved a model with ~730k trainable parameters capable of recognizing 7 emotions.

3.Prediction Pipeline

Developed helper functions to preprocess uploaded images and predict emotion.
Converted uploaded images to grayscale, normalized, resized, and reshaped for CNN input.
Used np.argmax to get predicted emotion and confidence score.
Integrated face detection using OpenCV Haar Cascade for robust predictions.

4.Interactive Streamlit App

Built a user-friendly web interface to upload images and display predictions.
Displayed detected face and predicted emotion with confidence.