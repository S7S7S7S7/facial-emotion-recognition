# imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training data
train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory("data/train", target_size=(64,64), batch_size=32, class_mode='categorical')

# train
model.fit(train_data, epochs=10)

# save model
model.save("../models/cnn_model.h5")
