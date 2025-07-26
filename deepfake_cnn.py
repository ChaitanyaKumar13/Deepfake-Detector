import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Data loading and augmentation
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    'data/train/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    'data/valid/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
if not os.path.exists('model'):
    os.makedirs('model')

model.save("model/deepfake_cnn.keras")
