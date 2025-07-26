# ✅ Updated train_model_v3.py with label check + class_weight support

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import random

# Constants
IMG_SIZE = (96, 96)
MAX_FRAMES = 25
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset Paths
BASE_DIR = "video_frames"
TRAIN_REAL = os.path.join(BASE_DIR, "train", "real")
TRAIN_FAKE = os.path.join(BASE_DIR, "train", "fake")
VALID_REAL = os.path.join(BASE_DIR, "valid", "real")
VALID_FAKE = os.path.join(BASE_DIR, "valid", "fake")


def load_video_sequence(folder):
    frames = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:MAX_FRAMES]
    images = []
    for f in frames:
        img = load_img(os.path.join(folder, f), target_size=IMG_SIZE)
        img = img_to_array(img).astype('float32') / 255.0
        images.append(img)
    # Padding
    while len(images) < MAX_FRAMES:
        images.append(np.zeros_like(images[0]))
    return np.array(images)


def load_dataset(real_dir, fake_dir):
    X, y = [], []
    print(f"Loading real: {real_dir}")
    for video in tqdm(os.listdir(real_dir)):
        path = os.path.join(real_dir, video)
        if os.path.isdir(path):
            X.append(load_video_sequence(path))
            y.append(1)
    print(f"Loading fake: {fake_dir}")
    for video in tqdm(os.listdir(fake_dir)):
        path = os.path.join(fake_dir, video)
        if os.path.isdir(path):
            X.append(load_video_sequence(path))
            y.append(0)
    return np.array(X), np.array(y)


print("\U0001F4E6 Loading datasets...")
X_train, y_train = load_dataset(TRAIN_REAL, TRAIN_FAKE)
X_valid, y_valid = load_dataset(VALID_REAL, VALID_FAKE)

# Shuffle
train = list(zip(X_train, y_train))
random.shuffle(train)
X_train, y_train = zip(*train)
X_train, y_train = np.array(X_train), np.array(y_train)

# Print label summary
print("\n\U0001F4DD Training label distribution:")
print(f"REAL: {np.sum(y_train==1)}, FAKE: {np.sum(y_train==0)}")

# Compute class weights (optional, tweakable)
import numpy as np
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\U0001F9F9 Class Weights: {class_weight_dict}")

# Model definition
print("\n\U0001F527 Building LSTM model...")
input_layer = Input(shape=(MAX_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3))
cnn_base = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, pooling='avg', weights='imagenet')
cnn_base.trainable = False
x = TimeDistributed(cnn_base)(input_layer)
x = LSTM(128)(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("\n🏋 Training...")
model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight_dict
)

print("\n📂 Saving model...")
os.makedirs("model", exist_ok=True)
model.save("model/final_deepfake_detector_v4.keras")
print("\u2705 Model saved to model/final_deepfake_detector_v4.keras")
