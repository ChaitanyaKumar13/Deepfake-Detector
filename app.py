import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime
from PIL import Image
import cv2
import tempfile
import time

# ---------------- CONFIGURATION ---------------- #
IMAGE_MODEL_PATH = "model/deepfake_cnn.keras"
VIDEO_MODEL_PATH = "model/final_deepfake_detector_v4.keras"
LOG_PATH = "predictions_log.csv"
IMG_SIZE = (96, 96)
FRAME_SAMPLE_COUNT = 25

# ---------------- LOAD MODELS ---------------- #
image_model = load_model(IMAGE_MODEL_PATH)
video_model = load_model(VIDEO_MODEL_PATH)

# ---------------- STREAMLIT SETUP ---------------- #
st.set_page_config(page_title="DeepFake Detection", layout="centered")
st.title("🧠 DeepFake Image & Video Detector")
st.write("Upload a face image or video to detect whether it's REAL or FAKE using AI models.")

# ---------------- IMAGE PREDICTION ---------------- #
st.subheader("🖼 Image DeepFake Detection")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="image")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = load_img("temp.jpg", target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict Image", key="predict_image"):
        prediction = image_model.predict(image_array)[0][0]
        label = "REAL" if prediction > 0.5 else "FAKE"
        confidence = prediction if label == "REAL" else 1 - prediction

        if label == "REAL":
            st.success(f"Prediction: REAL ({confidence:.4f})")
        else:
            st.error(f"Prediction: FAKE ({confidence:.4f})")

        new_entry = {
            "filename": uploaded_file.name,
            "label": label,
            "confidence": round(float(confidence), 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if os.path.exists(LOG_PATH):
            log_df = pd.read_csv(LOG_PATH)
            log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([new_entry])
        log_df.to_csv(LOG_PATH, index=False)
        st.subheader("📊 Recent Predictions")
        st.dataframe(log_df.tail(5))

# ---------------- VIDEO PREDICTION ---------------- #
st.subheader("📹 Video DeepFake Detection")
video_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov"], key="video")

def extract_face_frames(video_path, max_frames=FRAME_SAMPLE_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // max_frames, 1)

    frame_count = 0
    extracted = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, IMG_SIZE)
                frames.append(face)
                extracted += 1
                break
        frame_count += 1

    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    return frames.astype("float32") / 255.0

def predict_video(frames):
    if len(frames) == 0:
        return 0.0, [], []
    frames = preprocess_frames(frames)

    # Pad or trim
    if len(frames) < FRAME_SAMPLE_COUNT:
        pad = [np.zeros_like(frames[0], dtype=np.float32)] * (FRAME_SAMPLE_COUNT - len(frames))
        frames = list(frames) + pad
    else:
        frames = frames[:FRAME_SAMPLE_COUNT]

    frames_batch = np.expand_dims(np.array(frames), axis=0)  # (1, 25, 96, 96, 3)
    prediction = video_model.predict(frames_batch, verbose=0)[0][0]

    # Frame-wise voting breakdown
    frame_preds = []
    for f in frames:
        f_input = np.expand_dims(f, axis=0)
        f_input = np.expand_dims(f_input, axis=0)
        score = video_model.predict(f_input, verbose=0)[0][0]
        frame_preds.append(score)

    real_votes = sum(p >= 0.5 for p in frame_preds)
    fake_votes = FRAME_SAMPLE_COUNT - real_votes

    return float(prediction), frame_preds, (real_votes, fake_votes)

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    st.info("⏳ Extracting face frames...")
    frames = extract_face_frames(tmp_path)
    st.write(f"✅ {len(frames)} face-containing frames extracted.")

    if len(frames) == 0:
        st.error("❌ No detectable faces found in video.")
    else:
        st.info("🧠 Running frame-wise predictions...")
        start = time.time()
        score, frame_preds, (real_votes, fake_votes) = predict_video(frames)
        end = time.time()

        st.subheader("📈 Raw Prediction Score")
        st.code(f"Raw Model Output: {score:.6f}")

        if score >= 0.6:
            st.success(f"🎬 Video Prediction: REAL ({score:.2%})")
        elif score <= 0.4:
            st.error(f"🎬 Video Prediction: FAKE ({(1 - score):.2%})")
        else:
            st.warning(f"🎬 Prediction: UNSURE (Score: {score:.2%})")

        st.caption(f"Prediction completed in {end - start:.2f} seconds")

        # Frame Preview
        st.subheader("🖼 Preview of Sample Face Frames")
        num_preview = min(5, len(frames))
        preview_images = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in range(num_preview)]
        st.image(preview_images, width=128, caption=[f"Frame {i+1}" for i in range(num_preview)])

        # Voting Breakdown
        st.subheader("🗳️ Voting-Based Breakdown")
        st.info(f"Votes: 🟢 REAL = {real_votes} / 🔴 FAKE = {fake_votes}")
        st.line_chart(pd.DataFrame(frame_preds, columns=["Frame Confidence (Real)"]))

    os.unlink(tmp_path)
