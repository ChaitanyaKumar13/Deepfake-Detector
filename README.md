# 🧠 DeepFake Image & Video Detector

This is a Streamlit-based web app to detect deepfakes in both images and videos using CNN-based models trained on face data.

## 🚀 Features

- Upload and analyze **images** or **videos**
- View **face frame previews**
- Uses two AI models: one for images, another for videos
- Voting system for video frame classification
- Confidence score and visual feedback

## 📂 Project Structure

deepfake-detector/
├── app.py
├── model/
│ ├── deepfake_cnn.keras
│ └── final_deepfake_detector_v4.keras
├── requirements.txt
├── README.md

## 📦 Installation

# Clone the repo
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Create virtual env (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
