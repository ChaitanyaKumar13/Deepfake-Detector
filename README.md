# 🧠 DeepFake Image & Video Detector

A Streamlit-based web app that detects deepfake images and videos using CNN models. It performs real/fake classification on face images and videos with face-frame extraction and voting-based prediction.

---

## 📌 Project Features

- 🔍 **Image DeepFake Detection** — Upload an image and classify it as REAL or FAKE using a trained CNN.
- 🎥 **Video DeepFake Detection** — Upload a video; the app extracts face-containing frames, predicts on each, and uses a voting system to decide the final result.
- 📊 **Confidence Scores** — Detailed confidence levels shown with predictions.
- 📂 **Prediction History** — Saves and displays recent predictions in a CSV log.
- 🖼 **Frame Preview** — Displays sampled face frames from uploaded videos.
- ⚡ **Fast Inference** — Optimized for quick predictions on both images and videos.

---

## 🧠 Models Used

- `deepfake_cnn.keras`: A CNN-based image classifier trained to detect deepfake images.
- `final_deepfake_detector_v4.keras`: A CNN-only model that performs voting-based prediction on face frames extracted from videos.

---

## 🏗 Folder Structure

deepfake-detector/
├── app.py # Streamlit app
├── model/
│ ├── deepfake_cnn.keras # Image classifier
│ └── final_deepfake_detector_v4.keras # Video classifier
├── requirements.txt
├── predictions_log.csv # Optional: logs of predictions
├── assets/ # Optional: example images/videos
├── .gitignore
└── README.md

## ⚙️ Installation Instructions

### 🔹 1. Clone the Repository

git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

🔹 2. Create and Activate Environment (Optional)

python -m venv venv
source venv/bin/activate       # For Linux/macOS
venv\Scripts\activate          # For Windows

🔹 3. Install Dependencies

pip install -r requirements.txt

🚀 Run the App

streamlit run app.py
Once started, the app will open in your browser at http://localhost:8501.

💡 How It Works

📸 Image Prediction Flow:
Upload image

Preprocess and resize to 128x128

Run through deepfake_cnn.keras

Output label + confidence

🎞️ Video Prediction Flow:

Upload video (MP4/AVI/MOV)

Extract up to 25 face-containing frames (using Haar Cascade)

Resize to 96x96, normalize

Run all frames through final_deepfake_detector_v4.keras

Vote-based decision:

> 60% REAL → Classified as REAL

< 40% REAL → Classified as FAKE

40–60% → UNSURE

📦 Requirements
These are the main packages used:

streamlit
tensorflow
opencv-python
pillow
pandas
numpy

To install them:

pip install -r requirements.txt
📈 Example Outputs
✅ Real Video Prediction:
Raw Score: 0.97

Result: ✅ REAL (97%)

❌ Fake Video Prediction:
Raw Score: 0.23

Result: ❌ FAKE (77%)

📄 License
This project is for educational and research use only. Use responsibly.

🙌 Acknowledgements
Streamlit for app interface

OpenCV for face detection

TensorFlow/Keras for deep learning models

Celeb-DF and other open datasets for training data

👨‍💻 Author
Chaitanya Kumar
LinkedIn: https://www.linkedin.com/in/chaitanya-kumar-78a4b524b/
GitHub: https://github.com/ChaitanyaKumar131

