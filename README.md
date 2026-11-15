# 🎭 DeepFakeAI – Deepfake Video Detection System

**DeepFakeAI** is a machine learning-based system designed to detect deepfake videos using facial feature extraction and a custom Convolutional Neural Network (CNN). It uses real and fake video samples to train a classifier capable of distinguishing manipulated content with high accuracy.

---

## 📁 Dataset

- **Source**: [Real/Fake Video Dataset – Kaggle](https://www.kaggle.com/datasets/mohammadsarfrazalam/realfake-video-dataset)
- **Content**: 1400+ real and fake videos featuring celebrity faces
- **Preprocessing**:
  - Face extraction using OpenCV
  - Frame sampling and resizing
  - Labeling as `real` or `fake`

---

## 🛠️ Tech Stack

- Python
- OpenCV (for face and frame extraction)
- PyTorch (for CNN model training)
- NumPy, Pandas
- Flask (optional: for web interface)
- Streamlit (optional: for interactive UI)

---
 ![Screenshot 2025-06-02 213754](https://github.com/user-attachments/assets/26b731da-bb9d-4434-b3e7-11d128539fc1)
## 🧠 Model Architecture

A custom **CNN** built from scratch using PyTorch:
- 3 Convolutional layers with ReLU + MaxPooling
- Fully connected dense layers
- Softmax for final classification

You can later replace this with transfer learning using models like ResNet or EfficientNet for higher accuracy.

---

## 📈 Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Sample frame-level predictions for real/fake confidence

---


https://github.com/user-attachments/assets/6725ffe7-9551-47f3-beee-242b18f4bebf


## 🎯 Features

- Video upload and frame-by-frame face extraction
- Batch prediction on extracted face images
- Heatmap overlay for prediction confidence
- Optional UI with Streamlit for easy usage


![Screenshot 2025-06-02 213722](https://github.com/user-attachments/assets/21449e0d-07c3-4b50-999a-9920df6d0a6d)

---

## 🚀 How to Run

### 1. Set Up Virtual Environment & Dependencies:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the DeepGuardAI Web Application:
```bash
python app.py
```
Open your browser and navigate to:
- **Web App**: [http://127.0.0.1:5001](http://127.0.0.1:5001)
- **Public Verification Portal**: [http://127.0.0.1:5001/verify](http://127.0.0.1:5001/verify)

#### Default Demo Credentials:
- **Username**: `admin`
- **Password**: `admin123`
*(Or click "Sign up" on the login screen to register a new account)*

---

## 🏋️ How to Retrain or Evaluate the Model:
```bash
python main.py
```
This loads images from `Deepfake/dataset/`, trains the `SimpleCNN` network with GPU/MPS/CPU acceleration, saves the updated weights to `model.pth`, and prints evaluation accuracy.

---

## 🔌 API Usage:
Submit video or image media with your API Key:
```bash
curl -X POST http://127.0.0.1:5001/api/v1/detect \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "video=@path/to/video.mp4"
```


