# 🎤 Speech Emotion Recognition using Deep Learning

## 📌 Project Overview

This project is a Speech Emotion Recognition System developed using Deep Learning. It predicts human emotions from speech audio using a 1D Convolutional Neural Network (CNN).

The model is trained on the RAVDESS dataset and extracts audio features such as MFCC, Chroma, Mel Spectrogram, Zero Crossing Rate (ZCR), and RMS Energy.

---

## 🚀 Features

- Speech Emotion Recognition
- Audio Feature Extraction using Librosa
- Audio Data Augmentation
- 1D CNN Deep Learning Model
- Streamlit Web Application
- Real-time Emotion Prediction
- Confidence Score and Probability Display

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📂 Project Structure

```
Speech_Emotion_Recognition/
│
├── app.py
├── train_model.py
├── predict.py
├── preprocess.py
├── augment.py
├── requirements.txt
├── README.md
├── models/
│   ├── emotion_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── accuracy.png
└── loss.png
```

---

## 📊 Model Performance

- Dataset: RAVDESS
- Model: 1D CNN
- Final Test Accuracy: **76.91%**

---

## ▶️ Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Results

The project includes:

- Training Accuracy Graph
- Training Loss Graph
- Streamlit Web Interface
- Real-time Emotion Prediction

---

## 📌 Author

**Anwesha Barik**

B.Tech CSE (AI & ML)

GitHub: https://github.com/anwesha20062013

---