import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from predict import predict_emotion

st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="wide"
)

st.title("🎤 Speech Emotion Recognition using Deep Learning")

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    emotion, confidence, probabilities = predict_emotion(uploaded_file)

    st.success(f"Predicted Emotion : {emotion}")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    emotions = [
        "angry",
        "calm",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised"
    ]

    st.subheader("Prediction Probabilities")

    results = list(zip(emotions, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)

    for emotion_name, prob in results:
     st.progress(float(prob))
     st.write(f"{emotion_name} : {prob*100:.2f}%")

   