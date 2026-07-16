import librosa
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("models/emotion_model.keras")

scaler = joblib.load("models/scaler.pkl")

encoder = joblib.load("models/label_encoder.pkl")


def extract_feature(audio, sr):

    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(y=audio, sr=sr).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(y=audio, sr=sr).T,
        axis=0
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio).T,
        axis=0
    )

    rms = np.mean(
        librosa.feature.rms(y=audio).T,
        axis=0
    )

    return np.hstack([mfcc, chroma, mel, zcr, rms])


def predict_emotion(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        duration=3,
        offset=0.5
    )

    feature = extract_feature(audio, sr)

    feature = scaler.transform([feature])

    feature = feature.reshape(1,182,1)

    prediction = model.predict(feature, verbose=0)

    emotion = encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    confidence = np.max(prediction)*100

    probabilities = prediction[0]

    return emotion, confidence, probabilities