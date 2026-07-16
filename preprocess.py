import librosa
import numpy as np

from augment import add_noise, pitch_shift, time_stretch


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


def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        duration=3,
        offset=0.5
    )

    features = []

    # Original
    features.append(extract_feature(audio, sr))

    # Noise
    features.append(extract_feature(add_noise(audio), sr))

    # Pitch Shift
    features.append(extract_feature(pitch_shift(audio, sr), sr))

    # Time Stretch
    stretched = time_stretch(audio)
    features.append(extract_feature(stretched, sr))

    return features