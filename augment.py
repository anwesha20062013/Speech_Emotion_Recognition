import numpy as np

def add_noise(data):
    noise = 0.005 * np.random.randn(len(data))
    return data + noise


def pitch_shift(data, sr):
    import librosa
    return librosa.effects.pitch_shift(
        y=data,
        sr=sr,
        n_steps=2
    )


def time_stretch(data):
    import librosa
    return librosa.effects.time_stretch(
        y=data,
        rate=0.9
    )