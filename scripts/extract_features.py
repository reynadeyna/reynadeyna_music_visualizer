import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_audio_features(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1)
    tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
    rms = librosa.feature.rms(y=audio)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0].mean()

    features = np.hstack([chroma, [tempo, spectral_centroid, spectral_bandwidth, rolloff, rms, zcr]])

    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted features: {features}")

    assert len(features) == 18, f"Expected 18 features, got {len(features)}"

    return features

def augment_audio(y, sr):
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    y_speed = librosa.effects.time_stretch(y, rate=1.2)
    y_noise = y + 0.005 * np.random.randn(len(y))
    return [y, y_pitch, y_speed, y_noise]

def create_augmented_dataset(audio_dir):
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    dataset = []

    for file in audio_files:
        y, sr = librosa.load(file)
        augmented_audios = augment_audio(y, sr)

        for audio in augmented_audios:
            features = extract_audio_features(audio, sr)

            assert len(features) == 18, f"Expected 18 features, got {len(features)}"

            tempo, spectral_centroid, spectral_bandwidth, rolloff, rms, zcr = features[-6:]

            visual_params = [
                int(tempo % 800),                   # Center X
                int(spectral_centroid % 600),       # Center Y
                int(rms * 100 % 100),               # Radius
                tempo / 100,                        # Speed X (Angle offset)
                spectral_bandwidth / 100           # Speed Y (Scaling factor)
            ]

            dataset.append({'features': features.tolist(), 'visual_params': visual_params})

    return pd.DataFrame(dataset)

if __name__ == '__main__':
    audio_dir = 'audio_samples'
    df = create_augmented_dataset(audio_dir)
    df.to_csv('data/audio_visual_dataset_augmented.csv', index=False)
