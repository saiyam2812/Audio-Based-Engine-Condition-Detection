import os
import numpy as np
import pandas as pd
import librosa

# ---------- FEATURE EXTRACTION FUNCTION ----------
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Energy (loudness)
    rms = np.mean(librosa.feature.rms(y=y))

    # Noise / roughness
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Brightness of sound
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Spread of frequencies
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Combine all features into one vector
    return np.hstack([mfcc_mean, rms, zcr, centroid, bandwidth])


# ---------- DATASET CREATION ----------
data = []
labels = ["normal", "knocking", "sputtering", "silent"]

for label in labels:
    folder_path = f"../data/{label}"
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            features = extract_features(full_path)
            data.append(list(features) + [label])

# ---------- CREATE DATAFRAME ----------
columns = [f"mfcc_{i}" for i in range(13)] + [
    "rms", "zcr", "centroid", "bandwidth", "label"
]

df = pd.DataFrame(data, columns=columns)

# ---------- SAVE FEATURES ----------
os.makedirs("features", exist_ok=True)
df.to_csv("features/features.csv", index=False)

print("Feature extraction completed!")
print("Total samples:", len(df))
print(df.head())
