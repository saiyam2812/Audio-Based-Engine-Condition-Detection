import numpy as np
import soundfile as sf
import os

sr = 22050          # sample rate - SETTING THE FRQUENCY
duration = 3        # seconds - DUARTION OF THE SOUND
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

os.makedirs("data/normal", exist_ok=True)
os.makedirs("data/knocking", exist_ok=True)
os.makedirs("data/sputtering", exist_ok=True)
os.makedirs("data/silent", exist_ok=True)





def normal_engine():
    base = np.sin(2 * np.pi * 120 * t)
    print(base)
    harmonic = 0.5 * np.sin(2 * np.pi * 240 * t)
    noise = 0.02 * np.random.randn(len(t))
    return base + harmonic + noise

def knocking_engine():
    y = 0.6 * np.sin(2 * np.pi * 100 * t)
    print(y)
    for k in range(10):
        pos = np.random.randint(0, len(t))
        y[pos:pos+200] += np.hanning(200) * 2
    return y

def sputtering_engine():
    carrier = np.sin(2 * np.pi * 120 * t)
    modulator = np.sin(2 * np.pi * np.random.uniform(1, 5) * t)
    noise = 0.05 * np.random.randn(len(t))
    print(carrier)
    print(modulator)
    print(noise)
    return carrier * modulator + noise

def silent_engine():
    return 0.001 * np.random.randn(len(t))




for i in range(10):
    y = normal_engine()
    sf.write(f"data/normal/normal_{i}.wav", y, sr)

for i in range(10):
    y = knocking_engine()
    sf.write(f"data/knocking/knocking_{i}.wav", y, sr)
for i in range(10):
    y = sputtering_engine()
    sf.write(f"data/sputtering/sputtering_{i}.wav", y, sr)
for i in range(10):
    y = silent_engine()
    sf.write(f"data/silent/silent_{i}.wav", y, sr)


import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("data/knocking/knocking_0.wav")
plt.plot(y)
plt.title("Knocking waveform")
plt.show()

import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load("data/normal/normal_0.wav", sr=None)

plt.figure(figsize=(10,3))
plt.plot(y)
plt.title("NORMAL waveform")
plt.show()


