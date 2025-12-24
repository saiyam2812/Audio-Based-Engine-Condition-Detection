
# Audio-Based Engine Condition Detection

## Project Overview

This project demonstrates an end-to-end audio processing and machine learning pipeline for detecting vehicle engine conditions using short audio recordings. The system analyzes engine sound characteristics, extracts meaningful audio features, and classifies the engine state into one of four predefined conditions.

The primary goal of this project is to showcase audio signal processing, feature extraction, and supervised classification rather than to build a production-ready diagnostic system.

---

## Engine Condition Classes

The engine audio samples are classified into the following categories:

- **Normal** – smooth and stable engine hum  
- **Knocking** – irregular impact-like sounds  
- **Sputtering** – unstable or rough combustion sound  
- **Silent** – engine off or no signal  

---

## Dataset

- The dataset consists of **synthetically generated audio samples**.
- Synthetic signals are used to simulate distinct engine behaviors in a controlled manner.
- All audio samples used for training and evaluation are included in the repository to ensure reproducibility.

> **Note:** Synthetic data is used for proof-of-concept validation. Real-world deployment would require real engine recordings and environmental noise handling.

---

## Project Structure

```
audio-engine-condition-detection/
│
├── data/
│   ├── normal/
│   ├── knocking/
│   ├── sputtering/
│   └── silent/
│
├── features/
│   └── features.csv
│
├── models/
│   └── engine_condition_classifier.pkl
│
├── notebooks/
│   ├── audioexploration.ipynb
│   └── model.py
│
├── common/
│   └── utils.py
│
├── generate_engine_sounds.py
├── featureextraction.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Feature Extraction

Each audio file is converted into a numerical feature vector using the following audio features:

- **MFCCs (13 coefficients)** – capture the perceptual characteristics of sound  
- **RMS Energy** – represents signal loudness  
- **Zero Crossing Rate** – indicates signal noisiness  
- **Spectral Centroid** – measures sound brightness  
- **Spectral Bandwidth** – represents frequency spread  

These features are extracted using the `librosa` library and stored in a CSV file (`features.csv`) for model training.

---

## Model Training

- **Algorithm Used:** Random Forest Classifier  
- **Why Random Forest?**
  - Performs well on small datasets  
  - Handles non-linear decision boundaries  
  - Robust to feature scaling  
  - Simple to train and interpret  

The dataset is split into training and test sets using stratified sampling to preserve class balance.

---

## Model Evaluation

Model performance is evaluated using:

- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

Due to the controlled nature of the synthetic dataset, the model achieves high classification accuracy, demonstrating strong feature separability.

---

## Inference

The trained model predicts engine condition for a new audio file by:

1. Extracting audio features  
2. Feeding the features into the trained model  
3. Returning the predicted engine condition  

---

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Main libraries used:
- numpy
- pandas
- librosa
- soundfile
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## Limitations

- Trained on synthetic audio data  
- Real-world engine sounds may contain noise and overlapping conditions  
- Production use would require real engine recordings and robustness testing  

---

## Conclusion

This project demonstrates a complete audio-based engine condition classification pipeline, including sound generation, feature extraction, model training, evaluation, and inference. It serves as a proof-of-concept showcasing audio analytics and machine learning fundamentals.
