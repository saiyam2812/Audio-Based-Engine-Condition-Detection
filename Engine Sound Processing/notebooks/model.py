import sys
import os
sys.path.append(os.path.abspath(".."))


import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from common.utils import extract_features



# ---------------------------
# 1. LOAD DATASET
# ---------------------------
df = pd.read_csv("features/features.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------
# 2. TRAIN MODEL
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------
# 3. EVALUATE MODEL
# ---------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ---------------------------
# 4. SAVE MODEL
# ---------------------------
joblib.dump(model, "../models/engine_condition_classifier.pkl")


# ---------------------------
# 5. TEST PREDICTION (AFTER TRAINING)
# ---------------------------
def predict_engine_condition(audio_path):
    features = extract_features(audio_path)
    return model.predict([features])[0]


prediction = predict_engine_condition(
    "../data/knocking/knocking_1.wav"
)

print("\nTest Prediction:", prediction)
