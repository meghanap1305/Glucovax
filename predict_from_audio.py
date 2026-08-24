"""
predict_from_audio.py (portable version)

Same as before, but loads the model in a portable way (XGBoost native
.json format + separate scaler) instead of a single joblib-pickled
object -- this avoids "input stream corrupted" errors that can happen
when the pickling machine's Python/OS/xgboost build differs from the
loading machine's.
"""

import sys
import numpy as np
import librosa
import torch
import joblib
from xgboost import XGBClassifier

sys.path.insert(0, "serab-byols")
import serab_byols

MODEL_NAME = "default"
CHECKPOINT_PATH = "serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth"

print("Loading BYOL-S embedding model...")
byols_model = serab_byols.load_model(CHECKPOINT_PATH, MODEL_NAME)

print("Loading trained diabetes-risk classifier...")
clf = XGBClassifier()
clf.load_model("xgb_model.json")
scaler = joblib.load("scaler.pkl")


def extract_embedding(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    embedding = serab_byols.get_scene_embeddings(audio_tensor, byols_model)
    return embedding.detach().numpy()[0]


def predict(wav_path, age, bmi):
    embedding = extract_embedding(wav_path)
    features = np.concatenate([embedding, [age, bmi]]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    probability = clf.predict_proba(features_scaled)[0, 1]
    label = "higher_risk" if probability > 0.5 else "lower_risk"
    return label, float(probability)


if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    age = float(sys.argv[2]) if len(sys.argv) > 2 else 45
    bmi = float(sys.argv[3]) if len(sys.argv) > 3 else 26.0

    label, prob = predict(wav_path, age, bmi)
    print(f"\nFile: {wav_path}")
    print(f"Age: {age}, BMI: {bmi}")
    print(f"Prediction: {label} (probability: {prob:.3f})")
