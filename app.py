"""
app.py (portable version)

Same pipeline as before, loading the model in a portable way (XGBoost
native .json + separate scaler.pkl) to avoid cross-machine pickling
issues.
"""

import sys
import tempfile
import numpy as np
import librosa
import torch
import joblib
from xgboost import XGBClassifier
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, "serab-byols")
import serab_byols

app = FastAPI(title="Voice-Based Diabetes Risk Screening (Real Data)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "default"
CHECKPOINT_PATH = "serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth"

print("Loading BYOL-S embedding model (this takes a few seconds)...")
byols_model = serab_byols.load_model(CHECKPOINT_PATH, MODEL_NAME)

print("Loading classifier trained on 607 real participants...")
clf = XGBClassifier()
clf.load_model("xgb_model.json")
scaler = joblib.load("scaler.pkl")

print("Ready.")


def extract_embedding(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    embedding = serab_byols.get_scene_embeddings(audio_tensor, byols_model)
    return embedding.detach().numpy()[0]


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "Trained on 607 real Colive Voice / VOCADIAB participants (AUC 0.76 held-out)",
        "note": "Research prototype for a hackathon submission. NOT a medical device.",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    embedding = extract_embedding(tmp_path)
    features = np.concatenate([embedding, [age, bmi]]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    probability = float(clf.predict_proba(features_scaled)[0, 1])
    risk_label = "higher_risk" if probability > 0.5 else "lower_risk"

    return {
        "risk_label": risk_label,
        "risk_probability": round(probability, 3),
        "model_info": "Trained on 607 real participants (AUC 0.76 held-out test set)",
        "disclaimer": "Screening prototype only. Not a diagnosis. "
                       "Consult a doctor for an HbA1c or fasting glucose test.",
    }
