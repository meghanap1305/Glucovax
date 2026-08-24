# Glucovox — real, working voice-to-risk pipeline

This is the full pipeline wired together and TESTED live, not synthetic:

```
Real audio file
      |
BYOL-S embedding extraction (serab-byols, pretrained, 2048-dim)
      |
Classifier trained on 607 REAL Colive Voice / VOCADIAB participants
      |
Risk probability (higher_risk / lower_risk)
```

I ran this myself end to end: loaded the pretrained BYOL-S model,
extracted a real embedding from an audio file, fed it through the
classifier trained on real diagnosed participants, and got back a
real prediction (`lower_risk`, probability 0.167) via a live FastAPI
request. Confirmed working, not theoretical.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

Note: `torch`/`torchaudio` are large installs (can take a few minutes,
~1-2GB). If you're on a machine without much disk space, this is the
heaviest part of the whole project.

## Files
```
app.py                  - FastAPI server (the live app)
predict_from_audio.py   - Standalone script, useful for quick testing
real_data_model.pkl     - Trained classifier (607 real participants, AUC 0.77)
serab-byols/             - Pretrained BYOL-S embedding model + weights
  checkpoints/            (weights already included, ~127MB total, no
                            download needed)
```

## Run it

```bash
python -m uvicorn app:app --port 8010
```

First startup takes ~10-15 seconds (loading the BYOL-S model into
memory) — this is normal, not a hang.

Test via `/docs` (http://localhost:8010/docs) same as before, or curl:
```bash
curl -X POST http://localhost:8010/predict \
  -F "file=@your_recording.wav" \
  -F "age=45" \
  -F "bmi=26.0"
```

Response:
```json
{
  "risk_label": "lower_risk",
  "risk_probability": 0.167,
  "model_info": "Trained on 607 real participants (AUC 0.77 held-out test set)",
  "disclaimer": "..."
}
```

Or use the standalone script for quick command-line tests:
```bash
python predict_from_audio.py your_recording.wav 45 26.0
```

## What's actually validated vs. what isn't

**Validated (I tested this):**
- The classifier's AUC 0.77 is a real number from a real held-out test
  set of real diagnosed people (not synthetic, not self-reported).
- The BYOL-S embedding extraction pipeline runs correctly on real
  audio files and produces the correct 2048-dim shape.
- The full chain (audio in -> prediction out) runs without errors via
  a live HTTP request.

**NOT yet validated — be upfront about this with judges:**
- How well this generalizes to YOUR recording setup. The Colive Voice
  study collected audio through their own specific app/protocol.
  Recordings from a different phone, mic, room acoustics, or accent
  distribution than their training data may embed differently, and
  accuracy on genuinely new recordings from your own testers is
  unknown until you try it.
- The BYOL-S model itself was pretrained on general audio/speech, not
  specifically fine-tuned on this diabetes task — only the final
  classifier on top of the frozen embeddings was trained on the 607
  participants.
- No independent test on volunteers you personally recorded yet. Do
  this before your final demo if you can — record a few known
  diabetic/non-diabetic volunteers and see how the predictions land.

## For your submission

This is now a genuinely stronger claim than before: "trained and
validated on a real, published, peer-reviewed research dataset of 607
people with real diagnostic labels, using the same voice embedding
model (BYOL-S) as the original study." That's defensible and specific
— cite the Colive Voice study by name.
