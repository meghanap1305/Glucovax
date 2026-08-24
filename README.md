# Glucovox — Voice-Based Type 2 Diabetes Risk Screening

**iQOO Hackathon 2026 Submission**

Glucovox screens for type 2 diabetes risk from a short voice recording,
combined with basic clinical inputs (age, BMI). It's built on published,
peer-reviewed research — not a from-scratch guess — and trained on real
diagnostic data from an actual clinical voice study, not synthetic or
self-reported data.

---

## The problem

Nearly half of adults living with diabetes worldwide are undiagnosed —
240 million people, according to the International Diabetes Federation.
Standard screening (HbA1c, fasting glucose, OGTT) requires a clinic
visit, a blood draw, and cost that isn't accessible everywhere. If
voice — something a smartphone can already capture — carries a usable
signal, it opens a screening pathway that's free, non-invasive, and
scalable to anyone with a phone.

## What the research actually says (and what it doesn't)

Before writing a line of code, we went through the published literature
to understand what's actually been shown to work, rather than assume
voice-glucose detection is either solved or fantasy. Four papers shaped
our approach directly:

1. **Kaufman et al., "Acoustic Analysis and Prediction of Type 2
   Diabetes Mellitus Using Smartphone-Recorded Voice Segments,"**
   *Mayo Clinic Proceedings: Digital Health* (2023).
   Klick Labs recorded 267 participants speaking a short phrase into a
   smartphone and classified diabetic vs. non-diabetic status using
   acoustic features plus basic health data (age, sex, height, weight),
   reaching 89% accuracy in women and 86% in men. This told us two
   things: (a) classification, not an exact glucose number, is the
   tractable target, and (b) fusing voice with simple clinical fields
   measurably helps.

2. **Klick Labs, "Linear Effects of Glucose Levels on Voice Fundamental
   Frequency in Type 2 Diabetes and Individuals with Normoglycemia,"**
   *Scientific Reports* (2024).
   505 participants wore continuous glucose monitors for two weeks
   while recording their voice multiple times daily. The study found a
   real, linear relationship between glucose level and vocal pitch
   (F0) — but explicitly noted pitch alone isn't sufficient to predict
   an exact glucose value, and that predicting a precise mg/dL number
   from voice remains unsolved. This is why Glucovox predicts a **risk
   category**, not a glucose reading — we're not overclaiming past what
   the field has actually demonstrated.

3. **Guo, Peng, Hu, Lu, Chen, "A Novel Machine Learning-Driven Voice and
   Clinical Biomarkers Framework for Robust Prediction of Type 2
   Diabetes Mellitus"** (2025).
   The strongest classification result we found: 3,129 participants,
   88 openSMILE acoustic features fused with 30 clinical features,
   LASSO feature selection feeding an XGBoost classifier. Voice alone
   reached AUC 80.8%; voice **fused with clinical features** reached AUC
   95.2%. This is the specific architecture pattern (LASSO selection +
   XGBoost, voice+clinical fusion) our pipeline is modeled on.

4. **Elbéji, Pizzimenti, Aguayo, Fischer, Ayadi, Mauvais-Jarvis,
   Riveline, Despotovic, Fagherazzi, "A voice-based algorithm can
   predict type 2 diabetes status in USA adults: Findings from the
   Colive Voice study,"** *PLOS Digital Health* (2024).
   607 US participants (Colive Voice / VOCADIAB study, Luxembourg
   Institute of Health, ClinicalTrials.gov NCT04848623), each with a
   real clinical T2D diagnosis, age, BMI, and voice recordings
   processed into Hybrid BYOL-S embeddings. **This is the actual
   dataset our current model is trained and validated on** — real
   people, real diagnoses, publicly released by the study authors.

## What we built

Given that fusion and classification (not raw regression) is what the
literature actually supports, we built:

```
User records voice (4-part protocol: sentence, sustained "aaa",
sustained "iii", free-form description)
        |
Audio preprocessing (silence removal, normalization, filtering)
        |
Voice embedding extraction (Hybrid BYOL-S, 2048-dim — same
representation used in the Colive Voice study)
        |
Fusion with clinical inputs (age, BMI)
        |
XGBoost classifier, trained on 607 real Colive Voice / VOCADIAB
participants
        |
Risk probability + label (higher_risk / lower_risk)
```

The four-recording capture protocol (full sentence + two sustained
vowels + free speech) is our own design choice, going beyond a single
fixed phrase — sustained vowels isolate vocal cord behavior without
consonant/word confounds, a technique used in voice-biomarker research
more broadly.

## Real results, not synthetic

We didn't stop at "the code runs" — we validated the model on a real
held-out set of real people:

| Metric | Result |
|---|---|
| Dataset | 607 real participants (Colive Voice / VOCADIAB, PLOS Digital Health 2024) |
| Held-out test AUC | 0.76 |
| 5-fold cross-validation AUC | 0.77 ± 0.02 |
| Recall (sensitivity) | 0.67 |

This is lower than the fused-feature paper's 95.2% AUC, and that gap is
expected and worth explaining rather than hiding: our model uses only
age and BMI as clinical inputs (2 features), while the top-performing
paper fused 30 clinical features including biochemical and lifestyle
data we don't currently collect. The gap between our result and theirs
is a direct, honest measure of how much richer clinical input would
improve accuracy — a concrete, well-understood next step, not a mystery.

## What's validated vs. what's still open

**Validated:**
- Real 607-person dataset with real T2D diagnoses, not self-reported
  or synthetic
- Held-out AUC 0.76, 5-fold CV AUC 0.77 ± 0.02 — measured on data the
  model never saw during training
- Full pipeline (recording → embedding → classifier → risk output)
  runs end-to-end and returns a live prediction via our API

**Open, and stated plainly to avoid overclaiming:**
- Generalization to our own recording setup (different phones, mics,
  rooms, accents than the original study's cohort) is untested at
  scale — we're actively collecting a small volunteer test set to
  measure this before demo day
- Recall of 0.67 means roughly 1 in 3 actual diabetic cases in our
  held-out test are missed — this is a screening aid to prompt a real
  test, not a diagnostic replacement
- Only 2 clinical features are fused in currently (age, BMI); the
  literature suggests more (family history, hypertension, lifestyle
  factors) would materially close the gap to the 95% AUC ceiling

## Tech stack

- **Frontend:** React Native, Expo, Expo Router — 4-step voice capture
  flow
- **Backend:** FastAPI, Python
- **Voice embedding:** Hybrid BYOL-S (`serab-byols`), pretrained,
  2048-dim, matching the Colive Voice study's representation
- **Classifier:** XGBoost, trained on the Colive Voice / VOCADIAB
  public release
- **Audio processing:** librosa, soundfile

## Running it

See `SETUP.md` for full install and run instructions.

```bash
pip install -r requirements.txt
python -m uvicorn app:app --port 8010
```

## Data & code attribution

The Colive Voice / VOCADIAB dataset and baseline cross-validation code
are publicly released by the original study authors at
[github.com/LIHVOICE/Voice-and-diabetes-VOCADIAB](https://github.com/LIHVOICE/Voice-and-diabetes-VOCADIAB),
used here under their public release for research purposes. Our
contribution is the fused-feature classifier design, the multi-step
capture protocol, the portable model-serving pipeline, and the mobile
app built around it — not the underlying dataset or embedding model,
which we gratefully build on rather than claim as our own.

## Team

*(add your names here)*
