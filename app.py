# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import os
import logging
from fastapi.staticfiles import StaticFiles
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config - change MODEL_DIR if you saved models elsewhere
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
OHE_PATH = os.path.join(MODEL_DIR, "onehot_encoder.joblib")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")

# Load artifacts
try:
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    onehot_encoder    = joblib.load(OHE_PATH)
    xgb_model         = joblib.load(XGB_PATH)
    logger.info("Loaded TF-IDF, OHE, and XGB model.")
except Exception as e:
    logger.exception("Failed to load model artifacts. Check MODEL_DIR and file names.")
    # allow app to run but predictions will raise

# Pydantic request schema
class JobPost(BaseModel):
    text: str
    employment_type: str = 'Unknown'
    required_experience: str = 'Unknown'
    required_education: str = 'Unknown'
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0

app = FastAPI(title="Fake Job Detector API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://127.0.0.1:5500"] if using VS Code Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    ok = True
    try:
        ok = (tfidf_vectorizer is not None) and (onehot_encoder is not None) and (xgb_model is not None)
    except Exception:
        ok = False
    return {"healthy": ok}

def build_combined_row(post: JobPost):
    """Return combined csr_matrix for a single JobPost"""
    text_vec = tfidf_vectorizer.transform([post.text])
    cat_df = pd.DataFrame([{
        'employment_type': post.employment_type,
        'required_experience': post.required_experience,
        'required_education': post.required_education
    }])
    cat_vec = onehot_encoder.transform(cat_df)
    num_arr = np.array([[post.telecommuting, post.has_company_logo, post.has_questions]])
    num_sparse = csr_matrix(num_arr)
    combined = hstack([text_vec, cat_vec, num_sparse], format='csr')
    return combined

@app.post("/predict")
def predict(post: JobPost):
    try:
        combined = build_combined_row(post)
        try:
            proba = float(xgb_model.predict_proba(combined)[:,1][0])
        except Exception:
            # fallback convert to dense for single-row
            proba = float(xgb_model.predict_proba(combined.toarray())[:,1][0])
        threshold = float(os.environ.get("THRESHOLD", 0.5))
        label = "Fake" if proba >= threshold else "Real"
        return {"label": label, "proba_fake": proba, "threshold_used": threshold}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")

# Optional explain endpoint using XGBoost pred_contribs
@app.post("/explain")
def explain(post: JobPost, top_n: int = 10):
    try:
        combined = build_combined_row(post)
        booster = xgb_model.get_booster()
        dmat = xgb.DMatrix(combined)
        contribs = booster.predict(dmat, pred_contribs=True).ravel()
        bias = float(contribs[-1])
        feature_contribs = contribs[:-1]

        tfidf_feats = list(tfidf_vectorizer.get_feature_names_out())
        cat_feats = list(onehot_encoder.get_feature_names_out(['employment_type','required_experience','required_education']))
        num_feats = ['telecommuting','has_company_logo','has_questions']
        all_feats = tfidf_feats + cat_feats + num_feats

        pos_idx = np.argsort(feature_contribs)[-top_n:][::-1]
        neg_idx = np.argsort(feature_contribs)[:top_n]

        positive = [{"feature": all_feats[i], "contrib": float(feature_contribs[i])} for i in pos_idx]
        negative = [{"feature": all_feats[i], "contrib": float(feature_contribs[i])} for i in neg_idx]

        return {"bias": bias, "positive": positive, "negative": negative}
    except Exception as e:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail="Explain error")

app.mount("/", StaticFiles(directory="Frontend", html=True), name="Frontend")
