from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List
import numpy as np
import joblib
import tensorflow as tf

class Transaction(BaseModel):
    features: conlist(float, min_items=25, max_items=25)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Autoencoder + Ensemble + Calibration",
    version="1.0"
)

ae = None
encoder = None
scaler = None
stack_model = None
calibrator = None
threshold = None 

@app.on_event("startup")
def load_models():
    global ae, encoder, scaler, stack_model, calibrator, threshold
    ae = tf.keras.models.load_model("app/ae_model.keras")
    encoder = tf.keras.models.load_model("app/encoder_model.keras")
    scaler = joblib.load("app/scaler.pkl")
    stack_model = joblib.load("app/stack_model.pkl")
    calibrator = joblib.load("app/calibrator.pkl")
    threshold = 0.037 

def extract_features(x_raw):
    x_scaled = scaler.transform([x_raw])
    x_reconstructed = ae.predict(x_scaled, batch_size=1)
    reconstruction_error = np.mean(np.abs(x_scaled - x_reconstructed), axis=1).reshape(-1, 1)
    latent_features = encoder.predict(x_scaled, batch_size=1)
    final_features = np.hstack([latent_features, reconstruction_error])
    return final_features

@app.post("/predict_fraud")
async def predict(transaction: Transaction):
    try:
        input_data = np.array(transaction.features)
        features = extract_features(input_data)
        prob = stack_model.predict_proba(features)[:, 1]
        calibrated = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
        is_fraud = int(calibrated[0] >= threshold)
        return {
            "fraud_probability": float(calibrated[0]),
            "is_fraud": bool(is_fraud)
        }
    except Exception as e:
        return {"error": str(e)}
