# Credit Card Fraud Detection API
A real-time fraud detection system using Autoencoders for anomaly detection and a calibrated stacked ensemble classifier. Deployed using FastAPI and Ngrok. </br>

## Features
• Autoencoder for learning normal transaction patterns and extracting latent features </br>
• Reconstruction error used as an additional anomaly signal</br>
• Enriched feature set: latent features + reconstruction error</br>
• Ensemble models (Random Forest, AdaBoost, Gradient Boosting, LightGBM) with hyperparameter tuning</br>
• Stacked ensemble (LightGBM + Random Forest) using logistic regression as meta-learner</br>
• Platt Scaling (logistic regression) for probability calibration</br>
• MCC-based threshold tuning for optimal classification</br>
• Deployed using FastAPI with real-time inference support</br>
• Exposed via public URL using Ngrok</br>

## Model Pipeline
• Preprocessing: Performed EDA, dropped unused columns, scaled features, and visualized class separation with t-SNE</br>
• Autoencoder: Trained on legitimate transactions using a deep architecture with regularization and early stopping</br>
• Feature Engineering: Extracted latent features and reconstruction error, combined into a final feature set</br>
• Ensemble Classification: Trained and tuned multiple classifiers using StratifiedKFold and RandomizedSearchCV</br>
• Stacking & Calibration: Built a stacked model and calibrated outputs using Platt Scaling; optimized decision threshold via MCC</br>
• API Deployment: Exposed prediction endpoint /predict_fraud using FastAPI, tested via local and Ngrok tunnels</br>

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
cd app
uvicorn main:app --reload

# OR run in a Colab cell
!ngrok config add-authtoken <your-token>
!uvicorn main:app --host 0.0.0.0 --port 8000
```
## API Endpoint
```
POST /predict_fraud
```
Input JSON:
```
{
  "features": [0.1, -1.2, 0.44, ..., 0.91]  // 25 floats
}
```
Response:
```
{
  "fraud_probability": 0.8742,
  "is_fraud": true
}
```
## Project Structure
```bash
creditcard-fraud-detection-api/
├── app/
│   ├── main.py               # FastAPI backend
│   ├── *.keras / *.pkl       # Saved models
├── data/
│   ├── creditcard.csv
│   ├── processed_df.csv
├── notebooks/                # Jupyter notebooks (training + analysis)
├── outputs/                  # Results, CSVs, t-SNE images
├── predict.py                # Test script for API
├── requirements.txt
├── README.md
├── .gitignore
```
## Data Access

Due to GitHub's file size limitations, the following files are not included in this repository:</br>
- `data/creditcard.csv`</br>
- `data/processed_df.csv`</br>

To run the notebooks or retrain the models:</br>
1. Download the original dataset from [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).</br>
2. Place it in the `data/` folder as `creditcard.csv`.</br>
3. Run `1_data_preprocessing.ipynb` to generate `processed_df.csv`.</br>

Alternatively, if you're only testing the API, you can use the pretrained models already included.</br>

## Visualizations
tsne_initial_2d.png: t-SNE on raw features</br>
tsne_latent_2d.png: t-SNE on encoder output</br>

## Author
Saanvi Malik</br>
Credit Card Fraud Detection using Autoencoder + Ensemble + Calibration</br>
Made with ❤️ at IIT Roorkee</br>
