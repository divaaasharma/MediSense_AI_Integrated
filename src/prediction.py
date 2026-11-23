import joblib
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load trained models
random_forest_model = joblib.load(
    MODEL_DIR / "random_forest_diabetes.pkl"
)

logistic_model = joblib.load(
    MODEL_DIR / "logistic_regression_diabetes.pkl"
)

scaler = joblib.load(
    MODEL_DIR / "standard_scaler.pkl"
)

# Required input features in training order
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]


def predict_diabetes_risk(patient_data):

    # Convert input into DataFrame
    input_df = pd.DataFrame([patient_data], columns=FEATURES)

    # Scale input for Logistic Regression
    scaled_input = scaler.transform(input_df)

    # Predictions from both models
    rf_prediction = random_forest_model.predict(input_df)[0]
    lr_prediction = logistic_model.predict(scaled_input)[0]

    # Probability of higher-risk class
    rf_probability = random_forest_model.predict_proba(input_df)[0][1]
    lr_probability = logistic_model.predict_proba(scaled_input)[0][1]

    return {
        "Random Forest Prediction": int(rf_prediction),
        "Logistic Regression Prediction": int(lr_prediction),
        "Random Forest Probability": round(float(rf_probability) * 100, 2),
        "Logistic Regression Probability": round(float(lr_probability) * 100, 2)
    }