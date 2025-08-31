import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction")
st.caption("Educational demo — not a medical diagnosis.")


# Load artifacts 
artifactsDirectory = "artifacts"
preprocessorPath = os.path.join(artifactsDirectory, "preprocessor.joblib")
tunedModels = [
    ("Logistic Regression (tuned)", os.path.join(artifactsDirectory, "model_lr_tuned.joblib")),
    ("KNN (tuned)",                 os.path.join(artifactsDirectory, "model_knn_tuned.joblib")),
    ("Random Forest (tuned)",       os.path.join(artifactsDirectory, "model_rf_tuned.joblib")),
]


availableModels = [(name, path) for name, path in tunedModels if os.path.exists(path)]


preprocessor = joblib.load(preprocessorPath)

with st.sidebar:
    st.header("Model")
    modelName = st.selectbox("Choose a tuned model", [n for n, _ in availableModels], index=0)
    modelPath = dict(availableModels)[modelName]
    model = joblib.load(modelPath)

# Input form

rawFeatureOrder = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
    sex = st.selectbox("Sex", ["M", "F"])
    chestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    restingBp = st.number_input("Resting BP (mm Hg)", min_value=70, max_value=220, value=120, step=1)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
    fastingBs = 1 if st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No (0)", "Yes (1)"], horizontal=True) == "Yes (1)" else 0
with col2:
    restingEcg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxHr = st.number_input("MaxHR (60–202)", min_value=60, max_value=202, value=150, step=1)
    exerciseAngina = st.selectbox("Exercise-induced angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    stSlope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

rawInput = pd.DataFrame([{
    "Age": int(age),
    "Sex": sex,
    "ChestPainType": chestPainType,
    "RestingBP": int(restingBp),
    "Cholesterol": int(cholesterol),
    "FastingBS": int(fastingBs),
    "RestingECG": restingEcg,
    "MaxHR": int(maxHr),
    "ExerciseAngina": exerciseAngina,
    "Oldpeak": float(oldpeak),
    "ST_Slope": stSlope
}], columns=rawFeatureOrder)

st.divider()
st.write("### Your inputs")
st.dataframe(rawInput, use_container_width=True)


# Prediction
if st.button("Predict"):
    try:
        processedInput = preprocessor.transform(rawInput)
        probabilityClass1 = float(model.predict_proba(processedInput)[:, 1][0])
        probabilityClass1 = float(np.clip(probabilityClass1, 1e-8, 1 - 1e-8))  # guard rails
        predictedLabel = int(probabilityClass1 >= 0.50)

        st.success(f"Predicted probability of Heart Disease = **{probabilityClass1:.1%}**")
        if predictedLabel == 1:
            st.error("Prediction: Chances of **Heart Disease **")
        else:
            st.info("Prediction: **Normal**")

        with st.expander("Details"):
            st.write(f"Model: **{modelName}**")
            st.write("This demo is for educational purposes and not a medical diagnosis.")
    except Exception as e:
        st.exception(e)
