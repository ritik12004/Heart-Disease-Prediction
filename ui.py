

import streamlit as st
import pickle
import numpy as np

# Load model & scaler
# Ensure these files are in the same directory
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# ---- User Inputs ----
# 1. Numerical Features (These need scaling)
st.header("Patient Vitals")
age = st.number_input("Age", min_value=1, max_value=120, value=45)
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
max_hr = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)

# 2. Categorical Features (These need manual encoding)
st.header("Medical History")
sex = st.selectbox("Sex", ["Male", "Female"])
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)", "TA (Typical Angina)"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST (ST-T wave abnormality)", "LVH (Left Ventricular Hypertrophy)"])

# ---- Processing & Prediction ----
if st.button("Predict"):
    
    # --- A. Handle Numerical Scaling ---
    # The scaler expects exactly 5 columns: Age, BP, Chol, MaxHR, Oldpeak
    features_to_scale = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])
    scaled_features = scaler.transform(features_to_scale)
    
    # Extract scaled values for cleaner code usage below
    s_age = scaled_features[0][0]
    s_bp = scaled_features[0][1]
    s_chol = scaled_features[0][2]
    s_maxhr = scaled_features[0][3]
    s_oldpeak = scaled_features[0][4]

    # --- B. Manual One-Hot Encoding ---
    # We must match the EXACT columns your model was trained on.
    # Based on your previous snippet, the columns are:
    # ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','fastingBS',
    #  'sex','chestPainType_ASY','chestPainType_NAP','chestPainType_TA',
    #  'restingECG_Normal','restingECG_ST','exerciseAngina_YES',
    #  'st_Slope_Flat','st_Slope_Up']

    # 1. Sex (Assuming Male=1, Female=0)
    val_sex = 1 if sex == "Male" else 0
    
    # 2. Fasting BS (Assuming Yes=1, No=0)
    val_fbs = 1 if fasting_bs == "Yes" else 0

    # 3. Chest Pain (Model has: ASY, NAP, TA. 'ATA' is the hidden/dropped category)
    cp_asy = 1 if "ASY" in chest_pain else 0
    cp_nap = 1 if "NAP" in chest_pain else 0
    cp_ta  = 1 if "TA" in chest_pain else 0

    # 4. Resting ECG (Model has: Normal, ST. 'LVH' is likely the hidden category)
    ecg_norm = 1 if "Normal" in resting_ecg else 0
    ecg_st   = 1 if "ST" in resting_ecg else 0

    # 5. Exercise Angina (Model has: YES)
    ex_angina = 1 if exercise_angina == "Yes" else 0

    # 6. ST Slope (Model has: Flat, Up. 'Down' is likely the hidden category)
    slope_flat = 1 if st_slope == "Flat" else 0
    slope_up   = 1 if st_slope == "Up" else 0

    # --- C. Combine into Final Array ---
    # The order MUST match your model's training columns exactly
    final_input = np.array([[
        s_age, s_bp, s_chol, s_maxhr, s_oldpeak,  # 5 Scaled Numerical
        val_fbs,                                  # FastingBS
        val_sex,                                  # Sex
        cp_asy, cp_nap, cp_ta,                    # Chest Pain One-Hot
        ecg_norm, ecg_st,                         # ECG One-Hot
        ex_angina,                                # Exercise Angina
        slope_flat, slope_up                      # ST Slope One-Hot
    ]])

    # Debugging: Print shape to terminal to verify
    print(f"Input Shape: {final_input.shape}") 

    # --- D. Predict ---
    prediction = model.predict(final_input)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")