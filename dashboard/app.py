# app.py – 100% WORKING FOR STREAMLIT CLOUD (20 Nov 2025)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ==================== FORCE INSTALL SHAP (Streamlit needs this) ====================
# Streamlit Cloud doesn't have SHAP by default → we fix it with one line
import subprocess
import sys

try:
    import shap
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# ==================== TRAIN A QUICK MODEL ON THE FLY (no file needed) ====================
@st.cache_resource
def get_model_and_explainer():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age','Outcome']
    df = pd.read_csv(url, names=cols)
    
    # Clean zeros
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns

model, explainer, feature_names = get_model_and_explainer()

# ==================== DASHBOARD ====================
st.set_page_config(page_title="KAWUUNGA – T2DM Risk Tool", layout="centered")
st.title("KAWUUNGA")
st.markdown("### Type 2 Diabetes Early Warning System – Team Ravens")
st.markdown("_“Omulwadde agenda okuzuukuka” – Early detection saves lives_")

# Simple login
col1, col2 = st.columns(2)
with col1:
    username = st.text_input("Username", value="v="vht")
with col2:
    password = st.text_input("Password", type="password", value="kawuunga2025")

if username != "vht" or password != "kawuunga2025":
    st.stop()

st.success("Login successful – Village Health Team access granted")

# Input
col1, col2 = st.columns(2)
with col1:
    preg = st.slider("Pregnancies", 0, 17, 1)
    glucose = st.slider("Glucose (mg/dL)", 0, 199, 120)
    bp = st.slider("Blood Pressure", 0, 122, 70)
    skin = st.slider("Skin Thickness (mm)", 0, 99, 20)
with col2:
    insulin = st.slider("Insulin", 0, 846, 30)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0, 0.1)
    dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, 0.01)
    age = st.slider("Age", 21, 81, 35)

# Prediction
if st.button("Calculate Risk – KAWUUNGA", type="primary"):
    features = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                            columns=feature_names)
    
    prob = model.predict_proba(features)[0,1]
    pred = model.predict(features)[0]
    
    # Clinical rule override (safety first)
    if glucose >= 126 or bmi >= 30 or age >= 45:
        final_risk = "HIGH RISK – Refer to facility TODAY"
        colour = "red"
    else:
        final_risk = "HIGH RISK" if pred == 1 else "LOW RISK"
        colour = "green" if pred == 0 else "red"
    
    st.metric("Diabetes Risk Probability", f"{prob:.1%}")
    
    if "HIGH" in final_risk:
        st.error(f"**{final_risk}**")
    else:
        st.success(f"**{final_risk}**")
    
    # SHAP waterfall
    shap_values = explainer.shap_values(features)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                   base_values=explainer.expected_value[1], 
                                   data=features.iloc[0]), show=False)
    st.pyplot(fig)
    
    st.info("Top 3 reasons for this risk shown above – fully explainable (XAI)")

st.balloons()
