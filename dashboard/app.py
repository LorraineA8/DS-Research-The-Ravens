# app.py – FINAL VERSION FOR 25th NOV
import streamlit as st
import pandas as pd
import pickle
import shap
import joblib
import matplotlib.pyplot as plt

model = pickle.load(open('xgboost_t2dm.pkl', 'rb'))
explainer = joblib.load('shap_explainer.joblib')

st.set_page_config(page_title="Team Ravens T2DM Tool", layout="centered")
st.title("Team Ravens – T2DM Risk Prediction Tool")
st.markdown("### Uganda Ministry of Health • WHO Aligned • Explainable AI")

# Login (simple)
if st.text_input("Username") != "ravens" or st.text_input("Password", type="password") != "ncd2025":
    st.stop()

col1, col2 = st.columns(2)
with col1:
    preg = st.slider("Pregnancies", 0, 17, 1)
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
    bp = st.slider("Blood Pressure", 0, 122, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
with col2:
    insulin = st.slider("Insulin", 0, 846, 80)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0, 0.1)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5, 0.01)
    age = st.slider("Age", 21, 81, 30)

features = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                        columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age'])

if st.button("Predict Risk", type="primary"):
    prob = model.predict_proba(features)[0,1]
    pred = model.predict(features)[0]
    
    # Hybrid rule override
    if glucose >= 200 or bmi >= 40:
        final_risk = "HIGH RISK – IMMEDIATE REFERRAL"
    else:
        final_risk = "HIGH RISK" if pred == 1 else "LOW RISK"
    
    st.metric("Diabetes Risk Probability", f"{prob:.1%}")
    if "HIGH" in final_risk:
        st.error(final_risk)
    else:
        st.success(final_risk)
    
    # SHAP explanation
    shap_values = explainer.shap_values(features)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=features.iloc[0]), show=False)
    st.pyplot(fig)
    
    st.info("Top contributing factors shown above – fully explainable as required")
