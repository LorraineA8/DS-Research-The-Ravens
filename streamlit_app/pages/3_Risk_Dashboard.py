"""
Risk Dashboard Page for Diabetes CDSS

This page displays comprehensive risk analysis, visualizations,
and personalized recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    get_age_bracket_risk,
    get_recommendations,
    create_risk_gauge,
    create_feature_contribution_chart
)

# Page configuration
st.set_page_config(
    page_title="Risk Dashboard - Diabetes CDSS",
    layout="wide"
)

# Professional Theme CSS
def load_theme_css():
    """Load professional theme CSS"""
    return """
    <style>
    :root {
        --primary-color: #1a4d7a;
        --secondary-color: #2c7a7b;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --success-color: #2d8659;
        --danger-color: #c85d5d;
        --warning-color: #d4a574;
        --info-color: #4a90a4;
    }
    
    h1 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    </style>
    """
    
st.markdown(load_theme_css(), unsafe_allow_html=True)

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first to access this page.")
    st.stop()

# Check if prediction exists
if 'prediction_result' not in st.session_state:
    st.warning("No prediction data found. Please go to the **Input Form** page and make a prediction first.")
    st.stop()

# Get prediction results
prediction_result = st.session_state.prediction_result
user_input = st.session_state.user_input

# Title
st.title("Risk Dashboard")
st.markdown("Comprehensive analysis of diabetes risk prediction")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Risk Score",
        f"{prediction_result['risk_score']:.1f}%",
        delta=f"{prediction_result['risk_score'] - 50:.1f}%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Risk Level",
        prediction_result['risk_level'],
        delta=None
    )

with col3:
    st.metric(
        "Probability (Diabetes)",
        f"{prediction_result['probability_diabetes']*100:.1f}%"
    )

with col4:
    st.metric(
        "Probability (No Diabetes)",
        f"{prediction_result['probability_no_diabetes']*100:.1f}%"
    )

# Risk Gauge Chart
st.markdown("---")
st.markdown("### Risk Score Visualization")

col_gauge1, col_gauge2 = st.columns([2, 1])

with col_gauge1:
    risk_gauge = create_risk_gauge(prediction_result['risk_score'])
    st.plotly_chart(risk_gauge, use_container_width=True)

with col_gauge2:
    st.markdown("#### Risk Interpretation")
    
    risk_score = prediction_result['risk_score']
    if risk_score >= 70:
        st.error("""
        **High Risk (≥70%)**
        
        Immediate medical consultation recommended.
        Comprehensive diabetes screening advised.
        """)
    elif risk_score >= 50:
        st.warning("""
        **Moderate Risk (50-69%)**
        
        Proactive lifestyle changes recommended.
        Regular monitoring advised.
        """)
    else:
        st.success("""
        **Low Risk (<50%)**
        
        Continue healthy lifestyle habits.
        Regular checkups recommended.
        """)

# Age Bracket Risk Analysis
st.markdown("---")
st.markdown("### 👥 Age Bracket Risk Comparison")

age = user_input.get('Age', 0)
age_info = get_age_bracket_risk(age, prediction_result['risk_score'])

col_age1, col_age2 = st.columns(2)

with col_age1:
    st.markdown(f"**Age Bracket**: {age_info['age_bracket']}")
    st.markdown(f"**Your Risk**: {prediction_result['risk_score']:.1f}%")
    st.markdown(f"**Average Risk for Age Group**: {age_info['average_risk']:.1f}%")
    st.markdown(f"**Comparison**: {age_info['comparison_text']}")

with col_age2:
    # Create comparison chart
    fig_age = go.Figure()
    
    fig_age.add_trace(go.Bar(
        x=['Your Risk', 'Average for Age Group'],
        y=[prediction_result['risk_score'], age_info['average_risk']],
        marker_color=['#e74c3c' if prediction_result['risk_score'] > age_info['average_risk'] else '#2ecc71', '#3498db'],
        text=[f"{prediction_result['risk_score']:.1f}%", f"{age_info['average_risk']:.1f}%"],
        textposition='auto'
    ))
    
    fig_age.update_layout(
        title="Risk Comparison by Age Group",
        yaxis_title="Risk Score (%)",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig_age, use_container_width=True)

# Feature Contributions (if SHAP data available)
if 'shap_explanation' in st.session_state:
    st.markdown("---")
    st.markdown("### Feature Contributions")
    
    feature_contributions = st.session_state.shap_explanation.get('feature_contributions', {})
    
    if feature_contributions:
        contribution_chart = create_feature_contribution_chart(feature_contributions)
        st.plotly_chart(contribution_chart, use_container_width=True)
    else:
        st.info("Feature contribution data not available. Visit the SHAP Explanation page to generate it.")
else:
    st.markdown("---")
    st.markdown("### Feature Contributions")
    st.info("Visit the **SHAP Explanation** page to see detailed feature contributions to your risk prediction.")

# Patient Input Summary
st.markdown("---")
st.markdown("### Patient Input Summary")

col_sum1, col_sum2 = st.columns(2)

with col_sum1:
    st.markdown("#### Clinical Measurements")
    input_df = pd.DataFrame(list(user_input.items()), columns=['Feature', 'Value'])
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with col_sum2:
    st.markdown("#### Interpretation")
    
    # BMI interpretation
    bmi = user_input.get('BMI', 0)
    if bmi >= 30:
        bmi_status = "Obese"
    elif bmi >= 25:
        bmi_status = "Overweight"
    elif bmi >= 18.5:
        bmi_status = "Normal"
    else:
        bmi_status = "Underweight"
    
    # Glucose interpretation
    glucose = user_input.get('Glucose', 0)
    if glucose >= 126:
        glucose_status = "Diabetic Range"
    elif glucose >= 100:
        glucose_status = "Pre-diabetic"
    else:
        glucose_status = "Normal"
    
    # BP interpretation
    bp = user_input.get('BloodPressure', 0)
    if bp >= 90:
        bp_status = "High"
    elif bp >= 80:
        bp_status = "Elevated"
    else:
        bp_status = "Normal"
    
    st.markdown(f"""
    - **BMI**: {bmi_status} ({bmi:.1f})
    - **Glucose**: {glucose_status} ({glucose:.0f} mg/dL)
    - **Blood Pressure**: {bp_status} ({bp:.0f} mm Hg)
    """)

# Recommendations
st.markdown("---")
st.markdown("### Personalized Recommendations")

recommendations = get_recommendations(
    prediction_result,
    user_input.get('Age', 0),
    user_input.get('BMI', 0),
    user_input.get('Glucose', 0),
    user_input.get('BloodPressure', 0)
)

for i, rec in enumerate(recommendations, 1):
    st.markdown(rec)

# Action Items
st.markdown("---")
st.markdown("### Next Steps")

if prediction_result['risk_level'] == "High":
    st.error("""
    **Immediate Actions Required:**
    1. Schedule an appointment with a healthcare provider within 1 week
    2. Request comprehensive diabetes screening (HbA1c, fasting glucose, oral glucose tolerance test)
    3. Begin monitoring blood glucose levels daily
    4. Consider referral to an endocrinologist or diabetes specialist
    """)
elif prediction_result['risk_level'] == "Medium":
    st.warning("""
    **Recommended Actions:**
    1. Schedule a preventive health checkup within 3 months
    2. Start regular blood glucose monitoring (weekly)
    3. Implement lifestyle modifications (diet, exercise)
    4. Follow up with healthcare provider in 6 months
    """)
else:
    st.success("""
    **Maintenance Actions:**
    1. Continue healthy lifestyle habits
    2. Annual health checkup recommended
    3. Monitor risk factors regularly
    4. Maintain current healthy practices
    """)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for screening and educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare providers for medical decisions.
""")

