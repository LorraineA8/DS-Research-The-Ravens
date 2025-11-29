"""
SHAP Explanation Page for Diabetes CDSS

This page provides explainable AI insights using SHAP values
to understand why the model made a specific prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_model,
    load_shap_explainer,
    get_shap_explanation,
    create_feature_contribution_chart
)

# Page configuration
st.set_page_config(
    page_title="SHAP Explanation - Diabetes CDSS",
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
    }
    
    h1 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    .stExpander {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
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

# Title
st.title("SHAP Explanation - Understanding Your Prediction")
st.markdown("""
This page explains **why** the model predicted your diabetes risk using SHAP (SHapley Additive exPlanations) values.
SHAP values show how each feature contributes to the final prediction.
""")

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    """Load model and SHAP explainer."""
    model, model_metadata, selected_features = load_model()
    explainer_data = load_shap_explainer()
    return model, model_metadata, selected_features, explainer_data

try:
    model, model_metadata, selected_features, explainer_data = load_model_and_explainer()
except FileNotFoundError as e:
    st.error(f"Model files not found!")
    st.error(str(e))
    st.markdown("---")
    st.markdown("### How to Fix This")
    st.markdown("""
    You need to run the Jupyter notebooks first:
    
    1. **Notebook 1**: `notebooks/1_data_understanding_and_cleaning.ipynb`
    2. **Notebook 2**: `notebooks/2_model_training_and_evaluation.ipynb` **Critical!**
    3. **Notebook 3**: `notebooks/3_xai_and_fairness.ipynb` **Required for SHAP!**
    
    See `SETUP_INSTRUCTIONS.md` for detailed steps.
    """)
    st.stop()
except Exception as e:
    st.error(f"Error loading model or explainer: {str(e)}")
    st.exception(e)
    st.stop()

# Get user input and prediction
user_input = st.session_state.user_input
input_data = st.session_state.input_data
prediction_result = st.session_state.prediction_result

# Generate SHAP explanation
if explainer_data is None:
    st.error("SHAP explainer not found. Please run Notebook 3 to generate the explainer.")
    st.stop()

# Generate explanation
with st.spinner("Generating SHAP explanation..."):
    shap_explanation = get_shap_explanation(
        model,
        explainer_data,
        input_data,
        selected_features
    )

# Store in session state for dashboard
st.session_state.shap_explanation = shap_explanation

# Check for errors
if 'error' in shap_explanation:
    st.error(f"Error generating explanation: {shap_explanation['error']}")
    st.stop()

# Display explanation
st.markdown("---")
st.markdown("### Feature Contributions to Prediction")

# Feature contributions chart
if shap_explanation.get('feature_contributions'):
    contribution_chart = create_feature_contribution_chart(
        shap_explanation['feature_contributions']
    )
    st.plotly_chart(contribution_chart, use_container_width=True)

# Detailed feature contributions table
st.markdown("---")
st.markdown("### Detailed Feature Contributions")

# Create DataFrame for better display
contributions_data = []
for feature, contribution in shap_explanation['feature_contributions'].items():
    value = user_input.get(feature, 0)
    contributions_data.append({
        'Feature': feature,
        'Your Value': f"{value:.2f}",
        'Contribution': f"{contribution:+.4f}",
        'Impact': 'Increases Risk' if contribution > 0 else 'Decreases Risk',
        'Magnitude': abs(contribution)
    })

contributions_df = pd.DataFrame(contributions_data)
contributions_df = contributions_df.sort_values('Magnitude', ascending=False)

# Display table
st.dataframe(
    contributions_df[['Feature', 'Your Value', 'Contribution', 'Impact']],
    use_container_width=True,
    hide_index=True
)

# Natural language explanation
st.markdown("---")
st.markdown("### Explanation in Plain English")

st.markdown(shap_explanation.get('explanation_text', 'Explanation not available.'))

# Top contributing factors
st.markdown("---")
st.markdown("### Top Contributing Factors")

# Sort by absolute contribution
sorted_features = sorted(
    shap_explanation['feature_contributions'].items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

col1, col2 = st.columns(2)

for i, (feature, contribution) in enumerate(sorted_features[:6]):  # Top 6
    col = col1 if i % 2 == 0 else col2
    
    with col:
        value = user_input.get(feature, 0)
        contribution_pct = contribution * 100
        
        if contribution > 0:
            st.error(f"""
            **{feature}** = {value:.2f}
            
            **Increases risk by {abs(contribution_pct):.1f}%**
            """)
        else:
            st.success(f"""
            **{feature}** = {value:.2f}
            
            **Decreases risk by {abs(contribution_pct):.1f}%**
            """)

# SHAP Value Interpretation Guide
st.markdown("---")
with st.expander("Understanding SHAP Values"):
    st.markdown("""
    **What are SHAP Values?**
    
    SHAP (SHapley Additive exPlanations) values explain how each feature contributes 
    to the model's prediction. They are based on game theory and provide a fair way 
    to attribute the prediction to each input feature.
    
    **How to Read SHAP Values:**
    
    - **Positive SHAP Value** (Red): This feature increases your risk of diabetes
    - **Negative SHAP Value** (Blue): This feature decreases your risk of diabetes
    - **Larger Absolute Value**: The feature has a stronger impact on the prediction
    
    **Example:**
    - If Glucose has a SHAP value of +0.15, it means your glucose level increased 
      your diabetes risk by 15 percentage points
    - If BMI has a SHAP value of -0.08, it means your BMI decreased your risk 
      by 8 percentage points (relative to average)
    
    **Base Value:**
    The base value represents the average prediction across all patients. Your 
    individual features then push the prediction up or down from this base value.
    
    **Why This Matters:**
    Understanding which factors contribute most to your risk helps you:
    - Focus on the most important risk factors to address
    - Understand why the model made a specific prediction
    - Make informed decisions about lifestyle changes
    - Build trust in the AI system's recommendations
    """)

# Model Information
st.markdown("---")
st.markdown("### Model Information")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown(f"""
    **Model Type**: {model_metadata['model_name']}
    
    **Model Performance**:
    - Accuracy: {model_metadata['metrics']['accuracy']:.2%}
    - Precision: {model_metadata['metrics']['precision']:.2%}
    - Recall: {model_metadata['metrics']['recall']:.2%}
    - F1-Score: {model_metadata['metrics']['f1_score']:.2%}
    - ROC AUC: {model_metadata['metrics']['roc_auc']:.2%}
    """)

with col_info2:
    st.markdown(f"""
    **Training Data**:
    - Training samples: {model_metadata['training_samples']:,}
    - Test samples: {model_metadata['test_samples']:,}
    
    **Features Used**: {len(selected_features)}
    """)
    
    with st.expander("View all features"):
        for i, feature in enumerate(selected_features, 1):
            st.write(f"{i}. {feature}")

# Footer
st.markdown("---")
st.markdown("""
**Note**: SHAP explanations help understand model predictions but should be 
interpreted alongside clinical judgment. Always consult healthcare professionals 
for medical decisions.
""")

