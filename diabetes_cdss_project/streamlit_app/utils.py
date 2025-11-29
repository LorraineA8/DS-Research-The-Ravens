"""
Utility functions for the Diabetes CDSS Streamlit application.

This module contains helper functions for:
- Loading models and metadata
- Making predictions
- Generating counterfactual explanations
- Creating visualizations
- Risk assessment and recommendations
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import shap


def load_model():
    """
    Load the trained model from the models directory.
    
    Returns:
        model: The trained machine learning model
        model_metadata: Dictionary containing model information
        selected_features: List of feature names used by the model
    
    Raises:
        FileNotFoundError: If model files are not found (notebooks need to be run first)
    """
    # Get the base directory (parent of streamlit_app)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"Models directory not found at: {models_dir}\n\n"
            "Please run the Jupyter notebooks first:\n"
            "1. Run notebook 1_data_understanding_and_cleaning.ipynb\n"
            "2. Run notebook 2_model_training_and_evaluation.ipynb\n"
            "3. Run notebook 3_xai_and_fairness.ipynb\n"
            "These notebooks will generate the required model files."
        )
    
    # Load model
    model_path = os.path.join(models_dir, 'best_model_pipeline.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n\n"
            "Please run notebook 2_model_training_and_evaluation.ipynb first.\n"
            "This notebook trains the models and saves the best one."
        )
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Load metadata
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found at: {metadata_path}\n\n"
            "Please run notebook 2_model_training_and_evaluation.ipynb first."
        )
    
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    
    # Load selected features
    features_path = os.path.join(models_dir, 'selected_features.json')
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Selected features file not found at: {features_path}\n\n"
            "Please run notebook 1_data_understanding_and_cleaning.ipynb first."
        )
    
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
    
    return model, model_metadata, selected_features


def load_shap_explainer():
    """
    Load the SHAP explainer for model interpretability.
    
    Returns:
        explainer: SHAP explainer object or data for creating explainer
    """
    # Get the base directory (parent of streamlit_app)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    explainer_path = os.path.join(models_dir, 'shap_explainer.pkl')
    
    try:
        explainer_data = joblib.load(explainer_path)
        return explainer_data
    except FileNotFoundError:
        return None


def get_feature_ranges(selected_features):
    """
    Get reasonable ranges for each feature for counterfactual generation.
    
    Args:
        selected_features: List of feature names
    
    Returns:
        dict: Dictionary mapping feature names to (min, max, step) tuples
    """
    ranges = {
        'Age': (18, 100, 1),
        'BMI': (15.0, 50.0, 0.5),
        'BloodPressure': (40.0, 120.0, 1.0),
        'Glucose': (50.0, 300.0, 1.0),
        'DiabetesPedigreeFunction': (0.0, 2.5, 0.1),
        'Pregnancies': (0, 20, 1)
    }
    
    # Return only ranges for features that exist
    return {k: ranges[k] for k in selected_features if k in ranges}


def prepare_input_data(user_input, selected_features):
    """
    Prepare user input data for model prediction.
    
    Args:
        user_input: Dictionary containing user input values
        selected_features: List of feature names expected by the model
    
    Returns:
        pd.DataFrame: Prepared input data in the correct format
    """
    # Create DataFrame from user input
    input_df = pd.DataFrame([user_input])
    
    # Ensure all required features are present
    for feature in selected_features:
        if feature not in input_df.columns:
            # Use default value (0) for missing features
            # In production, you might want to use median/mean from training data
            input_df[feature] = 0
    
    # Reorder columns to match training data order
    input_df = input_df[selected_features]
    
    return input_df


def make_prediction(model, input_data):
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained machine learning model
        input_data: DataFrame with input features
    
    Returns:
        dict: Dictionary containing prediction results
    """
    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    prediction_class = model.predict(input_data)[0]
    
    # Calculate risk score (probability of diabetes)
    risk_score = probabilities[1] * 100
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 50:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        'prediction_class': int(prediction_class),
        'risk_score': risk_score,
        'risk_level': risk_level,
        'probability_diabetes': probabilities[1],
        'probability_no_diabetes': probabilities[0]
    }


def get_shap_explanation(model, explainer_data, input_data, selected_features):
    """
    Generate SHAP explanation for a prediction.
    
    Args:
        model: Trained machine learning model
        explainer_data: SHAP explainer or data to create explainer
        input_data: DataFrame with input features
        selected_features: List of feature names
    
    Returns:
        dict: Dictionary containing SHAP values and explanation
    """
    try:
        # Check if explainer_data is a direct explainer or needs to be created
        if isinstance(explainer_data, dict) and 'explainer_type' in explainer_data:
            # Recreate KernelExplainer
            explainer = shap.KernelExplainer(
                model.predict_proba,
                explainer_data['background_data']
            )
        else:
            # Use existing explainer (TreeExplainer)
            explainer = explainer_data
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_data)
        
        # Handle binary classification (get positive class values)
        # SHAP values can be:
        # - A list of arrays (one per class)
        # - A 3D array (samples, features, classes)
        # - A 2D array (samples, features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        elif len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes) - extract positive class
            shap_values = shap_values[:, :, 1]
        
        # Ensure shap_values is 2D (samples, features)
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # Get expected value - ensure it's a scalar (float)
        if isinstance(explainer.expected_value, list):
            expected_value = float(explainer.expected_value[1])
        elif isinstance(explainer.expected_value, np.ndarray):
            # If it's an array, extract the positive class value
            if len(explainer.expected_value) > 1:
                expected_value = float(explainer.expected_value[1])
            else:
                expected_value = float(explainer.expected_value[0])
        else:
            expected_value = float(explainer.expected_value)
        
        # Create feature contribution dictionary
        feature_contributions = {}
        # shap_values[0] should now be a 1D array of length len(selected_features)
        shap_vals_1d = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        if isinstance(shap_vals_1d, np.ndarray):
            shap_vals_1d = shap_vals_1d.flatten()
        
        for i, feature in enumerate(selected_features):
            if i < len(shap_vals_1d):
                feature_contributions[feature] = float(shap_vals_1d[i])
            else:
                feature_contributions[feature] = 0.0
        
        # Generate natural language explanation
        explanation_text = generate_explanation_text(
            input_data.iloc[0].to_dict(),
            feature_contributions,
            expected_value
        )
        
        # Convert shap_values[0] to list for JSON serialization
        shap_vals_list = shap_vals_1d.tolist() if isinstance(shap_vals_1d, np.ndarray) else list(shap_vals_1d)
        
        return {
            'shap_values': shap_vals_list,
            'expected_value': expected_value,
            'feature_contributions': feature_contributions,
            'explanation_text': explanation_text
        }
    except Exception as e:
        return {
            'error': str(e),
            'shap_values': None,
            'feature_contributions': {},
            'explanation_text': f"Error generating explanation: {str(e)}"
        }


def generate_explanation_text(input_values, feature_contributions, base_value):
    """
    Generate a natural language explanation of the prediction.
    
    Args:
        input_values: Dictionary of input feature values
        feature_contributions: Dictionary of SHAP contributions per feature
        base_value: Base expected value from SHAP
    
    Returns:
        str: Natural language explanation
    """
    # Sort features by absolute contribution
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    explanation_parts = []
    
    # Add top contributing factors
    explanation_parts.append("**Key Factors Affecting Your Risk:**\n\n")
    
    for feature, contribution in sorted_features[:3]:  # Top 3 factors
        value = input_values.get(feature, 0)
        if abs(contribution) > 0.01:  # Only mention significant contributions
            direction = "increased" if contribution > 0 else "decreased"
            explanation_parts.append(
                f"• **{feature}** ({value:.1f}): {direction.capitalize()} your risk "
                f"by {abs(contribution)*100:.1f}%.\n"
            )
    
    # Add interpretation
    total_contribution = sum(feature_contributions.values())
    final_value = base_value + total_contribution
    
    if final_value > 0.5:
        explanation_parts.append(
            "\n**Overall Assessment:** Your combined risk factors indicate "
            "an elevated risk of Type 2 Diabetes. Consider consulting with "
            "a healthcare provider for further evaluation and lifestyle modifications."
        )
    else:
        explanation_parts.append(
            "\n**Overall Assessment:** Your current risk factors suggest "
            "a lower risk of Type 2 Diabetes. Continue maintaining healthy "
            "lifestyle habits."
        )
    
    return "".join(explanation_parts)


def generate_counterfactual_explanations(model, input_data, user_input, selected_features, current_prediction):
    """
    Generate counterfactual explanations showing what-if scenarios.
    
    Args:
        model: Trained machine learning model
        input_data: DataFrame with current input features
        user_input: Dictionary of current user input values
        selected_features: List of feature names
        current_prediction: Dictionary with current prediction results
    
    Returns:
        dict: Dictionary containing counterfactual scenarios and explanations
    """
    try:
        current_risk = current_prediction['risk_score']
        current_level = current_prediction['risk_level']
        
        counterfactuals = []
        feature_ranges = get_feature_ranges(selected_features)
        
        # Generate counterfactuals for each feature
        for feature in selected_features:
            if feature not in feature_ranges:
                continue
                
            current_value = user_input.get(feature, 0)
            min_val, max_val, step = feature_ranges[feature]
            
            # Test scenarios: decrease and increase by meaningful amounts
            test_scenarios = []
            
            # For features that increase risk when high, test decreases
            if feature in ['BMI', 'Glucose', 'BloodPressure', 'Age', 'DiabetesPedigreeFunction']:
                # Test decreasing the value
                if feature == 'BMI':
                    test_values = [max(15.0, current_value - 2), max(15.0, current_value - 5)]
                elif feature == 'Glucose':
                    test_values = [max(50.0, current_value - 10), max(50.0, current_value - 20)]
                elif feature == 'BloodPressure':
                    test_values = [max(40.0, current_value - 5), max(40.0, current_value - 10)]
                elif feature == 'Age':
                    test_values = [max(18, current_value - 5), max(18, current_value - 10)]
                else:
                    test_values = [max(min_val, current_value - step * 2), max(min_val, current_value - step * 5)]
                
                for test_value in test_values:
                    if abs(test_value - current_value) < 0.1:  # Skip if too similar
                        continue
                    
                    # Create counterfactual input
                    cf_input = user_input.copy()
                    cf_input[feature] = test_value
                    
                    # Prepare data
                    cf_data = prepare_input_data(cf_input, selected_features)
                    
                    # Make prediction
                    cf_prediction = make_prediction(model, cf_data)
                    cf_risk = cf_prediction['risk_score']
                    cf_level = cf_prediction['risk_level']
                    
                    # Calculate change
                    risk_change = cf_risk - current_risk
                    level_changed = (cf_level != current_level)
                    
                    # Only include if it makes a meaningful difference
                    if abs(risk_change) > 1.0 or level_changed:
                        test_scenarios.append({
                            'feature': feature,
                            'current_value': current_value,
                            'counterfactual_value': test_value,
                            'change': test_value - current_value,
                            'current_risk': current_risk,
                            'counterfactual_risk': cf_risk,
                            'risk_change': risk_change,
                            'current_level': current_level,
                            'counterfactual_level': cf_level,
                            'level_changed': level_changed,
                            'direction': 'decrease'
                        })
            
            # Also test increases for features that might decrease risk when higher
            # (though most features increase risk when higher)
            
            counterfactuals.extend(test_scenarios)
        
        # Sort by impact (largest risk reduction first)
        counterfactuals.sort(key=lambda x: x['risk_change'])
        
        # Generate natural language explanation
        explanation_text = generate_counterfactual_explanation_text(
            counterfactuals,
            current_risk,
            current_level
        )
        
        return {
            'counterfactuals': counterfactuals,
            'current_risk': current_risk,
            'current_level': current_level,
            'explanation_text': explanation_text
        }
    except Exception as e:
        return {
            'error': str(e),
            'counterfactuals': [],
            'explanation_text': f"Error generating counterfactual explanations: {str(e)}"
        }


def generate_counterfactual_explanation_text(counterfactuals, current_risk, current_level):
    """
    Generate natural language explanation from counterfactual scenarios.
    
    Args:
        counterfactuals: List of counterfactual scenario dictionaries
        current_risk: Current risk score
        current_level: Current risk level
    
    Returns:
        str: Natural language explanation
    """
    if not counterfactuals:
        return "**No significant counterfactual scenarios found.**"
    
    explanation_parts = []
    explanation_parts.append("**What-If Scenarios - How Your Risk Could Change:**\n\n")
    
    # Get top 5 most impactful counterfactuals
    top_counterfactuals = counterfactuals[:5]
    
    for i, cf in enumerate(top_counterfactuals, 1):
        feature = cf['feature']
        current_val = cf['current_value']
        cf_val = cf['counterfactual_value']
        change = cf['change']
        risk_change = cf['risk_change']
        cf_level = cf['counterfactual_level']
        level_changed = cf['level_changed']
        
        # Format the explanation
        if change < 0:
            direction = "lower"
            change_desc = f"{abs(change):.1f} points lower"
        else:
            direction = "higher"
            change_desc = f"{abs(change):.1f} points higher"
        
        if level_changed:
            explanation_parts.append(
                f"**{i}. If {feature} were {change_desc}** (from {current_val:.1f} to {cf_val:.1f}):\n"
                f"   - Your risk would change from **{current_level}** ({current_risk:.1f}%) "
                f"to **{cf_level}** ({cf['counterfactual_risk']:.1f}%)\n"
                f"   - **Risk reduction: {abs(risk_change):.1f} percentage points**\n\n"
            )
        else:
            explanation_parts.append(
                f"**{i}. If {feature} were {change_desc}** (from {current_val:.1f} to {cf_val:.1f}):\n"
                f"   - Your risk would change from {current_risk:.1f}% to {cf['counterfactual_risk']:.1f}%\n"
                f"   - **Risk change: {risk_change:+.1f} percentage points**\n\n"
            )
    
    # Add actionable insights
    explanation_parts.append("\n**Actionable Insights:**\n\n")
    
    # Find the best counterfactual (largest risk reduction)
    best_cf = min(counterfactuals, key=lambda x: x['risk_change'])
    if best_cf['risk_change'] < -5:  # Significant reduction
        explanation_parts.append(
            f"• **Most impactful change**: Reducing **{best_cf['feature']}** by "
            f"{abs(best_cf['change']):.1f} points could lower your risk by "
            f"{abs(best_cf['risk_change']):.1f} percentage points.\n"
        )
    
    explanation_parts.append(
        "\n**Note**: These scenarios show potential risk changes but should be interpreted "
        "alongside clinical judgment. Consult with healthcare professionals before making "
        "significant lifestyle or treatment changes."
    )
    
    return "".join(explanation_parts)


def get_age_bracket_risk(age, risk_score):
    """
    Determine age bracket and associated risk information.
    
    Args:
        age: Patient age
        risk_score: Calculated risk score (0-100)
    
    Returns:
        dict: Age bracket information and risk comparison
    """
    if age < 30:
        age_bracket = "Young Adult (18-29)"
        avg_risk = 15  # Average risk for this age group
    elif age < 45:
        age_bracket = "Middle Age (30-44)"
        avg_risk = 25
    elif age < 60:
        age_bracket = "Middle Age (45-59)"
        avg_risk = 35
    else:
        age_bracket = "Senior (60+)"
        avg_risk = 45
    
    risk_comparison = risk_score - avg_risk
    
    return {
        'age_bracket': age_bracket,
        'average_risk': avg_risk,
        'risk_comparison': risk_comparison,
        'comparison_text': (
            f"{risk_comparison:+.1f}% compared to average for {age_bracket}"
        )
    }


def get_recommendations(prediction_result, age, bmi, glucose, bp):
    """
    Generate personalized recommendations based on prediction results.
    
    Args:
        prediction_result: Dictionary with prediction results
        age: Patient age
        bmi: Patient BMI
        glucose: Fasting glucose level
        bp: Blood pressure
    
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    risk_score = prediction_result['risk_score']
    risk_level = prediction_result['risk_level']
    
    # General recommendations based on risk level
    if risk_level == "High" or risk_score >= 70:
        recommendations.append(
            "**High Risk Detected:** Please consult with a healthcare provider "
            "immediately for comprehensive diabetes screening and management."
        )
        recommendations.append(
            "**Immediate Actions:**"
        )
        recommendations.append(
            "• Schedule a comprehensive medical evaluation including HbA1c test"
        )
        recommendations.append(
            "• Begin monitoring blood glucose levels regularly"
        )
        recommendations.append(
            "• Consider medication if recommended by your doctor"
        )
    elif risk_level == "Medium":
        recommendations.append(
            "**Moderate Risk:** You should take proactive steps to reduce your risk."
        )
        recommendations.append(
            "**Recommended Actions:**"
        )
        recommendations.append(
            "• Schedule a preventive health checkup within 3 months"
        )
        recommendations.append(
            "• Start regular blood glucose monitoring"
        )
    
    # BMI-specific recommendations
    if bmi >= 30:
        recommendations.append(
            "• **Weight Management:** Your BMI indicates obesity. Aim to lose "
            "5-10% of body weight through diet and exercise"
        )
    elif bmi >= 25:
        recommendations.append(
            "• **Weight Management:** Your BMI is in the overweight range. "
            "Consider losing weight through healthy diet and regular exercise"
        )
    
    # Glucose-specific recommendations
    if glucose >= 126:
        recommendations.append(
            "• **Blood Sugar:** Your glucose level is elevated. Follow a "
            "diabetes-friendly diet (low glycemic index foods, portion control)"
        )
    elif glucose >= 100:
        recommendations.append(
            "• **Blood Sugar:** Your glucose is in the pre-diabetic range. "
            "Focus on reducing sugar and refined carbohydrate intake"
        )
    
    # Blood pressure recommendations
    if bp >= 90:
        recommendations.append(
            "• **Blood Pressure:** Your blood pressure is elevated. "
            "Reduce sodium intake, increase physical activity, and manage stress"
        )
    
    # Lifestyle recommendations (always include)
    recommendations.append(
        "• **Physical Activity:** Aim for at least 150 minutes of moderate "
        "exercise per week (walking, cycling, swimming)"
    )
    recommendations.append(
        "• **Diet:** Follow a balanced diet rich in vegetables, whole grains, "
        "lean proteins, and healthy fats"
    )
    recommendations.append(
        "• **Sleep:** Ensure 7-9 hours of quality sleep per night"
    )
    recommendations.append(
        "• **Stress Management:** Practice stress-reduction techniques "
        "(meditation, yoga, deep breathing)"
    )
    
    return recommendations


def create_risk_gauge(risk_score):
    """
    Create a gauge chart showing risk score.
    
    Args:
        risk_score: Risk score (0-100)
    
    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    # Professional color scheme based on risk
    # Using professional healthcare colors
    if risk_score >= 70:
        color = '#c85d5d'  # Professional danger red
    elif risk_score >= 50:
        color = '#d4a574'  # Professional warning amber
    else:
        color = '#2d8659'  # Professional success green
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score (%)", 'font': {'size': 16, 'color': '#1a4d7a'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#64748b'},
            'bar': {'color': color},
            'bgcolor': '#f8fafc',
            'steps': [
                {'range': [0, 30], 'color': "#2d8659"},      # Professional green
                {'range': [30, 50], 'color': "#d4a574"},     # Professional amber
                {'range': [50, 70], 'color': "#d97757"},    # Professional coral
                {'range': [70, 100], 'color': "#c85d5d"}    # Professional red
            ],
            'threshold': {
                'line': {'color': "#c85d5d", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        font={'color': '#1e293b', 'family': 'Arial, sans-serif'}
    )
    return fig


def create_feature_contribution_chart(feature_contributions):
    """
    Create a bar chart showing feature contributions.
    
    Args:
        feature_contributions: Dictionary of feature names and SHAP contributions
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Sort by absolute value
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    features = [f[0] for f in sorted_features]
    contributions = [f[1] for f in sorted_features]
    
    # Professional color scheme: coral for positive (risk increase), teal for negative (risk decrease)
    colors = ['#d97757' if c > 0 else '#2c7a7b' for c in contributions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=contributions,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{c:+.2f}" for c in contributions],
            textposition='auto',
            marker_line_color='#1e293b',
            marker_line_width=0.5
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Feature Contributions to Risk Prediction",
            'font': {'size': 18, 'color': '#1a4d7a', 'family': 'Arial, sans-serif'}
        },
        xaxis_title="SHAP Value (Contribution)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=150, r=20, t=50, b=20),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font={'color': '#1e293b', 'family': 'Arial, sans-serif'},
        xaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0')
    )
    
    return fig


def create_counterfactual_chart(counterfactuals):
    """
    Create a bar chart showing counterfactual risk changes.
    
    Args:
        counterfactuals: List of counterfactual scenario dictionaries
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    if not counterfactuals:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No counterfactual scenarios available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get top 8 counterfactuals
    top_cfs = counterfactuals[:8]
    
    # Create labels
    labels = []
    risk_changes = []
    colors = []
    
    for cf in top_cfs:
        feature = cf['feature']
        change = cf['change']
        risk_change = cf['risk_change']
        
        if change < 0:
            label = f"{feature}\n({abs(change):.1f} lower)"
        else:
            label = f"{feature}\n({abs(change):.1f} higher)"
        
        labels.append(label)
        risk_changes.append(risk_change)
        
        # Green for risk reduction, red for risk increase
        if risk_change < 0:
            colors.append('#2d8659')  # Green for reduction
        else:
            colors.append('#c85d5d')  # Red for increase
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=risk_changes,
            marker_color=colors,
            text=[f"{rc:+.1f}%" for rc in risk_changes],
            textposition='auto',
            marker_line_color='#1e293b',
            marker_line_width=0.5
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Counterfactual Risk Changes - What-If Scenarios",
            'font': {'size': 18, 'color': '#1a4d7a', 'family': 'Arial, sans-serif'}
        },
        xaxis_title="Feature Change",
        yaxis_title="Risk Change (Percentage Points)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=100),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font={'color': '#1e293b', 'family': 'Arial, sans-serif'},
        xaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickangle=-45),
        yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0')
    )
    
    return fig

