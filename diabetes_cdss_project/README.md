# Diabetes Clinical Decision Support System (CDSS)

A comprehensive Clinical Decision Support System for Type 2 Diabetes risk prediction, designed for village health settings where advanced laboratory equipment may not be available.

## 🎯 Project Overview

This project implements a machine learning-based CDSS that:
- Predicts Type 2 Diabetes risk using easy-to-measure clinical variables
- Provides explainable AI insights using SHAP values
- Offers personalized recommendations for lifestyle changes and treatment
- Designed specifically for village health workers with limited resources

## 📁 Project Structure

```
diabetes_cdss_project/
│
├── data/
│   ├── health care diabetes.csv                # Original dataset
│   └── cleaned_diabetes_dataset.csv            # Cleaned dataset (generated)
│
├── notebooks/
│   ├── 1_data_understanding_and_cleaning.ipynb      # Data exploration & feature selection
│   ├── 2_model_training_and_evaluation.ipynb        # Model training (5 models)
│   ├── 3_xai_and_fairness.ipynb                     # SHAP explainability
│   └── 4_streamlit_integration_test.ipynb           # Integration testing
│
├── models/
│   ├── best_model_pipeline.pkl                 # Trained best model
│   ├── model_metadata.json                     # Model information
│   ├── selected_features.json                  # Selected features
│   ├── shap_explainer.pkl                      # SHAP explainer
│   └── shap_feature_importance.json            # Feature importance
│
├── streamlit_app/
│   ├── app.py                                  # Main Streamlit app (entry point)
│   ├── utils.py                                # Utility functions (model loading, predictions, visualizations)
│   ├── requirements.txt                        # Python dependencies
│   ├── theme.css                               # Professional healthcare theme styling
│   └── pages/
│       ├── 1_Login.py                          # Login page (authentication)
│       ├── 2_Input_Form.py                     # Patient data input form
│       ├── 3_Risk_Dashboard.py                 # Risk visualization and recommendations
│       └── 4_SHAP_Explain.py                  # SHAP explanations and interpretability
│
├── MoH-Logo.png                                # Ministry of Health logo
└── README.md                                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd diabetes_cdss_project
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

## 📊 Running the Notebooks

Execute the notebooks in order to train models and generate required files:

1. **Notebook 1: Data Understanding & Cleaning**
   - Loads and explores the dataset
   - Handles missing values
   - Performs feature selection using domain knowledge, statistics, and ML
   - Saves cleaned dataset

2. **Notebook 2: Model Training & Evaluation**
   - Trains 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
   - Evaluates models using multiple metrics
   - Selects best model based on ROC AUC
   - Saves best model using joblib

3. **Notebook 3: Explainable AI (SHAP)**
   - Creates SHAP explainer
   - Analyzes feature importance
   - Generates explanations for predictions
   - Saves explainer for Streamlit use

4. **Notebook 4: Streamlit Integration Test**
   - Tests model loading
   - Verifies predictions work
   - Checks all required files exist

## 🌐 Running the Streamlit Application

1. **Ensure all notebooks have been run** (to generate model files)

2. **Navigate to the streamlit_app directory**
   ```bash
   cd streamlit_app
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 📱 Using the Application

### 1. Login Page
- Use demo credentials:
  - Username: `admin` / Password: `admin123`
  - Username: `healthworker` / Password: `health123`
  - Username: `demo` / Password: `demo123`

### 2. Input Form
- Enter patient clinical measurements:
  - **Required Fields:**
    - Age (years, must be 18+)
    - Gender (Male/Female)
    - BMI (Body Mass Index)
    - Blood Pressure (diastolic, mm Hg)
    - Fasting Glucose (mg/dL)
    - Diabetes Pedigree Function (family history risk)
    - Pregnancies (for female patients)
  - **Optional Fields:**
    - Physical Activity Level (Low/Moderate/High)
    - Diet Quality (Poor/Fair/Good/Excellent)
    - Alcohol Use (None/Occasional/Regular/Heavy)
    - Smoking Status (Non-smoker/Former smoker/Current smoker)
    - Family History of Diabetes (Yes/No)
- Click "Predict Diabetes Risk" button to generate prediction

### 3. Risk Dashboard
- View comprehensive risk analysis:
  - **Risk Score Gauge**: Visual representation of diabetes risk (0-100%)
  - **Age Bracket Comparison**: Compare patient's risk with average for their age group
  - **Feature Contributions**: See which factors most impact the prediction
  - **Personalized Recommendations**: Tailored lifestyle and treatment advice based on:
    - Risk level (Low/Medium/High)
    - BMI status
    - Glucose levels
    - Blood pressure
    - Age-specific guidance
  - **Next Steps**: Actionable items based on risk assessment

### 4. SHAP Explanation
- Understand why the model made the prediction:
  - **Feature Contribution Charts**: Visual bar charts showing how each feature affects the prediction
  - **Natural Language Explanations**: Easy-to-understand text summaries of key factors
  - **Top Contributing Factors**: Identification of the 3 most important risk factors
  - **Detailed SHAP Values**: Exact numerical contributions of each feature
  - **Counterfactual Explanations**: What-if scenarios showing how risk would change with different values

## 🔬 Feature Selection Methodology

Our feature selection uses three complementary approaches:

### 1. Domain Knowledge
Prioritizes features that are:
- Easy to measure in village settings (no advanced equipment)
- Reliable and consistent
- Clinically relevant for Type 2 Diabetes

**Selected Features:**
- Age (very easy - self-reported)
- BMI (easy - scale and measuring tape)
- Blood Pressure (easy - standard BP cuff)
- Glucose (moderate - portable glucose meter)
- Diabetes Pedigree Function (easy - family history questionnaire)
- Physical Activity (easy - patient questionnaire)
- Diet Quality (easy - patient questionnaire)
- Alcohol Use (easy - patient questionnaire)
- Smoking (easy - patient questionnaire)

### 2. Statistical Methods
- Correlation analysis with target variable
- Chi-square tests for categorical relationships
- Features with significant statistical association

### 3. Machine Learning
- Random Forest feature importance
- Features with high predictive power
- Cross-validation performance

## 🤖 Models Trained

The system trains and compares 5 different models:

1. **Logistic Regression** - Linear model, interpretable
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting, high performance
4. **LightGBM** - Fast gradient boosting
5. **Support Vector Machine (SVM)** - Kernel-based classifier

The best model (based on ROC AUC) is selected and saved for deployment. The current best model is **XGBoost** with an ROC AUC of 96.78% and accuracy of 88.96%.

## 📈 Model Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (ability to distinguish classes)

## 🔍 Explainable AI (XAI)

The system uses SHAP (SHapley Additive exPlanations) to provide:
- **Global Feature Importance**: Which features matter most overall
- **Individual Explanations**: Why a specific prediction was made
- **Feature Contributions**: How each feature affects the prediction
- **Natural Language Explanations**: Easy-to-understand summaries

## ⚠️ Important Notes

### For Healthcare Workers:
- This tool is for **screening and educational purposes only**
- It is **not a substitute** for professional medical advice
- Always consult qualified healthcare providers for medical decisions
- Use clinical judgment alongside model predictions

### For Developers:
- Ensure all notebooks are run before deploying Streamlit app
- Model files must be in the `models/` directory
- Check that all dependencies are installed
- Test predictions before using in production

## 🛠️ Troubleshooting

### Model files not found
- **Solution**: Run all notebooks in order (1, 2, 3, 4)

### Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### SHAP explainer errors
- **Solution**: Run Notebook 3 to generate the SHAP explainer

### Streamlit not starting
- **Solution**: Check that you're in the `streamlit_app` directory and run `streamlit run app.py`

## 📝 Dataset Information

- **Dataset**: Health Care Diabetes Dataset
- **Target**: Type 2 Diabetes (Binary: 0 = No Diabetes, 1 = Diabetes)
- **Features**: 
  - Clinical measurements: Age, BMI, Blood Pressure, Glucose, Diabetes Pedigree Function
  - Lifestyle factors: Physical Activity, Diet Quality, Alcohol Use, Smoking
  - Demographics: Pregnancies (for female patients)
- **Use Case**: Village health tool - all measurements are easy to obtain with basic medical equipment

## 👥 Team Information

**Team Ravens** - Data Science Research Project

| Name | Registration Number | Access Number | Role |
|------|---------------------|---------------|------|
| Lorraine Paula Arinaitwe | M23B38/004 | B20729 | Model training & Streamlit Integration (Notebook 2 & 4) |
| Rugogamu Noela | S23B38/016 | B22775 | xAI & Fairness (Notebook 3) |
| Ssendi Aloysious Malon | S23B38/002 | B21258 | Data Understanding and Cleaning (Notebook 1) |

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: Health Care Diabetes Dataset
- Libraries: scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly, Pandas, NumPy
- Clinical guidance from healthcare professionals
- Ministry of Health (MoH) for support and resources

## 📧 Contact

For questions or issues, please contact the project team.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Active Development

---

## 📊 Model Performance

The deployed XGBoost model achieves the following performance metrics:

- **Accuracy**: 88.96%
- **Precision**: 83.64%
- **Recall**: 85.19%
- **F1-Score**: 84.40%
- **ROC AUC**: 96.78%

These metrics indicate strong predictive performance suitable for clinical decision support in village health settings.

