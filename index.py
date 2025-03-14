# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pickle
from io import BytesIO
from sklearn.ensemble import GradientBoostingRegressor

# Custom CSS styling
st.markdown("""
<style>
    .title {
        background-color: #0066cc;
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result {
        background-color: #4CAF50;
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        margin: 20px 0;
    }
    .section-header {
        color: #0066cc;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin: 20px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Page title with custom styling
st.markdown("<h1 class='title'>Medical Expenses Predictor</h1>", unsafe_allow_html=True)

# Sidebar styling and inputs
st.sidebar.markdown("<h2 style='color:#0066cc;'>Patient Details</h2>", unsafe_allow_html=True)

def user_input_features():
    with st.sidebar.expander("Demographic Information", expanded=True):
        AGE = st.number_input("Age", min_value=0, max_value=100, value=25)
        SEX = st.selectbox("Gender", ("Male", "Female"), format_func=lambda x: x)
        CHILDREN = st.selectbox("Number of Children", options=range(0, 6), index=0)
        REGION = st.selectbox(
            "Region",
            ("Southwest", "Southeast", "Northwest", "Northeast"),
            index=0
        )
    
    with st.sidebar.expander("Health Information", expanded=True):
        BMI = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        SMOKER = st.selectbox("Smoker", ("Yes", "No"), index=1)
    
    return pd.DataFrame({
        "age": AGE,
        "sex": 1 if SEX == "Male" else 0,
        "bmi": BMI,
        "children": CHILDREN,
        "smoker": 1 if SMOKER == "Yes" else 0,
        "region": ["southwest", "southeast", "northwest", "northeast"].index(REGION.lower())
    }, index=[0])

df = user_input_features()

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3 class='section-header'>Input Summary</h3>", unsafe_allow_html=True)
    styled_df = df.copy()
    styled_df.columns = [col.capitalize() for col in styled_df.columns]
    st.dataframe(styled_df.style.format(precision=2).set_properties(**{
        'background-color': '#f0f2f6',
        'color': '#0066cc',
        'border-color': 'white'
    }))

# Load model and make prediction
MODEL_URL = "https://github.com/sunilk872/Insurance-Expense-Prediction/raw/main/finalized_model.pkl"

try:
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        loaded_model = pickle.load(BytesIO(response.content))
        prediction = loaded_model.predict(df)
        prediction = np.round(prediction, 2)
        
        with col2:
            st.markdown("<h3 class='section-header'>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'>Estimated Medical Costs:<br>${prediction[0]:,.2f}</div>", 
                       unsafe_allow_html=True)
            
            # Visualization
            st.markdown("<h3 class='section-header'>Cost Drivers Analysis</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            features = ['Age', 'BMI', 'Smoker', 'Children', 'Region']
            importance = loaded_model.feature_importances_
            sns.barplot(x=importance, y=features, palette="Blues_d", ax=ax)
            ax.set_title("Feature Importance in Cost Prediction")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            
    else:
        st.error("Failed to load prediction model. Please try again later.")
except Exception as e:
    st.error(f"Error in prediction system: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; margin-top: 40px;'>"
            "Medical Cost Prediction System v1.0<br>"
            "Data Source: Insurance Dataset</div>", 
            unsafe_allow_html=True)
