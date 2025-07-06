import streamlit as st
import pandas as pd
import joblib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.preprocess import Preprocess

import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Financial Anomaly Detection App")

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_xgboost.pkl")

@st.cache_data
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    #Temporarily save to disk because Preprocess expects a path
    temp_path = "data/temp_input.csv"
    df.to_csv(temp_path, index=False)
    preprocessor = Preprocess(temp_path)
    X, _, _, _ = preprocessor.get_processed_data()
    return df, X

uploaded_file = st.file_uploader("Upload a transaction CSV file", type=["csv"])
if uploaded_file:
    with st.spinner("Preprocessing and predicting..."):
        df_raw, X_processed = preprocess_data(uploaded_file)
        model = load_model()
        preds = model.predict(X_processed)
        proba = model.predict_proba(X_processed)[:, 1]

        result_df = df_raw.copy()
        result_df['Fraud_Probability'] = proba
        result_df['Prediction'] = preds

        st.success("Predictions complete!")
        st.write("### Prediction Breakdown:")
        st.dataframe(result_df[['nameOrig', 'nameDest', 'amount', 'Prediction', 'Fraud_Probability']].head(100))

        st.write("### Prediction Summary:")
        st.bar_chart(result_df['Prediction'].value_counts())

        st.write("### SHAP Global Feature Importance:")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed[:100])
        plt.title("SHAP Summary Plot (first 100 rows)")
        shap.summary_plot(shap_values, X_processed[:100], show=False)
        st.pyplot(plt.gcf())
