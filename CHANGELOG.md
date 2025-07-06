# 📋 CHANGELOG.md

All notable changes to this project will be documented in this file. This project follows [Semantic Versioning](https://semver.org/).

---

## \[1.0.0] - 2025-07-06

###  Initial Release

* Full end-to-end fraud detection pipeline using PaySim dataset
* Preprocessing class with feature engineering (account behavior, balance errors, etc.)
* Train-test split with stratification
* Cross-validation with RandomForest and XGBoost models
* Model performance metrics (Precision, Recall, F1, AUC)
* SHAP explainability and feature importance plots
* Model saving with `joblib`
* Interactive `Streamlit` app to upload CSVs and visualize predictions

---

## \[0.1.0] - 2025-07-04

###  Setup and Infrastructure

* Project folder and file structure created
* Dataset added to `data/transaction_dataset.csv`
* Initial EDA and fraud class imbalance analysis

---

## \[Unreleased]

* Model versioning
* Streamlit UI enhancements (CSV export, filter, branding)
* Deployment pipeline (Docker or Streamlit Cloud)
* Integration with logging/alerting system
