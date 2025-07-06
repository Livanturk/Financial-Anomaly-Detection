# Financial Anomaly Detection for Internal Audit & Fraud Prevention

## 📌 Project Overview

This project simulates a real-world **financial fraud detection system** for internal audit teams. It uses the PaySim dataset to detect anomalous transactions using a robust ML pipeline with production-quality standards.

---

## 🎯 Objective

To build an end-to-end machine learning pipeline capable of identifying **fraudulent financial transactions** from a large volume of tabular data, with high accuracy, low false positives, and explainable outputs.

---

## 📂 Project Structure

```
financial-anomaly-detection/
├── data/
│   └── transaction_dataset.csv       # Input dataset (PaySim)
├── src/
│   ├── preprocess.py                 # Preprocessing class
│   ├── model.py                      # Training, evaluation, and explainability
│   └── EDA.ipynb                     # Exploratory Data Analysis (optional)
├── tests/
│   └── test_preprocess.py           # Unit tests for Preprocess class
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── CHANGELOG.md                     # Version history
```

---

## 🧠 Key Features

* **Stratified Train-Test Split** to preserve rare fraud class
* **Extensive Feature Engineering** including account behavior
* **Classical ML Models**: RandomForest and XGBoost
* **5-Fold Stratified Cross-Validation**
* **Explainability**:

  * Feature Importance Plot
  * SHAP Summary and Force Plot
* **Test Suite** for `Preprocess` class

---

## 📊 Model Results

**XGBoost Final Test Set Performance:**

* **Accuracy:** 99.98%
* **Fraud Recall:** 99.2%
* **Fraud Precision:** 94.4%
* **ROC AUC:** 0.999

> “Only 13 frauds missed out of 1,643. Only 97 non-frauds falsely flagged out of 552,000.”

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Model Pipeline

```bash
PYTHONPATH=src python src/model.py
```

---

## ✅ Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
```

---

## 📌 Acknowledgements

* Dataset: [PaySim - Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
* SHAP Explainability: Lundberg & Lee, 2017

---

## 🧾 License

MIT License (Optional)
