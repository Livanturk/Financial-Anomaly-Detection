import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer, precision_score, recall_score, f1_score

from preprocess import Preprocess


def load_data():
    p = Preprocess("data/transaction_dataset.csv")
    return p.get_processed_data()


def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    }


def evaluate_models(X, y, models, cv_folds=5):
    scoring = {
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score)
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {metric: np.mean(scores[f"test_{metric}"]) for metric in scoring}
        for metric, val in results[name].items():
            print(f"{metric.upper()}: {val:.4f}")

    return results


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()


def explain_with_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("Generating SHAP Summary Plot...")
    shap.summary_plot(shap_values, X_sample)

    print("Force plot for a specific instance...")
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0], matplotlib=True)


if __name__ == "__main__":
    #Pipelines

    #Load and prepare data
    X_train, X_test, y_train, y_test = load_data()

    #Train and evaluate
    models = get_models()
    results = evaluate_models(X_train, y_train, models)

    #Final model (XGBoost)
    model = models["XGBoost"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    #Feature Importance
    plot_feature_importance(model, X_train.columns)

    #SHAP Explanation (on a subset for performance)
    explain_with_shap(model, X_test.sample(100, random_state=42))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgboost.pkl")
    print("Model saved to models/fraud_xgboost.pkl")