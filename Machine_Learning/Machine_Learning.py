# ============================================================
# CUSTOMER CHURN PREDICTION - MACHINE LEARNING
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 1. LOAD ALREADY CLEANED DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Dataset")

FILE_PATH = os.path.join(
    DATA_DIR,
    "cleaned_churn_data.csv"
)

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("DATA LOADED SUCCESSFULLY")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. ENCODE TARGET VARIABLE
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# ============================================================
# 3. SELECT FEATURES
# ============================================================

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features]
y = df["Churn"]


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42,
        verbosity=-1
    )
}


# ============================================================
# 7. TRAIN + EVALUATE MODELS
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 8. MODEL ACCURACY COMPARISON
# ============================================================

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

results_df["Accuracy (%)"] = results_df["Accuracy"] * 100

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# ============================================================
# 9. ACCURACY BAR CHART
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy (%)"]
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.title("Customer Churn Model Accuracy Comparison")

plt.xticks(rotation=30)
plt.ylim(0, 100)
plt.savefig("Customer Churn Model Accuracy Comparison")
plt.tight_layout()
plt.show()


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy (%)"], 2), "%")


# ============================================================
# 11. CONFUSION MATRIX OF BEST MODEL
# ============================================================

best_model_name = best_model["Model"]

best_model_object = models[best_model_name]

best_predictions = best_model_object.predict(
    X_test_scaled
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)
plt.savefig("Confusion Matrix of Best Model")
plt.tight_layout()
plt.show()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

if best_model_name in [
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]:

    importance = best_model_object.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(feature_importance)

    plt.figure(figsize=(8, 5))

    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(
        f"Feature Importance - {best_model_name}"
    )
    plt.savefig("Feature Importance XGBoost")
    plt.tight_layout()
    plt.show()


print("\n" + "=" * 60)
print("MACHINE LEARNING COMPLETED SUCCESSFULLY")
print("=" * 60)