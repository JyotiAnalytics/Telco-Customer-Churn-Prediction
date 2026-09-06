import os
import glob
import pandas as pd


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "Dataset"
)

print("Project Folder:")
print(BASE_DIR)

print("\nDataset Folder:")
print(DATA_DIR)


# ============================================================
# FIND CSV FILE
# ============================================================

csv_files = glob.glob(
    os.path.join(DATA_DIR, "*.csv")
)

print("\nCSV Files Found:")
print(csv_files)


# ============================================================
# CHECK DATASET
# ============================================================

if not csv_files:
    raise FileNotFoundError(
        "No CSV dataset found inside Dataset folder."
    )


# ============================================================
# LOAD DATASET
# ============================================================

FILE_PATH = csv_files[0]

print("\nUsing Dataset:")
print(FILE_PATH)

df = pd.read_csv(FILE_PATH)

print("\nDataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())



# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after TotalCharges conversion:")
print(df.isnull().sum())

# Remove rows having missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("\nShape after cleaning:", df.shape)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

# CSV

CLEANED_FILE = os.path.join(
    DATA_DIR,
    "Cleaned_Churn_Data.csv"
)

df.to_csv(
    CLEANED_FILE,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(CLEANED_FILE)


# EXCEL

EXCEL_FILE = os.path.join(
    DATA_DIR,
    "Cleaned_Churn_Data.xlsx"
)

df.to_excel(
    EXCEL_FILE,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(EXCEL_FILE)
# ============================================================
# 5. DATA MANIPULATION
# ============================================================

# ------------------------------------------------------------
# A. Total number of Male Customers
# ------------------------------------------------------------

male_customers = (df["gender"] == "Male").sum()

print("\nTotal Male Customers:", male_customers)


# ------------------------------------------------------------
# B. Total number of DSL Customers
# ------------------------------------------------------------

dsl_customers = (
    df["InternetService"] == "DSL"
).sum()

print("Total DSL Customers:", dsl_customers)


# ------------------------------------------------------------
# C. Female Senior Citizens + Mailed Check
# ------------------------------------------------------------

new_customer = df[
    (df["gender"] == "Female") &
    (df["SeniorCitizen"] == 1) &
    (df["PaymentMethod"] == "Mailed check")
]

print("\nFemale Senior Citizens using Mailed Check:")
print(new_customer)

print("\nNumber of Customers:", len(new_customer))


# ------------------------------------------------------------
# D. Tenure < 10 OR TotalCharges < 500
# ------------------------------------------------------------

new_customer = df[
    (df["tenure"] < 10) |
    (df["TotalCharges"] < 500)
]

print("\nCustomers with Tenure < 10 OR TotalCharges < 500:")
print(new_customer)

print("\nNumber of Customers:", len(new_customer))


# ============================================================
# 6. DATA VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
# ------------------------------------------------------------
# A. CHURN PIE CHART
# ------------------------------------------------------------

churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Churn Distribution")

plt.show()


# ------------------------------------------------------------
# B. INTERNET SERVICE BAR PLOT
# ------------------------------------------------------------

internet_counts = df["InternetService"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    internet_counts.index,
    internet_counts.values
)

plt.title("Internet Service Distribution")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.show()


# ============================================================
# 7. CONVERT CHURN INTO 0 AND 1
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn after encoding:")
print(df["Churn"].value_counts())


# ============================================================
# MODEL 1
# FEATURE = TENURE
# TARGET = CHURN
#
# Input Layer  = 12 Nodes + ReLU
# Hidden Layer = 8 Nodes + ReLU
# Output       = 1 Node + Sigmoid
# Optimizer    = Adam
# Epochs       = 150
# ============================================================


# ------------------------------------------------------------
# 8. SELECT FEATURE AND TARGET
# ------------------------------------------------------------

X1 = df[["tenure"]]

y1 = df["Churn"]


print("\nModel 1 Feature:")
print(X1.head())

print("\nModel 1 Target:")
print(y1.head())


# ------------------------------------------------------------
# 9. TRAIN TEST SPLIT
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1,
    y1,
    test_size=0.20,
    random_state=42,
    stratify=y1
)

print("\nModel 1 Training Shape:", X1_train.shape)
print("Model 1 Testing Shape:", X1_test.shape)


# ------------------------------------------------------------
# 10. STANDARDIZATION
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()

X1_train_scaled = scaler1.fit_transform(X1_train)

X1_test_scaled = scaler1.transform(X1_test)


# ------------------------------------------------------------
# 11. BUILD MODEL 1
# ------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model1 = Sequential()

model1.add(
    Dense(
        12,
        activation="relu",
        input_shape=(1,)
    )
)

model1.add(
    Dense(
        8,
        activation="relu"
    )
)

model1.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ------------------------------------------------------------
# 12. COMPILE MODEL 1
# ------------------------------------------------------------

model1.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 13. MODEL 1 SUMMARY
# ------------------------------------------------------------

print("\n================ MODEL 1 SUMMARY ================")

model1.summary()


# ------------------------------------------------------------
# 14. TRAIN MODEL 1
# ------------------------------------------------------------

history1 = model1.fit(
    X1_train_scaled,
    y1_train,
    epochs=150,
    batch_size=32,
    validation_data=(
        X1_test_scaled,
        y1_test
    ),
    verbose=1
)


# ------------------------------------------------------------
# 15. PREDICTION MODEL 1
# ------------------------------------------------------------

y1_probability = model1.predict(
    X1_test_scaled
)

y1_pred = (
    y1_probability >= 0.5
).astype(int)


# ------------------------------------------------------------
# 16. MODEL 1 ACCURACY
# ------------------------------------------------------------
from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(
    y1_test,
    y1_pred
)

print("\nModel 1 Accuracy:", accuracy1)


# ------------------------------------------------------------
# 17. MODEL 1 CONFUSION MATRIX
# ------------------------------------------------------------
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(
    y1_test,
    y1_pred
)

print("\nModel 1 Confusion Matrix:")
print(cm1)
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(
    confusion_matrix=cm1,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("Model 1 - Confusion Matrix")
plt.savefig("model1_confusion_matrix")
plt.show()


# ------------------------------------------------------------
# 18. MODEL 1 CLASSIFICATION REPORT
# ------------------------------------------------------------
from sklearn.metrics import classification_report
print("\nModel 1 Classification Report:")

print(
    classification_report(
        y1_test,
        y1_pred
    )
)


# ------------------------------------------------------------
# 19. MODEL 1 ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history1.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history1.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model 1 - Accuracy vs Epochs"
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")
plt.savefig("model1_accuracy")
plt.legend()

plt.grid()

plt.show()


# ============================================================
# MODEL 2
# SAME FEATURE = TENURE
# SAME TARGET = CHURN
#
# Dropout after Input = 0.3
# Dropout after Hidden = 0.2
# ============================================================


# ------------------------------------------------------------
# 20. BUILD MODEL 2
# ------------------------------------------------------------

model2 = Sequential()

model2.add(
    Dense(
        12,
        activation="relu",
        input_shape=(1,)
    )
)

model2.add(
    Dropout(0.3)
)

model2.add(
    Dense(
        8,
        activation="relu"
    )
)

model2.add(
    Dropout(0.2)
)

model2.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ------------------------------------------------------------
# 21. COMPILE MODEL 2
# ------------------------------------------------------------

model2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 22. MODEL 2 SUMMARY
# ------------------------------------------------------------

print("\n================ MODEL 2 SUMMARY ================")

model2.summary()


# ------------------------------------------------------------
# 23. TRAIN MODEL 2
# ------------------------------------------------------------

history2 = model2.fit(
    X1_train_scaled,
    y1_train,
    epochs=150,
    batch_size=32,
    validation_data=(
        X1_test_scaled,
        y1_test
    ),
    verbose=1
)


# ------------------------------------------------------------
# 24. PREDICTION MODEL 2
# ------------------------------------------------------------

y2_probability = model2.predict(
    X1_test_scaled
)

y2_pred = (
    y2_probability >= 0.5
).astype(int)


# ------------------------------------------------------------
# 25. MODEL 2 ACCURACY
# ------------------------------------------------------------

accuracy2 = accuracy_score(
    y1_test,
    y2_pred
)

print("\nModel 2 Accuracy:", accuracy2)


# ------------------------------------------------------------
# 26. MODEL 2 CONFUSION MATRIX
# ------------------------------------------------------------

cm2 = confusion_matrix(
    y1_test,
    y2_pred
)

print("\nModel 2 Confusion Matrix:")

print(cm2)

ConfusionMatrixDisplay(
    confusion_matrix=cm2,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("Model 2 - Confusion Matrix")
plt.savefig("model2_confusion_matrix")
plt.show()


# ------------------------------------------------------------
# 27. MODEL 2 CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nModel 2 Classification Report:")

print(
    classification_report(
        y1_test,
        y2_pred
    )
)


# ------------------------------------------------------------
# 28. MODEL 2 ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history2.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history2.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model 2 - Accuracy vs Epochs"
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig("model2_accuracy")

plt.show()


# ============================================================
# MODEL 3
#
# FEATURES:
#   Tenure
#   MonthlyCharges
#   TotalCharges
#
# TARGET:
#   Churn
# ============================================================


# ------------------------------------------------------------
# 29. SELECT FEATURES
# ------------------------------------------------------------

X3 = df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

y3 = df["Churn"]


print("\nModel 3 Features:")
print(X3.head())


# ------------------------------------------------------------
# 30. TRAIN TEST SPLIT
# ------------------------------------------------------------

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3,
    y3,
    test_size=0.20,
    random_state=42,
    stratify=y3
)

print("\nModel 3 Training Shape:", X3_train.shape)
print("Model 3 Testing Shape:", X3_test.shape)


# ------------------------------------------------------------
# 31. STANDARDIZATION
# ------------------------------------------------------------

scaler3 = StandardScaler()

X3_train_scaled = scaler3.fit_transform(
    X3_train
)

X3_test_scaled = scaler3.transform(
    X3_test
)


# ------------------------------------------------------------
# 32. BUILD MODEL 3
# ------------------------------------------------------------

model3 = Sequential()

model3.add(
    Dense(
        12,
        activation="relu",
        input_shape=(3,)
    )
)

model3.add(
    Dense(
        8,
        activation="relu"
    )
)

model3.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ------------------------------------------------------------
# 33. COMPILE MODEL 3
# ------------------------------------------------------------

model3.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 34. MODEL 3 SUMMARY
# ------------------------------------------------------------

print("\n================ MODEL 3 SUMMARY ================")

model3.summary()


# ------------------------------------------------------------
# 35. TRAIN MODEL 3
# ------------------------------------------------------------

history3 = model3.fit(
    X3_train_scaled,
    y3_train,
    epochs=150,
    batch_size=32,
    validation_data=(
        X3_test_scaled,
        y3_test
    ),
    verbose=1
)


# ------------------------------------------------------------
# 36. PREDICTION MODEL 3
# ------------------------------------------------------------

y3_probability = model3.predict(
    X3_test_scaled
)

y3_pred = (
    y3_probability >= 0.5
).astype(int)


# ------------------------------------------------------------
# 37. MODEL 3 ACCURACY
# ------------------------------------------------------------

accuracy3 = accuracy_score(
    y3_test,
    y3_pred
)

print("\nModel 3 Accuracy:", accuracy3)


# ------------------------------------------------------------
# 38. MODEL 3 CONFUSION MATRIX
# ------------------------------------------------------------

cm3 = confusion_matrix(
    y3_test,
    y3_pred
)

print("\nModel 3 Confusion Matrix:")

print(cm3)

ConfusionMatrixDisplay(
    confusion_matrix=cm3,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("Model 3 - Confusion Matrix")

plt.savefig("model3_confusion_matrix")

plt.show()


# ------------------------------------------------------------
# 39. MODEL 3 CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nModel 3 Classification Report:")

print(
    classification_report(
        y3_test,
        y3_pred
    )
)


# ------------------------------------------------------------
# 40. MODEL 3 ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history3.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history3.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model 3 - Accuracy vs Epochs"
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig("model3_accuracy")

plt.show()


# ============================================================
# 41. COMPARE ALL 3 MODELS
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Model 1 - Tenure",
        "Model 2 - Tenure + Dropout",
        "Model 3 - 3 Features"
    ],

    "Accuracy": [
        accuracy1,
        accuracy2,
        accuracy3
    ]
})


print("\n================ MODEL COMPARISON ================")

print(comparison)


# ============================================================
# 42. MODEL COMPARISON BAR CHART
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title(
    "Comparison of Customer Churn Models"
)

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.ylim(0, 1)

plt.savefig("model_comparison")

plt.show()


# ============================================================
# 43. COMBINED ACCURACY VS EPOCHS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    history1.history["accuracy"],
    label="Model 1"
)

plt.plot(
    history2.history["accuracy"],
    label="Model 2"
)

plt.plot(
    history3.history["accuracy"],
    label="Model 3"
)

plt.title(
    "Accuracy vs Epochs - All Models"
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 44. FINAL RESULT
# ============================================================

print("\n================================================")
print("FINAL MODEL RESULTS")
print("================================================")

print(
    f"Model 1 Accuracy: {accuracy1:.4f}"
)

print(
    f"Model 2 Accuracy: {accuracy2:.4f}"
)

print(
    f"Model 3 Accuracy: {accuracy3:.4f}"
)

best_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = comparison["Accuracy"].max()

print(
    f"\nBest Model: {best_model}"
)

print(
    f"Best Accuracy: {best_accuracy:.4f}"
)

print("\n================================================")
print("PROJECT COMPLETED")
print("================================================")