# Telco-Customer-Churn-Prediction
# 📊 Telco Customer Churn Prediction — Data Science Project

An end-to-end customer churn analysis and prediction project for a telecom company, combining **SQL analysis**, **Power BI dashboarding**, **classical machine learning**, and **deep learning** to identify customers likely to churn and the key factors driving that churn.

---

## 🎯 Problem Statement

Customer churn directly impacts telecom revenue — acquiring a new customer costs significantly more than retaining an existing one. This project analyzes customer behavior, billing, contract, and service data to:

- Predict which customers are likely to churn
- Identify the strongest drivers of churn
- Surface actionable retention insights for the business

---

## 🗂️ Dataset

- **Source:** [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** ~7,043 customer records, 21 features
- **Target variable:** `Churn` (Yes/No)
- **Key features:** tenure, Contract, InternetService, PaymentMethod, MonthlyCharges, TotalCharges, SeniorCitizen, gender, Partner

---

## 🗃️ Project Structure

```
Customer_Churn_Prediction_Data_Science/
│
├── Dataset/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   ├── Cleaned_Churn_Data.csv                 # cleaned dataset
│   └── Cleaned_Churn_Data.xlsx
│
├── Deep_Learning/
│   ├── Deep_Learning_Model.py                 # data cleaning + 3 ANN models
│   └── Visualization/
│       ├── model1_accuracy.png / model1_confusion_matrix.png
│       ├── model2_accuracy.png / model2_confusion_matrix.png
│       ├── model3_accuracy.png / model3_confusion_matrix.png
│       └── model_comparison.png
│
├── Machine_Learning/
│   ├── Machine_Learning.py                    # 5-model ML comparison
│   └── Visualization/
│       ├── Confusion_Matrix_of_Best_Model.png
│       ├── Customer_Churn_Model_Accuracy_Comparison.png
│       └── Feature_Importance_XGBoost.png
│
├── PowerBI/
│   ├── Telco_Customer_Churn_Analysis.pbix
│   └── Telco_Customer_Churn_Analysis_Dashboard.png
│
└── SQL/
    └── Customer_Churn_Analysis.sql            # 17 business-insight queries
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Power BI |
| Classical ML | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow / Keras |
| Database | SQL Server (T-SQL) |
| Environment | VS Code |

---

## 🔍 Approach

### 1. Data Cleaning
- Converted `TotalCharges` to numeric, handled resulting nulls
- Removed rows with missing critical values
- Exported cleaned dataset to both CSV and Excel for downstream use

### 2. Exploratory Data Analysis
- Churn distribution, internet service distribution, and demographic breakdowns
- SQL-based business queries: churn by gender, contract type, internet service, payment method, tenure segment, and high-value customers

### 3. Machine Learning (`Machine_Learning.py`)
Trained and compared 5 classical models on `tenure`, `MonthlyCharges`, and `TotalCharges`:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~77.5% |
| Decision Tree | ~70.5% |
| Random Forest | ~74.0% |
| **XGBoost** ⭐ | **~78.6%** |
| LightGBM | ~77.5% |

**Best model:** XGBoost, selected via accuracy/F1 comparison across all 5 models.

**Feature Importance (XGBoost):**
1. `tenure` — ~64% importance
2. `MonthlyCharges` — ~27% importance
3. `TotalCharges` — ~9% importance

### 4. Deep Learning (`Deep_Learning_Model.py`)
Built 3 Keras Sequential ANNs (Dense layers + ReLU activation + Sigmoid output) to compare feature sets and regularization:

| Model | Features | Regularization | Accuracy |
|---|---|---|---|
| Model 1 | tenure | None | ~76.0% |
| Model 2 | tenure | Dropout (0.3 / 0.2) | ~76.0% |
| Model 3 | tenure, MonthlyCharges, TotalCharges | None | **~78.2%** |

Adding more features (Model 3) outperformed adding dropout regularization alone (Model 2) — reinforcing that feature richness mattered more than regularization for this dataset size.

### 5. Power BI Dashboard
Interactive dashboard with slicers (Contract, gender, Partner) and 6 visuals:
- Customers by Churn (pie)
- Internet Service by Churn
- Senior Citizens by Churn
- Contract Type by Churn
- Tenure by Churn
- Gender by Churn

---

## 💡 Key Business Insights

- **Overall churn rate is ~26.6%**, in line with typical telecom industry benchmarks
- **Month-to-month contracts churn far more** than one-year or two-year contracts — the single strongest retention lever
- **Fiber optic customers churn more** than DSL or no-internet customers, suggesting pricing or service-quality concerns
- **New customers (low tenure) are highest-risk** — churn drops sharply as tenure increases, consistent with the ML feature importance results
- **Tenure is the single biggest predictor of churn** across both classical ML and deep learning models

---

## 🚀 Future Improvements

- Incorporate categorical features (Contract, InternetService, PaymentMethod) into the ML/DL pipelines for higher accuracy
- Address class imbalance with SMOTE or class-weighting
- Hyperparameter tuning via GridSearchCV / Optuna
- Deploy the best model (XGBoost) as a REST API for real-time churn scoring
- Add cross-validation for more robust model evaluation

---

## 👤 Author

**Jyoti Ranjan Bhanja**
Aspiring Data Scientist & Data Analyst | Python · SQL · Machine Learning · Power BI
📍 Bengaluru, Karnataka, India


#DataScience #DataAnalytics #MachineLearning #DeepLearning #Python #SQL #PowerBI #Pandas #NumPy #ScikitLearn #XGBoost #LightGBM #TensorFlow #Keras #DataVisualization #ChurnPrediction #CustomerChurn #TelecomAnalytics #EDA #PredictiveAnalytics #GitHub #Portfolio #AspiringDataScientist
