use churned_data;

-- ============================================================
-- CUSTOMER CHURN ANALYSIS
-- SQL ANALYSIS
-- ============================================================



-- 1. VIEW DATA


select* from dbo.cleaned_churn_data;


-- 2. TOTAL CUSTOMERS


SELECT COUNT(*) AS Total_Customers
FROM cleaned_churn_data;



-- 3. TOTAL MALE CUSTOMERS


SELECT COUNT(*) AS Total_Male_Customers
FROM cleaned_churn_data
WHERE gender = 'Male';



-- 4. TOTAL FEMALE CUSTOMERS


SELECT COUNT(*) AS Total_Female_Customers
FROM cleaned_churn_data
WHERE gender = 'Female';



-- 5. TOTAL CHURNED CUSTOMERS


SELECT COUNT(*) AS Total_Churned_Customers
FROM cleaned_churn_data
WHERE churn = 1;


-- 6. CHURN BY GENDER

SELECT
    gender,
    Churn,
    COUNT(*) AS Customer_Count
FROM cleaned_churn_data
GROUP BY gender, Churn
ORDER BY Customer_Count DESC;


-- 7. CHURN BY CONTRACT

SELECT
    Contract,
    Churn,
    COUNT(*) AS Customer_Count
FROM cleaned_churn_data
GROUP BY Contract, Churn
ORDER BY Customer_Count DESC;


-- 8. CHURN BY INTERNET SERVICE

SELECT
    InternetService,
    Churn,
    COUNT(*) AS Customer_Count
FROM cleaned_churn_data
GROUP BY InternetService, Churn
ORDER BY Customer_Count DESC;



-- 9. CHURN BY PAYMENT METHOD

SELECT
    PaymentMethod,
    Churn,
    COUNT(*) AS Customer_Count
FROM cleaned_churn_data
GROUP BY PaymentMethod, Churn
ORDER BY Customer_Count DESC;


-- 10. AVERAGE MONTHLY CHARGES

SELECT
    AVG(MonthlyCharges) AS Average_Monthly_Charges
FROM cleaned_churn_data;


-- 11. TOTAL CHARGES

SELECT
    SUM(TotalCharges) AS Total_Revenue
FROM cleaned_churn_data;


-- 12. AVERAGE TENURE

SELECT
    AVG(tenure) AS Average_Tenure
FROM cleaned_churn_data;


-- 13. CUSTOMER SEGMENTATION

SELECT
    CustomerID,
    tenure,
    CASE
        WHEN tenure < 10 THEN 'New Customer'
        WHEN tenure BETWEEN 10 AND 30 THEN 'Medium Tenure'
        ELSE 'Long Term Customer'
    END AS Customer_Segment
FROM cleaned_churn_data;


-- 14. NEW CUSTOMERS WHO CHURNED

SELECT TOP 5
    CustomerID,
    tenure,
    MonthlyCharges,
    Churn
FROM cleaned_churn_data
WHERE tenure < 10
AND Churn = 0;

-- 15. HIGH MONTHLY CHARGE CUSTOMERS

SELECT
    CustomerID,
    MonthlyCharges,
    Churn
FROM cleaned_churn_data
WHERE MonthlyCharges > 70
ORDER BY MonthlyCharges DESC;


-- 16. CHURN RATE

SELECT
    COUNT(CASE WHEN Churn = 0 THEN 1 END) * 100.0
    / COUNT(*) AS Churn_Rate
FROM cleaned_churn_data;


-- 17. CHURN BY TENURE GROUP

SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 Months'
        WHEN tenure BETWEEN 12 AND 24 THEN '12-24 Months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 Months'
        ELSE '49+ Months'
    END AS Tenure_Group,
    Churn,
    COUNT(*) AS Customer_Count
FROM cleaned_churn_data
GROUP BY
    CASE
        WHEN tenure < 12 THEN '0-11 Months'
        WHEN tenure BETWEEN 12 AND 24 THEN '12-24 Months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 Months'
        ELSE '49+ Months'
    END,
    Churn
ORDER BY Customer_Count DESC;

