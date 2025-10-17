# Credit Card Default Prediction using AdaBoost & Streamlit

## Project Overview

This project aims to predict whether a **credit card client will default on payment next month** based on their past payment history, demographic details, and financial information.  
Using **AdaBoost Classifier** with hyperparameter tuning, we built a machine learning model that achieves robust performance in detecting potential defaults.  

## 📊 Dataset Description

**Dataset:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
**Streamlit Link:** https://kanika-07cs-task16-15-10-25--app-05t5md.streamlit.app/

| Feature | Description |
|----------|--------------|
| `LIMIT_BAL` | Amount of given credit (NT dollar) |
| `SEX` | Gender (1 = Male, 2 = Female) |
| `EDUCATION` | Education level (1 = Graduate school, 2 = University, 3 = High school, 4 = Others) |
| `MARRIAGE` | Marital status (1 = Married, 2 = Single, 3 = Others) |
| `AGE` | Age in years |
| `PAY_0` to `PAY_6` | Repayment status for months April–September (e.g., 0 = paid duly, 1 = payment delay for one month, etc.) |
| `BILL_AMT1` to `BILL_AMT6` | Bill statement amounts for the past six months |
| `PAY_AMT1` to `PAY_AMT6` | Amount of previous payments |
| `default.payment.next.month` | Target variable (1 = Default, 0 = No default) |

## 🧹 Data Preprocessing Steps

1. **Handling Skewness**
   - Applied log transformation to highly skewed columns.
   - Applied PowerTransformer (Yeo–Johnson) to stabilize variance and normalize data distribution.

2. **Outlier Capping**
   - Used the IQR method to cap extreme outliers for numeric features.

3. **Feature Scaling**
   - Standardized all numerical features using StandardScaler for consistent scale across features.

4. **Train-Test Split**
   - Split dataset into 80% training and 20% testing using stratified sampling to maintain target balance.

5. **Feature Selection**
   - Removed irrelevant features such as ID.

## ⚙️ Model Development

### 1️⃣ Base Model
- **Algorithm:** AdaBoostClassifier
- **Base Estimator:** DecisionTreeClassifier(max_depth=2)

### 2️⃣ Hyperparameter Tuning
Used `GridSearchCV` with 5-fold **StratifiedKFold** cross-validation.

**Parameter Grid:**
```python
param_grid = {
    'n_estimators': [100, 250, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1]
}
