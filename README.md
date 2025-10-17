# Credit Card Default Prediction using AdaBoost & Streamlit

## Project Overview

This project aims to predict whether a **credit card client will default on payment next month** based on their past payment history, demographic details, and financial information.  
Using **AdaBoost Classifier** with hyperparameter tuning, we built a machine learning model that achieves robust performance in detecting potential defaults.  

**Dataset:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

**Streamlit Link:** https://kanika-07cs-task16-15-10-25--app-05t5md.streamlit.app/

## Dataset Description
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

## Data Preprocessing Steps
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

## Model Development
1. Base Model
- **Algorithm:** AdaBoostClassifier
- **Base Estimator:** DecisionTreeClassifier(max_depth=2)
2. Hyperparameter Tuning
- Used GridSearchCV with 5-fold StratifiedKFold cross-validation.
- Used parameter grid

## Visualizations
- Confusion Matrix (to assess misclassifications)
- ROC Curve (to visualize model performance)
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/596fedc3-f819-4533-9479-34ebf6aa4b2f" />

## How to Run
1. Install Dependencies - pip install -r requirements.txt
2. Launch the Streamlit App - streamlit run app.py

## App screenshot
<img width="550" height="550" alt="image" src="https://github.com/user-attachments/assets/ac808e23-a016-49fa-8ef2-90e793f47557" />
<img width="550" height="550" alt="image" src="https://github.com/user-attachments/assets/8c176082-17f8-4d07-94c4-a4352bcfbae3" />

## Results & Insights
- AdaBoost performed well in handling the class imbalance and provided stable generalization.
- F1-score shows a balanced trade-off between precision and recall.
- Clients with high past due payments (PAY_0–PAY_6) and low credit limits have a higher likelihood of defaulting.

## Conclusion
- This project demonstrates how ensemble learning (AdaBoost) can effectively handle moderately imbalanced datasets like credit card defaults.
- Proper preprocessing—especially log and power transformation—significantly improved model stability.
- The Streamlit web app provides an intuitive interface for quick, real-time predictions.

