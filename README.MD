# Customer Churn Prediction Engine using Ensemble Learning

This project builds a **Customer Churn Prediction Model** using classical Machine Learning techniques and advanced ensemble methods.

The goal is to predict whether a telecom customer will churn (leave service) based on service usage, billing information, and contract details.

---

## Author

**Ishaan Sharma**  
MCA | Machine Learning & AI  
GitHub: https://github.com/ishaan0977  
LinkedIn: https://linkedin.com/in/ishaansh/

---

## Problem Statement

Customer churn is a critical business problem in subscription-based companies.

Each row in the dataset represents a customer and includes:

- Demographics (gender, senior citizen, partner, dependents)
- Service usage (internet, online security, tech support, streaming)
- Contract type
- Billing information (monthly charges, total charges)
- Tenure
- Target variable → `Churn` (Yes / No)

### Objective

Build a robust machine learning model that:

- Maximizes ROC-AUC
- Handles class imbalance properly
- Avoids data leakage
- Generalizes well to unseen test data

---

## Dataset

**IBM Telco Customer Churn Dataset**

- Total Samples: 7043  
- Features: 20  
- Target: Binary Classification (Churn: 0 or 1)

---

## Machine Learning Workflow

### 1. Data Preprocessing

- Dropped irrelevant ID column (`customerID`)
- Converted `TotalCharges` to numeric
- Handled missing values using median (computed on training data only)
- One-hot encoded categorical variables using `pd.get_dummies`
- Aligned train/test columns using `reindex`
- Scaled numerical features using `StandardScaler`
- Used stratified 80/20 train-test split

No data leakage was introduced.

---

### 2. Model Comparison using 5-Fold Cross Validation (ROC-AUC)

Models evaluated:

- Logistic Regression
- Decision Tree (Unconstrained)
- Decision Tree (Constrained)
- Random Forest
- Gradient Boosting

Evaluation metric:

scoring = "roc_auc"


---

### 3. Hyperparameter Tuning using GridSearchCV

Top models tuned:

- Logistic Regression (ElasticNet regularization using saga solver)
- Gradient Boosting

GridSearchCV configuration:

cv = 5 ,
scoring = "roc_auc" , 
n_jobs = -1


---

## Final Model

**Best Model: Gradient Boosting Classifier**

Best Parameters:

learning_rate = 0.05 ,
max_depth = 2 , 
n_estimators = 200


---

## Final Test Performance (Unseen Data)

Accuracy = 0.7991 , 
ROC-AUC = 0.8457 ,
PR-AUC = 0.6624

Cross-Validation ROC-AUC: 0.8493


Generalization gap is minimal → Stable model.

---

## Feature Importance (Top Drivers of Churn)

| Feature                          | Importance |
|----------------------------------|------------|
| Contract_Month-to-month          | 0.3548     |
| tenure                           | 0.1711     |
| InternetService_Fiber optic      | 0.1228     |
| OnlineSecurity_No                | 0.0811     |
| TechSupport_No                   | 0.0582     |
| PaymentMethod_Electronic check   | 0.0574     |
| TotalCharges                     | 0.0499     |
| MonthlyCharges                   | 0.0411     |

### Business Insight

- Month-to-month contracts have highest churn risk.
- Low tenure customers are more likely to churn.
- Lack of security or tech support increases churn probability.

---

## Key Concepts Demonstrated

- Bias vs Variance tradeoff
- Bagging vs Boosting
- Random Forest variance reduction
- Gradient Boosting bias reduction
- ROC-AUC vs PR-AUC comparison
- Cross Validation for model selection
- Hyperparameter tuning without test leakage
- Proper ML pipeline design
- Feature importance interpretation

---

## Project Structure

customer-churn-ml/
│
├── main.py
├── README.md
├── requirements.txt
└── data/


---

## How to Run

Clone repository:

```bash
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml

pip install -r requirements.txt

python main.py ```
