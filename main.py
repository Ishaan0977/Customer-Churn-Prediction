# About Dataset
# Context
# "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
#
# Content
# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
#
# The data set includes information about:
#
# Customers who left within the last month – the column is called Churn
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# Demographic info about customers – gender, age range, and if they have partners and dependents

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Step 1 -> Data cleaning and splitting into Training and Test split
df = pd.read_csv("/home/base/PycharmProjects/PythonProject/dataSet/Telco-Customer-Churn.csv")

# Understanding the dataset
# with pd.option_context("display.max_column",None,
#                        "display.width",None):
#     print(df.head(10))
# print(df.info())

# Dropping the unnecessary values
df = df.drop(columns=["customerID"])

# Converting numeric values which were stored as strings
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# Encoding Target column
df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

# Defining Features and Target Explicitly
targeted_column = "Churn"
X = df.drop(columns=[targeted_column])
y = df[targeted_column]

# print(targeted_column in X.columns)

# Train/Test Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handling missing value
# print(X_train.isna().sum())
# Using meadian to fill the missing value ( standard for billing data )
TotalCharges_median = X_train["TotalCharges"].median()
X_train["TotalCharges"] = X_train["TotalCharges"].fillna(TotalCharges_median)
X_test["TotalCharges"] = X_test["TotalCharges"].fillna(TotalCharges_median)

# print("Training data's missing values : \n",X_train.isna().sum())
# print("Test data's missing values : \n",X_test.isna().sum())

# Identifying Categorial and Numeric columns
categorial_cols = X_train.select_dtypes(include="object").columns
numeric_cols = X_train.select_dtypes(exclude="object").columns
# print(categorial_cols)
# print(numeric_cols)

# Categorial encoding
X_train_encoded = pd.get_dummies(
    X_train,
    columns = categorial_cols,
    drop_first = False
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns = categorial_cols,
    drop_first = False
)

# Syncing the columns of train with test
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value = 0
)

# Scaling numeric columns
scaler = StandardScaler()
# fit_transform(): # Computes mean and std from training data then scales training data
# transform(): Uses SAME mean/std and applies to test data to prevents leakage
X_train_encoded[numeric_cols] = scaler.fit_transform(
    X_train_encoded[numeric_cols]
)
X_test_encoded[numeric_cols] = scaler.transform(
    X_test_encoded[numeric_cols]
)

# Step 2 -> Initializing multiple models to see which one fits the best
# Initializing the Logistics Regression model
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# Training the model
cv_lr = cross_val_score(
    log_reg,
    X_train_encoded,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("CV score of Logistics Regression Model -> ",cv_lr.mean())

# Building a fully grown Decision Tree to check the variance
dt = DecisionTreeClassifier(
    random_state=42
)

# Training the model
cv_dt = cross_val_score(
    dt,
    X_train_encoded,
    y_train,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

print("CV score of Decision Tree -> ",cv_dt.mean())

# Constraining the Decision tree to reduce the variance
# Initializing the model
dt_c = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

# Training the model
cv_dt_c = cross_val_score(
    dt_c,
    X_train_encoded,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("CV score of Constrained Decision Tree -> ",cv_dt_c.mean())

# Using Random Forest to reduce the variance
# Initializing the model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Training the model
cv_rf = cross_val_score(
    rf,
    X_train_encoded,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("CV score of Random Forest -> ",cv_rf.mean())

# Using Gradient Boosting to reduce the bias
# Initializing the model
gb = GradientBoostingClassifier(
    n_estimators = 100,
    learning_rate= 0.1,
    max_depth= 3,
    random_state= 42
)

# Training the model
cv_gb = cross_val_score(
    gb,
    X_train_encoded,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("CV score of Gradient Boosting -> ",cv_gb.mean())

# Step 3 -> Using GridCV on the best 2 performed models i.e. Logistics Regression , Gradient Boosting
param_grid_lr = {
    "C":[0.01,0.1,1,10,100],
    "l1_ratio":[0,0.5,1]
}
grid_lr = GridSearchCV(
    LogisticRegression(
        max_iter=5000,
        solver="saga",
        random_state=42),
    param_grid=param_grid_lr,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

# Training the model
grid_lr.fit(X_train_encoded,y_train)

print("Best parameters of Logistics Regression -> ",grid_lr.best_params_)
print("Best score of Logistics Regression -> ",grid_lr.best_score_)

param_grid_gb = {
    "n_estimators":[100,200,300],
    "learning_rate":[0.01,0.05,0.1],
    "max_depth":[2,3,4]
}
grid_gb = GridSearchCV(
    GradientBoostingClassifier(
        random_state=42
    ),
    param_grid = param_grid_gb,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

# Training the model
grid_gb.fit(X_train_encoded,y_train)

print("Best parameters of Gradient Boosting -> ",grid_gb.best_params_)
print("Best score of Gradient Boosting -> ",grid_gb.best_score_)

# Grid Search wins with score of 0.8493021496845075

# Step 4 -> Final step , using the winner model on test set
best_gb = grid_gb.best_estimator_

# Analyzing the performance on test set
test_pred = best_gb.predict(X_test_encoded)
test_pred_prob = best_gb.predict_proba(X_test_encoded)[:,1]

test_acc = accuracy_score(y_test,test_pred)
test_roc = roc_auc_score(y_test,test_pred_prob)
precision , recall , _ = precision_recall_curve(y_test,test_pred_prob)
test_auc_pr = auc(recall,precision)

print("Accuracy score -> ",test_acc)
print("ROC-AUC score -> ",test_roc)
print("PR score -> ",test_auc_pr)

# Feature importance identified by the model
feature_importance = pd.Series(
    best_gb.feature_importances_,
    index=X_train_encoded.columns
).sort_values(ascending=False)

print(feature_importance.head(10))
