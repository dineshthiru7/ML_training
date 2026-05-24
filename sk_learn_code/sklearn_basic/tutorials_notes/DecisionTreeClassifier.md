import pandas as pd
# ===============================================================
# IMPORT REQUIRED LIBRARIES
# ===============================================================
# train_test_split:
# Used to divide dataset into training data and testing data.
# Training data = used to teach model
# Testing data = used to check model performance

from sklearn.model_selection import train_test_split
# DecisionTreeClassifier:
# Classification algorithm used when output is category.
#
# Works using rule-based questions.
#
# Example:
# CreditScore > 600 ?
# Income > 40 ?
#
# Final output:
# 0 = Rejected
# 1 = Approved

from sklearn.tree import DecisionTreeClassifier
# Metrics:
# Used to evaluate classification model performance.

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# ===============================================================
# STEP 1 : CREATE DATASET
# ===============================================================
# Description:
# Sample loan approval dataset
# Columns:
# Income       -> customer income
# Age          -> customer age
# CreditScore  -> customer credit score
# Approved     -> target output
#                0 = Rejected
#                1 = Approved

df = pd.DataFrame({
    'Income': [20,25,30,35,40,50,60,70,80,90],
    'Age': [22,25,28,30,32,35,38,40,45,50],
    'CreditScore': [300,350,400,450,500,600,650,700,750,800],
    'Approved': [0,0,0,0,1,1,1,1,1,1]
})

print("Original Dataset")
print(df)
# ===============================================================
# STEP 2 : SELECT FEATURES AND TARGET
# ===============================================================
# X = Input columns
# y = Output column

X = df[['Income', 'Age', 'CreditScore']]
y = df['Approved']

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)
# ===============================================================
# STEP 3 : SPLIT DATA INTO TRAIN AND TEST
# ===============================================================
# 80% -> training
# 20% -> testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)
# ===============================================================
# STEP 4 : FEATURE SCALING
# ===============================================================
# Description:
# Decision Tree does NOT need scaling.
# Why?
# Tree compares values using conditions:
# CreditScore < 500 ?
# Income > 60 ?
# It does not use distance.

print("\nFeature Scaling Not Required for Decision Tree")
# ===============================================================
# STEP 5 : CREATE MODEL
# ===============================================================
# max_depth:
# Controls maximum tree levels
# Small tree -> simple model
# Large tree -> may overfit

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

print("\nModel Created")
# ===============================================================
# STEP 6 : TRAIN MODEL
# ===============================================================

model.fit(X_train, y_train)
# ===============================================================
# INTERNAL WORKING OF fit()
# ===============================================================
# Tree tries many questions:
# Income < 45 ?
# Age < 30 ?
# CreditScore < 550 ?
# It chooses best split using impurity reduction.
# Common methods:
# Gini Index
# Entropy
# Example:
# If CreditScore < 500:
#       class = 0
# Else:
#       class = 1
# Repeats recursively.

print("\nTree Depth:")
print(model.get_depth())

print("\nNumber of Leaves:")
print(model.get_n_leaves())
# ===============================================================
# STEP 7 : PREDICT CLASS
# ===============================================================

pred = model.predict(X_test)
# ===============================================================
# INTERNAL WORKING OF predict()
# ===============================================================
# Each test row moves through tree:
# Example:
# CreditScore = 700 -> right branch
# Income = 60 -> right branch
# Final leaf gives class.

print("\nPredicted Class:")
print(pred)
# ===============================================================
# STEP 8 : PREDICT PROBABILITY
# ===============================================================

prob = model.predict_proba(X_test)
# Output:
# [P(class0), P(class1)]

print("\nPredicted Probability:")
print(prob)
# ===============================================================
# STEP 9 : EVALUATION
# ===============================================================
# ---------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------

acc = accuracy_score(y_test, pred)

print("\nAccuracy:")
print(acc)
# ---------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------

cm = confusion_matrix(y_test, pred)
# Format:
# [[TN FP]
#  [FN TP]]

print("\nConfusion Matrix:")
print(cm)
# ---------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------

report = classification_report(y_test, pred)

print("\nClassification Report:")
print(report)
# ===============================================================
# STEP 10 : FEATURE IMPORTANCE
# ===============================================================
# Description:
# Shows most useful columns.

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print("\nFeature Importance:")
print(importance)
# ===============================================================
# STEP 11 : PREDICT NEW CUSTOMER
# ===============================================================
# Example:
# Income = 55
# Age = 35
# CreditScore = 650

new_customer = [[55, 35, 650]]

result = model.predict(new_customer)

print("\nLoan Approval Prediction:")
print(result)
# ===============================================================
# FINAL SUMMARY
# ===============================================================
# 1. Create dataset
# 2. Select X and y
# 3. Split train/test
# 4. No scaling required
# 5. Create Decision Tree model
# 6. Train using rule-based splits
# 7. Predict class
# 8. Evaluate using metrics
# 9. Check feature importance
# 10. Predict new customer
# Best for:
# Explainable models
# Rule based learning
# Risk:
# Can overfit if tree grows too much