import pandas as pd
# ===============================================================
# IMPORT REQUIRED LIBRARIES
# ===============================================================
# train_test_split:
# Used to divide dataset into training data and testing data.
# Training data = used to teach model
# Testing data = used to check model performance

from sklearn.model_selection import train_test_split
# StandardScaler:
# Used to normalize numeric columns.
# Converts different ranges into same scale.
#
# Important for KNN because KNN uses distance.

from sklearn.preprocessing import StandardScaler
# KNeighborsClassifier:
# Classification algorithm used when output is category.
#
# Predicts based on nearest neighbors.
#
# Example:
# If nearby customers got loan approved,
# new customer may also be approved.

from sklearn.neighbors import KNeighborsClassifier
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
# KNN uses distance formula.
#
# If one feature has huge values,
# it dominates distance.
# Example:
# CreditScore = 800
# Age = 40
# So scaling is required.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaled X_train:")
print(X_train)

print("\nScaled X_test:")
print(X_test)
# ===============================================================
# STEP 5 : CREATE MODEL
# ===============================================================
# n_neighbors = K value
#
# Means:
# Look at nearest 3 rows.

model = KNeighborsClassifier(n_neighbors=3)

print("\nModel Created")
# ===============================================================
# STEP 6 : TRAIN MODEL
# ===============================================================

model.fit(X_train, y_train)
# ===============================================================
# INTERNAL WORKING OF fit()
# ===============================================================
# KNN has almost no traditional training.
# It mainly stores training data.
# During prediction it uses:
# X_train values
# y_train labels
# So fit() means:
# Save data for future comparison.

print("\nTraining Data Stored")
# ===============================================================
# STEP 7 : PREDICT CLASS
# ===============================================================

pred = model.predict(X_test)
# ===============================================================
# INTERNAL WORKING OF predict()
# ===============================================================
# For each test row:
# 1. Calculate distance to all training rows
# Usually Euclidean distance:
# sqrt((x1-a1)^2 + (x2-a2)^2 + ...)
# 2. Find nearest K rows
# Example K=3:
# Neighbor labels:
# [1,1,0]
# 3. Majority vote:
# Final class = 1

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

print("\nConfusion Matrix:")
print(cm)
# ---------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------

report = classification_report(y_test, pred)

print("\nClassification Report:")
print(report)
# ===============================================================
# STEP 10 : PREDICT NEW CUSTOMER
# ===============================================================
# Example:
# Income = 55
# Age = 34
# CreditScore = 620

new_customer = [[55, 34, 620]]
# Scale first

new_customer_scaled = scaler.transform(new_customer)

result = model.predict(new_customer_scaled)

print("\nLoan Approval Prediction:")
print(result)
# ===============================================================
# FINAL SUMMARY
# ===============================================================
# 1. Create dataset
# 2. Select X and y
# 3. Split train/test
# 4. Scale features
# 5. Create KNN model
# 6. fit() stores training data
# 7. Predict using nearest neighbors
# 8. Evaluate using accuracy metrics
# 9. Predict new customer
# Best for:
# Small datasets
# Pattern based classification
# Risk:
# Slow on very large data
# Sensitive to scaling