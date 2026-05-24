import pandas as pd
# ===============================================================
# IMPORT REQUIRED LIBRARIES
# ===============================================================
# train_test_split:
# Used to divide dataset into training data and testing data.
# Training data = used to teach model
# Testing data = used to check model performance

from sklearn.model_selection import train_test_split
# RandomForestRegressor:
# Regression algorithm used when output is numeric value.
#
# Random Forest = Many Decision Trees together
#
# Final prediction = Average of all tree outputs
#
# Strong model for real-world problems.

from sklearn.ensemble import RandomForestRegressor
# Metrics:
# Used to evaluate regression model performance.

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ===============================================================
# STEP 1 : CREATE DATASET
# ===============================================================
# Description:
# Sample house price dataset
# Columns:
# Area        -> square feet
# Bedrooms    -> number of bedrooms
# HouseAge    -> age of house
# Price       -> target output

df = pd.DataFrame({
    'Area': [1000,1200,1500,1800,2000,2200,2500,2800,3000,3500],
    'Bedrooms': [2,2,3,3,4,4,4,5,5,6],
    'HouseAge': [20,18,15,12,10,8,7,5,4,2],
    'Price': [30,35,45,55,65,72,80,92,100,120]
})

print("Original Dataset")
print(df)
# ===============================================================
# STEP 2 : SELECT FEATURES AND TARGET
# ===============================================================
# X = Input columns
# y = Output column

X = df[['Area', 'Bedrooms', 'HouseAge']]
y = df['Price']

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
# Random Forest does NOT need scaling.
# Why?
# Because trees use rules like:
# Area < 2200 ?
# Bedrooms >= 4 ?
# Trees do not depend on feature magnitude.

print("\nFeature Scaling Not Required for Random Forest")
# ===============================================================
# STEP 5 : CREATE MODEL
# ===============================================================
# n_estimators:
# Number of trees
# max_depth:
# Max depth of each tree

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
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
# Random Forest builds many trees.
# Tree 1:
# Uses random sample of rows
# Tree 2:
# Uses another random sample
# Tree 3:
# Uses another sample
# Also each split may use random subset of features.
# Example:
# Tree1 may use Area + Bedrooms
# Tree2 may use Area + HouseAge
# Every tree learns separately.
# Final model stores all trees.

print("\nNumber of Trees:")
print(len(model.estimators_))
# ===============================================================
# STEP 7 : PREDICT OUTPUT
# ===============================================================

pred = model.predict(X_test)
# ===============================================================
# INTERNAL WORKING OF predict()
# ===============================================================
# Each tree predicts price.
# Example:
# Tree1 = 88
# Tree2 = 91
# Tree3 = 90
# ...
# Final Prediction =
# Average of all tree predictions

print("\nPredicted Prices:")
print(pred)
# ===============================================================
# STEP 8 : EVALUATION
# ===============================================================
# ---------------------------------------------------------------
# MAE
# ---------------------------------------------------------------

mae = mean_absolute_error(y_test, pred)

print("\nMAE:")
print(mae)
# ---------------------------------------------------------------
# MSE
# ---------------------------------------------------------------

mse = mean_squared_error(y_test, pred)

print("\nMSE:")
print(mse)
# ---------------------------------------------------------------
# R2 SCORE
# ---------------------------------------------------------------

r2 = r2_score(y_test, pred)

print("\nR2 Score:")
print(r2)
# ===============================================================
# STEP 9 : FEATURE IMPORTANCE
# ===============================================================
# Description:
# Shows which feature was most useful.

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print("\nFeature Importance:")
print(importance)
# ===============================================================
# STEP 10 : PREDICT NEW HOUSE PRICE
# ===============================================================
# Example:
# Area = 2400
# Bedrooms = 4
# HouseAge = 6

new_house = [[2400, 4, 6]]

new_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(new_price)
# ===============================================================
# FINAL SUMMARY
# ===============================================================
# 1. Create dataset
# 2. Select X and y
# 3. Split train/test
# 4. No scaling required
# 5. Create Random Forest model
# 6. Train many trees using fit()
# 7. Predict using average of trees
# 8. Evaluate using MAE, MSE, R2
# 9. Check feature importance
# 10. Predict new house price
# Best for:
# High accuracy
# Complex data
# Real-world tabular datasets
# Better than single tree:
# Less overfitting
# More stable predictions