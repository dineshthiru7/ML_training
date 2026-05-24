import pandas as pd
# ===============================================================
# IMPORT REQUIRED LIBRARIES
# ===============================================================
# train_test_split:
# Used to divide dataset into training data and testing data.
# Training data = used to teach model
# Testing data = used to check model performance

from sklearn.model_selection import train_test_split
# DecisionTreeRegressor:
# Regression algorithm used when output is numeric value.
#
# Works using rule-based splits.
# Example:
# If Area < 2000 ?
# If Bedrooms >= 4 ?
#
# Good for non-linear data.

from sklearn.tree import DecisionTreeRegressor
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
# Decision Tree does NOT need scaling.
# Why?
# Because tree compares values:
# Area < 2000 ?
# HouseAge < 10 ?
# It uses splits, not distance or weights.

print("\nFeature Scaling Not Required for Decision Tree")
# ===============================================================
# STEP 5 : CREATE MODEL
# ===============================================================
# max_depth:
# Maximum levels of tree.
#
# Small depth  -> simpler tree
# Large depth  -> can overfit

model = DecisionTreeRegressor(max_depth=3)

print("\nModel Created")
# ===============================================================
# STEP 6 : TRAIN MODEL
# ===============================================================

model.fit(X_train, y_train)
# ===============================================================
# INTERNAL WORKING OF fit()
# ===============================================================
# Tree tries many questions:
# Area < 1600 ?
# Area < 2100 ?
# Bedrooms < 4 ?
# HouseAge < 10 ?
# For each split it calculates error reduction.
# Usually uses:
# Mean Squared Error reduction
# Chooses BEST split.
# Then repeats for child branches.
# Example:
# If Area < 1900:
#     predict lower price
# Else:
#     predict higher price
# Continues until:
# max_depth reached
# or data too small
# or pure nodes formed

print("\nTree Depth:")
print(model.get_depth())

print("\nNumber of Leaves:")
print(model.get_n_leaves())
# ===============================================================
# STEP 7 : PREDICT OUTPUT
# ===============================================================

pred = model.predict(X_test)
# Internally:
# Each row travels tree path.
# Example:
# Area=3000 -> right branch
# Bedrooms=5 -> right branch
# Final leaf price = 100

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
# STEP 9 : PREDICT NEW HOUSE PRICE
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
# 5. Create Decision Tree model
# 6. Train using fit()
# 7. Predict prices
# 8. Evaluate using MAE, MSE, R2
# 9. Predict new house price
# Best for:
# Non-linear relationships
# Rule based learning
# Risk:
# Can overfit if tree too deep