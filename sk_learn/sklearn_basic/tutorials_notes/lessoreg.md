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

from sklearn.preprocessing import StandardScaler
# Lasso:
# Regression algorithm used when output is numeric value.
# It is Linear Regression + L1 Regularization.
#
# Helps reduce overfitting.
# Can make some weights exactly zero.
# Useful for feature selection.

from sklearn.linear_model import Lasso
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
# We create sample house price dataset.
# Columns:
# Area        -> house square feet
# Bedrooms    -> number of bedrooms
# HouseAge    -> age of house in years
# Price       -> target output (house price in lakhs)

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
# Lasso uses weights, so scaling is recommended.
# Different feature ranges:
# Area = large values
# Bedrooms = small values
# HouseAge = medium values
# Scaling avoids domination by large columns.

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
# alpha = regularization strength
#
# low alpha  -> behaves closer to Linear Regression
# high alpha -> stronger penalty, more weights may become zero

model = Lasso(alpha=0.1)

print("\nModel Created")
# ===============================================================
# STEP 6 : TRAIN MODEL
# ===============================================================

model.fit(X_train, y_train)
# ===============================================================
# INTERNAL WORKING OF fit()
# ===============================================================
# Lasso uses equation:
# y = w1*x1 + w2*x2 + w3*x3 + b
# Goal:
# Find best weights + bias
# But Lasso adds penalty:
# Cost =
# MSE + alpha(|w1| + |w2| + |w3|)
# MSE = average((actual - predicted)^2)
# L1 penalty can shrink some weights to zero.
# Example:
# If Bedrooms is not useful:
# weight may become 0
# Benefits:
# 1. Reduces overfitting
# 2. Performs feature selection
# 3. Simpler model

print("\nLearned Weights:")
print(model.coef_)

print("\nBias / Intercept:")
print(model.intercept_)
# ===============================================================
# STEP 7 : PREDICT OUTPUT
# ===============================================================

pred = model.predict(X_test)
# Internally:
# Uses learned equation:
# Price = w1*x1 + w2*x2 + w3*x3 + b

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
# Use same scaler

new_house_scaled = scaler.transform(new_house)

new_price = model.predict(new_house_scaled)

print("\nPredicted Price for New House:")
print(new_price)
# ===============================================================
# FINAL SUMMARY
# ===============================================================
# 1. Create dataset
# 2. Select X and y
# 3. Split train/test
# 4. Scale features
# 5. Create Lasso model
# 6. Train using fit()
# 7. Predict prices
# 8. Evaluate using MAE, MSE, R2
# 9. Predict new house price
# Lasso = Best when many features exist
# and some features may be useless.