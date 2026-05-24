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
# LinearRegression:
# Regression algorithm used when output is numeric value.
# Example:
# House Price
# Salary
# Sales
# Temperature

from sklearn.linear_model import LinearRegression
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
# Description:
# X = Input columns
# y = Output column
# Features used to predict price:
# Area, Bedrooms, HouseAge

X = df[['Area', 'Bedrooms', 'HouseAge']]
# Target:
# Price

y = df['Price']

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)
# ===============================================================
# STEP 3 : SPLIT DATA INTO TRAIN AND TEST
# ===============================================================
# Description:
# Model should not learn and test on same data.
# 80% -> Training data
# 20% -> Testing data
# random_state=42 gives same split every run.

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

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)
# ===============================================================
# STEP 4 : FEATURE SCALING
# ===============================================================
# Description:
# Linear Regression can work without scaling,
# but scaling helps when features have different ranges.
# Current ranges:
# Area      = 1000 to 3500
# Bedrooms  = 2 to 6
# HouseAge  = 2 to 20
# StandardScaler formula:
# z = (x - mean) / std

scaler = StandardScaler()
# fit_transform:
# fit -> learn mean/std
# transform -> scale values

X_train = scaler.fit_transform(X_train)
# Use same training mean/std for test data

X_test = scaler.transform(X_test)

print("\nScaled X_train:")
print(X_train)

print("\nScaled X_test:")
print(X_test)
# ===============================================================
# STEP 5 : CREATE MODEL
# ===============================================================
# Description:
# Create empty Linear Regression model.

model = LinearRegression()
# Initially:
# weights = 0
# bias = 0
# Weight meaning:
# How much each feature affects price.

print("\nModel Created")
# ===============================================================
# STEP 6 : TRAIN MODEL
# ===============================================================
# Description:
# fit() teaches model using training data.

model.fit(X_train, y_train)
# ===============================================================
# INTERNAL WORKING OF fit()
# ===============================================================
# Linear Regression uses equation:
# y = w1*x1 + w2*x2 + w3*x3 + b
# where:
# x1 = Area
# x2 = Bedrooms
# x3 = HouseAge
# Goal:
# Find best weights and bias.
# It minimizes error between:
# Actual Price vs Predicted Price
# Cost Function usually based on MSE:
# MSE = average((actual - predicted)^2)
# sklearn LinearRegression generally uses:
# Least Squares Method
# It finds best weights mathematically.

print("\nLearned Weights:")
print(model.coef_)

print("\nBias / Intercept:")
print(model.intercept_)
# ===============================================================
# STEP 7 : PREDICT OUTPUT
# ===============================================================
# Description:
# Use test data to predict house prices.

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
# Mean Absolute Error
# Description:
# Average prediction mistake.

mae = mean_absolute_error(y_test, pred)
# Formula:
# average(abs(actual - predicted))

print("\nMAE:")
print(mae)
# ---------------------------------------------------------------
# MSE
# ---------------------------------------------------------------
# Mean Squared Error
# Description:
# Bigger errors get more penalty.

mse = mean_squared_error(y_test, pred)

print("\nMSE:")
print(mse)
# ---------------------------------------------------------------
# R2 SCORE
# ---------------------------------------------------------------
# Description:
# How well model explains data.
# 1.0 = Perfect
# 0.0 = Poor
# Negative = Very bad

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
# Scale using old scaler

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
# 5. Create model
# 6. Train using fit()
# 7. Predict prices
# 8. Evaluate using MAE, MSE, R2
# 9. Predict new house price