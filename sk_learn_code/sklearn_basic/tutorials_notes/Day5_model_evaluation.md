# 📅 Day 5 — Model Evaluation
## Why Model Evaluation Matters
Training a model is only half the job.
You must measure how good it actually is — and the right metric depends on the problem.

```
Wrong metric → wrong conclusion → wrong decision

Example:
  Dataset: 95% healthy, 5% cancer
  A model that predicts "healthy" for everyone → 95% accuracy!
  But it catches ZERO cancer cases → completely useless.
  Accuracy was the wrong metric here.
```

This is why you need to understand **multiple metrics** — each one reveals a different aspect.

## Two Sections Today
| Section | Task | Metrics |
|---------|------|---------|
| Regression Metrics | Predicting numbers | MAE, MSE, RMSE, R² |
| Classification Metrics | Predicting categories | Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report |

# REGRESSION METRICS
# MAE — Mean Absolute Error
## What is it?
MAE measures the **average size of prediction errors**, ignoring direction (positive or negative).

It answers: "On average, how far off is the model's prediction from the actual value?"

## Formula
`MAE = (1/n) × Σ |yᵢ - ŷᵢ|`

- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $| \cdot |$ = absolute value (makes negatives positive)

## Step-by-Step Calculation
```
Actual:    [200000, 350000, 150000, 400000]
Predicted: [210000, 330000, 160000, 390000]

Errors:     [10000, -20000, 10000, -10000]
Absolute:   [10000,  20000, 10000,  10000]
Mean:       (10000 + 20000 + 10000 + 10000) / 4 = 12500

MAE = 12,500

Meaning: On average the model is off by ₹12,500
```

## What Does it Output?
```python
from sklearn.metrics import mean_absolute_error

y_actual    = [200000, 350000, 150000, 400000]
y_predicted = [210000, 330000, 160000, 390000]

mae = mean_absolute_error(y_actual, y_predicted)
print("MAE:", mae)
# 12500.0
```

## Properties
| Property | Detail |
|----------|--------|
| Unit | Same as target variable (dollars, kg, etc.) |
| Sensitivity to outliers | Low — outliers contribute equally per error |
| Interpretability | Very easy — directly interpretable |
| Best when | You want a simple average error |

## Real-life Example
**House Price Prediction:**
- MAE = ₹5,00,000 → model is off by ₹5 lakhs on average
- MAE = ₹50,000 → model is very accurate

# MSE — Mean Squared Error
## What is it?
MSE measures the **average of squared errors**.

Squaring does two things:
1. Makes all errors positive (no cancelling)
2. Penalizes large errors much more heavily than small ones

## Formula
`MSE = (1/n) × Σ(yᵢ - ŷᵢ)²`

## Step-by-Step Calculation
```
Actual:    [200000, 350000, 150000, 400000]
Predicted: [210000, 330000, 160000, 390000]

Errors:    [10000,  -20000,  10000,  -10000]
Squared:   [1e8,    4e8,     1e8,    1e8   ]
Mean:      (1e8 + 4e8 + 1e8 + 1e8) / 4 = 1.75e8 = 175,000,000

MSE = 175,000,000
```

Notice: the -20000 error got squared to 4e8 — it contributed 4x more than the 10000 errors.
MSE amplifies large mistakes.

## What Does it Output?
```python
from sklearn.metrics import mean_squared_error

y_actual    = [200000, 350000, 150000, 400000]
y_predicted = [210000, 330000, 160000, 390000]

mse = mean_squared_error(y_actual, y_predicted)
print("MSE:", mse)
# 175000000.0
```

## Properties
| Property | Detail |
|----------|--------|
| Unit | Squared units (dollars², kg²) — hard to interpret directly |
| Sensitivity to outliers | Very high — one big error dominates |
| Use | Loss function for training models, comparing models |
| Best when | Large errors are especially bad (medical, finance) |

# RMSE — Root Mean Squared Error
## What is it?
RMSE is simply the **square root of MSE**.

It fixes the unit problem — RMSE is in the same units as the target variable.

## Formula
`RMSE = √MSE = √( (1/n) × Σ(yᵢ - ŷᵢ)² )`

## Step-by-Step Calculation
```
From above: MSE = 175,000,000
RMSE = sqrt(175,000,000) = 13,228

Meaning: On average the model is off by ₹13,228
         (penalizing large errors more than MAE does)
```

## What Does it Output?
```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_actual    = [200000, 350000, 150000, 400000]
y_predicted = [210000, 330000, 160000, 390000]

rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
print("RMSE:", rmse)
# 13228.75
# OR in newer sklearn:
rmse = mean_squared_error(y_actual, y_predicted, squared=False)
```

## MAE vs RMSE
| Scenario | Better Metric |
|----------|--------------|
| Outliers exist and are expected | MAE (not affected as much) |
| Large errors are unacceptable | RMSE (punishes big mistakes) |
| Want easy interpretation | MAE |
| Training neural networks | MSE/RMSE (differentiable) |

# R² Score — Coefficient of Determination
## What is it?
R² measures **how well your model explains the variation in the data**.

It compares your model's errors to a "dumb baseline" that always predicts the mean.
A perfect model has R²=1. A model equal to the baseline has R²=0.

## Formula
`R² = 1 - (SS_res / SS_tot)`

Where:

`SS_res = Σ(yᵢ - ŷᵢ)²    (your model's total squared error)`

`SS_tot = Σ(yᵢ - ȳ)²    (total variation in y)`

## Step-by-Step Calculation
```
Actual y:   [200000, 350000, 150000, 400000]
Predicted:  [210000, 330000, 160000, 390000]
Mean y:     275000

SS_res = (200000-210000)² + (350000-330000)² + (150000-160000)² + (400000-390000)²
       = 100M + 400M + 100M + 100M = 700M

SS_tot = (200000-275000)² + (350000-275000)² + (150000-275000)² + (400000-275000)²
       = 5625M + 5625M + 15625M + 15625M = 42500M

R² = 1 - (700M / 42500M) = 1 - 0.0165 = 0.9835
```

**R² = 0.98 → model explains 98% of the variance in prices. Excellent.**

## What Does it Output?
```python
from sklearn.metrics import r2_score

y_actual    = [200000, 350000, 150000, 400000]
y_predicted = [210000, 330000, 160000, 390000]

r2 = r2_score(y_actual, y_predicted)
print("R²:", r2)
# 0.9835...
```

## R² Interpretation Table
| R² Value | Meaning | Action |
|----------|---------|--------|
| 1.00 | Perfect (may indicate data leakage) | Investigate |
| 0.90+ | Excellent | Deploy |
| 0.70–0.90 | Good | Use with confidence |
| 0.50–0.70 | Average | Try more features or better model |
| 0.20–0.50 | Weak | Major improvements needed |
| 0 | Equal to predicting the mean always | Useless |
| Negative | Worse than predicting the mean | Something is wrong |

## Complete Regression Metrics Example
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
X = np.random.rand(200, 3) * 100
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("=" * 35)
print(f"MAE  (avg error):     {mae:>10.2f}")
print(f"MSE  (squared error): {mse:>10.2f}")
print(f"RMSE (root MSE):      {rmse:>10.2f}")
print(f"R²   (explained var): {r2:>10.4f}")
print("=" * 35)
```

# CLASSIFICATION METRICS
## The Confusion Matrix — Foundation of All Classification Metrics
Before learning individual metrics, you must understand the confusion matrix.

For a binary classifier (positive=1, negative=0):

```
                    PREDICTED
                  Positive   Negative
ACTUAL Positive |    TP    |    FN   |
       Negative |    FP    |    TN   |
```

| Term | Full Name | Meaning |
|------|-----------|---------|
| TP | True Positive | Actually positive, predicted positive — CORRECT |
| TN | True Negative | Actually negative, predicted negative — CORRECT |
| FP | False Positive | Actually negative, predicted positive — WRONG (Type I error) |
| FN | False Negative | Actually positive, predicted negative — WRONG (Type II error) |

# Accuracy Score
## What is it?
Accuracy = fraction of **all predictions that are correct** (both TP and TN).
## Formula
`Accuracy = (TP + TN) / (TP + TN + FP + FN)`
## Step-by-Step
```
TP=50, TN=30, FP=10, FN=10  (total=100)
Accuracy = (50+30) / 100 = 0.80 = 80%
```
## What Does it Output?
```python
from sklearn.metrics import accuracy_score

y_actual    = [1, 0, 1, 1, 0, 1, 0, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0]

acc = accuracy_score(y_actual, y_predicted)
print("Accuracy:", acc)   # 0.75
```
## When Accuracy is Misleading
```
Dataset: 950 "no cancer", 50 "cancer"
Model predicts "no cancer" for EVERYONE.
Accuracy = 950/1000 = 95%  ← looks great!
But it catches ZERO cancer cases → useless for the actual task.
```

Use Precision, Recall, and F1 for imbalanced data.

# Precision Score
## What is it?
Precision = of all cases the model **predicted as positive**, how many were actually positive?

It answers: "When the model says YES, how often is it right?"
## Formula
`Precision = TP / (TP + FP)`
## Step-by-Step
```
Model predicted "cancer" for 60 patients.
Of those 60: 50 actually had cancer (TP), 10 did not (FP).

Precision = 50 / (50+10) = 50/60 = 0.833

The model is right 83% of the time when it says "cancer".
```
## What Does it Output?
```python
from sklearn.metrics import precision_score

y_actual    = [1, 0, 1, 1, 0, 1, 0, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0]

prec = precision_score(y_actual, y_predicted)
print("Precision:", prec)
```
## When to Prioritize Precision
Use when **false positives are costly**:
- Spam filter — you do not want to incorrectly mark real emails as spam
- Fraud detection — you do not want to block legitimate transactions

# Recall Score
## What is it?
Recall = of all cases that were **actually positive**, how many did the model catch?

It answers: "Out of all real positives, how many did the model find?"
Also called **Sensitivity** or **True Positive Rate**.
## Formula
`Recall = TP / (TP + FN)`
## Step-by-Step
```
100 patients actually have cancer.
Model correctly detected 80 (TP), missed 20 (FN).

Recall = 80 / (80+20) = 80/100 = 0.80

Model catches 80% of all cancer cases.
```
## What Does it Output?
```python
from sklearn.metrics import recall_score

y_actual    = [1, 0, 1, 1, 0, 1, 0, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0]

rec = recall_score(y_actual, y_predicted)
print("Recall:", rec)
```
## When to Prioritize Recall
Use when **false negatives are costly**:
- Cancer detection — missing a cancer is very dangerous
- COVID testing — a missed positive can spread disease
- Security systems — missing an intrusion is worse than a false alarm

## Precision vs Recall Trade-off
There is always a **trade-off** between precision and recall.

```
Lower threshold (e.g. 0.3 instead of 0.5):
→ Model says "positive" more often
→ Catches more real positives → Recall goes UP
→ But also more false positives → Precision goes DOWN

Higher threshold (e.g. 0.7):
→ Model only says "positive" when very confident
→ Fewer false positives → Precision goes UP
→ But misses more real positives → Recall goes DOWN
```

You cannot maximize both simultaneously with one threshold.
F1 Score balances them.

# F1 Score
## What is it?
F1 Score is the **harmonic mean** of Precision and Recall.

It gives a single balanced number when both precision and recall matter.
## Formula
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`
## Why Harmonic Mean and Not Regular Average?
Regular average can be misleading:
```
Precision = 1.0, Recall = 0.0
Regular mean = 0.5  ← looks okay
Harmonic mean (F1) = 0  ← correctly shows the model is broken
```

Harmonic mean punishes extreme imbalances between precision and recall.
## Step-by-Step
```
Precision = 0.833, Recall = 0.80

F1 = 2 * (0.833 * 0.80) / (0.833 + 0.80)
   = 2 * 0.6664 / 1.633
   = 0.816
```
## What Does it Output?
```python
from sklearn.metrics import f1_score

y_actual    = [1, 0, 1, 1, 0, 1, 0, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0]

f1 = f1_score(y_actual, y_predicted)
print("F1 Score:", f1)
```

For multi-class:
```python
f1 = f1_score(y_actual, y_predicted, average="macro")    # equal weight per class
f1 = f1_score(y_actual, y_predicted, average="weighted") # weight by class size
f1 = f1_score(y_actual, y_predicted, average="micro")    # global TP/FP/FN
```

# Confusion Matrix
## What is it?
The confusion matrix is a **grid that shows all prediction outcomes** — TP, TN, FP, FN in one view.

It gives the full picture of where the model is succeeding and failing.
## What Does it Output?
```python
from sklearn.metrics import confusion_matrix

y_actual    = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]

cm = confusion_matrix(y_actual, y_predicted)
print(cm)
# [[3 2]    ← TN=3, FP=2  (actual negative row)
#  [1 4]]   ← FN=1, TP=4  (actual positive row)
```
## Reading the Matrix
```
Row 0 = actual class 0 (negative)
Row 1 = actual class 1 (positive)
Col 0 = predicted class 0 (negative)
Col 1 = predicted class 1 (positive)

cm[0][0] = TN = 3  → correctly predicted negative
cm[0][1] = FP = 2  → predicted positive but actually negative
cm[1][0] = FN = 1  → predicted negative but actually positive
cm[1][1] = TP = 4  → correctly predicted positive
```
## Visualizing with Heatmap
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot()
plt.title("Confusion Matrix — Cancer Detection")
plt.show()
```

# Classification Report
## What is it?
`classification_report()` prints **precision, recall, and F1 score for every class** in one table.
It is the most complete single summary of classification performance.
## What Does it Output?
```python
from sklearn.metrics import classification_report

y_actual    = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_predicted = [0, 1, 1, 0, 1, 2, 0, 2, 2, 0]

print(classification_report(y_actual, y_predicted, target_names=["Cat", "Dog", "Fish"]))
```

```
              precision    recall  f1-score   support

         Cat       1.00      1.00      1.00         4
         Dog       0.67      0.67      0.67         3
        Fish       0.67      0.67      0.67         3

    accuracy                           0.80        10
   macro avg       0.78      0.78      0.78        10
weighted avg       0.80      0.80      0.80        10
```
## Reading the Report
| Column | Meaning |
|--------|---------|
| `precision` | Of all predicted as this class, how many were correct |
| `recall` | Of all actual this class, how many were caught |
| `f1-score` | Harmonic mean of precision and recall |
| `support` | How many actual samples of this class in y_test |
| `accuracy` | Overall accuracy |
| `macro avg` | Average of all classes equally weighted |
| `weighted avg` | Average weighted by support (class size) |

## Day 5 — All Metrics at a Glance
### Regression
| Metric | Formula | Good value | Unit |
|--------|---------|------------|------|
| MAE | mean(|actual - pred|) | Lower is better | Same as y |
| MSE | mean((actual - pred)²) | Lower is better | Squared units |
| RMSE | sqrt(MSE) | Lower is better | Same as y |
| R² | 1 - SS_res/SS_tot | Closer to 1 | Unitless (0 to 1) |
### Classification
| Metric | What it measures | Use when |
|--------|-----------------|---------|
| Accuracy | Overall correctness | Balanced classes |
| Precision | Correctness of positive predictions | FP is costly (spam) |
| Recall | Coverage of actual positives | FN is costly (cancer) |
| F1 | Balance of precision and recall | Imbalanced data |
| Confusion Matrix | Full breakdown of all outcomes | Detailed analysis |
| Classification Report | All metrics per class | Multi-class analysis |

## Complete Evaluation Project — All Metrics Together
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

print("=" * 50)
print("REGRESSION EVALUATION")
print("=" * 50)

X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred_reg):.4f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_reg)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_reg):.4f}")

print("\n" + "=" * 50)
print("CLASSIFICATION EVALUATION")
print("=" * 50)

X, y = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_clf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_clf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_clf):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_clf):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_clf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_clf))
```

> Day 5 Complete.
> You now understand every major evaluation metric — what it measures, how it is calculated, and when to use each one.
>
> Next: **Day 6 — Model Improvement**
> (Cross Validation, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV)