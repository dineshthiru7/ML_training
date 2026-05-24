# Machine Learning Notes — Complete Beginner to Intermediate Guide

## Table of Contents
1. [Linear Regression](#1-linear-regression)
2. [Polynomial Regression](#2-polynomial-regression)
3. [Ridge Regression](#3-ridge-regression)
4. [Lasso Regression](#4-lasso-regression)
5. [Logistic Regression](#5-logistic-regression)
6. [Decision Tree](#6-decision-tree)
7. [Random Forest](#7-random-forest)
8. [K-Nearest Neighbors (KNN)](#8-k-nearest-neighbors-knn)
9. [Naive Bayes](#9-naive-bayes)
10. [Support Vector Machine (SVM)](#10-support-vector-machine-svm)
11. [Gradient Boosting](#11-gradient-boosting)
12. [K-Means Clustering](#12-k-means-clustering)
13. [Hierarchical Clustering](#13-hierarchical-clustering)
14. [DBSCAN](#14-dbscan)
15. [HDBSCAN](#15-hdbscan)
16. [Apriori Algorithm](#16-apriori-algorithm)
17. [FP-Growth Algorithm](#17-fp-growth-algorithm)
18. [ECLAT Algorithm](#18-eclat-algorithm)

---

# 1. Linear Regression

## What is Linear Regression?
Linear Regression is a **supervised machine learning algorithm** used to predict **continuous numerical values**.

Examples:
- House price prediction
- Salary prediction
- Temperature prediction
- Student marks prediction

## Main Goal
Find the **best fit line** that minimizes prediction error — a line that best represents the relationship between input `x` and output `y`.

## Equation

**Simple Linear Regression (1 feature):**
```
y = m*x + b

Where:
  y = predicted value (output)
  m = slope (weight / coefficient)
  x = input feature
  b = intercept (bias)
```

**Multiple Linear Regression (more than 1 feature):**
```
y = m1*x1 + m2*x2 + ... + mn*xn + b

Where:
  m1, m2, ... = weights for each feature
  x1, x2, ... = input features
  b            = intercept
```

## How Linear Regression Works — Full Step-by-Step

### Step 1 — Initialize m and b Randomly
We start with initial guesses (random or zero):
```
m = 0
b = 0
(or small random values like m=0.5, b=1)

These are just starting points — not meaningful yet.
```

### Step 2 — Predict Values
Pass every input `x` into the equation to get predicted values:
```
y_hat = m*x + b

Example:
  x = 1  → y_hat = 0*1 + 0 = 0
  x = 2  → y_hat = 0*2 + 0 = 0
  x = 3  → y_hat = 0*3 + 0 = 0
(All zeros at start — we haven't learned anything yet)
```

### Step 3 — Calculate Loss Function
**Loss = error for a single prediction**
```
Loss for one point = (y - y_hat)^2   (squared error)

Why squared?
  - Makes all errors positive
  - Penalizes large errors more heavily
  - Smooth and differentiable (needed for gradient descent)
  - Clean mathematical solution exists
```

Example:
```
x   Actual y   Predicted y_hat   Loss = (y - y_hat)^2
1      3              1               (3-1)^2 = 4
2      5              2               (5-2)^2 = 9
3      7              3               (7-3)^2 = 16
```

### Step 4 — Calculate Cost Function
**Cost = average of all losses across the entire dataset**
```
J(m,b) = (1/n) * SUM[ (yi - y_hat_i)^2 ]

This is called Mean Squared Error (MSE).

Key distinction:
  Loss  → error for ONE data point
  Cost  → average error across the ENTIRE dataset

Total cost at start = (4 + 9 + 16) / 3 = 9.67
```

### Step 5 — Perform Gradient Descent
Gradient descent reduces the cost function step by step.

**What is a gradient?**
```
The gradient (derivative) tells:
  1. Which DIRECTION reduces the error (increase or decrease m/b?)
  2. How STEEP the error curve is (how big a step to take?)

Think of it like standing on a hill:
  - Gradient tells which direction is downhill fastest
  - Gradient descent keeps moving downhill until minimum error is reached
```

**Why gradients are needed:**
```
Cost function J(m,b) is a curve (bowl shape / convex).
At any point on the curve:
  - Positive gradient → we are on right side → decrease m/b
  - Negative gradient → we are on left side → increase m/b
  - Zero gradient     → we are at the bottom → minimum found!
```

**Gradient formulas:**
```
dJ/dm = -(2/n) * SUM[ xi * (yi - y_hat_i) ]
dJ/db = -(2/n) * SUM[ (yi - y_hat_i) ]
```

### Step 6 — Update m and b Using Learning Rate
```
m = m - alpha * (dJ/dm)
b = b - alpha * (dJ/db)

Where:
  alpha = learning rate (how big a step we take, e.g. 0.01)
  dJ/dm = gradient of cost w.r.t. m
  dJ/db = gradient of cost w.r.t. b
```

**Effect of learning rate:**
```
alpha too large  → overshoots minimum → cost increases → diverges
alpha too small  → very slow convergence → takes too long
alpha just right → smoothly reaches minimum (typical: 0.01 or 0.001)
```

### Step 7 — Repeat Until Convergence
```
This looping process is called: iterations / epochs

Loop:
  1. Compute y_hat using current m, b
  2. Compute cost J
  3. Compute gradients dJ/dm and dJ/db
  4. Update m and b
  5. Check: has cost stopped changing significantly?
     → YES: stop (converged)
     → NO:  go back to step 1

The process continues until the cost function CONVERGES,
meaning the change in error becomes extremely small.
```

**Concrete iteration example:**
```
Iter    m        b       Cost
0       0.000    0.000   9.670  ← start
10      0.430    0.120   3.210
50      0.890    0.450   0.980
100     1.920    0.210   0.120
200     1.990    0.050   0.003  ← nearly converged
250     1.999    0.001   0.0001 ← converged (cost barely changes)

Final: m ≈ 2.0, b ≈ 0.0
Line: y = 2x + 0 → correctly maps x to 2x
```

## Example Dataset
```
Hours   Marks
  1       2
  2       4
  3       6
  4       8

Pattern: y = 2*x → m=2, b=0 is the perfect fit
```

## Gradient Descent Variants
```
Batch Gradient Descent:
  Uses entire dataset to compute gradient each step
  Stable but slow for large datasets

Stochastic Gradient Descent (SGD):
  Uses ONE random sample per step
  Fast but noisy updates

Mini-Batch Gradient Descent:
  Uses small batches (e.g. 32 samples)
  Best of both worlds — most commonly used in practice
```

## Simple vs Multiple Linear Regression
| | Simple | Multiple |
|---|---|---|
| Features | 1 | 2 or more |
| Equation | `y = m*x + b` | `y = m1*x1 + m2*x2 + ... + b` |
| Visual shape | Line in 2D | Hyperplane in N+1 dimensions |
| Coefficients | 1 weight | N weights |

## When to Use
- Linear relationship between input and output
- As a baseline model before trying complex models
- When interpretability matters

## When NOT to Use
- Non-linear relationships → use Decision Tree or Random Forest
- Many correlated features → use Ridge or Lasso
- Data has many outliers → linear regression is very sensitive

## Code Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Simple linear regression
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # must be 2D
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])        # 2.0
print("Intercept (b):", model.intercept_)  # 0.0
print("Predict x=6:", model.predict([[6]])) # 12.0
```

## Advantages
- Simple to understand and implement
- Fast training
- Interpretable (you can see each feature's weight)
- Works well when relationship is truly linear

## Limitations
- Only captures linear relationships
- Sensitive to outliers (one bad point can skew the line)
- Assumes no multicollinearity between features

---

# 2. Polynomial Regression

## What is Polynomial Regression?
Polynomial Regression is an **extension of linear regression** that models **non-linear (curved) relationships** between input and output.

It still uses linear regression internally — it just adds polynomial features first.

## Why Use Polynomial Regression?
When data follows a curved pattern instead of a straight line:
- Population growth
- Stock price trends
- Temperature variations over time
- Drug dosage vs response

## Equation
```
y = b + m1*x + m2*x^2 + m3*x^3 + ...

Degree 1: y = b + m1*x              (straight line)
Degree 2: y = b + m1*x + m2*x^2    (parabola / curve)
Degree 3: y = b + m1*x + m2*x^2 + m3*x^3  (more flexible curve)
```

## How it Works
```
Step 1: Transform original feature x into polynomial features:
  Original: [x]
  Degree 2: [x, x^2]
  Degree 3: [x, x^2, x^3]

Step 2: Apply standard Linear Regression on these transformed features.

The model finds the best weights for x, x^2, x^3 etc.
```

## Degree of Polynomial
| Degree | Shape | Risk |
|--------|-------|------|
| 1 | Straight line | Underfitting |
| 2 | Single curve (parabola) | Good for moderate curves |
| 3 | More flexible curve | Moderate |
| 5+ | Very wiggly | High overfitting risk |

## Overfitting Warning
Higher degree = more flexibility BUT:
```
Degree 2:  Captures real trend well
Degree 10: Perfectly fits training data, fails on new data → OVERFITTING
```
Always use cross-validation to choose the right degree.

## Code Example
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # y = x^2

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
model.fit(X, y)
print("Predict x=6:", model.predict([[6]]))  # ~36
```

## Advantages
- Captures non-linear patterns
- Still interpretable
- Uses same linear regression math internally

## Limitations
- Hard to choose the right degree
- Overfits easily at high degrees
- Computationally heavier than linear regression

---

# 3. Ridge Regression

## What is Ridge Regression?
Ridge Regression is **Linear Regression with L2 regularization**.

It adds a penalty on large weights to prevent overfitting and handle correlated features.

## Why Ridge Regression?
**Problems it solves:**
- Overfitting (model too complex)
- Multicollinearity (features are correlated with each other)
- Unstable weights that change drastically with small data changes

## Loss Function
```
Ridge Loss = MSE + lambda * SUM(w^2)

= (1/n) * SUM(yi - y_hat_i)^2  +  lambda * (w1^2 + w2^2 + ... + wn^2)

Two competing goals:
  Goal 1: Minimize prediction error (MSE)
  Goal 2: Keep all weights small (penalty term)
```

## Lambda (alpha) — Regularization Strength
```
lambda = 0    → identical to plain Linear Regression (no penalty)
lambda = 1    → mild regularization, small weight reduction
lambda = 10   → medium regularization, noticeable weight shrinkage
lambda = 100  → strong regularization, weights very small
lambda → inf  → all weights forced to ~0 (predicts only the mean)

In sklearn the parameter is called `alpha` (not lambda).
```

## Key Behaviour
```
Feature           Linear Regression   Ridge (alpha=10)
sqft              120.5               98.3    ← shrunk
bedrooms          49800.0             41200.0 ← shrunk
bathrooms         29950.0             25100.0 ← shrunk

ALL weights shrunk → but NO weight becomes exactly zero.
All features are still kept.
```

## Why Ridge Never Zeroes Weights
The L2 penalty creates a **sphere-shaped constraint** in weight space.
The optimal solution always lands on the smooth curve of the sphere — almost never exactly at zero.
(Compare with Lasso's diamond shape — see Lasso section.)

## When to Use
- Many features
- Features are correlated
- Linear regression is overfitting

## Code Example
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = Ridge(alpha=10.0)
model.fit(X_train_s, y_train)
print("R2:", model.score(X_test_s, y_test))
print("Weights:", model.coef_)
```

## Advantages
- Prevents overfitting
- Handles multicollinearity
- Stable weights

## Limitations
- Does not remove any features (all weights just shrink)
- Still linear — cannot capture non-linear patterns
- Need to tune alpha

---

# 4. Lasso Regression

## What is Lasso Regression?
Lasso = **Least Absolute Shrinkage and Selection Operator**

Linear Regression with **L1 regularization** that can drive some weights **exactly to zero**, performing automatic feature selection.

## Loss Function
```
Lasso Loss = MSE + lambda * SUM(|w|)

= (1/n) * SUM(yi - y_hat_i)^2  +  lambda * (|w1| + |w2| + ... + |wn|)

Uses absolute value of weights (L1 norm).
```

## Ridge vs Lasso — Key Geometric Difference
```
Ridge (L2) constraint shape: CIRCLE (sphere)
  → Solution touches smooth curve → weights shrink but NEVER reach zero

Lasso (L1) constraint shape: DIAMOND (rotated square)
  → Solution likely touches a CORNER of diamond
  → Corners have w1=0 or w2=0 → weights are ZEROED OUT

This is why Lasso performs feature selection and Ridge does not.
```

## Effect of Alpha
```
alpha = 0.01 → few features zeroed, close to Linear Regression
alpha = 0.1  → moderate feature elimination
alpha = 1.0  → many features zeroed (default)
alpha = 10.0 → aggressive elimination, very few features kept
```

## Example Output
```
After fit() with alpha=0.5 on 15 features (only 5 truly matter):

Feature 0:  w =  34.2   ← KEPT   (strong signal)
Feature 1:  w =   0.0   ← ZEROED (weak, useless)
Feature 2:  w = -18.7   ← KEPT
Feature 3:  w =   0.0   ← ZEROED
Feature 4:  w =  52.1   ← KEPT
...

model.coef_ = [34.2, 0.0, -18.7, 0.0, 52.1, ..., 0.0]
Features kept: 5 / 15
```

## Ridge vs Lasso Summary
| | Ridge | Lasso |
|---|---|---|
| Penalty | L2 (sum of squared weights) | L1 (sum of absolute weights) |
| Effect | Shrinks all weights | Some weights go to exactly zero |
| Feature selection | No | Yes |
| Algorithm | Closed-form | Coordinate descent (iterative) |
| Best when | Many correlated features | Many irrelevant features |

## Code Example
```python
from sklearn.linear_model import Lasso
import numpy as np

model = Lasso(alpha=0.5)
model.fit(X_train_s, y_train)
print("R2:", model.score(X_test_s, y_test))
print("Non-zero weights:", np.sum(model.coef_ != 0), "/", len(model.coef_))
for i, w in enumerate(model.coef_):
    print(f"  Feature {i}: {w:.3f}  {'KEPT' if w != 0 else 'REMOVED'}")
```

## Advantages
- Automatic feature selection (removes useless features)
- Prevents overfitting
- Simpler final model

## Limitations
- With many correlated features, keeps only one (arbitrarily)
- Coordinate descent is slower than Ridge's closed-form solution
- Cannot handle non-linear patterns

---

# 5. Logistic Regression

## What is Logistic Regression?
Logistic Regression is a **classification algorithm** (despite "regression" in the name).

It predicts the **probability** that a sample belongs to a class, then classifies it using a threshold.

Examples:
- Email → Spam (1) or Not Spam (0)
- Tumor → Malignant (1) or Benign (0)
- Customer → Will Churn (1) or Stay (0)

## The Four Steps

### Step 1 — Linear Combination
```
z = w1*x1 + w2*x2 + ... + wn*xn + b
(Same as linear regression — gives a raw score, any value from -inf to +inf)
```

### Step 2 — Sigmoid Function
```
sigmoid(z) = 1 / (1 + e^(-z))

Maps any z value to a probability between 0 and 1:
  z = -10  → 0.00005  (nearly 0% → class 0)
  z = -1   → 0.27     (27%      → class 0)
  z =  0   → 0.50     (exactly 50%)
  z =  1   → 0.73     (73%      → class 1)
  z = +10  → 0.99995  (nearly 100% → class 1)
```

### Step 3 — Apply Threshold
```
if P(y=1) >= 0.5  → predict class 1
if P(y=1) <  0.5  → predict class 0

Threshold is adjustable:
  Cancer detection → use 0.3 (be more cautious, catch more positives)
  Spam detection   → use 0.7 (reduce false positives)
```

### Step 4 — Gradient Descent on Log Loss
```
Log Loss = -(1/n) * SUM[ yi*log(p_hat) + (1-yi)*log(1-p_hat) ]

Penalizes confident wrong predictions heavily:
  Actual=1, Predicted=0.99 → loss ≈ 0.01  (tiny — correct and confident)
  Actual=1, Predicted=0.01 → loss ≈ 4.60  (huge — wrong and confident)

Gradient descent updates weights to reduce this loss iteratively.
```

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 1.0 | Inverse regularization strength. Small C = more regularization |
| `penalty` | `"l2"` | L1, L2, elasticnet, or None |
| `max_iter` | 100 | Increase if ConvergenceWarning appears (use 1000) |
| `solver` | `"lbfgs"` | Optimization algorithm |

## Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
print("Accuracy:", model.score(X_test_s, y_test))
print("Probabilities:", model.predict_proba(X_test_s[:3]))
```

## Advantages
- Fast training
- Outputs probabilities
- Works well for linearly separable data
- Built-in regularization

## Limitations
- Only linear decision boundary
- Needs feature scaling
- Poor performance on complex non-linear patterns

---

# 6. Decision Tree

## What is a Decision Tree?
A Decision Tree builds a **tree of if-else rules** to make predictions.

Works for both **classification** and **regression**.

```
         outlook == Sunny?
         /               \
       YES                NO
  humidity > 70?       windy == TRUE?
  /        \             /         \
REFUSE    PLAY       REFUSE        PLAY
```

## How it Works — The Splitting Criterion

### For Classification — Gini Impurity
```
Gini(node) = 1 - SUM(pk^2)

Where pk = proportion of class k in the node.

Pure node (all same class): Gini = 0        (best)
Maximally impure (50/50):   Gini = 0.5      (worst for binary)

Example (10 samples: 7 class-A, 3 class-B):
  Gini = 1 - ((7/10)^2 + (3/10)^2) = 1 - (0.49 + 0.09) = 0.42
```

### For Regression — MSE Reduction
```
At each node: find the split that reduces MSE the most.
Leaf node stores: MEAN of all y values in that leaf.
```

## Overfitting Problem
```
max_depth=None → tree grows until each leaf has 1 sample
  Training accuracy = 100% (memorized every point)
  Test accuracy = 60%        (fails on new data)

max_depth=5 → at most 32 leaves
  Training accuracy = 88%
  Test accuracy = 85%   ← good generalization
```

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | None | Limit tree depth to prevent overfitting |
| `criterion` | `"gini"` | Split quality measure |
| `min_samples_leaf` | 1 | Minimum samples per leaf |
| `min_samples_split` | 2 | Minimum samples to split a node |

## Code Example
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
print("Feature importances:", model.feature_importances_)
```

## Advantages
- Easy to interpret and visualize
- No need for feature scaling
- Handles non-linear relationships
- Works with mixed feature types

## Limitations
- Overfits easily without depth limits
- Unstable — small data change → completely different tree
- Single tree less accurate than ensemble methods

---

# 7. Random Forest

## What is Random Forest?
Random Forest builds **many Decision Trees** and combines their results:
- **Classification** → majority vote
- **Regression** → average of all tree predictions

**"Wisdom of the crowd"** — individual trees make errors, but averaged/voted results are much more accurate.

## Two Core Techniques

### 1. Bootstrap Sampling (Bagging)
```
For each of 100 trees:
  Sample n training rows WITH REPLACEMENT
  → Each tree sees different data
  → Each tree makes different errors
  → Errors cancel out when combined

63% unique rows per bootstrap sample on average.
37% are "Out-of-Bag" (OOB) → free validation set!
```

### 2. Feature Randomness
```
At each split node:
  Instead of trying ALL features → randomly pick sqrt(n_features) features
  Find best split among ONLY those features

Forces trees to use different features → less correlated trees → better ensemble.
```

## predict() — Majority Vote (Classification)
```
Test sample → run through all 100 trees
Tree 1 → class 1 (Spam)
Tree 2 → class 0 (Not Spam)
Tree 3 → class 1 (Spam)
...
Tree 100 → class 1 (Spam)

Votes: Spam=67, Not Spam=33  → Final: Spam
```

## Feature Importance
```
model.feature_importances_
→ Shows which features caused the most MSE/Gini reduction across all trees
→ Sums to 1.0
→ Higher value = more important feature
```

## OOB Score
```python
model = RandomForestClassifier(oob_score=True)
model.fit(X_train, y_train)
print(model.oob_score_)  # Free accuracy estimate, no test set needed
```

## Code Example
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                oob_score=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))
print("OOB Accuracy:", model.oob_score_)
print("Feature importances:", model.feature_importances_)
```

## Advantages
- Very accurate on most datasets
- Handles non-linear relationships
- Robust to noise and outliers
- Built-in feature importance
- Free OOB validation

## Limitations
- Slower than single decision tree
- Less interpretable than a single tree
- More memory usage

---

# 8. K-Nearest Neighbors (KNN)

## What is KNN?
KNN classifies (or regresses) a new point by looking at the **K nearest training points** and taking a majority vote (classification) or average (regression).

No actual training — it memorizes all data. Called a **lazy learner**.

## How predict() Works — Step by Step
```
New point arrives: X_test[0]

Step 1: Compute distance from X_test[0] to EVERY training point
        Default: Euclidean distance = sqrt(SUM((xi - xj)^2))

Step 2: Sort all training points by distance (nearest first)

Step 3: Pick the K nearest neighbors

Step 4 (Classification): Majority vote among K labels → predicted class
Step 4 (Regression):     Average of K target values  → predicted value
```

## Why Feature Scaling is CRITICAL for KNN
```
Without scaling (raw values):
  feature1: glucose  70-200  → max distance = 130
  feature2: bmi      18-45   → max distance =  27
  feature3: age      20-70   → max distance =  50

  Euclidean distance dominated by glucose (130 >> 27, 50)
  → KNN ignores bmi and age → wrong neighbors!

With StandardScaler:
  All features → mean=0, std=1 → same range ~4
  → All features contribute equally → correct neighbors
```

## Effect of K
```
K=1   → Exact nearest neighbor → training accuracy 100% → overfits
K=5   → Vote among 5 → noise cancels out → default, usually good
K=15  → Smoother boundary, less noise sensitivity
K=n   → Always predicts majority class → useless

Use ODD K for binary classification to avoid 50/50 tie.
```

## Code Example
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train_s, y_train)
print("Accuracy:", model.score(X_test_s, y_test))
```

## Advantages
- Simple, no training needed
- Naturally handles non-linear boundaries
- No assumptions about data distribution

## Limitations
- Slow prediction on large datasets (O(n*d) per point)
- Must store all training data in memory
- MUST scale features before use

---

# 9. Naive Bayes

## What is Naive Bayes?
Naive Bayes is a **probabilistic classification algorithm** based on Bayes' Theorem.

"Naive" = it assumes all features are **independent** of each other (which is rarely true, but it still works well in practice).

Used heavily in:
- Text classification (spam detection)
- Sentiment analysis
- Document categorization

## Bayes' Theorem
```
P(class | features) = P(features | class) * P(class) / P(features)

In plain language:
  "Probability of a class given the features
   = how likely those features appear in that class × how common the class is"

For prediction we compare P(class=0 | X) vs P(class=1 | X)
and predict whichever is higher.
```

## How it Works — Spam Example
```
Training data: 100 emails
  60 are Spam,   40 are Not Spam

P(Spam)     = 60/100 = 0.6
P(Not Spam) = 40/100 = 0.4

In Spam emails:
  "FREE"  appears in 55/60 = 91.7%
  "WIN"   appears in 48/60 = 80.0%

In Not Spam emails:
  "FREE"  appears in  5/40 = 12.5%
  "WIN"   appears in  2/40 =  5.0%

New email: contains "FREE" and "WIN"

P(Spam      | FREE, WIN) ∝ 0.6 × 0.917 × 0.800 = 0.440
P(Not Spam  | FREE, WIN) ∝ 0.4 × 0.125 × 0.050 = 0.003

Spam wins → predict SPAM
```

## Types of Naive Bayes
| Type | When to Use | Assumes |
|------|-------------|---------|
| `GaussianNB` | Continuous features | Features follow Gaussian (normal) distribution |
| `MultinomialNB` | Text / count data | Features are counts (word frequencies) |
| `BernoulliNB` | Binary features | Features are 0 or 1 (word present or not) |

## Code Example
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
print("Probabilities:", model.predict_proba(X_test[:3]))
```

## Advantages
- Extremely fast training and prediction
- Works well with small training data
- Handles high-dimensional data (text) very well
- No need for feature scaling

## Limitations
- Independence assumption rarely holds in practice
- Continuous features must follow Gaussian distribution (GaussianNB)
- Poor when features are strongly correlated

---

# 10. Support Vector Machine (SVM)

## What is SVM?
SVM finds the **hyperplane that best separates classes** with the **maximum margin**.

The points closest to the boundary are called **support vectors** — these alone define the boundary.

## The Core Idea — Maximum Margin
```
Many lines can separate two classes.
SVM finds the one that is FARTHEST from the nearest points on both sides.

       Class 0 (*)            Class 1 (o)

  *  *  *  *                  o  o  o  o
     [*] ---- BOUNDARY ---- [o]
  *  *  *  *                  o  o  o  o

  [*] and [o] = support vectors (nearest points to boundary)
  Margin = distance from boundary to each support vector line.
  SVM goal: MAXIMISE this margin width.
```

## Parameter C — Soft Margin
Real data has overlapping classes. C controls the trade-off:
```
High C (e.g. 100):
  "Classify every training point correctly, even if margin is narrow"
  → Narrow margin → complex boundary → risk of overfitting

Low C (e.g. 0.01):
  "Allow some misclassifications to get a wider, safer margin"
  → Wide margin → simpler boundary → better generalization

C=1.0 (default) → balanced trade-off
```

## Kernels — Handling Non-linear Data
When classes are not linearly separable, kernels transform data into higher dimensions:

| Kernel | When to Use | Formula |
|--------|-------------|---------|
| `"linear"` | Linearly separable data | `K(x,x') = x dot x'` |
| `"rbf"` | Non-linear data (DEFAULT) | `K(xi,xj) = exp(-gamma * ||xi-xj||^2)` |
| `"poly"` | Polynomial patterns | `K(x,x') = (gamma*x dot x' + r)^degree` |
| `"sigmoid"` | Neural network-like | `K(x,x') = tanh(gamma*x dot x' + r)` |

## RBF Gamma Parameter
```
High gamma (e.g. 10):
  → Each training point influences only a tiny nearby region
  → Very complex boundary → likely overfits

Low gamma (e.g. 0.001):
  → Each point influences a wide region
  → Very smooth boundary → may underfit

gamma="scale" (default) = 1 / (n_features * X.var())  ← automatic good choice
```

## Code Example
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_s, y_train)
print("Accuracy:", model.score(X_test_s, y_test))
print("Support vectors per class:", model.n_support_)
```

## Advantages
- Very effective in high-dimensional spaces
- Works well with complex, non-linear boundaries (with kernels)
- Memory efficient (only stores support vectors)

## Limitations
- Slow on large datasets (n > 100k rows)
- Needs feature scaling
- Hard to interpret
- Kernel and C/gamma must be tuned carefully

---

# 11. Gradient Boosting

## What is Gradient Boosting?
Gradient Boosting builds trees **sequentially** — each new tree tries to **correct the errors** of the previous trees.

Unlike Random Forest (parallel trees), Gradient Boosting is **additive and sequential**.

Popular implementations:
- `sklearn.ensemble.GradientBoostingClassifier`
- **XGBoost** (extreme gradient boosting)
- **LightGBM** (light gradient boosting machine)
- **CatBoost** (categorical boosting)

## Core Idea — Learning from Mistakes
```
Step 1: Start with a simple prediction (e.g. mean of y)
Step 2: Compute residuals (errors) = actual - predicted
Step 3: Build a small decision tree to PREDICT these residuals
Step 4: Add this tree's prediction to the current prediction
Step 5: Compute new residuals
Step 6: Build another tree on NEW residuals
...repeat for n_estimators trees

Each tree patches the weaknesses of all previous trees.
```

## Example — Iteration by Iteration
```
True price: 300,000

Iteration 0 (initial guess):
  Prediction = 250,000
  Residual   = 50,000

Tree 1 predicts residual: 30,000
  New prediction = 250,000 + 0.1 * 30,000 = 253,000
  (0.1 = learning rate — small cautious step)
  Residual = 47,000

Tree 2 predicts new residual: 28,000
  New prediction = 253,000 + 0.1 * 28,000 = 255,800
  Residual = 44,200

...after 100 trees:
  Prediction ≈ 299,800  ← very close to true value
```

## Learning Rate Trade-off
```
High learning rate (e.g. 0.5):
  → Big steps → converges fast → may overshoot → overfitting risk

Low learning rate (e.g. 0.01):
  → Small steps → more stable → needs more trees → better generalization

Typical: learning_rate=0.05 to 0.1, n_estimators=100-500
```

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of trees (boosting rounds) |
| `learning_rate` | 0.1 | Step size for each tree's contribution |
| `max_depth` | 3 | Depth of each individual tree (keep small: 3-5) |
| `subsample` | 1.0 | Fraction of data used per tree (0.8 adds randomness) |

## Code Example
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                    max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

## Random Forest vs Gradient Boosting
| | Random Forest | Gradient Boosting |
|---|---|---|
| Tree order | Parallel (independent) | Sequential (each corrects previous) |
| Speed | Faster | Slower |
| Overfitting | Harder to overfit | Can overfit without tuning |
| Accuracy | High | Often higher |
| Tuning effort | Low | Higher |

## Advantages
- Often achieves the highest accuracy on tabular data
- Handles non-linear relationships
- Built-in feature importance
- Works well out of the box with XGBoost/LightGBM

## Limitations
- Slower to train than Random Forest
- More hyperparameters to tune
- Less interpretable
- Can overfit on noisy data

---

# 12. K-Means Clustering

## What is K-Means?
K-Means is an **unsupervised clustering algorithm** that divides data into **K groups (clusters)** based on similarity.

No labels needed — it discovers structure in data automatically.

Examples:
- Customer segmentation
- Document grouping
- Image compression
- Anomaly detection

## How K-Means Works — Step by Step
```
Step 1: Choose K (number of clusters)

Step 2: Initialize K centroids randomly (or with K-Means++ for better start)

Step 3: Assign each point to its NEAREST centroid
  (using Euclidean distance)

Step 4: Recalculate each centroid = MEAN of all points assigned to it

Step 5: Repeat steps 3-4 until centroids stop moving (convergence)
```

## Concrete Example
```
K=2, 6 data points: [1,2], [1,3], [2,2], [8,7], [8,8], [9,7]

Iteration 1:
  Centroids initialized: C1=[1,2], C2=[9,7]
  Assign points:
    [1,2] → C1 (dist=0),   [1,3] → C1, [2,2] → C1
    [8,7] → C2,             [8,8] → C2, [9,7] → C2

  Update centroids:
    C1 = mean([[1,2],[1,3],[2,2]]) = [1.33, 2.33]
    C2 = mean([[8,7],[8,8],[9,7]]) = [8.33, 7.33]

Iteration 2:
  Re-assign (same assignments → no change)
  Centroids stable → STOP

Result: Cluster 1 = {[1,2],[1,3],[2,2]},  Cluster 2 = {[8,7],[8,8],[9,7]}
```

## How to Choose K — Elbow Method
```
Run K-Means for K=1,2,3,...,10
Plot WCSS (Within-Cluster Sum of Squares) vs K:
  K=1: WCSS very high
  K=2: WCSS drops sharply
  K=3: WCSS drops moderately
  K=4: WCSS barely drops      ← ELBOW POINT → choose K=3 or K=4

WCSS = SUM of squared distances from each point to its cluster centroid
```

## Code Example
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Find optimal K using elbow method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_s)
    wcss.append(km.inertia_)

# Fit final model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_s)
print("Cluster labels:", model.labels_)
print("Centroids:", model.cluster_centers_)
```

## Advantages
- Simple and fast
- Scales well to large datasets
- Easy to interpret

## Limitations
- Must choose K in advance
- Sensitive to initial centroid placement (use `n_init=10`)
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Must scale features

---

# 13. Hierarchical Clustering

## What is Hierarchical Clustering?
Hierarchical Clustering builds a **tree of clusters (dendrogram)** showing how data merges from individual points up to one big cluster.

No need to specify K in advance.

Two types:
- **Agglomerative** (bottom-up): start with each point as its own cluster → merge
- **Divisive** (top-down): start with one cluster → split

## Agglomerative — How it Works
```
Step 1: Every point is its own cluster (n clusters)

Step 2: Find the TWO closest clusters and merge them into one
  → n-1 clusters now

Step 3: Repeat merging → n-2 clusters

Step 4: Continue until all points are in ONE cluster

Step 5: Cut the dendrogram at desired height → choose number of clusters
```

## Linkage Methods — How Cluster Distance is Measured
| Linkage | Distance Between Clusters | Use When |
|---------|--------------------------|----------|
| `"ward"` | Minimizes within-cluster variance | Generally best, compact clusters |
| `"complete"` | Maximum distance between any two points | Compact clusters |
| `"average"` | Average of all pairwise distances | Balanced |
| `"single"` | Minimum distance between any two points | Elongated/chain clusters |

## Code Example
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Plot dendrogram to visualize merges
Z = linkage(X_s, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.ylabel("Distance")
plt.show()

# Fit with chosen number of clusters
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_s)
print("Cluster labels:", labels)
```

## Advantages
- No need to specify K upfront
- Dendrogram gives full picture of cluster structure
- Works with any distance metric

## Limitations
- Slow for large datasets (O(n^2) to O(n^3))
- Cannot undo a bad merge (agglomerative)
- Sensitive to outliers and linkage choice

---

# 14. DBSCAN

## What is DBSCAN?
DBSCAN = **Density-Based Spatial Clustering of Applications with Noise**

It finds clusters based on **density** — groups of points that are close together — and labels sparse points as **noise/outliers**.

Key advantage: **automatically detects number of clusters AND outliers**.

## Two Key Parameters
```
eps       = neighborhood radius — how far to look for neighbors
min_samples = minimum points within eps to form a dense region (core point)
```

## Three Types of Points
```
Core point:   Has >= min_samples neighbors within eps radius
              → Starts a cluster

Border point: Has < min_samples neighbors within eps
              but is a neighbor of a core point
              → Joins the cluster, but does not extend it

Noise point:  Not a core point, not close to any core point
              → Labeled as -1 (outlier)
```

## How it Works
```
Step 1: Pick any unvisited point
Step 2: Count how many points are within eps distance
Step 3: If count >= min_samples → CORE POINT → start a cluster
        Add all neighbors to the cluster
        For each neighbor that is also a core point → expand recursively
Step 4: If count < min_samples → mark as NOISE (tentatively)
Step 5: Move to next unvisited point → repeat

Noise points near a later-found cluster may get reassigned as border points.
```

## Code Example
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_s)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print("Clusters found:", n_clusters)
print("Noise points:  ", n_noise)
print("Labels:", labels)
```

## Advantages
- Automatically finds number of clusters
- Detects outliers natively
- Handles arbitrarily shaped clusters (not just spherical)
- No need for K in advance

## Limitations
- eps and min_samples are hard to tune
- Struggles with varying-density clusters
- Slow on very large datasets

---

# 15. HDBSCAN

## What is HDBSCAN?
HDBSCAN = **Hierarchical Density-Based Spatial Clustering of Applications with Noise**

An improved version of DBSCAN that:
- **Automatically handles varying density clusters**
- **Does not require eps parameter** (only min_cluster_size)
- **More robust to parameter choices**

## Key Difference from DBSCAN
```
DBSCAN:   Single global eps → fails when clusters have different densities
HDBSCAN:  Builds a hierarchy of densities → extracts stable clusters automatically
```

## How it Works (Overview)
```
Step 1: Build mutual reachability graph (handles varying densities)
Step 2: Build minimum spanning tree of this graph
Step 3: Construct cluster hierarchy (dendrogram)
Step 4: Extract stable clusters by measuring cluster persistence
Step 5: Assign labels → sparse points become noise (-1)
```

## Code Example
```python
import hdbscan  # pip install hdbscan

model = hdbscan.HDBSCAN(min_cluster_size=10)
labels = model.fit_predict(X_s)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Clusters found:", n_clusters)
print("Noise points:", list(labels).count(-1))
```

## DBSCAN vs HDBSCAN
| | DBSCAN | HDBSCAN |
|---|---|---|
| Parameters | eps + min_samples | min_cluster_size only |
| Varying density | Struggles | Handles well |
| Stability | Sensitive to eps | More robust |
| Speed | Fast | Slightly slower |
| Use when | Uniform density | Mixed density clusters |

---

# 16. Apriori Algorithm

## What is Apriori?
Apriori is an algorithm for **frequent itemset mining** and **association rule learning**.

It discovers items that frequently appear together in transactions.

Examples:
- {Milk, Bread} → often bought together
- {Phone} → {Charger} often bought together
- Used in: recommendation systems, market basket analysis

## Key Concepts
```
Support:    How often an itemset appears
            Support({Milk,Bread}) = transactions containing both / total transactions

Confidence: How often rule is correct
            Confidence(Milk → Bread) = Support({Milk,Bread}) / Support({Milk})

Lift:       How much better than random
            Lift = Confidence(A → B) / Support(B)
            Lift > 1 = positive association
```

## How Apriori Works
```
Step 1: Set minimum support threshold (e.g. 30%)

Step 2: Scan database → find all frequent 1-itemsets
  {Milk}=60%  {Bread}=55%  {Butter}=40%  {Juice}=20%<30% → PRUNED

Step 3: Generate 2-itemsets from frequent 1-itemsets (join)
  {Milk,Bread}  {Milk,Butter}  {Bread,Butter}

Step 4: Scan database → count support for 2-itemsets → prune below threshold

Step 5: Repeat for 3-itemsets → 4-itemsets → ... until no more frequent itemsets

Step 6: Generate association rules from frequent itemsets
```

## Apriori Property (Anti-Monotonicity)
```
"If an itemset is INFREQUENT, ALL its supersets are also infrequent."

{Juice} is infrequent → {Milk,Juice}, {Bread,Juice}, etc. can ALL be pruned immediately.
This is the key pruning trick that makes Apriori efficient.
```

## Code Example
```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# One-hot encode transactions
basket = pd.get_dummies(transactions.explode('items')).groupby(level=0).max()

frequent_items = apriori(basket, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)
print(rules[['antecedents','consequents','support','confidence','lift']])
```

## Advantages
- Easy to understand and implement
- Foundation for association rule mining
- Uses pruning to reduce candidates

## Limitations
- Many database scans (one per itemset level)
- Generates many candidate itemsets
- Slow on large, dense datasets

---

# 17. FP-Growth Algorithm

## What is FP-Growth?
FP-Growth = **Frequent Pattern Growth**

An improved alternative to Apriori that mines frequent itemsets **without candidate generation** using a compressed data structure called an **FP-tree**.

## Key Advantage Over Apriori
```
Apriori: many database scans + many candidate itemsets → slow
FP-Growth: only 2 database scans + no candidate generation → much faster
```

## How it Works
```
Step 1: Scan database once → count support of each item → prune infrequent items

Step 2: Sort frequent items by support (descending) in each transaction

Step 3: Insert each transaction into FP-tree (prefix tree / trie structure)
  → Common prefixes share nodes → tree is compact

Step 4: For each frequent item → extract conditional pattern base
        (all prefix paths in the FP-tree that end with that item)

Step 5: Build conditional FP-tree for each item → mine recursively

Step 6: Combine prefixes → extract all frequent itemsets
```

## FP-Tree Structure
```
Example transactions (after pruning infrequent items):
  T1: Milk, Bread, Butter
  T2: Milk, Bread
  T3: Milk, Butter
  T4: Bread, Butter

FP-tree (root):
  root
  ├── Milk:3
  │   ├── Bread:2
  │   │   └── Butter:1
  │   └── Butter:1
  └── Bread:1
      └── Butter:1

Common prefix "Milk,Bread" shared in one path → memory efficient.
```

## Apriori vs FP-Growth
| | Apriori | FP-Growth |
|---|---|---|
| Database scans | Many | Only 2 |
| Candidate generation | Yes (many) | No |
| Data structure | Transaction list | FP-tree |
| Speed | Slower | Much faster |
| Memory | Lower | Higher (FP-tree) |

## Code Example
```python
from mlxtend.frequent_patterns import fpgrowth, association_rules

frequent_items = fpgrowth(basket, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
print(rules.sort_values('lift', ascending=False).head(10))
```

## Advantages
- Much faster than Apriori for large datasets
- Only 2 database scans
- No candidate generation
- Handles dense datasets well

## Limitations
- FP-tree may not fit in memory for very large datasets
- More complex to implement than Apriori

---

# 18. ECLAT Algorithm

## What is ECLAT?
ECLAT = **Equivalence Class Clustering and bottom-up Lattice Traversal**

An algorithm for **frequent itemset mining** that uses a completely different approach to Apriori and FP-Growth — it uses **vertical database format** and **set intersections**.

## Main Idea — Vertical Data Format
```
Normal (Horizontal) format:
  Transaction → Items
  T1 → {Milk, Bread, Butter}
  T2 → {Bread, Butter}

ECLAT (Vertical) format:
  Item → Transaction IDs (TID set)
  Milk   → {T1, T3, T4, T5}
  Bread  → {T1, T2, T3, T5}
  Butter → {T1, T2, T4, T5}

Why vertical?
  Support calculation becomes: just count the TID set size.
  Multi-item support = just INTERSECT TID sets!
```

## Full Step-by-Step Example

**Transactions:**
```
T1: Milk, Bread, Butter
T2: Bread, Butter
T3: Milk, Bread
T4: Milk, Butter
T5: Milk, Bread, Butter

Minimum support = 3
```

### Step 1 — Convert to Vertical Format
```
Item     TID Set              Support
Milk   → {T1, T3, T4, T5}       4   ✓ frequent
Bread  → {T1, T2, T3, T5}       4   ✓ frequent
Butter → {T1, T2, T4, T5}       4   ✓ frequent
```

### Step 2 — Frequent 1-Itemsets
All three survive (support ≥ 3).

### Step 3 — Generate 2-Itemsets via TID Intersections
```
(Milk, Bread):
  Milk TIDs:   {T1, T3, T4, T5}
  Bread TIDs:  {T1, T2, T3, T5}
  Intersection: {T1, T3, T5}     Support = 3  ✓ frequent

(Milk, Butter):
  Milk TIDs:   {T1, T3, T4, T5}
  Butter TIDs: {T1, T2, T4, T5}
  Intersection: {T1, T4, T5}     Support = 3  ✓ frequent

(Bread, Butter):
  Bread TIDs:  {T1, T2, T3, T5}
  Butter TIDs: {T1, T2, T4, T5}
  Intersection: {T1, T2, T5}     Support = 3  ✓ frequent
```

### Step 4 — Generate 3-Itemsets
```
(Milk, Bread, Butter):
  (Milk,Bread) TIDs:  {T1, T3, T5}
  Butter TIDs:        {T1, T2, T4, T5}
  Intersection:       {T1, T5}       Support = 2  ✗ FAILS (< 3)  → DISCARD
```

### Final Frequent Itemsets
```
1-Itemsets: {Milk}, {Bread}, {Butter}
2-Itemsets: {Milk,Bread}, {Milk,Butter}, {Bread,Butter}
3-Itemsets: NONE (all fail minimum support)
```

## ECLAT Uses Depth-First Search (DFS)
```
Apriori style (BFS):
  All 1-itemsets → All 2-itemsets → All 3-itemsets

ECLAT style (DFS):
  Milk
    → Milk, Bread
         → Milk, Bread, Butter
    → Milk, Butter
  Bread
    → Bread, Butter

DFS reduces memory usage — no need to store all itemsets at once.
```

## Important Intuition
```
ECLAT realizes:
  "Instead of repeatedly scanning transactions for items,
   directly store which transactions contain each item,
   then find frequent itemsets efficiently using recursive set intersections."

Support(A ∪ B) = |TID(A) ∩ TID(B)|   ← just count intersection size
```

## ECLAT vs Apriori vs FP-Growth
| Feature | Apriori | FP-Growth | ECLAT |
|---------|---------|-----------|-------|
| Database format | Horizontal | Horizontal | Vertical |
| Main operation | Candidate generation | FP-tree mining | TID intersection |
| Database scans | Many | 2 | Few |
| Search style | BFS | Recursive | DFS |
| Speed | Slower | Fast | Fast |
| Memory | Low | Medium | Can be high |

## Advantages
- Faster than Apriori (fewer database scans)
- Support calculation is simple intersection
- DFS reduces memory
- Elegant and efficient for dense datasets

## Limitations
- Large TID sets consume memory on sparse datasets
- Intersections less efficient on very sparse data
- More complex implementation than Apriori

---

# Summary — Algorithm Selection Guide

## By Task
| Task | Algorithms to Try |
|------|------------------|
| Regression (predict a number) | LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting |
| Classification (predict a category) | LogisticRegression, DecisionTree, RandomForest, GradientBoosting, SVM, KNN |
| Clustering (find groups, no labels) | K-Means, DBSCAN, HDBSCAN, Hierarchical |
| Association mining (find co-occurrences) | Apriori, FP-Growth, ECLAT |

## Model Evaluation Metrics — Quick Reference
**For Classification:**
```
Accuracy    = correct predictions / total predictions
Precision   = TP / (TP + FP)    ← how many predicted positives are actually positive
Recall      = TP / (TP + FN)    ← how many actual positives did we catch
F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
ROC-AUC     = area under the ROC curve (1.0 = perfect, 0.5 = random)
```

**For Regression:**
```
MAE  = (1/n) * SUM(|yi - y_hat_i|)               ← mean absolute error
MSE  = (1/n) * SUM((yi - y_hat_i)^2)             ← mean squared error
RMSE = sqrt(MSE)                                  ← same units as y
R2   = 1 - SUM((yi-y_hat)^2) / SUM((yi-y_bar)^2) ← 1.0 is perfect
```

## Key Concepts to Learn Next
| Topic | Why Important |
|-------|--------------|
| Bias vs Variance | Understand underfitting vs overfitting |
| Cross Validation | Reliable model evaluation (k-fold) |
| Feature Engineering | Create better features from raw data |
| Dimensionality Reduction | PCA, t-SNE for high-dimensional data |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Deep Learning Basics | Neural networks, activation functions |

## Topics Connection Map
```
Linear Regression  ←→  Gradient Descent  ←→  Logistic Regression
Decision Trees     ←→  Random Forest     ←→  Gradient Boosting
K-Means            ←→  Distance Metrics  ←→  KNN
DBSCAN/HDBSCAN     ←→  Density-based clustering
Apriori/FP-Growth/ECLAT  ←→  Frequent pattern mining

You are building concepts in the right order.
"Why/how" questions are the correct way to learn ML deeply.
```

---
*Notes compiled from supervised, unsupervised, ensemble, and association rule mining concepts.*
*Next steps: Model evaluation metrics → Bias-Variance → Feature Engineering → Deep Learning basics.*
