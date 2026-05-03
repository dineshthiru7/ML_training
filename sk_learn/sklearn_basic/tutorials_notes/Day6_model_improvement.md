# 📅 Day 6 — Model Improvement
## Why Model Improvement?
After building a model and measuring its score, two problems commonly appear:

```
Problem 1 — OVERFITTING:
  Training score: 0.99
  Test score:     0.65
  Gap is too large → model memorized training data, fails on new data

Problem 2 — WRONG HYPERPARAMETERS:
  You tried max_depth=5 manually.
  But max_depth=12 might give a much better result.
  Searching manually is inefficient and unreliable.
```

Day 6 tools solve both problems:

| Tool | Solves |
|------|--------|
| `cross_val_score()` | Get a more reliable, honest score |
| `KFold` | Control how cross validation splits data |
| `StratifiedKFold` | Better cross validation for imbalanced classification |
| `GridSearchCV` | Automatically find the best hyperparameters |
| `RandomizedSearchCV` | Faster hyperparameter search over large spaces |

# Cross Validation
## What is Cross Validation?
Cross validation is a technique to get a **more reliable estimate of model performance**
by training and testing on multiple different splits of your data.
### The Problem with a Single Train/Test Split
```
Full dataset: 1000 rows

Split A: train on rows 1–800, test on 801–1000  → score = 0.82
Split B: train on rows 1–600 + 801–1000, test on 601–800 → score = 0.79
Split C: train on rows 201–1000, test on 1–200 → score = 0.86

Which score is "the truth"?
A single split can be lucky or unlucky depending on what ended up in test.
```
### The Cross Validation Solution
Use **all the data for both training and testing** — just at different times.

The most common type: **K-Fold Cross Validation**

```
K=5 (5-Fold Cross Validation):

Fold 1: [TEST] [train] [train] [train] [train]  → score 0.82
Fold 2: [train] [TEST] [train] [train] [train]  → score 0.79
Fold 3: [train] [train] [TEST] [train] [train]  → score 0.84
Fold 4: [train] [train] [train] [TEST] [train]  → score 0.81
Fold 5: [train] [train] [train] [train] [TEST]  → score 0.83

Final CV Score = mean([0.82, 0.79, 0.84, 0.81, 0.83]) = 0.818 ± 0.018
```

This is much more reliable than a single split.

# cross_val_score()
## What is it?
`cross_val_score()` runs K-Fold cross validation for you automatically and returns the score
from each fold.

## Syntax
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
```

## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `model` | sklearn estimator | Any fitted or unfitted sklearn model |
| `X` | 2D array | Your full input features (not split yet) |
| `y` | 1D array | Your full target values |
| `cv` | int or splitter | Number of folds (default 5), or a KFold object |
| `scoring` | string | Metric to compute per fold |
| `n_jobs` | int | `-1` uses all CPU cores (parallel) |

## What Happens Inside — Step by Step
1. `cross_val_score` receives the **full dataset** X and y
2. Internally creates a `KFold` splitter with the specified `cv` splits
3. For each fold:
   - Sets aside 1/cv fraction as test set
   - Trains a **fresh copy** of the model on the remaining folds
   - Evaluates on the test fold using the `scoring` metric
4. Returns an array of scores — one per fold

**Key point:** The model is retrained from scratch in each fold.
The original model object you passed is used only as a template — it is **not modified**.

## What Does it Output?
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("Scores per fold:", scores)
# [0.9667 1.0000 0.9333 0.9667 1.0000]

print("Mean:", scores.mean())     # 0.9733
print("Std:", scores.std())       # 0.0249

print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
# CV Score: 0.9733 ± 0.0249
```

The ± tells you **consistency**: small std = model is stable, large std = model varies a lot.

## Scoring Options
| Task | Scoring string |
|------|---------------|
| Classification | `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"roc_auc"` |
| Regression | `"r2"`, `"neg_mean_absolute_error"`, `"neg_mean_squared_error"` |

Note: Regression metrics use `neg_` prefix (sklearn returns negative values internally for minimization compatibility).

```python
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
mae_scores = -scores   # negate to get positive MAE values
print("MAE per fold:", mae_scores)
print("Mean MAE:", mae_scores.mean())
```

## Real-life Example
```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
# Use pipeline so scaling is done correctly inside each fold
lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
rf_model    = RandomForestClassifier(n_estimators=100, random_state=42)

for name, m in [("Logistic Regression", lr_pipeline), ("Random Forest", rf_model)]:
    scores = cross_val_score(m, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

# KFold
## What is it?
`KFold` is a **splitter object** that defines exactly how to divide data into K folds.

You use it when you want manual control over how the cross validation splits work.

## Syntax
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_splits` | `5` | Number of folds (K) |
| `shuffle` | `False` | Shuffle rows before splitting |
| `random_state` | `None` | Seed for shuffle. Only used if shuffle=True |

## What Happens Inside — Step by Step
KFold works like this for n_splits=5 and 10 rows:

```
Row indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Fold 1: test=[0,1]  train=[2,3,4,5,6,7,8,9]
Fold 2: test=[2,3]  train=[0,1,4,5,6,7,8,9]
Fold 3: test=[4,5]  train=[0,1,2,3,6,7,8,9]
Fold 4: test=[6,7]  train=[0,1,2,3,4,5,8,9]
Fold 5: test=[8,9]  train=[0,1,2,3,4,5,6,7]
```

With `shuffle=True`: rows are randomly shuffled first before dividing.

## What Does it Output?
`KFold` is an **iterator** that yields `(train_indices, test_indices)` tuples:

```python
from sklearn.model_selection import KFold
import numpy as np

X = np.arange(10).reshape(5, 2)  # 5 rows
y = np.array([0, 1, 0, 1, 0])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}:")
    print(f"  Train indices: {train_idx}")
    print(f"  Test  indices: {test_idx}")
```

## Using KFold Manually
```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)
    scores.append(score)
    print(f"Fold {fold+1}: R² = {score:.4f}")

print(f"\nMean R²: {np.mean(scores):.4f}")
print(f"Std R²:  {np.std(scores):.4f}")
```

## When to Use KFold Manually?
Use manual KFold (instead of `cross_val_score`) when you need to:
- Apply preprocessing **inside** each fold (to avoid data leakage)
- Save predictions from each fold (for stacking)
- Log detailed info per fold
- Use custom logic that `cross_val_score` does not support

# StratifiedKFold
## What is it?
`StratifiedKFold` is like `KFold`, but it **preserves the class proportions** in each fold.

For imbalanced datasets, regular KFold might put all rare-class samples in training
and none in testing. Stratification prevents this.

## Syntax
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## How it Works Internally — Step by Step
**Problem with regular KFold on imbalanced data:**

```
Dataset: 90 "no fraud", 10 "fraud" (10% fraud)

KFold Fold 1 (without stratification):
  test set might get: 19 no-fraud, 1 fraud  ← 5% fraud
  or worse: 20 no-fraud, 0 fraud ← NO fraud in test!
```

**StratifiedKFold solution:**

1. Separates samples by class: [all class 0 rows] and [all class 1 rows]
2. Applies KFold split independently on each class
3. Combines the splits to ensure each fold has the same class ratio

```
With StratifiedKFold (5-fold):
Each fold: 18 no-fraud, 2 fraud → always 10% fraud in every fold
```

## What Does it Output?
```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.zeros((100, 3))
y = np.array([0]*90 + [1]*10)   # 90% class 0, 10% class 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):  # note: y is passed here
    y_test_fold = y[test_idx]
    fraud_pct = y_test_fold.mean() * 100
    print(f"Fold {fold+1}: test size={len(test_idx)}, fraud%={fraud_pct:.0f}%")
# All folds: fraud% ≈ 10%  ← balanced!
```

## StratifiedKFold vs KFold
| | KFold | StratifiedKFold |
|--|-------|-----------------|
| Preserves class ratio | No | Yes |
| Works with regression | Yes | No (y must be discrete) |
| Works with imbalanced data | Poorly | Well |
| Default in cross_val_score | For regression | For classification |
**Note:** `cross_val_score()` automatically uses StratifiedKFold for classifiers.
You only need to manually use StratifiedKFold when doing a custom loop.

## Real-life Example
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Imbalanced dataset: 90% class 0, 10% class 1
X, y = make_classification(
    n_samples=500,
    n_features=10,
    weights=[0.9, 0.1],   # 90% class 0
    random_state=42
)

print(f"Class distribution: {np.bincount(y)}")   # [450, 50]

model = RandomForestClassifier(n_estimators=100, random_state=42)
# Stratified CV — ensures each fold has 10% class 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="f1")

print(f"Stratified CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

# GridSearchCV
## What is it?
`GridSearchCV` **automatically searches all combinations** of hyperparameters you specify
and finds the best combination using cross validation.

It is the systematic, exhaustive way to tune a model.

## Syntax
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
```

## How Does it Work Internally — Step by Step
### Step 1 — Build all combinations
```
n_estimators: [50, 100, 200]
max_depth:    [5, 10, 20]

All combinations (Cartesian product):
(50,5)  (50,10)  (50,20)
(100,5) (100,10) (100,20)
(200,5) (200,10) (200,20)

Total: 3 × 3 = 9 combinations
```
### Step 2 — For each combination, run K-fold CV
With cv=5:
- 9 combinations × 5 folds = **45 model fits** total

For each combination:
1. Train model with those hyperparameters on 4 folds
2. Evaluate on the 5th fold
3. Repeat 5 times, average the scores
### Step 3 — Select the best combination
The combination with the highest mean CV score wins.
### Step 4 — Refit on full training data
After finding best params, sklearn refits one final model on the **entire X_train** using those params.
This final model is stored in `grid_search.best_estimator_`.

## What Does it Output?
```python
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
# {'max_depth': 10, 'n_estimators': 200}

print("Best CV Score:", grid_search.best_score_)
# 0.9267

print("Test Score:", grid_search.score(X_test, y_test))
# Score on actual test set using best model
# Best model ready to use
best_model = grid_search.best_estimator_
# Full results for all combinations
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(results[["param_n_estimators", "param_max_depth",
               "mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False))
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `estimator` | required | The sklearn model to tune |
| `param_grid` | required | Dictionary of parameter names and value lists |
| `cv` | `5` | Number of cross validation folds |
| `scoring` | model default | Metric to optimize |
| `n_jobs` | `1` | `-1` runs all combinations in parallel |
| `verbose` | `0` | `1` prints progress, `2` prints more detail |
| `refit` | `True` | Refit best model on full training data after search |
| `return_train_score` | `False` | Include training scores in results |

## Real-life Example — Tuning Random Forest for Cancer Detection
```python
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5],
}
# Total: 3 × 3 × 2 = 18 combinations × 5 folds = 90 model fits

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:  ", grid_search.best_score_)
print("Test Score:     ", grid_search.score(X_test, y_test))
# Compare with default model
default_model = RandomForestClassifier(n_estimators=100, random_state=42)
default_model.fit(X_train, y_train)
print("Default Test Score:", default_model.score(X_test, y_test))
```

# RandomizedSearchCV
## What is it?
`RandomizedSearchCV` searches hyperparameters by **randomly sampling** from the parameter space,
instead of trying every single combination.

It is faster than GridSearchCV when the search space is large.

## Syntax
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth":    [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5]
}

rand_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,        # try 20 random combinations (not all 180)
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rand_search.fit(X_train, y_train)
```

## How Does it Work Internally — Step by Step
### Step 1 — Define the search space
You provide lists of values OR statistical distributions (from scipy.stats):

```python
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators":      randint(50, 500),         # random integer between 50 and 500
    "max_depth":         randint(3, 30),            # random integer between 3 and 30
    "max_features":      uniform(0.1, 0.9),         # random float between 0.1 and 1.0
    "min_samples_leaf":  randint(1, 20),
}
```
### Step 2 — Randomly sample `n_iter` combinations
With n_iter=20: randomly picks 20 combinations from all possible options.
Each combination is drawn independently — no systematic grid.

```
Combination 1: n_estimators=234, max_depth=12, max_features=0.45, min_samples_leaf=7
Combination 2: n_estimators=89,  max_depth=5,  max_features=0.71, min_samples_leaf=3
...
Combination 20: n_estimators=445, max_depth=18, max_features=0.29, min_samples_leaf=15
```
### Step 3 — Evaluate each combination with K-fold CV
20 combinations × 5 folds = **100 model fits** total.
Compare: GridSearchCV with 180 combinations × 5 folds = **900 model fits**.
### Step 4 — Return best combination and refit
Same as GridSearchCV — returns best_params_, best_score_, best_estimator_.

## What Does it Output?
```python
rand_search.fit(X_train, y_train)

print("Best Parameters:", rand_search.best_params_)
print("Best CV Score:  ", rand_search.best_score_)
print("Test Score:     ", rand_search.score(X_test, y_test))

best_model = rand_search.best_estimator_
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `estimator` | required | The sklearn model to tune |
| `param_distributions` | required | Dict of parameter names and value lists or distributions |
| `n_iter` | `10` | How many random combinations to try |
| `cv` | `5` | Number of cross validation folds |
| `scoring` | model default | Metric to optimize |
| `n_jobs` | `1` | Parallel jobs |
| `random_state` | `None` | Seed for reproducibility |
| `verbose` | `0` | Progress logging |

## GridSearchCV vs RandomizedSearchCV
| | GridSearchCV | RandomizedSearchCV |
|--|-------------|-------------------|
| Tries | ALL combinations | n_iter random combinations |
| Search space | Must be discrete lists | Lists OR continuous distributions |
| Time | Slow for large grids | Much faster |
| Misses best? | Never (exhaustive) | Might miss optimal (but usually finds good) |
| Best for | Small grids (< 50 combos) | Large grids (hundreds+) |

**Rule of thumb:**
- Less than 50 combinations → use GridSearchCV
- More than 50 combinations → use RandomizedSearchCV

## Real-life Example — Large Hyperparameter Space
```python
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Large search space — GridSearchCV would take too long
param_dist = {
    "n_estimators":       randint(50, 500),
    "max_depth":          [None] + list(range(3, 30)),
    "min_samples_split":  randint(2, 20),
    "min_samples_leaf":   randint(1, 10),
    "max_features":       ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap":          [True, False],
}

rand_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,        # try 50 random combinations
    cv=5,
    scoring="f1",     # optimize for F1 (good for balanced metric)
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rand_search.fit(X_train, y_train)

print("\n" + "="*45)
print("Best Parameters found:")
for k, v in rand_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest CV F1 Score: {rand_search.best_score_:.4f}")
print(f"Test F1 Score:    {rand_search.score(X_test, y_test):.4f}")
# Use best model for full evaluation
best = rand_search.best_estimator_
y_pred = best.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Day 6 — Summary Table
| Tool | What it does | When to use |
|------|-------------|-------------|
| `cross_val_score()` | Get reliable CV score | Always — instead of single train/test |
| `KFold` | Manual control over folds | Custom training loops |
| `StratifiedKFold` | Balanced folds for imbalanced data | Classification with class imbalance |
| `GridSearchCV` | Exhaustive hyperparameter search | Small search space |
| `RandomizedSearchCV` | Random hyperparameter search | Large search space |

## Complete Day 6 Project — Full Model Tuning Pipeline
```python
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# ── 1. Load data ────────────────────────────────────────────
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ── 2. Baseline cross validation score ──────────────────────
baseline = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"Baseline CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# ── 3. Grid Search on small grid first ──────────────────────
small_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [None, 5, 10],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    small_grid, cv=skf, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"\nGridSearch Best Params: {grid_search.best_params_}")
print(f"GridSearch Best CV:     {grid_search.best_score_:.4f}")
# ── 4. Randomized Search on large space ─────────────────────
large_space = {
    "n_estimators":      randint(50, 500),
    "max_depth":         [None] + list(range(3, 25)),
    "min_samples_split": randint(2, 15),
    "min_samples_leaf":  randint(1, 8),
}
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    large_space, n_iter=40, cv=skf,
    scoring="accuracy", n_jobs=-1, random_state=42
)
rand_search.fit(X_train, y_train)
print(f"\nRandomSearch Best Params: {rand_search.best_params_}")
print(f"RandomSearch Best CV:     {rand_search.best_score_:.4f}")
# ── 5. Final Evaluation ──────────────────────────────────────
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n" + "="*45)
print("FINAL TEST EVALUATION (Best Model)")
print("="*45)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"Test Accuracy: {best_model.score(X_test, y_test):.4f}")
```

> Day 6 Complete.
> You now know how to get reliable scores with cross validation, and how to automatically
> find the best hyperparameters using GridSearchCV and RandomizedSearchCV.
>
> You have completed all 6 days of the sklearn learning plan.
>
> Summary of all 6 days:
> - Day 1: sklearn workflow, fit, predict, score, train_test_split, datasets
> - Day 2: SimpleImputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
> - Day 3: LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor
> - Day 4: LogisticRegression, KNN, DecisionTreeClassifier, RandomForestClassifier, SVC
> - Day 5: MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report
> - Day 6: cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV