# 📅 Day 3 — Regression Models
## What is Regression?
Regression is a type of machine learning where the **output is a continuous number**.

Examples:
- Predict house price → ₹45,00,000
- Predict temperature → 32.5°C
- Predict exam score → 87.3

If the output is a category (yes/no, spam/not spam) → that is Classification (Day 4).
If the output is a number → that is Regression.

## Models Covered Today
| Model | Best For | Key Idea |
|-------|----------|----------|
| `LinearRegression` | Simple relationships, baseline | Finds best straight line |
| `Ridge` | Many features, prevent overfitting | Linear + L2 penalty on weights |
| `Lasso` | Feature selection needed | Linear + L1 penalty, kills weak features |
| `DecisionTreeRegressor` | Non-linear patterns | Splits data into boxes |
| `RandomForestRegressor` | Best general purpose | Many trees, averages their answers |

# LinearRegression
## What is it?
`LinearRegression` finds the **best straight line** (or hyperplane in multiple dimensions)
that fits your data — and uses it to predict continuous output values.

It is the simplest and most fundamental regression model.

## Syntax
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## What Input Does it Accept?
```python
# X_train: 2D array — numeric, no NaN
# y_train: 1D array — continuous numeric values (not categories)

X_train = [[1500, 3, 2],   # sqft, bedrooms, bathrooms
           [2000, 4, 3],
           [900,  2, 1]]
y_train = [250000, 400000, 150000]   # house prices
```

## How Does it Work Internally?
### The Model Equation
For multiple features:

$$\hat{y} = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b$$

- $w_1, w_2, \ldots$ = coefficients (weights) for each feature
- $b$ = intercept (bias)
- $\hat{y}$ = predicted output
### How fit() Finds the Best Weights — OLS
Goal: find $w$ and $b$ that minimize the total squared error (MSE):

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

For Linear Regression, sklearn solves this **exactly** using the OLS (Ordinary Least Squares) formula:

$$W = (X^T X)^{-1} X^T y$$

No iteration needed. One matrix calculation gives the optimal weights directly.
### Step-by-step what happens after fit():
```
Input  → X_train (800, 3), y_train (800,)
Step 1 → Compute X^T * X  → shape (3, 3) matrix
Step 2 → Compute inverse of that matrix
Step 3 → Multiply by X^T * y
Step 4 → Result W = best weights

Stored in:
model.coef_      = [120.0, 50000.0, 30000.0]  ← weight per feature
model.intercept_ = 10000.0                     ← bias
```

## What Does it Output After Training?
```python
model.fit(X_train, y_train)

print(model.coef_)         # [120.  50000.  30000.]
print(model.intercept_)    # 10000.0
# Meaning:
# price = 120*sqft + 50000*bedrooms + 30000*bathrooms + 10000
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `fit_intercept` | `True` | Whether to calculate intercept (b). Set False if data is already centered |
| `copy_X` | `True` | Whether to copy X before fitting |
| `n_jobs` | `None` | Number of CPU cores for computation |
| `positive` | `False` | Force all coefficients to be positive |

## When to Use and When Not to Use
**Use when:**
- You believe a linear relationship exists between X and y
- As a baseline model before trying complex models
- Data is not too noisy

**Do not use when:**
- There is a non-linear relationship (use Decision Tree or Random Forest)
- You have many correlated features (use Ridge or Lasso)
- Data has many outliers (outliers heavily affect the line)

## Real-life Example — House Price Prediction
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Sample house data
np.random.seed(42)
n = 200
sqft      = np.random.randint(800, 3000, n)
bedrooms  = np.random.randint(1, 6, n)
price     = 120 * sqft + 50000 * bedrooms + 10000 + np.random.randn(n) * 20000

X = np.column_stack([sqft, bedrooms])
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficient for sqft:    ", model.coef_[0])    # ~120
print("Coefficient for bedrooms:", model.coef_[1])    # ~50000
print("Intercept:               ", model.intercept_)  # ~10000

y_pred = model.predict(X_test)
print("R² Score:", model.score(X_test, y_test))
# Predict a new house: 1500 sqft, 3 bedrooms
new = np.array([[1500, 3]])
print("Predicted price:", model.predict(new)[0])
```

# Ridge Regression
## What is it?
`Ridge` is Linear Regression with an added **penalty on large weights**.

It is used when:
- You have many features
- Features are correlated (multicollinearity)
- Your linear model is overfitting

## Syntax
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
Regular Linear Regression minimizes:

$$\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$

Ridge adds an **L2 penalty** — it also tries to keep weights small:

$$\text{Ridge Loss} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} w_j^2$$

- $\alpha$ = regularization strength (you set this)
- $\sum w_j^2$ = sum of squared weights
- Higher $\alpha$ → model forced to use smaller weights → simpler model
### What does this penalty do physically?
Without penalty: model might assign weight=50000 to one feature, ignoring others.
With L2 penalty: model is forced to spread weights more evenly across all features.
### Effect of alpha:
```
alpha = 0    → same as plain LinearRegression
alpha = 1    → mild regularization
alpha = 10   → medium regularization
alpha = 100  → strong regularization, weights very small
alpha = ∞    → all weights = 0 (useless model)
```
### Ridge closed-form solution:
$$W = (X^T X + \alpha I)^{-1} X^T y$$

Adding $\alpha I$ (identity matrix) stabilizes the matrix inversion — this is especially useful
when features are correlated and $X^T X$ would otherwise be poorly invertible.

## Key Difference from LinearRegression
Ridge **shrinks** all weights toward zero — but never makes them exactly zero.
All features are kept, just with smaller weights.

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `alpha` | `1.0` | Regularization strength. Higher = more penalty |
| `fit_intercept` | `True` | Include bias term |
| `solver` | `"auto"` | Algorithm to use: `"svd"`, `"cholesky"`, `"lsqr"`, etc. |
| `max_iter` | `None` | Max iterations for iterative solvers |

## Real-life Example — Salary Prediction with Many Features
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
# Dataset with 20 features (many may be correlated)
X, y = make_regression(n_samples=300, n_features=20, noise=30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
# Compare Linear vs Ridge
lr = LinearRegression()
lr.fit(X_train_s, y_train)
print("LinearRegression R²:", lr.score(X_test_s, y_test))
print("Max weight (Linear):", np.max(np.abs(lr.coef_)))

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_s, y_train)
print("Ridge R²:", ridge.score(X_test_s, y_test))
print("Max weight (Ridge):", np.max(np.abs(ridge.coef_)))
# Ridge weights will be smaller but score often similar or better
```

# Lasso Regression
## What is it?
`Lasso` is Linear Regression with an **L1 penalty**.

The key difference from Ridge: Lasso can drive some weights **exactly to zero**, effectively removing those features.
This makes Lasso perform **automatic feature selection**.

## Syntax
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
Lasso Loss:

$$\text{Lasso Loss} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |w_j|$$

- Uses **absolute values** of weights (L1 norm) instead of squared values (L2)
- The L1 penalty creates a geometric condition where the optimal solution often lands
  exactly at a corner of the constraint region — where one or more weights = 0
### Why L1 forces zeros but L2 does not?
**L2 penalty** creates a circular constraint region.
The solution typically touches the circle at a point where no weight is exactly zero.

**L1 penalty** creates a diamond-shaped constraint region.
The solution often hits a corner of the diamond — corners have some weights = 0.

```
L2 (Ridge): all weights shrink, none become exactly zero
L1 (Lasso): some weights become exactly zero → feature removed
```

Sklearn uses **coordinate descent** algorithm to minimize Lasso loss (not a simple closed form).

## Key Difference from Ridge
| Feature | Ridge | Lasso |
|---------|-------|-------|
| Penalty | L2 (sum of squared weights) | L1 (sum of absolute weights) |
| Effect | Shrinks all weights | Some weights go to zero |
| Feature selection | No | Yes — removes useless features |
| When many useless features | Not ideal | Great |

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `alpha` | `1.0` | Regularization strength. Higher = more features removed |
| `fit_intercept` | `True` | Include bias term |
| `max_iter` | `1000` | Max iterations for coordinate descent |
| `tol` | `1e-4` | Convergence tolerance |

## Real-life Example — Identifying Important Features
```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
# 15 features — but only 5 actually matter
X, y = make_regression(
    n_samples=300,
    n_features=15,
    n_informative=5,   # only 5 features really affect y
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lasso = Lasso(alpha=0.5)
lasso.fit(X_train_s, y_train)

print("Lasso R²:", lasso.score(X_test_s, y_test))
print("\nWeights per feature:")
for i, w in enumerate(lasso.coef_):
    status = "ACTIVE" if w != 0 else "REMOVED (zero)"
    print(f"  Feature {i:2d}: {w:8.3f}  ← {status}")

print(f"\nFeatures kept: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")
# Should show ~5 active features, rest zeroed out
```

# DecisionTreeRegressor
## What is it?
`DecisionTreeRegressor` builds a **tree of if-else rules** to predict continuous values.

It does not assume any linear relationship. It simply divides the data into smaller and smaller
groups (rectangular regions) and predicts the average value in each region.

## Syntax
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
### Building the Tree — Step by Step
**Step 1:** Start with all training data at the root.

**Step 2:** For every feature and every possible split point, calculate the **MSE reduction** (called impurity reduction):

$$\text{MSE reduction} = \text{MSE}_{\text{parent}} - \left(\frac{n_{\text{left}}}{n} \cdot \text{MSE}_{\text{left}} + \frac{n_{\text{right}}}{n} \cdot \text{MSE}_{\text{right}}\right)$$

**Step 3:** Choose the feature and split value that gives the **maximum MSE reduction**.

**Step 4:** Split the data into left and right subsets.

**Step 5:** Repeat Steps 2–4 recursively on each subset until:
- max_depth is reached, OR
- minimum samples per leaf is reached, OR
- no more improvement possible

**Step 6:** At each leaf node, store the **mean of y values** in that region.
### Example Tree Structure:
```
                  sqft > 1500?
                 /              \
          No (≤1500)          Yes (>1500)
          /                        \
   bedrooms > 2?              bedrooms > 3?
   /          \               /           \
 avg=$180k  avg=$250k    avg=$380k     avg=$520k
```
### Making a Prediction:
Follow the tree rules from root to leaf. Return the average value stored at the leaf.

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `max_depth` | `None` | Max levels in tree. None = grow fully (overfits) |
| `min_samples_split` | `2` | Min samples needed to split a node |
| `min_samples_leaf` | `1` | Min samples needed in a leaf node |
| `max_features` | `None` | Max features to consider at each split |
| `random_state` | `None` | Seed for reproducibility |

## Overfitting Problem in Decision Trees
A fully grown tree (no max_depth) memorizes training data exactly — every single training point
can get its own leaf node.

```
max_depth=None  → Training score: 1.00 (100%) but Test score: 0.60  ← OVERFITTING
max_depth=5     → Training score: 0.88 but Test score: 0.85         ← GOOD
```

Always set `max_depth` or `min_samples_leaf` to control overfitting.

## Real-life Example
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=5, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Compare different depths
for depth in [None, 2, 5, 10]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_sc = dt.score(X_train, y_train)
    test_sc  = dt.score(X_test, y_test)
    print(f"max_depth={str(depth):4s} → Train: {train_sc:.3f}  Test: {test_sc:.3f}")
# Output:
# max_depth=None → Train: 1.000  Test: 0.601  ← overfitting
# max_depth=2    → Train: 0.702  Test: 0.681  ← underfitting
# max_depth=5    → Train: 0.912  Test: 0.823  ← good balance
# max_depth=10   → Train: 0.987  Test: 0.712  ← slight overfit
```

# RandomForestRegressor
## What is it?
`RandomForestRegressor` builds **many Decision Trees** and averages their predictions.

"Wisdom of the crowd" — one person guessing wrong, many people average to the right answer.

It is one of the most reliable and widely used regression models in practice.

## Syntax
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
### Two Key Techniques Make It Work:
#### 1. Bagging (Bootstrap Aggregating)
For each of the `n_estimators` trees:
1. Draw a **random sample of training rows** (with replacement) — called a bootstrap sample
2. Typically 63% of rows appear (37% are left out — called "out-of-bag" samples)
3. Train one Decision Tree on this bootstrap sample

Each tree is trained on **different data** → each tree makes different mistakes.
#### 2. Feature Randomness
At each split inside a tree, instead of considering all features:
- Randomly select `max_features` features (default: n_features / 3 for regression)
- Only split on the best of those randomly selected features

This ensures trees are **uncorrelated** with each other.
### Prediction:
```
Run X_test through all 100 trees.
Each tree gives one prediction.
Final prediction = average of all 100 predictions.

Tree 1 predicts: 250,000
Tree 2 predicts: 265,000
Tree 3 predicts: 242,000
...
Tree 100 predicts: 258,000

Final = mean([250000, 265000, 242000, ..., 258000]) = ~255,000
```

Averaging cancels out individual tree errors → much more stable and accurate.

## Feature Importance
Random Forest automatically tells you which features were most useful:

```python
model.feature_importances_
# array([0.45, 0.30, 0.15, 0.07, 0.03])
# Feature 0 (sqft) is most important — 45% contribution
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_estimators` | `100` | Number of trees. More = better but slower |
| `max_depth` | `None` | Max depth per tree |
| `min_samples_split` | `2` | Min samples to split a node |
| `min_samples_leaf` | `1` | Min samples in leaf |
| `max_features` | `"sqrt"` | Features considered at each split |
| `bootstrap` | `True` | Whether to use bootstrap samples |
| `n_jobs` | `None` | Parallel processing (-1 = use all cores) |
| `random_state` | `None` | Seed for reproducibility |

## Real-life Example — House Price Prediction with Feature Importance
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 500
sqft        = np.random.randint(800, 4000, n)
bedrooms    = np.random.randint(1, 6, n)
bathrooms   = np.random.randint(1, 4, n)
age         = np.random.randint(1, 50, n)
school_rate = np.random.uniform(1, 10, n)

price = (120 * sqft + 50000 * bedrooms + 30000 * bathrooms
         - 1000 * age + 15000 * school_rate + np.random.randn(n) * 20000)

X = np.column_stack([sqft, bedrooms, bathrooms, age, school_rate])
feature_names = ["sqft", "bedrooms", "bathrooms", "age", "school_rate"]
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("R² Score:", model.score(X_test, y_test))

print("\nFeature Importances:")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {name:12s}: {imp:.4f}  {bar}")
```

## LinearRegression vs Ridge vs Lasso vs DecisionTree vs RandomForest
| Model | Linear? | Handles Non-linear? | Feature Selection? | Overfits easily? | Speed |
|-------|---------|--------------------|--------------------|-----------------|-------|
| `LinearRegression` | Yes | No | No | Yes (many features) | Very fast |
| `Ridge` | Yes | No | No | Less than Linear | Fast |
| `Lasso` | Yes | No | Yes (L1 zeros) | Less than Linear | Fast |
| `DecisionTreeRegressor` | No | Yes | No (but gives importance) | Very easily | Fast |
| `RandomForestRegressor` | No | Yes | No (but gives importance) | Rarely | Slower |

## Day 3 — Complete Project: Predicting Student Exam Score
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 400

study_hours     = np.random.uniform(1, 10, n)
attendance_pct  = np.random.uniform(50, 100, n)
prev_score      = np.random.uniform(40, 100, n)
sleep_hours     = np.random.uniform(4, 9, n)
# Real formula — study and prev score matter most
score = (5 * study_hours + 0.3 * attendance_pct + 0.4 * prev_score
         + 0.5 * sleep_hours + np.random.randn(n) * 5)
score = np.clip(score, 0, 100)

X = np.column_stack([study_hours, attendance_pct, prev_score, sleep_hours])
y = score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "LinearRegression":       LinearRegression(),
    "Ridge(alpha=1)":         Ridge(alpha=1.0),
    "Lasso(alpha=0.1)":       Lasso(alpha=0.1),
    "DecisionTree(depth=5)":  DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest(100 trees)":RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

print(f"{'Model':<30} {'R² Score':>10}")
print("-" * 42)
for name, m in models.items():
    m.fit(X_train_s, y_train)
    score_val = m.score(X_test_s, y_test)
    print(f"{name:<30} {score_val:>10.4f}")
```

> Day 3 Complete.
> You now understand how all 5 regression models work — from the simple straight line to an ensemble of 100 trees.
>
> Next: **Day 4 — Classification Models**
> (LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, SVC)