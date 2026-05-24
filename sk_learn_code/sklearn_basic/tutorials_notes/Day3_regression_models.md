# Day 3 — Regression Models
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

## Simple vs Multiple Linear Regression

### Simple Linear Regression (1 feature)
Only ONE input feature (x) → fits a straight line in 2D:
```
ŷ = w × x + b

Example: predict price from sqft only
  w = slope    → how much price changes per 1 sqft increase
  b = intercept → price when sqft = 0 (base value)

Data:
  sqft   price
  1000   180,000
  1500   250,000
  2000   340,000

Line fitted: price = 120 × sqft + 60,000

Predict sqft=1800: ŷ = 120 × 1800 + 60,000 = 276,000
```

**How fit() finds w and b for Simple Linear Regression:**
```
slope formula (w):
  w = Σ( (xᵢ - x̄) × (yᵢ - ȳ) ) / Σ( (xᵢ - x̄)² )

intercept formula (b):
  b = ȳ - w × x̄

Step by Step with actual numbers:
  x  = [1000, 1500, 2000]      y  = [180000, 250000, 340000]
  x̄  = (1000+1500+2000)/3 = 1500    ȳ = (180000+250000+340000)/3 = 256667

  Numerator of w:
    (1000-1500)×(180000-256667) = (-500)×(-76667) = +38,333,500
    (1500-1500)×(250000-256667) = (0)×(-6667)     = 0
    (2000-1500)×(340000-256667) = (500)×(83333)   = +41,666,500
    Total numerator = 38,333,500 + 0 + 41,666,500 = 80,000,000

  Denominator of w:
    (1000-1500)² = 250,000
    (1500-1500)² = 0
    (2000-1500)² = 250,000
    Total denominator = 500,000

  w = 80,000,000 / 500,000 = 160.0   ← slope
  b = 256,667 - 160 × 1500 = 256,667 - 240,000 = 16,667  ← intercept

  Final model: price = 160 × sqft + 16,667
```

### Multiple Linear Regression (2+ features)
More than one input feature → fits a hyperplane in N-dimensional space:
```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Example: predict price from sqft AND bedrooms AND bathrooms
  ŷ = 120×sqft + 50000×bedrooms + 30000×bathrooms + 10000

Each coefficient = effect of that feature holding all others constant.
  w₁=120 means: for every 1 sqft increase (bedrooms/baths fixed), price increases by 120
```

**How fit() finds all weights for Multiple Linear Regression (OLS formula):**
```
Cannot use the simple slope formula — must solve for all weights simultaneously.
sklearn uses the Ordinary Least Squares (OLS) matrix formula:

  W = (XᵀX)⁻¹ × Xᵀy

This finds the EXACT weights that minimise total squared error in ONE calculation.
No iterations needed — the matrix formula gives the optimal answer directly.

Difference in what fit() does:
  Simple  (1 feature):  solves 2 equations  → finds w, b
  Multiple (n features): solves n+1 equations → finds w₁,w₂,...,wₙ, b  (simultaneously)
```

**Key distinction:**
| | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| Features | 1 | 2 or more |
| Equation | `ŷ = w×x + b` | `ŷ = w₁x₁ + w₂x₂ + ... + b` |
  | Visual shape | Straight line in 2D | Hyperplane in N+1 dimensions |
| `model.coef_` | Array of 1 value | Array of N values |
| fit() method | Same `model.fit(X, y)` — sklearn handles both automatically |

**sklearn uses the SAME API for both — it detects automatically:**
```python
# Simple — 1 feature (X must still be 2D)
X = sqft.reshape(-1, 1)   # shape (n, 1)
model.fit(X, y)
print(model.coef_)        # [160.0]       ← 1 weight

# Multiple — 3 features
X = np.column_stack([sqft, bedrooms, bathrooms])   # shape (n, 3)
model.fit(X, y)
print(model.coef_)        # [120.0, 50000.0, 30000.0]  ← 3 weights
```

## How Does it Work Internally?
**The Model Equation**
`ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

- `w₁, w₂, ...` = coefficients (weights) — how much each feature contributes to the prediction
- `b` = intercept (bias) — baseline value when all features are zero
- `ŷ` = predicted output value

**What fit(X_train, y_train) does — Step by Step**

Say you call: `model.fit(X_train, y_train)` with this data:
```
X_train (3 rows, 2 features):       y_train:
  sqft   bedrooms
  1500     3        →   250,000
  2000     4        →   400,000
   900     2        →   150,000
```

**Step 1 — Prepend bias column**
sklearn internally adds a column of 1s to X_train so the intercept b is handled as a weight:
```
X_aug (augmented):
  [1, 1500, 3]
  [1, 2000, 4]
  [1,  900, 2]
```
Now the equation is: `ŷ = b×1 + w₁×sqft + w₂×bedrooms`

**Step 2 — Compute XᵀX (X-transpose multiplied by X)**
```
XᵀX = X_aug.T  @  X_aug   →   shape (3, 3)

Captures:
  [ sum(1),         sum(sqft),       sum(bed)      ]
  [ sum(sqft),      sum(sqft²),      sum(sqft×bed) ]
  [ sum(bed),       sum(sqft×bed),   sum(bed²)     ]

Numeric result for our 3-row example:
  [    3,   4400,    9   ]
  [ 4400, 7225000, 13400 ]
  [    9,  13400,   29   ]
```
This matrix tells the model how features relate to each other.

**Step 3 — Compute Xᵀy (X-transpose multiplied by y)**
```
Xᵀy = X_aug.T  @  y_train   →   shape (3,)

= [ sum(y),            ]     [ 800000        ]
  [ sum(sqft × y),     ]  =  [ 1,350,000,000 ]
  [ sum(bedrooms × y), ]     [ 2,750,000     ]
```
This captures how each feature correlates with the target.

**Step 4 — Solve for W using OLS formula**
`W = (XᵀX)⁻¹ × Xᵀy`

- Invert the XᵀX matrix (like dividing for scalars)
- Multiply by Xᵀy
- Result: the exact weights that minimize total squared error

No iteration needed — one matrix calculation gives the globally optimal weights.
```
W = [ b=10000.0, w₁=120.0, w₂=50000.0 ]
```

**Step 5 — Store results**
```
model.coef_      = [120.0, 50000.0]   ← weight for sqft, weight for bedrooms
model.intercept_ = 10000.0            ← bias
```

**What sklearn actually uses (implementation detail)**
sklearn does NOT compute `(XᵀX)⁻¹` directly — that can be numerically unstable.
Instead it uses **LAPACK SVD-based least squares** (`gelsd` routine) which gives the same
mathematical result but is much more numerically robust.

**What predict(X_test) does — Step by Step**
```
X_test[0] = [1600, 3]   ← new house: 1600 sqft, 3 bedrooms

Step 1: ŷ = w₁×x₁ + w₂×x₂ + b
Step 2: ŷ = 120.0×1600 + 50000.0×3 + 10000
Step 3: ŷ = 192,000 + 150,000 + 10,000
Step 4: ŷ = 352,000   ← predicted price
```
Just a dot product (multiply each feature value by its weight, sum them all, add bias).

**Why MSE as the loss — not absolute error?**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Squared error is smooth and differentiable everywhere → clean analytical solution exists.
Absolute error |yᵢ - ŷᵢ| has a sharp corner at 0 → no closed-form solution.
```

**What score(X_test, y_test) returns**
`model.score()` returns R² (coefficient of determination):
`R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²`
- R²=1.0 → perfect predictions
- R²=0.0 → model is as good as just predicting the mean
- R²<0  → model is worse than predicting the mean

## What Does it Output After Training?
```python
model.fit(X_train, y_train)

print(model.coef_)         # [120.  50000.]
print(model.intercept_)    # 10000.0
# Meaning: price = 120×sqft + 50000×bedrooms + 10000
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
**The Problem Ridge Solves**

Plain LinearRegression minimizes only prediction error:
`MSE = (1/n) × Σ(yᵢ - ŷᵢ)²`

When features are correlated or there are many features, the OLS weights become very large and
unstable (tiny change in data → huge change in weights). This is called multicollinearity.

**Ridge adds an L2 penalty** — it penalises large weights to keep them small:
`Ridge Loss = (1/n) Σ(yᵢ - ŷᵢ)² + α × Σwⱼ²`

- `α` (alpha) = regularization strength — YOU set this hyperparameter
- `Σwⱼ²` = sum of all squared weights
- The model now has two goals: (1) fit the data, (2) keep weights small
- These two goals are in tension — alpha controls the balance

**What fit(X_train, y_train) does — Step by Step**

Say you have 20 features (sqft, bedrooms, age, school_rating, ...) and alpha=10.

**Step 1 — Same as LinearRegression**: build augmented X, compute XᵀX and Xᵀy

**Step 2 — Add αI to XᵀX before inverting**
```
LinearRegression:   W = (XᵀX)⁻¹ Xᵀy
Ridge:              W = (XᵀX + αI)⁻¹ Xᵀy

αI = alpha × Identity matrix:
  [ α  0  0  0  ... ]
  [ 0  α  0  0  ... ]
  [ 0  0  α  0  ... ]
  [ 0  0  0  α  ... ]
```

Adding αI to the diagonal of XᵀX does two things:
1. Makes the matrix always invertible (even when features are correlated)
2. Pulls all weights toward zero by penalising their magnitude

**Step 3 — Solve for W** — same formula but with the modified matrix
```
alpha=0   → W = same as LinearRegression
alpha=1   → W slightly shrunk
alpha=10  → W moderately shrunk
alpha=100 → W heavily shrunk, all weights very small
```

**Concrete weight comparison (with 3 features after scaling):**
```
Feature           LinearReg weight    Ridge(α=10) weight
-----------------------------------------------------------
sqft              120.5               98.3    ← shrunk
bedrooms          49800.0             41200.0 ← shrunk
bathrooms         29950.0             25100.0 ← shrunk
```
All weights got smaller. No weight became exactly zero — all features still contribute.

**What fit() stores:**
```
model.coef_      = [98.3, 41200.0, 25100.0]   ← shrunk weights
model.intercept_ = 10000.0
```

**What predict(X_test) does**
Identical to LinearRegression:
```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```
Just uses the (smaller) Ridge weights instead of OLS weights.

**Effect of alpha — full picture:**
```
alpha = 0      → identical to LinearRegression (no penalty)
alpha = 1      → mild regularization, small reduction in weights
alpha = 10     → medium regularization, weights noticeably smaller
alpha = 100    → strong regularization, weights very small, simple model
alpha → ∞      → all weights forced to ~0 (predicts only the mean of y)
```

**Why Ridge never zeroes out weights**
The L2 penalty `Σwⱼ²` creates a sphere-shaped constraint region in weight space.
The optimal solution lands on the surface of this sphere — which almost never has any weight exactly at zero.
(Contrast with Lasso's diamond shape where corners force zeros — see Lasso section.)

## Key Difference from LinearRegression
Ridge **shrinks** all weights toward zero — but never makes them exactly zero.
All features are kept, just with smaller, more stable weights.

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
**The Lasso Loss Function**
`Lasso Loss = (1/n) Σ(yᵢ - ŷᵢ)² + α × Σ|wⱼ|`

- Uses **absolute values** of weights (L1 norm) instead of squared values (L2)
- Same two competing goals as Ridge: fit data + keep weights small
- But the absolute value penalty has a fundamentally different geometric effect

**Why L1 forces some weights to exactly zero (but L2 does not)**

Think of the weights as a point in 2D space (w₁, w₂):
```
Ridge constraint region (L2):          Lasso constraint region (L1):
         ●                                      ●
        /|\                                    /|\
       / | \                                  / | \
  ────/──●──\────                        ────●────●────
       \   /                                  \   /
        \ /                                    \ /
         ●                                      ●

Shape: CIRCLE                           Shape: DIAMOND (rotated square)
```
The optimal solution is where the loss ellipse touches the constraint region.
- On a CIRCLE: touches at a smooth curved edge → almost never at a corner → no zeros
- On a DIAMOND: very likely to touch at a sharp CORNER → corners have w₁=0 or w₂=0 → zeros!

This is why Lasso zeroes out useless features and Ridge does not.

**Algorithm: Coordinate Descent (not closed-form like Ridge)**

Since |w| is not differentiable at 0, there is no clean matrix formula.
sklearn uses **coordinate descent** — optimise one weight at a time, cycling through all weights:

```
Iteration 1:
  Fix w₂, w₃, ..., wₙ  →  find the best w₁  (1D optimisation)
  Fix w₁, w₃, ..., wₙ  →  find the best w₂
  Fix w₁, w₂, w₄, ..., wₙ → find the best w₃
  ...
  Fix w₁, ..., w_{n-1}  →  find the best wₙ

Iteration 2: repeat — each pass refines all weights further

Keep iterating until weights stop changing (convergence)
```

**The Soft-Threshold Update Rule (what happens for each weight)**

For each weight wⱼ, the coordinate descent update applies a "soft threshold":
```
Let ρⱼ = correlation of feature j with the current residual

If ρⱼ > α/2  →  wⱼ = ρⱼ - α/2   (positive weight, reduced by penalty)
If ρⱼ < -α/2 →  wⱼ = ρⱼ + α/2   (negative weight, reduced by penalty)
If -α/2 ≤ ρⱼ ≤ α/2 →  wⱼ = 0    ← ZEROED OUT (feature not strong enough)
```

**Practical example — 15 features, only 5 matter:**
```
After fit(X_train, y_train) with alpha=0.5:

Feature 0:  w =  34.2   ← KEPT   (strong correlation with y)
Feature 1:  w =   0.0   ← ZEROED (weak, useless feature)
Feature 2:  w = -18.7   ← KEPT
Feature 3:  w =   0.0   ← ZEROED
Feature 4:  w =  52.1   ← KEPT
...
Feature 14: w =   0.0   ← ZEROED

model.coef_ = [34.2, 0.0, -18.7, 0.0, 52.1, ..., 0.0]
```

**What fit() stores:**
```
model.coef_      = array — weights (many will be exactly 0.0)
model.intercept_ = float — bias
model.n_iter_    = int   — how many iterations until convergence
```

**What predict(X_test) does**
Same dot product as LinearRegression — but most weights are zero so those features are ignored:
```
ŷ = w₀×x₀ + 0×x₁ + w₂×x₂ + 0×x₃ + w₄×x₄ + ...
  = w₀×x₀          + w₂×x₂           + w₄×x₄ + ...
```

**Effect of alpha:**
```
alpha = 0.01 → few features zeroed, model close to LinearRegression
alpha = 0.1  → moderate feature elimination
alpha = 1.0  → many features zeroed (default)
alpha = 10.0 → aggressive elimination, very few features kept
```

## Key Difference from Ridge
| Feature | Ridge | Lasso |
|---------|-------|-------|
| Penalty | L2 (sum of squared weights) | L1 (sum of absolute weights) |
| Effect | Shrinks all weights | Some weights go to exactly zero |
| Feature selection | No | Yes — removes useless features |
| Algorithm | Closed-form (one calculation) | Coordinate descent (iterative) |
| When many useless features | Not ideal | Great |

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `alpha` | `1.0` | Regularization strength. Higher = more features removed |
| `fit_intercept` | `True` | Include bias term |
| `max_iter` | `1000` | Max iterations for coordinate descent |
| `tol` | `1e-4` | Convergence tolerance — stop when weight changes < tol |

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
print(f"Iterations to converge: {lasso.n_iter_}")
print("\nWeights per feature:")
for i, w in enumerate(lasso.coef_):
    status = "ACTIVE" if w != 0 else "REMOVED (zero)"
    print(f"  Feature {i:2d}: {w:8.3f}  <- {status}")

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
**Core Idea**
Instead of fitting a line, the tree finds rectangular boxes in the feature space and assigns each
box a single predicted value (mean of training points in that box).

**What fit(X_train, y_train) does — Step by Step**

Say you have 6 houses:
```
sqft   bedrooms   price
1000      2       180,000
1200      2       200,000
1500      3       260,000
1800      3       290,000
2200      4       390,000
2500      4       420,000
```

**Step 1 — Start at the root node with all data**
```
Root node: 6 samples
MSE of root = variance(prices) = large number
```

**Step 2 — Try every possible split on every feature**

For feature "sqft", try every value between consecutive sorted sqft values:
```
Split: sqft <= 1100?    → left: [180k]           right: [200k, 260k, 290k, 390k, 420k]
Split: sqft <= 1350?    → left: [180k, 200k]     right: [260k, 290k, 390k, 420k]
Split: sqft <= 1650?    → left: [180k, 200k, 260k] right: [290k, 390k, 420k]
...

For feature "bedrooms", try: bedrooms <= 2, bedrooms <= 3
```

**Step 3 — Calculate MSE Reduction for each split**

For split `sqft <= 1650`:
```
Left group  [180k, 200k, 260k]: mean=213k, MSE_left  = ((180-213)²+(200-213)²+(260-213)²)/3 = 1,089
Right group [290k, 390k, 420k]: mean=367k, MSE_right = ((290-367)²+(390-367)²+(420-367)²)/3 = 2,889

Weighted MSE = (3/6)×1089 + (3/6)×2889 = 1989

Root MSE = variance of all prices = large

MSE Reduction = Root_MSE - Weighted_MSE
```

**Step 4 — Choose the split with the MAXIMUM MSE Reduction**
The split that creates the most homogeneous (similar prices) children is chosen.

**Step 5 — Recurse on each child**
Apply the same split-finding process to the left child and right child independently.
Keep splitting until `max_depth` is reached or too few samples remain.

**Step 6 — At each leaf node, store the MEAN of y values**
```
Leaf 1: sqft<=1500 AND bedrooms<=2 → mean([180k, 200k]) = 190,000
Leaf 2: sqft<=1500 AND bedrooms>2  → mean([260k])       = 260,000
Leaf 3: sqft>1500  AND bedrooms<=3 → mean([290k])       = 290,000
Leaf 4: sqft>1500  AND bedrooms>3  → mean([390k, 420k]) = 405,000
```

**What the final tree looks like:**
```
              sqft > 1500?
             /             \
         No                 Yes
    bedrooms > 2?       bedrooms > 3?
    /         \           /         \
  190k       260k       290k        405k
```

**What predict(X_test) does — Step by Step**
```
New house: sqft=1600, bedrooms=3

Step 1: sqft > 1500?  →  YES  → go right
Step 2: bedrooms > 3? →  NO   → go left
Step 3: Reached leaf → predict 290,000
```
Just traverse the tree using if-else rules. No math — just comparisons.

**Why fully grown trees overfit:**
```
max_depth=None: the tree grows until each leaf has only 1 training sample.
→ Every training point gets its own leaf → training error = 0%
→ Any new point falls into a leaf made for a different sample → terrible generalization

Example:
  Training R² = 1.00 (100%) ← memorized everything
  Test R²     = 0.60        ← fails on new data
```

**How max_depth controls this:**
```
max_depth=2:  Only 2 levels, 4 leaves max.  Simple rules, may underfit.
max_depth=5:  32 leaves max.               Good balance for most datasets.
max_depth=10: 1024 leaves max.             Can still overfit on small data.
```

**What fit() stores:**
```
model.tree_              ← the internal tree structure (nodes + rules)
model.feature_importances_ ← how much each feature reduced MSE
model.max_features_      ← actual features used
```

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
**The Two Core Techniques**

Random Forest uses two sources of randomness to make its trees diverse and uncorrelated.
Uncorrelated trees make different mistakes → averaging them cancels errors out.

**Technique 1: Bootstrap Sampling (Bagging)**

For each of the `n_estimators=100` trees, before training:
```
Step 1: Take your training data (e.g. 800 rows)
Step 2: Draw 800 rows RANDOMLY WITH REPLACEMENT

Example: Original indices [0,1,2,3,4,5,6,7,...]
Bootstrap sample A: [0,0,3,5,5,7,2,4,...] ← row 0 appears twice, row 1 missing
Bootstrap sample B: [1,3,3,6,0,2,8,1,...] ← different random selection
Bootstrap sample C: [4,7,0,5,5,1,3,2,...] ← another different selection

Each tree trains on a DIFFERENT 800-row sample
→ Each tree sees slightly different data
→ Each tree makes different errors
```

On average, each bootstrap sample contains about **63% unique rows** (37% are never selected
— these are called "out-of-bag" (OOB) samples and can be used for free validation).

**Technique 2: Feature Randomness (Random Subspace)**

At each node split inside a tree, instead of checking ALL features:
```
You have 10 features total.
max_features = n_features/3 ≈ 3  (for regression, default)

At node 1: randomly pick features [sqft, age, school_rate]  → best split: sqft>1500
At node 2: randomly pick features [bedrooms, bathrooms, age] → best split: bedrooms>3
At node 3: randomly pick features [sqft, bathrooms, year]    → best split: year<2000
```
Each split considers only a random SUBSET of features.
This forces trees to use different features → trees are more diverse → less correlated.

**What fit(X_train, y_train) does — Full Step by Step**
```
for i in range(n_estimators=100):
    1. Draw bootstrap sample from X_train (800 rows with replacement)
    2. Build a DecisionTreeRegressor on that bootstrap sample
       - At each node: try only max_features=3 random features
       - Choose the split that minimises weighted MSE
       - Keep splitting until max_depth reached
    3. Store tree_i

After loop: model has 100 trees stored internally.
```

**What predict(X_test) does — Full Step by Step**
```
New house: X_test[0] = [1600, 3, 2, 25, 8.5]  (sqft, bed, bath, age, school)

Step 1: Run X_test[0] through Tree 1   → prediction = 340,000
Step 2: Run X_test[0] through Tree 2   → prediction = 365,000
Step 3: Run X_test[0] through Tree 3   → prediction = 342,000
...
Step 100: Run X_test[0] through Tree 100 → prediction = 358,000

Step 101: Final = AVERAGE of all 100 predictions
         = (340000 + 365000 + 342000 + ... + 358000) / 100
         = ~352,000
```

**Why averaging beats a single tree:**
```
Single DecisionTree error: large variance — one bad split ruins everything
RandomForest error: individual tree errors cancel out when averaged
→ More stable, more accurate, much less sensitive to noise
```

**Feature Importance — What it Actually Measures**
```
model.feature_importances_ = [0.45, 0.30, 0.15, 0.07, 0.03]

For each feature, importance = total MSE reduction that feature caused across ALL trees / total MSE reduction
→ sqft caused 45% of all splits' MSE reduction → most important
→ school_rate caused 3% → least important
```

**Out-of-Bag (OOB) Score — Free Validation**
```python
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X_train, y_train)
print(model.oob_score_)  # R² on out-of-bag samples — no test set needed
```
Each tree was not trained on its OOB rows → use those for validation for free.

**What fit() stores:**
```
model.estimators_          ← list of 100 fitted DecisionTreeRegressor objects
model.feature_importances_ ← importance of each feature
model.oob_score_           ← OOB R² score (only if oob_score=True)
```

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
| `oob_score` | `False` | Whether to compute out-of-bag R² score |
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

model = RandomForestRegressor(n_estimators=100, max_depth=10, oob_score=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("R² Score (test):", model.score(X_test, y_test))
print("OOB Score:      ", model.oob_score_)

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
