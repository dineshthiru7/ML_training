# 📅 Day 1 — Scikit-learn Basics
## What is Scikit-learn?
Scikit-learn (sklearn) is a Python library built specifically for machine learning.

It provides ready-made implementations of algorithms like Linear Regression, Decision Tree,
Random Forest, SVM, and many more. It also gives you tools for preprocessing data,
evaluating models, and improving them.

You do not need to write math or algorithms from scratch.
You just import the tool, give it your data, and it does the work.

**Real-life Analogy:**
Think of sklearn like a hospital with all the specialist doctors already available.
You do not need to train doctors yourself.
You just bring the patient (data), and the right doctor (algorithm) handles it.

## Sklearn Workflow — The Standard Process
Every machine learning project in sklearn follows the same pattern.
Understanding this flow is the most important thing on Day 1.

```
Raw Data
   ↓
Select Features (X) and Target (y)
   ↓
Split into Train and Test sets
   ↓
Create Model object
   ↓
fit(X_train, y_train)     ← Model learns
   ↓
predict(X_test)           ← Model answers
   ↓
score(X_test, y_test)     ← Check how good it is
```
### Full Workflow with Code:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 1. Load data
df = pd.read_csv("house_prices.csv")
# 2. Select features and target
X = df[["bedrooms", "bathrooms", "sqft"]]   # input  — shape: (1000, 3)
y = df["price"]                              # output — shape: (1000,)
# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train: (800, 3), X_test: (200, 3)
# 4. Create model
model = LinearRegression()
# 5. Train
model.fit(X_train, y_train)
# 6. Predict
predictions = model.predict(X_test)
# 7. Score
print(model.score(X_test, y_test))
```

# fit()
## What is fit()?
`fit()` is the function that **trains the model**.

When you call `fit()`, you are giving the model your training data and telling it:
"Look at these inputs and their correct answers. Learn the pattern."

The model then figures out the best internal settings (weights and bias) to map inputs to outputs.

## Syntax
```python
model.fit(X_train, y_train)
```

| Argument | Type | What it is |
|----------|------|------------|
| `X_train` | 2D array or DataFrame | Input features — rows are samples, columns are features |
| `y_train` | 1D array or Series | Correct output values for each row in X_train |

## What Input Does fit() Accept?
```python
# X_train — 2D array, shape (n_samples, n_features)
# Example: 800 houses, 3 features each
X_train = [
    [3, 2, 1500],   # house 1: 3 bed, 2 bath, 1500 sqft
    [4, 3, 2000],   # house 2
    [2, 1, 900],    # house 3
    ...             # 800 rows total
]
# y_train — 1D array, shape (n_samples,)
# Example: actual price of each house
y_train = [250000, 400000, 150000, ...]
```

**Important rules:**
- `X_train` must be **2D** (rows x columns). If you have a single feature, reshape it: `X.reshape(-1, 1)`
- `y_train` must have the **same number of rows** as `X_train`
- No `NaN` (missing) values allowed — sklearn will throw an error
- All values must be **numeric** — no strings

## What Happens Inside fit() — Step by Step
Let us use Linear Regression as the example (the simplest case).
### Step 1 — Model is initialized with random or zero weights
Before `fit()`, the model has no learned values.
Internally, it has:
- `coef_` (weights) → all zeros or unset
- `intercept_` (bias) → zero or unset
### Step 2 — fit() receives X_train and y_train
```
X_train shape: (800, 3)   → 800 rows, 3 features
y_train shape: (800,)     → 800 correct price values
```
### Step 3 — Model computes the best weights using OLS
For Linear Regression, sklearn does NOT use slow trial-and-error.
It uses a direct mathematical formula called **Ordinary Least Squares (OLS)**:

$$W = (X^T X)^{-1} X^T y$$

- $X^T$ = transpose of input matrix
- $(X^T X)^{-1}$ = matrix inverse
- The result $W$ is the set of weights that minimizes total squared error

This is done in one shot — no iteration needed for Linear Regression.
### Step 4 — Weights and bias are stored inside the model
After `fit()` finishes, the model stores:
- `model.coef_` → the learned weight for each feature
- `model.intercept_` → the learned bias value

```
model.coef_      → [50000.0, 30000.0, 120.0]
                    (bedrooms weight, bathrooms weight, sqft weight)
model.intercept_ → 10000.0
```

This means internally the model learned:

$$\text{price} = 50000 \times \text{bedrooms} + 30000 \times \text{bathrooms} + 120 \times \text{sqft} + 10000$$
### Step 5 — fit() returns the model object itself
`fit()` returns `self` — which means the model object with updated weights.
You can chain it: `model.fit(X_train, y_train).predict(X_test)`

## What Does fit() Output?
```python
result = model.fit(X_train, y_train)

print(type(result))          # <class 'sklearn.linear_model._base.LinearRegression'>
print(model.coef_)           # [50000.  30000.    120.]  ← learned weights
print(model.intercept_)      # 10000.0                   ← learned bias
```

**fit() does NOT return predictions.**
It only updates the model's internal parameters.
You call `predict()` separately to get predictions.

## The Loss Function fit() Minimizes
The goal of `fit()` is to find weights that minimize prediction error.
For Linear Regression, the error measure is **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- $y_i$ = actual price of house $i$
- $\hat{y}_i$ = predicted price of house $i$
- Squaring makes all errors positive and penalizes large errors more

`fit()` finds the $W$ and $b$ that make this MSE as small as possible.

## Real-life Example of fit()
**Scenario:** You are building a system to predict house prices.

```
Training data you pass to fit():

| bedrooms | bathrooms | sqft | price (y) |
|----------|-----------|------|-----------|
|    3     |     2     | 1500 |  250,000  |
|    4     |     3     | 2000 |  400,000  |
|    2     |     1     |  900 |  150,000  |
...800 rows total

fit() reads all 800 rows.
It calculates: which weight for bedrooms? which for sqft? what bias?
Answer: weights that best predict price from those 3 features.
```

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.array([
    [3, 2, 1500],
    [4, 3, 2000],
    [2, 1, 900],
    [5, 4, 2500],
])
y_train = np.array([250000, 400000, 150000, 500000])

model = LinearRegression()
model.fit(X_train, y_train)

print("Learned weights:", model.coef_)
# e.g. [50000. 30000. 120.]

print("Learned bias:", model.intercept_)
# e.g. 10000.0
```

## What fit() Does NOT Do
- It does NOT make predictions (use `predict()` for that)
- It does NOT evaluate accuracy (use `score()` for that)
- It does NOT modify your original X_train or y_train data
- It does NOT work if there are NaN values in your data

# predict()
## What is predict()?
`predict()` uses the **already trained model** to produce output values for new input data.

You pass new rows (that the model has never seen) and it returns the predicted output for each row.

## Syntax
```python
predictions = model.predict(X_test)
```

| Argument | Type | What it is |
|----------|------|------------|
| `X_test` | 2D array or DataFrame | New input rows you want predictions for |
**Note:** You must call `fit()` before `predict()`. Otherwise sklearn raises `NotFittedError`.

## What Input Does predict() Accept?
```python
# X_test — 2D array, shape (n_samples, n_features)
# Must have the SAME number of features as X_train

X_test = [
    [3, 2, 1600],   # new house 1
    [5, 3, 2200],   # new house 2
]
# shape: (2, 3) — 2 houses, 3 features each
```
**Rules:**
- Must be 2D (same shape rules as X_train)
- Must have the **exact same number of columns** as X_train
- No NaN values
- All numeric

## What Happens Inside predict() — Step by Step
### Step 1 — predict() receives X_test
```
X_test shape: (200, 3)  → 200 new houses, 3 features each
```
### Step 2 — It retrieves the stored weights from fit()
The model reads `model.coef_` and `model.intercept_` that were stored during `fit()`.

```
coef_      = [50000., 30000., 120.]
intercept_ = 10000.0
```
### Step 3 — It runs matrix multiplication for every row
For each row in X_test, it computes:

$$\hat{y}_i = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b$$

Example for house: bedrooms=3, bathrooms=2, sqft=1600

$$\hat{y} = 50000 \times 3 + 30000 \times 2 + 120 \times 1600 + 10000$$
$$= 150000 + 60000 + 192000 + 10000 = 412000$$

In matrix form for all rows at once:

$$\hat{Y} = X_{\text{test}} \cdot W^T + b$$
### Step 4 — Returns array of predictions
One predicted value per row.

## What Does predict() Output?
```python
predictions = model.predict(X_test)

print(type(predictions))    # <class 'numpy.ndarray'>
print(predictions.shape)    # (200,)  — one value per row
print(predictions[:5])      # [412000. 520000. 180000. 350000. 290000.]
```

For **regression** → returns continuous values (prices, temperatures, scores)
For **classification** → returns class labels (0, 1, 2, "cat", "dog")

## Real-life Example of predict()
**Scenario:** A new customer visits your house price website and enters their house details.

```python
import numpy as np
# New house: 4 bedrooms, 3 bathrooms, 2200 sqft
new_house = np.array([[4, 3, 2200]])
# IMPORTANT: double brackets [[]] — must be 2D

predicted_price = model.predict(new_house)
print("Predicted Price: $", predicted_price[0])
# Output: Predicted Price: $ 484000.0
```

**Multiple new houses at once:**

```python
new_houses = np.array([
    [4, 3, 2200],   # house A
    [2, 1, 1000],   # house B
    [6, 4, 3000],   # house C
])

prices = model.predict(new_houses)
print(prices)
# [484000.  190000.  720000.]
```

## What predict() Does NOT Do
- It does NOT learn or update any weights
- It does NOT compare with actual values (use `score()` for that)
- It does NOT return probabilities for basic classifiers (use `predict_proba()` for that)

# score()
## What is score()?
`score()` measures how well the model's predictions match the actual values.

It returns a single number that tells you the quality of your model.

## Syntax
```python
result = model.score(X_test, y_test)
```

| Argument | Type | What it is |
|----------|------|------------|
| `X_test` | 2D array | Test input features |
| `y_test` | 1D array | True/actual output values for X_test |

## What Input Does score() Accept?
```python
# X_test — same 2D format as predict()
X_test = [[3, 2, 1600], [5, 3, 2200], ...]   # shape (200, 3)
# y_test — actual correct answers for those rows
y_test = [400000, 510000, ...]                # shape (200,)
```

## What Happens Inside score() — Step by Step
### Step 1 — score() internally calls predict()
It runs `model.predict(X_test)` to get predicted values.

```
y_predicted = model.predict(X_test)
# [412000. 520000. 180000. ...]
```
### Step 2 — It compares predicted values to actual values (y_test)
```
y_test      = [400000, 510000, 190000, ...]
y_predicted = [412000, 520000, 180000, ...]
```
### Step 3 — For Regression: Calculates R² Score
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Breaking it down:

- **Numerator** $\sum(y_i - \hat{y}_i)^2$ = total squared error of your model's predictions
- **Denominator** $\sum(y_i - \bar{y})^2$ = total squared variation in actual values (baseline)
- **$\bar{y}$** = mean of all actual values in y_test

**What this ratio means:**
- If your model predicts perfectly → numerator = 0 → R² = 1.0
- If your model is no better than predicting the mean every time → R² = 0.0
- If your model is worse than predicting the mean → R² is negative
### Step 3 — For Classification: Calculates Accuracy
$$\text{Accuracy} = \frac{\text{Number of rows where } \hat{y}_i = y_i}{\text{Total number of rows}}$$
### Step 4 — Returns a single float
## What Does score() Output?
```python
score = model.score(X_test, y_test)

print(type(score))   # <class 'float'>
print(score)         # 0.8812
```

**How to interpret the R² score:**

| R² Value | What it means | Action |
|----------|--------------|--------|
| 1.00 | Perfect — every prediction is exact | (Rarely happens, could mean data leakage) |
| 0.90 – 0.99 | Excellent model | Ready to use |
| 0.70 – 0.90 | Good model | Fine for most cases |
| 0.50 – 0.70 | Average model | Try adding more features |
| 0.20 – 0.50 | Weak model | Something is wrong |
| Below 0 | Terrible — worse than guessing average | Start over |

## Real-life Example of score()
**Scenario:** You trained a house price model. Now you want to know if it is reliable.

```python
# After training
score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")
# If score = 0.88
# It means: 88% of the variation in house prices is explained by your model.
# Only 12% is unexplained (noise, missing features, etc.)
```

**Understanding with real numbers:**

```
y_test (actual prices):    [400000, 510000, 190000, 350000, 280000]
y_predicted (model guess): [412000, 505000, 188000, 362000, 275000]

These are close → low error → high R² → good model
```

## What score() Does NOT Do
- It does NOT give you detailed breakdown (use `classification_report()` for that)
- It does NOT tell you where the model is wrong
- For regression it uses R² — this can be misleading for some data patterns
- It does NOT modify the model

# train_test_split()
## What is train_test_split()?
`train_test_split()` splits your full dataset into two separate parts:

- **Training set** — you give this to `fit()` so the model can learn
- **Testing set** — you give this to `predict()` and `score()` to measure real performance

## Syntax
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## What Input Does train_test_split() Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `X` | 2D array or DataFrame | All your input features |
| `y` | 1D array or Series | All your output/target values |
| `test_size` | float (0 to 1) | Fraction of data to use for testing |
| `random_state` | int | Seed number for reproducible splits |
| `shuffle` | bool | Shuffle rows before splitting (default: True) |
| `stratify` | array | Keep class proportions equal in both splits |

```python
# Example input
X = df[["bedrooms", "bathrooms", "sqft"]]   # shape: (1000, 3)
y = df["price"]                              # shape: (1000,)
```

## What Happens Inside train_test_split() — Step by Step
### Step 1 — Creates an index array
Internally it creates an array of row indices: `[0, 1, 2, 3, ..., 999]`
### Step 2 — Shuffles the indices randomly
Using `random_state` as the seed, it randomly shuffles the index array:

```
Before: [0, 1, 2, 3, 4, ..., 999]
After:  [532, 17, 891, 44, 3, ..., 221]   ← shuffled
```

With `random_state=42`, this shuffle is the same every single run.
Without `random_state`, the shuffle is different each run → different results each time.
### Step 3 — Splits the shuffled indices by test_size
With `test_size=0.2` and 1000 rows:
- First 800 shuffled indices → training set
- Last 200 shuffled indices → test set
### Step 4 — Uses those indices to select rows from X and y
```
X_train = X at training indices   → shape (800, 3)
X_test  = X at test indices       → shape (200, 3)
y_train = y at training indices   → shape (800,)
y_test  = y at test indices       → shape (200,)
```
### Step 5 — Returns all 4 arrays
## What Does train_test_split() Output?
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)   # (800, 3)
print(X_test.shape)    # (200, 3)
print(y_train.shape)   # (800,)
print(y_test.shape)    # (200,)
# The rows in X_train and y_train match each other
# The rows in X_test and y_test match each other
```

## Why Not Just Use All Data for Training?
This is the most important concept on Day 1.

**Problem — Overfitting:**

> If you train on 1000 rows and also test on the same 1000 rows,
> the model could just memorize all the answers.
> Score = 100% on training data.
> But give it a new house it has never seen → it fails completely.

**Analogy:**
> A student memorizes exactly the questions and answers from last year's paper.
> They score 100% if the same paper is given.
> But if one new question appears → they cannot answer it.
> They learned the answers, not the concept.

**The fix:**
> Split the data. Train on 800. Test on 200 the model has never seen.
> Now the test score reflects **real-world performance**.

## The stratify Parameter — When to Use It?
In classification, if you have unbalanced classes, a random split might put most of one
class into training and very few into test.

`stratify=y` ensures both train and test have the same class ratios.

```python
# Without stratify — might get imbalanced split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# With stratify — guaranteed balanced split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

**Example:**
Dataset: 900 "not spam" emails, 100 "spam" emails (10% spam)
With `stratify=y`:
- Train set: 720 not-spam, 80 spam → 10% spam
- Test set: 180 not-spam, 20 spam → 10% spam

## Real-life Example
**Scenario:** You have 1000 house sale records.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("house_prices.csv")
X = df[["bedrooms", "bathrooms", "sqft"]]
y = df["price"]

print("Full dataset:", X.shape)          # (1000, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 200 rows for testing
    random_state=42      # same result every time you run this code
)

print("Training set:", X_train.shape)    # (800, 3)
print("Testing set:", X_test.shape)      # (200, 3)
# These 200 test houses will NEVER be shown to the model during training
# So testing them gives a fair measure of real performance
```

# datasets module
## What is the datasets module?
`sklearn.datasets` is a built-in module containing **ready-to-use sample datasets**.

No CSV files needed. No internet. No cleaning. Just import and get data instantly.

## Why Use It?
When learning machine learning, finding a clean dataset is half the struggle.
The datasets module solves this — every dataset inside is already:
- Clean (no missing values)
- Properly formatted (all numeric)
- Well-documented
- Sized appropriately for learning

## Types of Functions in datasets
### 1. Loaders — Load real classic datasets
```python
from sklearn import datasets

iris          = datasets.load_iris()           # flower classification
digits        = datasets.load_digits()         # handwritten digit recognition
wine          = datasets.load_wine()           # wine quality classification
breast_cancer = datasets.load_breast_cancer()  # cancer detection
```
### 2. Generators — Create fake (synthetic) data
```python
X, y = datasets.make_regression(n_samples=1000, n_features=5, noise=10)
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)
X, y = datasets.make_blobs(n_samples=500, centers=3)   # for clustering
```

## What Input Does load functions Accept?
```python
iris = datasets.load_iris(
    return_X_y=False   # If True, returns (X, y) tuple directly instead of Bunch object
)
```

For generators:

```python
X, y = datasets.make_regression(
    n_samples=1000,    # how many rows
    n_features=5,      # how many input columns
    n_informative=3,   # how many features actually matter
    noise=10.0,        # how much random noise to add
    random_state=42    # for reproducibility
)
```

## What Happens Inside — Step by Step
### For Loaders (e.g. load_iris()):
1. sklearn reads the dataset stored inside the package installation files
2. It converts it into NumPy arrays
3. It wraps everything inside a **Bunch object** (a special dictionary-like container)
4. Returns it
### For Generators (e.g. make_regression()):
1. Uses NumPy random number generation (seeded by `random_state`)
2. Creates a random feature matrix X
3. Creates a target y by multiplying X by random weights and adding noise
4. Returns X and y as NumPy arrays

## What Does the datasets Output Look Like?
### Bunch Object Structure:
```python
iris = datasets.load_iris()

print(type(iris))               # <class 'sklearn.utils.Bunch'>

print(iris.data.shape)          # (150, 4) — 150 rows, 4 features
print(iris.target.shape)        # (150,)   — 150 labels

print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']

print(iris.target[:10])
# [0 0 0 0 0 0 0 0 0 0]   — 0 means setosa

print(iris.DESCR[:300])         # text description of the dataset
```
### How to Extract X and y:
```python
X = iris.data      # input features — 2D array, shape (150, 4)
y = iris.target    # labels         — 1D array, shape (150,)
# OR — shortcut
X, y = datasets.load_iris(return_X_y=True)
```

## Available Datasets Summary
| Function | Task | Rows | Features | Classes / Target |
|----------|------|------|----------|-----------------|
| `load_iris()` | Classification | 150 | 4 | 3 flower types |
| `load_digits()` | Classification | 1797 | 64 | digits 0–9 |
| `load_wine()` | Classification | 178 | 13 | 3 wine types |
| `load_breast_cancer()` | Classification | 569 | 30 | benign / malignant |
| `make_regression()` | Regression | custom | custom | continuous value |
| `make_classification()` | Classification | custom | custom | 2 or more classes |
| `make_blobs()` | Clustering | custom | custom | cluster centers |

## Real-life Example — Using iris dataset end to end
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Step 1: Load dataset
iris = datasets.load_iris()
X = iris.data    # shape (150, 4)
y = iris.target  # shape (150,) — values: 0, 1, 2

print("Classes:", iris.target_names)    # ['setosa' 'versicolor' 'virginica']
print("Features:", iris.feature_names)  # 4 measurement names
# Step 2: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_train: (120, 4), X_test: (30, 4)
# Step 3: Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Step 4: Predict
predictions = model.predict(X_test)
print("Predictions:", predictions)
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
# Step 5: Score
print("Accuracy:", model.score(X_test, y_test))
# Accuracy: 1.0 (100% — iris is a clean easy dataset)
```

## Day 1 — Full Summary
### Every Function at a Glance:
| Function | You Pass In | What Happens Inside | You Get Back |
|----------|------------|---------------------|--------------|
| `fit(X_train, y_train)` | Training features + labels | Model calculates best weights | Updated model with coef_ and intercept_ |
| `predict(X_test)` | New input rows | Multiplies inputs by learned weights | Array of predicted values |
| `score(X_test, y_test)` | Test features + actual labels | Calls predict, then computes R² or accuracy | Single float between -∞ and 1.0 |
| `train_test_split(X, y)` | Full dataset | Shuffles and splits by test_size | X_train, X_test, y_train, y_test |
| `datasets.load_iris()` | Nothing | Reads stored numpy arrays | Bunch object with .data, .target, .feature_names |

## Complete End-to-End Project
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# ── 1. Create data ──────────────────────────────────────────
# Simulates: 1000 houses, 3 features (bedrooms, sqft, age)
X, y = make_regression(
    n_samples=1000,
    n_features=3,
    noise=20,
    random_state=42
)
print("Full dataset shape:", X.shape, y.shape)
# (1000, 3)  (1000,)
# ── 2. Split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
# Train: (800, 3) | Test: (200, 3)
# ── 3. Create model ─────────────────────────────────────────
model = LinearRegression()
print("Model before training — coef_ exists?", hasattr(model, 'coef_'))
# False — not yet fitted
# ── 4. fit() ────────────────────────────────────────────────
model.fit(X_train, y_train)
print("Model after training — coef_:", model.coef_)
print("Intercept:", model.intercept_)
# coef_: [w1 w2 w3]  intercept_: b
# ── 5. predict() ────────────────────────────────────────────
predictions = model.predict(X_test)
print("Predictions shape:", predictions.shape)   # (200,)
print("First 5 predictions:", predictions[:5])
# ── 6. score() ──────────────────────────────────────────────
r2 = model.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")
# e.g. 0.9912 → excellent model
# ── 7. Predict for one new house ────────────────────────────
new_house = np.array([[1.5, -0.3, 2.1]])   # must be 2D
result = model.predict(new_house)
print(f"Predicted value for new input: {result[0]:.2f}")
```

> Day 1 Complete.
> You now know exactly what each function accepts as input, how it processes the data internally, and what it returns as output.
>
> Next: **Day 2 — Data Preprocessing**
> (SimpleImputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler)