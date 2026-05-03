# 📅 Day 4 — Classification Models
## What is Classification?
Classification is a type of machine learning where the **output is a category (class label)**.

Examples:
- Email → "Spam" or "Not Spam"
- Tumor → "Benign" or "Malignant"
- Customer → "Will Churn" or "Will Stay"
- Flower → "Setosa", "Versicolor", or "Virginica"

**Binary Classification** → only 2 classes (yes/no, 0/1)
**Multi-class Classification** → 3 or more classes

## Models Covered Today
| Model | Best For | Key Idea |
|-------|----------|----------|
| `LogisticRegression` | Linear boundary problems, baseline | Sigmoid function maps to probability |
| `KNeighborsClassifier` | Non-linear, small datasets | Looks at nearest neighbors |
| `DecisionTreeClassifier` | Interpretable, non-linear | Tree of if-else rules |
| `RandomForestClassifier` | General purpose, high accuracy | Many trees vote |
| `SVC` | High-dimensional, complex boundaries | Finds best separating hyperplane |

# LogisticRegression
## What is it?
`LogisticRegression` is a **classification** model (despite the name "regression").

It predicts the **probability** that a sample belongs to a class,
then classifies it based on a threshold (default 0.5).

## Syntax
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## What Input Does it Accept?
```python
# X_train: 2D numeric array, no NaN, scaled preferred
# y_train: 1D array of class labels (integers or strings)

X_train = [[25, 50000], [35, 80000], [22, 30000]]
y_train = [0, 1, 0]   # 0 = no churn, 1 = churn
```

## How Does it Work Internally?
### Step 1 — Compute a Linear Combination
Same as Linear Regression first:

$$z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b$$
### Step 2 — Apply the Sigmoid Function
The raw value $z$ can be any number (-∞ to +∞).
The sigmoid function squishes it to a probability between 0 and 1:

$$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

| z value | Sigmoid output | Meaning |
|---------|---------------|---------|
| Very negative (e.g. -10) | ≈ 0.00 | Almost certainly class 0 |
| 0 | 0.50 | Exactly 50/50 |
| Very positive (e.g. +10) | ≈ 1.00 | Almost certainly class 1 |
### Step 3 — Apply Threshold
```
If P(y=1) >= 0.5  →  predict class 1
If P(y=1) <  0.5  →  predict class 0
```
### Step 4 — How fit() Finds Weights
Uses **Maximum Likelihood Estimation (MLE)** with gradient descent.
Minimizes the **Log Loss (Binary Cross-Entropy)**:

$$\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

- If actual=1 and predicted probability=0.99 → very small loss
- If actual=1 and predicted probability=0.01 → very large loss

fit() adjusts weights to minimize this log loss.

## What Does it Output?
```python
model.fit(X_train, y_train)
# predict() returns class labels
labels = model.predict(X_test)
print(labels)             # [0 1 0 1 1]
# predict_proba() returns probability for each class
probs = model.predict_proba(X_test)
print(probs)
# [[0.82, 0.18],   ← 82% class 0, 18% class 1
#  [0.23, 0.77],   ← 23% class 0, 77% class 1
#  ...]
# Coefficients (weights)
print(model.coef_)         # shape: (1, n_features) for binary
print(model.intercept_)    # bias
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `C` | `1.0` | Inverse of regularization strength. Smaller C = more regularization |
| `penalty` | `"l2"` | Type of regularization: "l1", "l2", "elasticnet", None |
| `solver` | `"lbfgs"` | Optimization algorithm |
| `max_iter` | `100` | Max iterations. Increase if model does not converge |
| `multi_class` | `"auto"` | Strategy for multi-class |

## Real-life Example — Bank Loan Default Prediction
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 500

age       = np.random.randint(20, 65, n)
income    = np.random.randint(20000, 120000, n)
loan_amt  = np.random.randint(5000, 50000, n)
# Will default = 1 if loan_amt is too high relative to income
default = ((loan_amt / income) > 0.4).astype(int)

X = np.column_stack([age, income, loan_amt])
y = default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

print("Accuracy:", model.score(X_test_s, y_test))
print("Weights:", model.coef_)
# Predict for a new customer
new_customer = scaler.transform([[35, 50000, 25000]])
print("Predicted class:", model.predict(new_customer)[0])
print("Probabilities:", model.predict_proba(new_customer)[0])
# e.g. [0.72, 0.28] → 72% chance no default, 28% chance default
```

# KNeighborsClassifier (KNN)
## What is it?
`KNeighborsClassifier` classifies a new point by looking at the **K nearest training points** and taking a majority vote.

No training phase — it memorizes all training data and uses it at prediction time.
This is called a **lazy learner**.

## Syntax
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## How Does it Work Internally?
### fit(X_train, y_train):
KNN does **almost nothing** during fit.
It just stores X_train and y_train in memory.
### predict(X_test):
For each new point in X_test:

**Step 1:** Calculate the distance from the new point to **every** training point.

**Default: Euclidean distance**:

$$d = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2 + \ldots + (x_n - x_n')^2}$$

**Step 2:** Sort all training points by distance. Take the K closest ones.

**Step 3:** Among those K neighbors, count votes for each class. Return majority class.

```
New point: [28, 60000]

5 Nearest Neighbors:
  distance=1.2 → class 1
  distance=1.5 → class 1
  distance=2.1 → class 0
  distance=2.4 → class 1
  distance=2.8 → class 0

Vote: class 1 gets 3 votes, class 0 gets 2 votes
→ Prediction: class 1
```

## The Effect of K
```
K=1  → Very sensitive, jagged boundary, overfits easily
K=3  → Better, some smoothing
K=5  → Default, good balance (most used)
K=20 → Very smooth boundary, may underfit
K=n  → Predicts the majority class for everything (useless)
```

Odd K is preferred for binary classification to avoid ties.

## Why Scaling is Critical for KNN?
Without scaling:
```
Age:    25 vs 30  → distance contribution = 5
Salary: 30000 vs 80000 → distance contribution = 50000

Salary dominates completely. Age becomes irrelevant.
```

With StandardScaler:
```
Age:    -0.5 vs 0.0  → distance contribution = 0.5
Salary: -1.2 vs 0.3  → distance contribution = 1.5

Both features now contribute fairly.
```

**Always scale your data before using KNN.**

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_neighbors` | `5` | Number of neighbors to consider |
| `weights` | `"uniform"` | `"uniform"` = all neighbors equal, `"distance"` = closer neighbors vote more |
| `metric` | `"minkowski"` | Distance metric. p=2 → Euclidean, p=1 → Manhattan |
| `algorithm` | `"auto"` | Data structure to find neighbors: `"ball_tree"`, `"kd_tree"`, `"brute"` |
| `n_jobs` | `None` | Parallel jobs for prediction |

## Real-life Example — Diagnosing Diabetes
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 400

glucose    = np.random.uniform(70, 200, n)
bmi        = np.random.uniform(18, 45, n)
age        = np.random.randint(20, 70, n)

has_diabetes = ((glucose > 140) | (bmi > 35)).astype(int)

X = np.column_stack([glucose, bmi, age])
y = has_diabetes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
# Try different K values
print(f"{'K':>4}  {'Accuracy':>10}")
print("-" * 18)
for k in [1, 3, 5, 7, 11, 15, 21]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f"  {k:2d}    {acc:.4f}")
```

# DecisionTreeClassifier
## What is it?
`DecisionTreeClassifier` builds a tree of if-else rules to classify samples.
At each node it asks a question about one feature, splits the data, and repeats until a decision is made.

## Syntax
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
### Building the Tree
At each node, it tries every feature and every possible split value.
For each candidate split, it measures how "pure" the resulting children are.

**Gini Impurity** (default):

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

- $p_k$ = proportion of class k in node t
- Gini = 0 → perfectly pure (all same class)
- Gini = 0.5 → 50/50 split (maximally impure for binary)

**Gini Gain** — how much splitting improves purity:

$$\text{Gain} = \text{Gini}_{\text{parent}} - \left(\frac{n_L}{n} \cdot \text{Gini}_L + \frac{n_R}{n} \cdot \text{Gini}_R\right)$$

The split with the **highest Gini Gain** is chosen.

**Alternative: Entropy (Information Gain)**:

$$\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

Choose with `criterion="entropy"`.
### Prediction:
Follow the tree from root to a leaf. Return the **majority class** in that leaf.
### Example:
```
Root: glucose > 140?
    Yes (45 diabetes, 5 healthy)  → Mostly Diabetes
    No  (10 diabetes, 90 healthy) → Mostly Healthy

Left node: bmi > 35?
    Yes → Diabetes (very confident)
    No  → Probably Diabetes (still check more)
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `max_depth` | `None` | Max tree levels. None = fully grown (overfits) |
| `criterion` | `"gini"` | Split quality measure: `"gini"` or `"entropy"` |
| `min_samples_split` | `2` | Min samples to split a node |
| `min_samples_leaf` | `1` | Min samples allowed in a leaf |
| `max_features` | `None` | Features to consider at each split |
| `class_weight` | `None` | Handle imbalanced classes with `"balanced"` |

## Real-life Example — Iris Flower Classification
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
# See feature importances
print("\nFeature Importances:")
for name, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")
# Predict a new flower
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])   # sepal/petal measurements
prediction = model.predict(new_flower)
print("\nPredicted class:", iris.target_names[prediction[0]])
```

# RandomForestClassifier
## What is it?
`RandomForestClassifier` builds many Decision Trees and uses **majority voting** to classify.

It is one of the most used and reliable classification models in practice.

## Syntax
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
Same two techniques as RandomForestRegressor (Day 3) — but for classification:
### 1. Bagging
For each of 100 trees:
- Sample ~63% of rows randomly with replacement (bootstrap)
- Train one full Decision Tree on those rows
### 2. Feature Randomness
At each split, only `sqrt(n_features)` random features are considered.
Default for classification: `max_features = sqrt(n_features)`
### Prediction — Majority Vote:
```
100 trees each predict a class for the new point:

Tree 1: "Spam"
Tree 2: "Not Spam"
Tree 3: "Spam"
...
Tree 100: "Spam"

Count: Spam=67, Not Spam=33
→ Final Prediction: "Spam"
```

For probabilities: `predict_proba()` returns the fraction of trees that voted for each class.

```python
proba = model.predict_proba(X_test)
# [[0.67, 0.33],   ← 67 trees said Spam, 33 said Not Spam
#  [0.12, 0.88],
#  ...]
```

## Parameters
Same as RandomForestRegressor (Day 3):

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_estimators` | `100` | Number of trees |
| `max_depth` | `None` | Max depth per tree |
| `max_features` | `"sqrt"` | Features per split (sqrt for classification) |
| `class_weight` | `None` | Use `"balanced"` for imbalanced datasets |
| `n_jobs` | `None` | `-1` uses all CPU cores |
| `random_state` | `None` | Seed |

## Real-life Example — Email Spam Detection
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# Simulate email features: word frequencies, email length, etc.
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
# Feature importances
importances = model.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("\nTop 5 most important features:")
for i in top5_idx:
    print(f"  Feature {i:2d}: {importances[i]:.4f}")
# Predict with probability
sample = X_test[0].reshape(1, -1)
print("\nSample prediction:", model.predict(sample)[0])
print("Probability [not_spam, spam]:", model.predict_proba(sample)[0])
```

# SVC (Support Vector Classifier)
## What is it?
`SVC` finds the **best hyperplane** (decision boundary) that separates classes with the **maximum margin**.

It works well in high-dimensional spaces and with complex, non-linear boundaries (using kernels).

## Syntax
```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
model.fit(X_train, y_train)
```

## How Does it Work Internally?
### The Core Idea — Maximum Margin
Consider two classes: "spam" and "not spam" plotted in 2D.
Many lines could separate them. SVC finds the line that:
1. Separates the classes correctly
2. Is **as far as possible** from the nearest points of both classes

The nearest points from each class that define the margin are called **Support Vectors**.

```
Class 0 points:  ●  ●  ●  ●
                          |  ← maximum margin hyperplane
Class 1 points:              ○  ○  ○
                   ↑
            Support Vectors (closest points to boundary)
```

**Margin** = distance from the boundary to the nearest support vector on each side.
SVC maximizes this margin.
### The Optimization Problem:
Minimize:

$$\frac{1}{2} \|w\|^2$$

Subject to:

$$y_i (w \cdot x_i + b) \geq 1 \quad \forall i$$
### Soft Margin — Parameter C
In real data, classes often overlap. C controls how much violation is allowed:

```
High C (e.g. 100) → tries to classify everything correctly → narrow margin, risk of overfit
Low C  (e.g. 0.1) → allows some misclassification → wider margin, more generalization
```
### Kernels — Handling Non-linear Data
If data cannot be separated by a straight line, kernels transform data into higher dimensions
where separation IS possible:

| Kernel | When to Use | What it Does |
|--------|-------------|--------------|
| `"linear"` | Data is linearly separable | Standard hyperplane |
| `"rbf"` | Data is non-linear (DEFAULT) | Gaussian radial basis function |
| `"poly"` | Polynomial patterns | Polynomial feature space |
| `"sigmoid"` | Neural network-like | Hyperbolic tangent |

**RBF Kernel** (most common):

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

- $\gamma$ controls how far the influence of each training point reaches
- High $\gamma$ → small influence area → complex boundary → overfits
- Low $\gamma$ → large influence area → smoother boundary

## What Does it Output?
```python
model.fit(X_train, y_train)
# Predict class
predictions = model.predict(X_test)   # [0 1 0 1 1 ...]
# Predict probability (only if probability=True)
proba = model.predict_proba(X_test)   # [[0.82, 0.18], ...]
# Support vectors (training points closest to boundary)
print("Number of support vectors:", model.n_support_)
# e.g. [45, 52]  ← 45 from class 0, 52 from class 1
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `C` | `1.0` | Regularization. High = fit training better, may overfit |
| `kernel` | `"rbf"` | Type of kernel function |
| `gamma` | `"scale"` | RBF kernel coefficient. `"scale"` = 1/(n_features * X.var()) |
| `degree` | `3` | Degree for polynomial kernel only |
| `probability` | `False` | Set True to enable predict_proba() |
| `class_weight` | `None` | Use `"balanced"` for imbalanced data |
**Note:** SVC does NOT scale well to very large datasets (slow for n > 100k rows). Use `LinearSVC` instead.

## Real-life Example — Cancer Classification
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target   # 569 samples, 30 features
# y: 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scaling is critical for SVC
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
# Compare kernels
for kernel in ["linear", "rbf", "poly"]:
    svc = SVC(kernel=kernel, C=1.0, probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    acc = svc.score(X_test_s, y_test)
    print(f"Kernel={kernel:8s} → Accuracy: {acc:.4f}")
# Best model with predict_proba
best_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
best_model.fit(X_train_s, y_train)

new_patient = X_test[0].reshape(1, -1)
new_patient_scaled = scaler.transform(new_patient)

print("\nPrediction:", cancer.target_names[best_model.predict(new_patient_scaled)[0]])
print("Probabilities:", best_model.predict_proba(new_patient_scaled)[0])
```

## Day 4 — Full Comparison
| Model | Training Speed | Prediction Speed | Needs Scaling | Handles Non-linear | Probability Output |
|-------|--------------|-----------------|--------------|-------------------|-------------------|
| `LogisticRegression` | Fast | Fast | Yes | No (use poly features) | Yes (built-in) |
| `KNeighborsClassifier` | Instant | Slow (large data) | Yes (critical) | Yes | Yes |
| `DecisionTreeClassifier` | Fast | Fast | No | Yes | Yes |
| `RandomForestClassifier` | Medium | Medium | No | Yes | Yes |
| `SVC` | Slow (large data) | Medium | Yes (critical) | Yes (with kernels) | Only if probability=True |

## Day 4 — Complete Project: Predicting Customer Churn
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
# Simulate: customer features → will they churn?
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "LogisticRegression":      LogisticRegression(max_iter=1000),
    "KNN (k=5)":               KNeighborsClassifier(n_neighbors=5),
    "DecisionTree (depth=5)":  DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest (100)":      RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC (rbf)":               SVC(kernel="rbf", probability=True, random_state=42),
}

print(f"{'Model':<28} {'Accuracy':>10}")
print("-" * 40)
for name, m in models.items():
    m.fit(X_train_s, y_train)
    print(f"{name:<28} {m.score(X_test_s, y_test):>10.4f}")
```

> Day 4 Complete.
> You now understand all 5 classification models — from logistic regression to SVM with kernels.
>
> Next: **Day 5 — Model Evaluation**
> (MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report)