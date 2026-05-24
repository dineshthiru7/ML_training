# Day 4 — Classification Models
## What is Classification?
Classification is a type of machine learning where the **output is a category (class label)**.

Examples:
- Email -> "Spam" or "Not Spam"
- Tumor -> "Benign" or "Malignant"
- Customer -> "Will Churn" or "Will Stay"
- Flower -> "Setosa", "Versicolor", or "Virginica"

**Binary Classification** -> only 2 classes (yes/no, 0/1)
**Multi-class Classification** -> 3 or more classes

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
**The Four Steps**

**Step 1 — Compute Linear Combination (same as LinearRegression first)**
```
z = w1*x1 + w2*x2 + ... + wn*xn + b

Example: customer with age=35, income=80000 (scaled)
z = 0.5 * age_scaled + 1.2 * income_scaled + (-0.3)
z = 1.4    <- a raw score, can be any number from -inf to +inf
```

**Step 2 — Apply the Sigmoid Function (squish z to 0-1 probability)**

The raw z can be any number. Sigmoid maps it to a probability between 0 and 1:
```
sigmoid(z) = 1 / (1 + e^(-z))

z = -10  -> sigmoid(-10) approx 0.00005  -> nearly 0%    -> class 0
z = -1   -> sigmoid(-1)  approx 0.27     -> 27%          -> class 0
z =  0   -> sigmoid(0)   = 0.50          -> exactly 50%
z =  1   -> sigmoid(1)   approx 0.73     -> 73%          -> class 1
z = +10  -> sigmoid(10)  approx 0.99995  -> nearly 100%  -> class 1
```

The sigmoid curve looks like an "S" -- gradually transitions from 0 to 1.

**Step 3 — Apply Threshold to Make a Decision**
```
If P(y=1|X) >= 0.5  ->  predict class 1  (z >= 0)
If P(y=1|X) <  0.5  ->  predict class 0  (z < 0)
```
You can change this threshold. For cancer detection you might use 0.3 to be more cautious.

**Step 4 — How fit() finds the best weights (Gradient Descent)**

Unlike LinearRegression which uses a matrix formula, LogisticRegression uses **iterative optimization**.

The loss function is **Log Loss (Binary Cross-Entropy)**:
```
Log Loss = -(1/n) * SUM[ yi*log(pi_hat) + (1-yi)*log(1-pi_hat) ]

For each training sample:
  Actual y=1, predicted p_hat=0.99 -> loss approx 0.01  (very small, correct confident)
  Actual y=1, predicted p_hat=0.01 -> loss approx 4.60  (very large, wrong confident)
  Actual y=0, predicted p_hat=0.01 -> loss approx 0.01  (very small, correct confident)
  Actual y=0, predicted p_hat=0.99 -> loss approx 4.60  (very large, wrong confident)
```
Log loss heavily penalises confident wrong predictions.

**Gradient Descent — how fit() updates weights step by step:**
```
Iteration 1:
  Initialize: w1=0, w2=0, b=0  (or small random values)
  Forward pass: compute z = X_train @ w + b  for all training rows
  Apply sigmoid: p_hat = sigmoid(z)
  Compute Log Loss over all n training samples
  Compute gradient (partial derivatives of loss w.r.t. each weight):
    gradient_w = (1/n) * X_train.T @ (p_hat - y_train)
    gradient_b = (1/n) * sum(p_hat - y_train)
  Update weights (move in direction that REDUCES loss):
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

Iteration 2:
  New weights -> new predictions -> new loss -> new gradients -> update again

...repeat up to max_iter times or until loss stops decreasing (convergence)
```

**Concrete example of weight evolution:**
```
Iteration    w1        w2        Loss
0            0.000     0.000     0.693
10           0.120     0.550     0.511
100          0.340     1.180     0.342
500          0.510     1.270     0.298  <- converged (weights stable)
```

**Convergence warning** -- if you see "ConvergenceWarning: lbfgs failed to converge":
-> Increase `max_iter` (default 100 is often too small, use 1000) or scale your features.

**Regularization in LogisticRegression:**
```
Built-in via parameter C:
  C = 1/alpha (C is INVERSE of regularization strength)
  Small C (e.g. 0.01) -> strong regularization -> weights close to zero -> simpler model
  Large C (e.g. 100)  -> weak regularization  -> weights can be large -> complex model
  C = 1.0 default
```

**What fit() stores:**
```
model.coef_      -> shape (1, n_features) for binary -- weights per feature
model.intercept_ -> shape (1,) -- bias
model.n_iter_    -> int -- how many gradient descent iterations until convergence
model.classes_   -> array -- unique class labels
```

**What predict(X_test) does:**
```
For each row in X_test:
  1. z = w1*x1 + w2*x2 + ... + wn*xn + b   (linear combination)
  2. p = 1 / (1 + exp(-z))                  (sigmoid -> probability)
  3. if p >= 0.5: label = classes_[1]  (positive class)
     else:        label = classes_[0]  (negative class)

Return array of class labels.
```

**What predict_proba(X_test) does:**
```
Same sigmoid calculation, but return both probabilities:

[[0.82, 0.18],   <- 82% chance class 0, 18% chance class 1
 [0.23, 0.77],   <- 23% chance class 0, 77% chance class 1
 [0.67, 0.33]]

Column 0 = P(class 0),  Column 1 = P(class 1)
Row sums always = 1.0
```

## What Does it Output?
```python
model.fit(X_train, y_train)
labels = model.predict(X_test)      # [0 1 0 1 1]
probs  = model.predict_proba(X_test)
# [[0.82, 0.18], [0.23, 0.77], ...]
print(model.coef_)        # shape: (1, n_features) for binary
print(model.intercept_)   # bias
print(model.n_iter_)      # iterations until convergence
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
print("Converged in iterations:", model.n_iter_)
new_customer = scaler.transform([[35, 50000, 25000]])
print("Predicted class:", model.predict(new_customer)[0])
print("Probabilities:", model.predict_proba(new_customer)[0])
```

# KNeighborsClassifier (KNN)
## What is it?
`KNeighborsClassifier` classifies a new point by looking at the **K nearest training points** and taking a majority vote.

No training phase -- it memorizes all training data and uses it at prediction time.
This is called a **lazy learner**.

## Syntax
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## How Does it Work Internally?
**What fit(X_train, y_train) does:**
```
Step 1: Validate input shapes
Step 2: Store X_train in memory  (e.g. shape 400 x 3)
Step 3: Store y_train in memory  (e.g. shape 400)
Step 4: Build spatial data structure for fast distance lookup:
        algorithm="ball_tree" -> BallTree (spherical partitioning)
        algorithm="kd_tree"   -> KD-Tree (axis-aligned partitioning)
        algorithm="brute"     -> no structure, compute all distances at predict time
        algorithm="auto"      -> sklearn picks best based on data size and shape

Training time is O(n) -- just storing data and optionally building index.
```

**What predict(X_test) does -- Step by Step with actual numbers:**

Say your scaled training data looks like this:
```
Index  glucose_s  bmi_s  label
  0     -1.2      -0.8     0  (healthy)
  1      0.5       0.3     1  (diabetic)
  2      1.8       1.1     1  (diabetic)
  3     -0.4      -0.5     0  (healthy)
  4      0.9       0.7     1  (diabetic)

New test patient scaled: [0.7, 0.5]

Step 1: Compute Euclidean distance to EVERY training point:
  d(test, 0) = sqrt((0.7-(-1.2))^2 + (0.5-(-0.8))^2) = sqrt(3.61+1.69) = 2.30
  d(test, 1) = sqrt((0.7-0.5)^2   + (0.5-0.3)^2)   = sqrt(0.04+0.04) = 0.28
  d(test, 2) = sqrt((0.7-1.8)^2   + (0.5-1.1)^2)   = sqrt(1.21+0.36) = 1.25
  d(test, 3) = sqrt((0.7-(-0.4))^2 + (0.5-(-0.5))^2) = sqrt(1.21+1.0) = 1.49
  d(test, 4) = sqrt((0.7-0.9)^2   + (0.5-0.7)^2)   = sqrt(0.04+0.04) = 0.28

Step 2: Sort by distance, take K=3 nearest:
  Rank 1: index_1  distance=0.28  label=1 (diabetic)
  Rank 2: index_4  distance=0.28  label=1 (diabetic)
  Rank 3: index_2  distance=1.25  label=1 (diabetic)

Step 3: Majority vote among 3 neighbors:
  Class 0: 0 votes
  Class 1: 3 votes

Step 4: Return class 1 (diabetic)
```

**Why you MUST scale data before KNN:**
```
Without scaling (raw values):
  glucose: 70 to 200     -> max single-feature distance = 130
  bmi:     18 to 45      -> max single-feature distance =  27
  age:     20 to 70      -> max single-feature distance =  50

  Euclidean distance dominated by glucose (130 >> 27, 50)
  KNN effectively ignores bmi and age -> wrong nearest neighbors

With StandardScaler (all features -> mean=0, std=1):
  glucose_s: -2.1 to 2.1  -> range about 4
  bmi_s:     -1.8 to 1.8  -> range about 4
  age_s:     -1.6 to 1.6  -> range about 4

  All features contribute equally -> correct nearest neighbors
```

**What predict_proba(X_test) does (with weights="uniform"):**
```
K=5, 5 nearest neighbors have labels: [1, 1, 0, 1, 0]

P(class=0) = 2/5 = 0.40
P(class=1) = 3/5 = 0.60  -> predict class 1

Return: [0.40, 0.60]
```

**With weights="distance" (closer neighbors count more):**
```
neighbor 1: distance=0.3  -> weight=1/0.3=3.33  -> class 1
neighbor 2: distance=0.5  -> weight=1/0.5=2.00  -> class 1
neighbor 3: distance=0.9  -> weight=1/0.9=1.11  -> class 0
neighbor 4: distance=1.2  -> weight=1/1.2=0.83  -> class 1
neighbor 5: distance=1.5  -> weight=1/1.5=0.67  -> class 0

Weighted P(class=1) = (3.33+2.00+0.83)/(3.33+2.00+1.11+0.83+0.67) = 6.16/7.94 = 0.78
Weighted P(class=0) = (1.11+0.67)/7.94 = 0.22
-> predict class 1 with higher confidence than uniform weights
```

**Computational cost:**
```
fit():     O(n)      -- just store data
predict(): O(n*d)    per test point (n=training rows, d=features)
           -> Slow for large training sets! n=100,000 means 100,000 distances per test point
```

## The Effect of K
```
K=1   -> Exact nearest neighbor -> training accuracy = 100% -> usually overfits
K=3   -> Vote among 3 -> noise starts to cancel out
K=5   -> Default, good balance (most used in practice)
K=15  -> Smooth boundary, less sensitive to noise
K=n   -> Always predicts majority class -> useless model
Tip: Use odd K for binary classification to avoid 50/50 tie votes
```

## Why Scaling is Critical for KNN
Without scaling, features with large numeric ranges dominate distance calculation.
Always apply StandardScaler or MinMaxScaler before fitting KNN.

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_neighbors` | `5` | Number of neighbors to consider |
| `weights` | `"uniform"` | `"uniform"` = all neighbors equal, `"distance"` = closer neighbors vote more |
| `metric` | `"minkowski"` | Distance metric. p=2 Euclidean, p=1 Manhattan |
| `algorithm` | `"auto"` | Data structure: `"ball_tree"`, `"kd_tree"`, `"brute"` |
| `n_jobs` | `None` | Parallel jobs for prediction |

## Real-life Example — Diagnosing Diabetes
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 400

glucose      = np.random.uniform(70, 200, n)
bmi          = np.random.uniform(18, 45, n)
age          = np.random.randint(20, 70, n)
has_diabetes = ((glucose > 140) | (bmi > 35)).astype(int)

X = np.column_stack([glucose, bmi, age])
y = has_diabetes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

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
**What fit(X_train, y_train) does -- Step by Step with actual Gini numbers**

Say you have 4 patients:
```
glucose  bmi   label
  70      20     0 (healthy)
  85      22     0 (healthy)
 150      30     1 (diabetic)
 160      38     1 (diabetic)
```

**Step 1 -- Calculate Gini at root node:**
```
Root has 4 samples: 2 class-0, 2 class-1
p0 = 2/4 = 0.50,  p1 = 2/4 = 0.50
Gini(root) = 1 - (0.50^2 + 0.50^2) = 1 - 0.50 = 0.50
```
Gini=0.50 = maximally impure. We need to split to reduce this.

**Step 2 -- Try every possible split on every feature**

For feature "glucose" (sorted unique values: 70, 85, 150, 160):
Candidate splits at midpoints: 77.5, 117.5, 155.0

**Split: glucose <= 117.5?**
```
Left  (glucose <= 117.5): samples 70->0, 85->0     = 2 class-0, 0 class-1
Right (glucose > 117.5):  samples 150->1, 160->1   = 0 class-0, 2 class-1

Gini(left)  = 1 - (1.0^2 + 0.0^2) = 0.00   <- PURE (all healthy)
Gini(right) = 1 - (0.0^2 + 1.0^2) = 0.00   <- PURE (all diabetic)

Weighted Gini = (2/4)*0.00 + (2/4)*0.00 = 0.00

Gini Gain = 0.50 - 0.00 = 0.50  <- maximum possible gain
```

**Split: glucose <= 77.5?**
```
Left  (glucose <= 77.5):  sample 70->0             = 1 class-0, 0 class-1
Right (glucose > 77.5):   samples 85->0, 150->1, 160->1 = 1 class-0, 2 class-1

Gini(left)  = 1 - (1.0^2 + 0.0^2) = 0.00
Gini(right) = 1 - ((1/3)^2 + (2/3)^2) = 1 - (0.111 + 0.444) = 0.444

Weighted Gini = (1/4)*0.00 + (3/4)*0.444 = 0.333

Gini Gain = 0.50 - 0.333 = 0.167  <- smaller gain, worse split
```

**Step 3 -- Choose the split with maximum Gini Gain**
glucose <= 117.5 wins with Gini Gain = 0.50

**Step 4 -- Recurse on left and right children**
Both children are already pure (Gini=0) -> no further splitting needed.

**Resulting tree:**
```
        glucose <= 117.5?
        /                     YES                NO
  Gini=0.00           Gini=0.00
  class=0 (certain)   class=1 (certain)
```

**Entropy as alternative criterion:**
```
Entropy(node) = -SUM( pk * log2(pk) )

Root: p0=0.5, p1=0.5
Entropy(root) = -(0.5*log2(0.5) + 0.5*log2(0.5))
              = -(0.5*(-1) + 0.5*(-1)) = 1.0

After glucose<=117.5 split:
  Entropy(left)  = 0.0  (pure)
  Entropy(right) = 0.0  (pure)
  Information Gain = 1.0 - (2/4*0 + 2/4*0) = 1.0

Gini vs Entropy: almost always give same tree. Gini is faster (no log). Use default Gini.
```

**What predict(X_test) does:**
```
New patient: glucose=130, bmi=28

Step 1: At root -> glucose <= 117.5?  -> 130 > 117.5  -> go RIGHT
Step 2: At right leaf -> majority class = 1 (diabetic)
Return: class 1
```
Pure traversal -- just follow if-else rules until you hit a leaf.

**What predict_proba(X_test) does:**
```
At the leaf reached, look at training samples in that leaf:
  Leaf has: [45 diabetic, 5 healthy]
  P(class=0) = 5/(45+5)  = 0.10
  P(class=1) = 45/(45+5) = 0.90
Return: [0.10, 0.90]
```

**Why trees overfit without max_depth:**
```
Without limit: tree grows until every leaf has exactly 1 training sample
-> Training accuracy = 100% (every sample in its own leaf)
-> Test accuracy poor (new samples fall into leaves built for 1 specific training point)

max_depth=5 -> at most 32 leaves, each leaf has several training samples -> generalizes
```

**What fit() stores:**
```
model.tree_                <- internal binary tree (nodes, rules, leaf values)
model.feature_importances_ <- total Gini reduction caused by each feature
model.classes_             <- unique class labels in internal order
model.n_classes_           <- number of classes
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `max_depth` | `None` | Max tree levels. None = fully grown (overfits) |
| `criterion` | `"gini"` | Split quality: `"gini"` or `"entropy"` |
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
print("\nFeature Importances:")
for name, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_flower)
proba = model.predict_proba(new_flower)
print("\nPredicted class:", iris.target_names[prediction[0]])
print("Probabilities:", proba[0])
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
**Technique 1: Bootstrap Sampling (Bagging)**
```
For each of the 100 trees:
  Take training data (e.g. 800 rows x 20 features)
  Sample 800 rows WITH REPLACEMENT -> bootstrap sample

  Bootstrap A: rows [0, 0, 3, 5, 5, 12, 7, ...]   <- row 0 twice, row 1 missing
  Bootstrap B: rows [1, 3, 3, 6, 0, 2, 8, ...]    <- completely different
  Bootstrap C: rows [7, 2, 0, 5, 5, 1, 3, ...]    <- another different sample

Result: each tree sees slightly different data -> makes different errors -> diverse ensemble
On average each bootstrap sample contains ~63% unique rows (37% are out-of-bag / OOB)
```

**Technique 2: Feature Randomness (Random Subspace)**
```
With 20 features: max_features = sqrt(20) = 4 (default for classification)

At node 1 of tree 7:   randomly pick [F3, F9, F14, F17]  -> best split on F9
At node 2 of tree 7:   randomly pick [F1, F5, F11, F18]  -> best split on F5
At node 1 of tree 23:  randomly pick [F2, F6, F13, F19]  -> best split on F13

Each tree uses different feature subsets at each node
-> Trees are uncorrelated -> averaging/voting is more powerful
```

**What fit(X_train, y_train) does -- Full Loop:**
```
for i in range(n_estimators=100):
  1. Draw bootstrap sample (800 rows from X_train, with replacement)
  2. Build DecisionTreeClassifier on that bootstrap sample:
       - At each node: randomly pick max_features=4 features
       - Split on the feature+threshold with highest Gini Gain
       - Keep splitting until max_depth or min_samples_leaf reached
  3. Store tree_i

After loop: model.estimators_ = list of 100 DecisionTreeClassifier objects
```

**What predict(X_test) does -- Majority Voting:**
```
New email: X_test[0] = [word_free=0.6, has_html=1, link_count=12, ...]

Tree 1   -> class 1 (Spam)
Tree 2   -> class 0 (Not Spam)
Tree 3   -> class 1 (Spam)
...
Tree 100 -> class 1 (Spam)

Count votes:  Spam=67,  Not Spam=33
Final prediction: class 1 (Spam)  <- majority winner
```

**What predict_proba(X_test) does:**
```
P(Not Spam) = 33/100 = 0.33
P(Spam)     = 67/100 = 0.67

predict_proba returns: [[0.33, 0.67]]
```

**Out-of-Bag (OOB) Score -- free validation:**
```
For each training sample, average predictions only from trees that did NOT train on it.
-> Free accuracy estimate without needing a separate test set.

model = RandomForestClassifier(oob_score=True, random_state=42)
model.fit(X_train, y_train)
print(model.oob_score_)   # accuracy on out-of-bag samples
```

**Feature Importance:**
```
model.feature_importances_[i] =
    (total Gini reduction caused by feature i across all splits in all 100 trees)
    / (total Gini reduction from all features in all trees)

Values sum to 1.0.   Higher value = more important feature.
```

**What fit() stores:**
```
model.estimators_          <- list of 100 fitted DecisionTreeClassifier objects
model.feature_importances_ <- importance array (sums to 1.0)
model.classes_             <- class labels in internal order
model.oob_score_           <- OOB accuracy (only if oob_score=True)
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `n_estimators` | `100` | Number of trees |
| `max_depth` | `None` | Max depth per tree |
| `max_features` | `"sqrt"` | Features per split: sqrt(n_features) for classification |
| `class_weight` | `None` | Use `"balanced"` for imbalanced datasets |
| `oob_score` | `False` | Compute free OOB accuracy estimate |
| `n_jobs` | `None` | `-1` uses all CPU cores |
| `random_state` | `None` | Seed |

## Real-life Example — Email Spam Detection
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Test Accuracy:", model.score(X_test, y_test))
print("OOB Accuracy: ", model.oob_score_)

importances = model.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("\nTop 5 most important features:")
for i in top5_idx:
    print(f"  Feature {i:2d}: {importances[i]:.4f}")

sample = X_test[0].reshape(1, -1)
print("\nSample prediction:", model.predict(sample)[0])
print("Probability [class0, class1]:", model.predict_proba(sample)[0])
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
**The Core Idea -- Maximum Margin**

Many hyperplanes can separate two classes. SVC finds the one that:
1. Separates classes correctly
2. Has the **maximum margin** -- is farthest from the nearest points of both classes

```
Class 0 ( * ):              Class 1 ( o ):

   *  *  *  *               o  o  o  o
      [*] ---- boundary ---- [o]
   *  *  *  *               o  o  o  o

   Support vectors:  [*] and [o] -- the closest points to the boundary on each side.

Margin = distance from boundary to each support vector line.
SVC goal: maximise this margin width.
```

**Support Vectors -- what they are:**
```
Support vectors are the ONLY training points that define the boundary.
All other training points could be removed and the model would not change at all.

This is why SVC stores model.support_vectors_ (a subset of X_train)
-- typically only 5-20% of training data are support vectors.
```

**The optimization problem:**
```
Minimize:    (1/2) * ||w||^2                    (wide margin = small w)
Subject to:  yi * (w.dot(xi) + b) >= 1          (all points correctly classified)
```

**Parameter C -- Soft Margin (real data has overlapping classes):**
```
High C (e.g. C=100):
  "Classify EVERY training point correctly even if margin is very narrow"
  -> Very narrow margin -> complex boundary -> risk of overfitting

Low C (e.g. C=0.01):
  "Allow some misclassifications to achieve a wider, more general margin"
  -> Wide margin -> simpler boundary -> better generalization

C=1.0 default: balanced
```

**Kernels -- Handling Non-linear Data:**

If classes are not linearly separable, kernels transform data into higher dimensions
where a hyperplane CAN separate them.

```
Original 2D (no line works):    After kernel transform (3D):
   o  *  o  *                      o  o  o  (on one layer)
   *  o  *  o                      *  *  *  (on another layer)
   o  *  o  *                  -> a flat plane separates them perfectly
```

The "kernel trick": compute these high-dimensional dot products WITHOUT actually
transforming the data points (computationally feasible this way).

| Kernel | When to Use | Formula |
|--------|-------------|---------|
| `"linear"` | Linearly separable, high features | K(x,x') = x dot x' |
| `"rbf"` | Non-linear data (DEFAULT) | K(xi,xj) = exp(-gamma * ||xi-xj||^2) |
| `"poly"` | Polynomial pattern | K(x,x') = (gamma*x dot x' + r)^degree |
| `"sigmoid"` | Neural network-like | K(x,x') = tanh(gamma*x dot x' + r) |

**RBF Kernel -- what gamma controls:**
```
K(xi, xj) = exp( -gamma * ||xi - xj||^2 )

High gamma (e.g. gamma=10):
  Kernel drops to 0 quickly as distance increases
  -> Each training point influences only a tiny nearby region
  -> Very complex, wiggly boundary -> likely overfits

Low gamma (e.g. gamma=0.001):
  Kernel stays non-zero over large distances
  -> Each training point influences a wide region
  -> Very smooth boundary -> may underfit

gamma="scale" (default) = 1 / (n_features * X.var())  <- automatic good choice
gamma="auto"            = 1 / n_features
```

**What fit(X_train, y_train) does -- Step by Step:**
```
Step 1: Validate input. Scale expected (SVC very sensitive to feature scale).

Step 2: Compute the kernel matrix K of shape (n_train x n_train):
        K[i,j] = kernel(X_train[i], X_train[j])
        For RBF: K[i,j] = exp(-gamma * ||X_train[i] - X_train[j]||^2)
        This matrix captures similarity between every pair of training points.

Step 3: Solve the quadratic programming optimization problem (using libsvm internally):
        Find dual coefficients alpha_i for each training point.
        Most alpha_i = 0  -> these points are NOT support vectors (far from boundary)
        A few alpha_i > 0 -> these ARE support vectors (close to / on boundary)

Step 4: Store only the support vectors:
        model.support_vectors_  <- X_train rows where alpha_i > 0
        model.dual_coef_        <- alpha_i * y_i for each support vector
        model.intercept_        <- bias b
        model.support_          <- indices into X_train of support vectors
        model.n_support_        <- number of support vectors per class e.g. [45, 52]
```

**What predict(X_test) does:**
```
New point: x_new = X_test[0]

Step 1: Compute kernel between x_new and EVERY support vector (only these, not all of X_train):
        k_i = kernel(x_new, support_vectors_[i])

Step 2: Decision function:
        f(x) = SUM_i( dual_coef_[i] * k_i ) + intercept_

Step 3: Apply sign:
        if f(x) >= 0 -> class +1 (positive class)
        if f(x) <  0 -> class -1 (negative class)
```

**Why SVC is slow on large datasets:**
```
Kernel matrix K is n_train x n_train
For n_train=100,000: K = 10 billion entries -> 80GB RAM
Solving the QP problem is O(n^2) to O(n^3)
-> SVC impractical for n > ~100,000 rows

Alternative: LinearSVC (uses liblinear, no kernel, much faster for large n)
```

**What fit() stores:**
```
model.support_          <- indices into X_train of support vectors
model.support_vectors_  <- actual X values of support vectors (subset of X_train)
model.n_support_        <- [n_sv_class0, n_sv_class1] number per class
model.dual_coef_        <- alpha_i * y_i weights for each support vector
model.intercept_        <- bias b
```

## What Does it Output?
```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)         # [0 1 0 1 1 ...]
proba = model.predict_proba(X_test)         # [[0.82, 0.18], ...]  (needs probability=True)
print("Support vector counts:", model.n_support_)   # e.g. [45, 52]
print("Support vector indices:", model.support_[:5])
```

## Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `C` | `1.0` | Regularization. High = fit training better, may overfit |
| `kernel` | `"rbf"` | Type of kernel function |
| `gamma` | `"scale"` | RBF kernel coefficient |
| `degree` | `3` | Degree for polynomial kernel only |
| `probability` | `False` | Set True to enable predict_proba() |
| `class_weight` | `None` | Use `"balanced"` for imbalanced data |

**Note:** SVC is slow for n > 100k rows. Use `LinearSVC` for large datasets.

## Real-life Example — Cancer Classification
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target   # 569 samples, 30 features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

for kernel in ["linear", "rbf", "poly"]:
    svc = SVC(kernel=kernel, C=1.0, probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    acc = svc.score(X_test_s, y_test)
    nsv = sum(svc.n_support_)
    print(f"kernel={kernel:8s}  Accuracy={acc:.4f}  SupportVectors={nsv}")

best = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
best.fit(X_train_s, y_train)

new_pt = scaler.transform(X_test[0].reshape(1, -1))
print("\nPrediction:", cancer.target_names[best.predict(new_pt)[0]])
print("Probabilities:", best.predict_proba(new_pt)[0])
print("Support vector counts per class:", best.n_support_)
```

## Day 4 — Full Comparison
| Model | Training Speed | Prediction Speed | Needs Scaling | Handles Non-linear | Probability Output |
|-------|--------------|-----------------|--------------|-------------------|-------------------|
| `LogisticRegression` | Fast | Fast | Yes | No | Yes (built-in) |
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

X, y = make_classification(
    n_samples=1000, n_features=8, n_informative=5, n_redundant=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "LogisticRegression":      LogisticRegression(max_iter=1000),
    "KNN(k=5)":                KNeighborsClassifier(n_neighbors=5),
    "DecisionTree(depth=5)":   DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest(100 trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVC(rbf)":                SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

print(f"{'Model':<28}  {'Accuracy':>10}")
print("-" * 42)
for name, m in models.items():
    m.fit(X_train_s, y_train)
    acc = m.score(X_test_s, y_test)
    print(f"{name:<28}  {acc:>10.4f}")
```

> Day 4 Complete.
> You now understand how all 5 classification models work internally -- what fit() and predict() do step by step.
>
> Next: **Day 5 -- Model Evaluation**
> (MAE, MSE, RMSE, R-squared, Confusion Matrix, Precision, Recall, F1, ROC-AUC)
