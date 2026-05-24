# 📅 Day 7 — Advanced Scikit-learn
## What You Will Learn Today
| Tool | What it does |
|------|-------------|
| `Pipeline` | Chain preprocessing + model into one object |
| `make_pipeline` | Shortcut to build Pipeline without naming steps |
| `ColumnTransformer` | Apply different preprocessing to different columns |
| Feature Selection | Remove useless features automatically |
| `PCA` | Reduce number of features while keeping information |
| `KMeans` | Group data into clusters without labels |
| `joblib` | Save and load a trained model to disk |

# Pipeline
## What is it?
A `Pipeline` chains multiple steps — preprocessing + model — into a **single object**
that you can fit, predict, and evaluate like any normal sklearn model.
### The Problem Without Pipeline
```python
# WRONG way — causes data leakage in cross validation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learns mean/std from train
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
model.score(X_test_scaled, y_test)
```

This looks correct, but if you use `cross_val_score`:

```python
# BUG — scaler was fit on ALL data before the CV loop started
# The test fold's information leaked into the scaler
cross_val_score(model, X_train_scaled, y_train, cv=5)
```
### The Solution — Pipeline
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression())
])
# Now cross_val_score is correct — scaler is fit inside each fold
cross_val_score(pipe, X, y, cv=5)
```

Pipeline guarantees that preprocessing only sees training data in each fold.

## Syntax
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ("step_name_1", TransformerObject1()),
    ("step_name_2", TransformerObject2()),
    ("step_name_n", ModelObject())    # last step must be an estimator
])
```

Each step is a tuple: `("name", object)`.
You choose the names — they are used to access steps later.

## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `steps` | list of (name, object) tuples | Transformers first, estimator last |
| `memory` | path or None | Cache fitted transformers (speeds up GridSearchCV) |
| `verbose` | bool | Print step names during fit |

Rules:
- All steps except the last must implement `fit_transform()` (transformers)
- The last step must implement `fit()` and `predict()` (estimator)
- Names must be unique and contain no double underscores `__`

## What Happens Inside — Step by Step
### During `pipe.fit(X_train, y_train)`:
```
Step 1 — scaler.fit_transform(X_train)      → scaled X_train
Step 2 — imputer.fit_transform(scaled)      → imputed X_train
Step 3 — model.fit(imputed, y_train)        → model learns
```

Each step receives the **output of the previous step** as input.
### During `pipe.predict(X_test)`:
```
Step 1 — scaler.transform(X_test)           ← uses stats learned from X_train
Step 2 — imputer.transform(scaled_test)     ← uses fill values from X_train
Step 3 — model.predict(imputed_test)        → final predictions
```

Note: During predict, each step uses `.transform()` only — not `.fit_transform()`.
The pipeline never refits on test data.

## What Does it Output?
`pipe.fit()` → returns the fitted pipeline (same object)
`pipe.predict()` → returns predictions (same as model.predict)
`pipe.score()` → returns score (same as model.score)
`pipe.named_steps["scaler"]` → access any individual step

```python
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))        # e.g. 0.9473
# Access internal step after fitting
scaler_used = pipe.named_steps["scaler"]
print(scaler_used.mean_)                 # learned mean values
```

## Parameters Table
| Method | What it does |
|--------|-------------|
| `pipe.fit(X, y)` | Fit all steps in order |
| `pipe.predict(X)` | Transform then predict |
| `pipe.predict_proba(X)` | Transform then predict_proba |
| `pipe.score(X, y)` | Transform then score |
| `pipe.named_steps` | Dict of all steps by name |
| `pipe.steps` | List of all (name, object) tuples |
| `pipe[0]` or `pipe["scaler"]` | Access step by index or name |

## Real-life Example
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
# Introduce some NaN to simulate real data
rng = np.random.default_rng(42)
X_noisy = X.copy().astype(float)
X_noisy[rng.integers(0, X.shape[0], 30), rng.integers(0, X.shape[1], 30)] = np.nan

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
# Build pipeline: fill NaN → scale → classify
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
print("Test Score:", pipe.score(X_test, y_test))
# CV is now fully correct — no leakage
cv_scores = cross_val_score(pipe, X_noisy, y, cv=5, scoring="accuracy")
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

# make_pipeline
## What is it?
`make_pipeline` is a **shortcut** for creating a Pipeline.
It auto-generates step names from the class names — you do not need to name them manually.

## Syntax
```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
```

This is exactly the same as:

```python
Pipeline([
    ("standardscaler",    StandardScaler()),
    ("logisticregression", LogisticRegression(max_iter=1000))
])
```

The names are generated as lowercase class names.

## Pipeline vs make_pipeline
| | `Pipeline` | `make_pipeline` |
|--|-----------|-----------------|
| Name each step | Yes, manually | No, auto-generated |
| Access step by name | `pipe.named_steps["scaler"]` | `pipe.named_steps["standardscaler"]` |
| GridSearchCV param names | `"scaler__with_mean"` | `"standardscaler__with_mean"` |
| Best for | Complex pipelines, GridSearchCV | Quick pipelines |

## What Does it Output?
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
pipe.fit(X_train, y_train)

print("Steps:", pipe.named_steps)
# {'standardscaler': StandardScaler(), 'randomforestclassifier': RandomForestClassifier(...)}

print("Score:", pipe.score(X_test, y_test))
# 0.9667
```

# ColumnTransformer
## What is it?
`ColumnTransformer` applies **different transformations to different columns** in one step.

Real datasets have mixed types:
- Numeric columns (age, salary) → need StandardScaler
- Categorical columns (city, gender) → need OneHotEncoder

Without ColumnTransformer you must split, process separately, and manually recombine.
ColumnTransformer handles it all in one object.

## Syntax
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),  numeric_columns),
    ("cat", OneHotEncoder(),   categorical_columns)
])
```

Each transformer is a tuple: `("name", transformer, columns)`.

## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `transformers` | list of (name, transformer, columns) | Define what to apply where |
| `remainder` | `"drop"` or `"passthrough"` | What to do with columns not mentioned |
| `sparse_threshold` | float | When to return sparse matrix |

`columns` can be:
- A list of column names: `["age", "salary"]`
- A list of integer indices: `[0, 1, 2]`
- A boolean mask

## What Happens Inside — Step by Step
```
Input DataFrame:
  age   salary   city      gender
  25    50000    Delhi     Male
  32    80000    Mumbai    Female
  28    60000    Chennai   Male

Numeric columns: ["age", "salary"]
Categorical columns: ["city", "gender"]

Step 1: StandardScaler fits on age and salary → z-scores
Step 2: OneHotEncoder fits on city and gender → binary columns

Output (after transform):
  age_scaled  salary_scaled  city_Delhi  city_Mumbai  city_Chennai  gender_Male  gender_Female
  -1.22       -1.22          1           0             0             1            0
   1.07        1.07          0           1             0             0            1
   0.15        0.15          0           0             1             1            0
```

Columns that are not in any transformer are handled by `remainder`:
- `"drop"` → removed (default)
- `"passthrough"` → kept as-is

## What Does it Output?
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

data = pd.DataFrame({
    "age":    [25, 32, 28, 45],
    "salary": [50000, 80000, 60000, 120000],
    "city":   ["Delhi", "Mumbai", "Chennai", "Delhi"],
    "gender": ["Male", "Female", "Male", "Female"]
})

numeric_cols     = ["age", "salary"]
categorical_cols = ["city", "gender"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),             numeric_cols),
    ("cat", OneHotEncoder(drop="first"),  categorical_cols)
])

X_transformed = preprocessor.fit_transform(data)
print("Shape before:", data.shape)           # (4, 4)
print("Shape after:", X_transformed.shape)  # (4, 5)  — 2 scaled + 3 OHE columns
```

## Using ColumnTransformer Inside a Pipeline
This is the most common real-world pattern:

```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Sample real-world-style dataset
data = pd.DataFrame({
    "age":    [25, 32, np.nan, 45, 28, 50, 35, 22],
    "salary": [50000, 80000, 60000, 120000, 55000, 95000, 70000, 40000],
    "city":   ["Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai"],
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
    "bought": [0, 1, 0, 1, 1, 1, 0, 0]
})

X = data.drop("bought", axis=1)
y = data["bought"]

numeric_cols     = ["age", "salary"]
categorical_cols = ["city", "gender"]
# Numeric: fill NaN → scale
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
# Categorical: fill NaN → encode
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])
# Full pipeline
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

full_pipeline.fit(X_train, y_train)
print("Score:", full_pipeline.score(X_test, y_test))
```

# Feature Selection
## What is it?
Feature Selection removes **useless or redundant features** from your dataset.

Too many features causes:
- Overfitting — model learns noise
- Slower training
- Worse generalization

sklearn has several automatic methods:

| Method | What it does |
|--------|-------------|
| `SelectKBest` | Keep top K features by statistical score |
| `SelectPercentile` | Keep top N% features |
| `RFE` (Recursive Feature Elimination) | Remove features one by one using model importance |
| `SelectFromModel` | Keep features above importance threshold from any model |

## SelectKBest
### What is it?
Scores every feature using a statistical test and keeps the top K.
### Syntax
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train)
```
### Score Functions
| score_func | Use for |
|------------|---------|
| `f_classif` | Classification with numeric features (F-test ANOVA) |
| `chi2` | Classification with non-negative integer features |
| `f_regression` | Regression (F-test) |
| `mutual_info_classif` | Classification, captures non-linear relationships |
### What Does it Output?
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

print("Original shape:", X.shape)      # (569, 30)
print("Selected shape:", X_new.shape)  # (569, 10)
# Which features were selected?
selected_mask = selector.get_support()
selected_features = np.array(cancer.feature_names)[selected_mask]
print("Selected features:", selected_features)
# Feature scores
scores = selector.scores_
print("All feature scores:", np.round(scores, 1))
```

## SelectFromModel
### What is it?
Uses a model's feature importances (like `feature_importances_` from Random Forest)
to automatically drop features below a threshold.
### Syntax
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean"    # keep features with importance > mean importance
)
selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)
```
### What Does it Output?
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean"
)
selector.fit(X, y)
X_selected = selector.transform(X)

print("Original:", X.shape)           # (569, 30)
print("After selection:", X_selected.shape)

kept = cancer.feature_names[selector.get_support()]
print("Kept features:", kept)
```

## Feature Selection Inside a Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

pipe = Pipeline([
    ("scaler",    StandardScaler()),
    ("selector",  SelectKBest(score_func=f_classif, k=10)),
    ("model",     LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

# PCA — Principal Component Analysis
## What is it?
PCA is a **dimensionality reduction** technique.
It transforms many correlated features into a smaller set of uncorrelated components
while keeping as much information (variance) as possible.

Use PCA when:
- You have too many features (hundreds or thousands)
- Features are correlated (redundant)
- You want to speed up training
- You want to visualize high-dimensional data in 2D or 3D

## Syntax
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)   # reduce to 2 components
X_reduced = pca.fit_transform(X)
```

## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `n_components` | int or float | Number of components to keep. If float (e.g. 0.95) → keep enough components to explain 95% of variance |
| `whiten` | bool | Normalize component variances |
| `random_state` | int | For reproducibility |

## What Happens Inside — Step by Step
### Step 1 — Center the data
`X_centered = X - X̄`

Subtract the mean of each feature so the data is centered at the origin.
### Step 2 — Compute the Covariance Matrix
`C = (1/(n-1)) × X_centeredᵀ × X_centered`

This matrix captures how much pairs of features vary together.
### Step 3 — Find Eigenvectors and Eigenvalues
Solve:

`C × v = λ × v`

- Each **eigenvector** ($\mathbf{v}$) is a direction in feature space
- Each **eigenvalue** ($\lambda$) tells how much variance lies in that direction
- Sort by eigenvalue descending → principal components in order of importance
### Step 4 — Project data onto top K eigenvectors
`X_reduced = X_centered × Vₖ`

Where $V_k$ is the matrix of the top K eigenvectors.
### Example
```
Original: 30 features (breast cancer dataset)

PCA finds 30 directions (principal components).
PC1 explains 44% of variance
PC2 explains 19% of variance
PC3 explains 9% of variance
...
First 10 PCs explain 95% of total variance.

After PCA(n_components=10): 30 features → 10 components
Information kept: 95%
Training is faster and less overfitting.
```

## What Does it Output?
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
X = cancer.data

pca = PCA(n_components=0.95)   # keep 95% of variance
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)           # (569, 30)
print("Reduced shape: ", X_reduced.shape)   # (569, 10) — may vary

print("Components kept:", pca.n_components_)
print("Variance explained per component:", np.round(pca.explained_variance_ratio_, 3))
print("Total variance kept:", pca.explained_variance_ratio_.sum().round(3))
```

## Parameters
| Attribute | What it tells you |
|-----------|------------------|
| `pca.n_components_` | Number of components actually used |
| `pca.explained_variance_ratio_` | Fraction of variance each component explains |
| `pca.components_` | The eigenvectors (direction of each PC) |
| `pca.explained_variance_` | Variance (eigenvalue) of each component |

## Visualizing Data with PCA (2D plot)
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
# Always scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(8, 5))
colors = ["red", "green", "blue"]
for class_id, color in enumerate(colors):
    mask = y == class_id
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=iris.target_names[class_id], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris Dataset — PCA 2D Projection")
plt.legend()
plt.tight_layout()
plt.show()
```

## PCA Inside a Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=0.95)),       # keep 95% variance
    ("model",  LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy with PCA: {scores.mean():.4f} ± {scores.std():.4f}")
```

# KMeans — Clustering
## What is it?
KMeans is an **unsupervised learning** algorithm that groups data into K clusters.

Unsupervised means: **there is no y (target label)**.
The model finds natural groupings in the data by itself.

Real-life uses:
- Customer segmentation (group buyers by behavior)
- Image compression (group similar pixel colors)
- Anomaly detection (samples far from any cluster)
- Document grouping (group similar articles)

## Syntax
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_         # cluster assignment per sample
centers = kmeans.cluster_centers_  # center of each cluster
```

## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `n_clusters` | int | Number of clusters K to form |
| `init` | `"k-means++"` or `"random"` | How to initialize centroids |
| `n_init` | int | Times to run with different seeds (best result kept) |
| `max_iter` | int | Max iterations per run |
| `random_state` | int | For reproducibility |

## What Happens Inside — Step by Step
### The KMeans Algorithm
**Step 1 — Initialize:** Randomly place K centroids in the feature space.
(With `init="k-means++"`, centroids are spread out intelligently for better start.)

**Step 2 — Assign:** For each data point, find the nearest centroid using Euclidean distance:

`d(x, c) = √( Σ(xⱼ - cⱼ)² )`

Assign the point to that centroid's cluster.

**Step 3 — Update:** Move each centroid to the **mean position** of all points assigned to it:

`cₖ = (1/|Cₖ|) × Σ x   (mean of all points in cluster k)`

**Step 4 — Repeat:** Go back to Step 2. Continue until centroids stop moving (convergence).
### Visual walkthrough
```
Initial:   K=3 centroids placed randomly
           Data points scattered in 2D space

Iteration 1:
  Each point assigned to nearest centroid
  Centroids move to mean of their assigned points

Iteration 2:
  Assignments may change as centroids moved
  Centroids move again

...converges (centroids barely move)

Final clusters: 3 groups
```

KMeans minimizes the **Within-Cluster Sum of Squares (WCSS)**:

`WCSS = Σₖ Σ(x ∈ Cₖ) ||x - cₖ||²`

## What Does it Output?
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1, 2], [1, 4], [1, 0],   # cluster 1
    [10, 2], [10, 4], [10, 0] # cluster 2
])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

print("Cluster labels:", kmeans.labels_)
# [1 1 1 0 0 0]  — each point's cluster

print("Cluster centers:")
print(kmeans.cluster_centers_)
# [[10.  2.]
#  [ 1.  2.]]

print("Inertia (WCSS):", kmeans.inertia_)
# Lower = tighter clusters
# Predict cluster for new data
new_point = np.array([[0, 0]])
print("New point cluster:", kmeans.predict(new_point))
# [1]
```

## Choosing the Right K — Elbow Method
There is no y to tell you the right K. Use the **Elbow Method**:

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method — Finding Optimal K")
plt.xticks(K_range)
plt.tight_layout()
plt.show()
# Look for the "elbow" — where inertia stops dropping sharply
# That K is the optimal choice
```

## Real-life Example — Customer Segmentation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Simulated customer data: annual income vs spending score
np.random.seed(42)
income   = np.concatenate([np.random.normal(30, 5, 50),
                            np.random.normal(60, 5, 50),
                            np.random.normal(90, 5, 50)])
spending = np.concatenate([np.random.normal(70, 5, 50),
                            np.random.normal(40, 5, 50),
                            np.random.normal(80, 5, 50)])

X = np.column_stack([income, spending])
# Scale first (KMeans is distance-based — scale matters!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
# Plot
colors = ["red", "blue", "green"]
plt.figure(figsize=(8, 5))
for k in range(3):
    mask = labels == k
    plt.scatter(X[mask, 0], X[mask, 1],
                c=colors[k], label=f"Segment {k+1}", alpha=0.7)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation — KMeans (K=3)")
plt.legend()
plt.tight_layout()
plt.show()

print("Cluster sizes:", np.bincount(labels))
```

# joblib — Save and Load Models
## What is it?
`joblib` is a library that **saves a trained model to a file** so you can load it
later without retraining.

In production, you train a model once and then deploy it.
Saving and loading with joblib is how you do that in sklearn.

## Syntax
```python
import joblib
# Save
joblib.dump(model, "model_filename.pkl")
# Load
model = joblib.load("model_filename.pkl")
```

## What Input Does dump() Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `value` | any Python object | The trained model, pipeline, scaler, etc. |
| `filename` | str or path | File path to save to (`.pkl` or `.joblib` extension) |
| `compress` | int 0–9 | Compression level. 0=no compress, 3=good balance |

## What Happens Inside
`joblib.dump()` **serializes** the Python object — converts it from in-memory object
to a binary file that can be saved on disk.

`joblib.load()` **deserializes** it back — reads the binary file and recreates the Python object
with all its learned parameters (weights, means, thresholds, etc.) intact.

## What Does it Output?
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Score before save:", model.score(X_test, y_test))   # e.g. 1.0
# Save to disk
joblib.dump(model, "iris_rf_model.pkl")
print("Model saved.")
# --- Later, in another script or session ---
# Load from disk
loaded_model = joblib.load("iris_rf_model.pkl")
print("Score after load:", loaded_model.score(X_test, y_test))   # same: 1.0
# Predict with loaded model
import numpy as np
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(new_flower)
print("Predicted class:", iris.target_names[prediction[0]])   # e.g. "setosa"
```

## Saving a Full Pipeline
You should always save the **full pipeline** (preprocessing + model), not just the model.
This ensures the exact same transformations are applied at prediction time.

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)
# Build and train full pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=100, random_state=42))
])
pipe.fit(X_train, y_train)
print("Pipeline score:", pipe.score(X_test, y_test))
# Save the ENTIRE pipeline
joblib.dump(pipe, "cancer_pipeline.pkl", compress=3)
print("Pipeline saved.")
# Load and use in production
loaded_pipe = joblib.load("cancer_pipeline.pkl")
predictions = loaded_pipe.predict(X_test)
print("Loaded pipeline predictions:", predictions[:10])
print("Loaded pipeline score:", loaded_pipe.score(X_test, y_test))
```

## Parameters
| Parameter | Detail |
|-----------|--------|
| Extension | Use `.pkl` or `.joblib` (both work, `.joblib` is more explicit) |
| `compress=3` | Good balance of file size and speed |
| `compress=9` | Maximum compression, slowest |
| `compress=0` | No compression, fastest, largest file |

## Important Notes
```
1. The sklearn version must be the same (or compatible) when loading.
   A model saved with sklearn 1.2 may not load in sklearn 0.24.

2. Never load .pkl files from untrusted sources.
   Pickle files can execute arbitrary code on load. Only load files you created.

3. Save the scaler/pipeline too — not just the model.
   If you only save the model, you cannot reproduce the preprocessing step.
```

## Day 7 — All Tools at a Glance
| Tool | Category | What it solves |
|------|----------|----------------|
| `Pipeline` | Workflow | Chains steps, prevents data leakage |
| `make_pipeline` | Workflow | Shortcut for Pipeline |
| `ColumnTransformer` | Preprocessing | Different transforms per column |
| `SelectKBest` | Feature Selection | Keep top K statistical features |
| `SelectFromModel` | Feature Selection | Keep features by model importance |
| `PCA` | Dimensionality Reduction | Compress many features into fewer |
| `KMeans` | Clustering (Unsupervised) | Group unlabeled data into K clusters |
| `joblib` | Deployment | Save and load trained models |

## Complete Day 7 Project — End-to-End ML System
```python
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
# ── 1. Create a mixed dataset ──────────────────────────────
np.random.seed(42)
n = 500
data = pd.DataFrame({
    "age":      np.random.randint(18, 65, n).astype(float),
    "income":   np.random.normal(60000, 20000, n),
    "score":    np.random.normal(650, 80, n),
    "city":     np.random.choice(["Delhi", "Mumbai", "Chennai", "Kolkata"], n),
    "edu":      np.random.choice(["UG", "PG", "PhD"], n),
    "approved": np.random.choice([0, 1], n, p=[0.4, 0.6])
})
# Introduce some missing values
data.loc[np.random.choice(n, 30), "age"]    = np.nan
data.loc[np.random.choice(n, 20), "income"] = np.nan

X = data.drop("approved", axis=1)
y = data["approved"]
# ── 2. Define column groups ────────────────────────────────
numeric_cols     = ["age", "income", "score"]
categorical_cols = ["city", "edu"]
# ── 3. Build preprocessing for each column type ───────────
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipe,      numeric_cols),
    ("cat", categorical_pipe,  categorical_cols)
])
# ── 4. Full pipeline with feature selection + model ────────
full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("selector",     SelectKBest(score_func=f_classif, k=8)),
    ("model",        RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ── 5. Cross validation before tuning ─────────────────────
cv_scores = cross_val_score(full_pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"Baseline CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# ── 6. GridSearchCV ────────────────────────────────────────
param_grid = {
    "selector__k":              [5, 8, "all"],
    "model__n_estimators":      [100, 200],
    "model__max_depth":         [None, 10],
}

grid = GridSearchCV(full_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\nBest Params: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.4f}")
# ── 7. Final test evaluation ───────────────────────────────
best_pipe = grid.best_estimator_
y_pred = best_pipe.predict(X_test)
print("\nTest Evaluation:")
print(classification_report(y_test, y_pred))
# ── 8. Save full pipeline ──────────────────────────────────
joblib.dump(best_pipe, "loan_approval_pipeline.pkl", compress=3)
print("Pipeline saved to loan_approval_pipeline.pkl")
# ── 9. Load and verify ────────────────────────────────────
loaded = joblib.load("loan_approval_pipeline.pkl")
print("Loaded pipeline score:", loaded.score(X_test, y_test))
```

> Day 7 Complete.
> You now know the full sklearn toolkit from raw data to deployed model:
>
> - Pipeline / make_pipeline → clean, leak-free workflows
> - ColumnTransformer → handle mixed data types correctly
> - Feature Selection → drop useless features automatically
> - PCA → compress high-dimensional data
> - KMeans → discover natural groups without labels
> - joblib → save and reload trained models for production
>
> All 7 Days Complete. You have covered the entire core of scikit-learn.