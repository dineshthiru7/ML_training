# 📅 Day 2 — Data Preprocessing
## Why Preprocessing?
Raw data is almost never clean. Before training any model you must fix:

| Problem | Example | Tool |
|---------|---------|------|
| Missing values | Age column has NaN | `SimpleImputer` |
| Text categories | "Male"/"Female" as strings | `LabelEncoder` |
| Multiple categories | City: "Delhi", "Mumbai", "Chennai" | `OneHotEncoder` |
| Features on different scales | Age: 25, Salary: 50000 | `StandardScaler` |
| Need values strictly 0 to 1 | Pixel values, neural nets | `MinMaxScaler` |

If you skip preprocessing:
- Models give wrong results
- Distance-based models (KNN, SVM) get dominated by large-scale features
- Tree models fail on NaN values
- Linear models fail on string inputs

# SimpleImputer
## What is SimpleImputer?
`SimpleImputer` fills in **missing values (NaN)** in your dataset with a calculated replacement value.

Most sklearn models throw an error if they see NaN.
SimpleImputer solves this by replacing NaN with mean, median, most frequent value, or a constant.

## Syntax
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)
X_train_clean = imputer.transform(X_train)
X_test_clean  = imputer.transform(X_test)
```

## Parameters
| Parameter | Options | What it does |
|-----------|---------|--------------|
| `strategy` | `"mean"` | Replace NaN with column average |
| | `"median"` | Replace NaN with column median |
| | `"most_frequent"` | Replace NaN with most common value |
| | `"constant"` | Replace NaN with a fixed value |
| `fill_value` | any value | Used only when strategy="constant" |
| `missing_values` | `np.nan` | What to treat as missing (default NaN) |

## What Input Does fit() Accept?
```python
import numpy as np

X_train = np.array([
    [25,  50000],
    [30,  np.nan],   # missing salary
    [np.nan, 60000], # missing age
    [22,  45000],
])
# shape: (4, 2)
# NaN values are present in both columns
```

## What Happens Inside — Step by Step
### fit(X_train):
1. Scans every column of X_train
2. For each column, computes the statistic based on `strategy`:
   - `"mean"` → calculates `np.nanmean()` ignoring NaN
   - `"median"` → calculates `np.nanmedian()`
   - `"most_frequent"` → finds the value that appears most times
3. Stores these computed values in `imputer.statistics_`

```
For X_train above with strategy="mean":
Column 0 (age):    mean of [25, 30, 22]    = 25.67
Column 1 (salary): mean of [50000, 60000, 45000] = 51666.67

imputer.statistics_ = [25.67, 51666.67]
```
### transform(X):
1. Goes through every cell in X
2. Wherever it finds NaN → replaces with the stored statistic for that column
3. Returns a new array with no NaN values

```
Input:
[[25,    50000  ],
 [30,    NaN    ],   ← NaN salary → replaced with 51666.67
 [NaN,   60000  ],   ← NaN age   → replaced with 25.67
 [22,    45000  ]]

Output:
[[25,    50000  ],
 [30,    51666.67],
 [25.67, 60000  ],
 [22,    45000  ]]
```

## What Does it Output?
```python
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)

print(imputer.statistics_)
# [25.666...  51666.666...]  ← computed fill values per column

X_clean = imputer.transform(X_train)
print(type(X_clean))   # <class 'numpy.ndarray'>
print(X_clean)
# No NaN values in output
```

## Important Rule — fit on train, transform both
```python
# CORRECT
imputer.fit(X_train)              # learn statistics from training data ONLY
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)   # use SAME statistics for test
# WRONG — data leakage
imputer.fit(X_test)    # Never fit on test data
```

**Why?** If you fit on test data, the model indirectly sees test information during training.
That is called **data leakage** and gives falsely high scores.

## Real-life Example
**Scenario:** Hospital patient dataset. Some patients didn't record their age or blood pressure.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

data = {
    "age":      [45, np.nan, 55, 38, np.nan, 60, 42],
    "bp":       [80, 90, np.nan, 70, 85, np.nan, 78],
    "cholesterol": [200, 220, 180, np.nan, 210, 190, 205]
}
df = pd.DataFrame(data)
print("Before:\n", df)

X = df.values
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
# Use median (better for skewed data like age or BP)
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)

print("\nMedian values learned:", imputer.statistics_)

X_train_clean = imputer.transform(X_train)
X_test_clean  = imputer.transform(X_test)

print("\nAfter imputation:\n", X_train_clean)
```

## When to Use Which Strategy?
| Strategy | Use When |
|----------|---------|
| `mean` | Data is normally distributed (no extreme outliers) |
| `median` | Data has outliers (salary, house price) |
| `most_frequent` | Categorical columns with NaN |
| `constant` | You want to explicitly mark missing as a special value like 0 or "Unknown" |

# LabelEncoder
## What is LabelEncoder?
`LabelEncoder` converts **text/string category labels into numbers**.

```
"cat" → 0
"dog" → 1
"fish" → 2
```

Sklearn models only understand numbers. LabelEncoder gives each unique category a unique integer.

## Syntax
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)
y_train_encoded = le.transform(y_train)
```

## What Input Does fit() Accept?
```python
y = ["cat", "dog", "cat", "fish", "dog", "cat"]
# 1D array of string labels — this is what LabelEncoder works on
```
**Important:** LabelEncoder is designed for the **target column (y)**, not for input features (X).
For input features with categories, use `OneHotEncoder`.

## What Happens Inside — Step by Step
### fit(y):
1. Finds all unique values in y: `["cat", "dog", "fish"]`
2. Sorts them alphabetically: `["cat", "dog", "fish"]`
3. Assigns integers starting from 0:
   - `"cat"` → 0
   - `"dog"` → 1
   - `"fish"` → 2
4. Stores this mapping in `le.classes_`

```
le.classes_ = ["cat", "dog", "fish"]
Index:              0       1      2
```
### transform(y):
Goes through each label in y and replaces it with the stored integer:

```
Input:  ["cat", "dog", "cat", "fish", "dog", "cat"]
Output: [0,     1,     0,     2,      1,     0    ]
```
### inverse_transform(encoded):
Converts integers back to original labels:

```
Input:  [0, 1, 2, 0]
Output: ["cat", "dog", "fish", "cat"]
```

## What Does it Output?
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = ["cat", "dog", "cat", "fish", "dog"]

le.fit(y)
print("Classes:", le.classes_)       # ['cat' 'dog' 'fish']
print("Encoded:", le.transform(y))   # [0 1 0 2 1]
print("Back:", le.inverse_transform([0, 1, 2]))  # ['cat' 'dog' 'fish']
```

## Real-life Example
**Scenario:** Email spam classifier. Target column is "spam" or "not_spam".

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

y = ["not_spam", "spam", "not_spam", "not_spam", "spam",
     "spam", "not_spam", "spam", "not_spam", "not_spam"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)   # fit + transform in one step

print("Classes:", le.classes_)        # ['not_spam' 'spam']
print("Encoded:", y_encoded)          # [0 1 0 0 1 1 0 1 0 0]
# After prediction, convert back to labels
predictions_numeric = [0, 1, 0, 1]
predictions_labels = le.inverse_transform(predictions_numeric)
print("Predictions:", predictions_labels)   # ['not_spam' 'spam' 'not_spam' 'spam']
```

## When NOT to Use LabelEncoder on Features (X)?
**Problem:** LabelEncoder assigns numbers like 0, 1, 2.
Models may think 2 > 1 > 0, implying an order/ranking that does not exist.

```
"Paris"  → 0
"Delhi"  → 1
"Tokyo"  → 2

Model might think: Tokyo > Delhi > Paris  ← WRONG, no such ranking
```

For features (X columns), always use **OneHotEncoder** instead.

# OneHotEncoder
## What is OneHotEncoder?
`OneHotEncoder` converts a categorical column into **multiple binary columns** — one per category.

```
Color: ["Red", "Blue", "Green"]

Becomes:
color_Red  color_Blue  color_Green
    1           0           0       ← was Red
    0           1           0       ← was Blue
    0           0           1       ← was Green
```

Each row has exactly one 1 and all other columns are 0. That is why it is called "one-hot".

## Syntax
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X_train[["city"]])
X_encoded = ohe.transform(X_train[["city"]])
```

## What Input Does fit() Accept?
```python
X = [
    ["Delhi"],
    ["Mumbai"],
    ["Chennai"],
    ["Delhi"],
    ["Mumbai"]
]
# 2D array — even for a single column, must be 2D
```

## What Happens Inside — Step by Step
### fit(X):
1. Scans all unique values in each column
2. Stores them in sorted order in `ohe.categories_`

```
ohe.categories_ = [array(['Chennai', 'Delhi', 'Mumbai'])]
Index:                        0          1        2
```
### transform(X):
For each row, creates a binary row:

```
"Delhi"   → [0, 1, 0]  (Chennai=0, Delhi=1, Mumbai=0)
"Mumbai"  → [0, 0, 1]
"Chennai" → [1, 0, 0]
```

Result is a matrix with shape (n_rows, n_unique_categories).
### drop="first" — Avoiding Dummy Variable Trap
When you encode 3 cities into 3 columns, knowing 2 is enough to figure out the 3rd.
This redundancy can cause issues in linear models.

```python
ohe = OneHotEncoder(drop="first", sparse_output=False)
# Chennai is dropped (becomes the reference)
# Delhi   → [1, 0]
# Mumbai  → [0, 1]
# Chennai → [0, 0]
```

## What Does it Output?
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

X = np.array([["Delhi"], ["Mumbai"], ["Chennai"], ["Delhi"]])

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X)

print("Categories:", ohe.categories_)
# [array(['Chennai', 'Delhi', 'Mumbai'], dtype='<U7')]

result = ohe.transform(X)
print(result)
# [[0. 1. 0.]   ← Delhi
#  [0. 0. 1.]   ← Mumbai
#  [1. 0. 0.]   ← Chennai
#  [0. 1. 0.]]  ← Delhi
```

## Real-life Example
**Scenario:** House price prediction. Dataset has a "city" column as string.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "city":     ["Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai", "Chennai"],
    "bedrooms": [3, 4, 2, 5, 3, 4],
    "price":    [250, 400, 180, 500, 380, 200]
}
df = pd.DataFrame(data)
# Encode city column
ohe = OneHotEncoder(sparse_output=False, drop="first")
city_encoded = ohe.fit_transform(df[["city"]])
# shape: (6, 2)  — 3 cities, drop first → 2 columns
# Combine with numeric columns
X_numeric = df[["bedrooms"]].values
X = np.hstack([X_numeric, city_encoded])
y = df["price"].values

print("Feature matrix:\n", X)
# [[3. 1. 0.]   ← Delhi, 3 beds
#  [4. 0. 1.]   ← Mumbai, 4 beds
#  ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
```

## LabelEncoder vs OneHotEncoder — When to Use Which?
| Situation | Use |
|-----------|-----|
| Target column (y) with categories | `LabelEncoder` |
| Input feature (X) with 2 categories | `LabelEncoder` is fine |
| Input feature (X) with 3+ categories | `OneHotEncoder` |
| Ordinal data (low/medium/high) | `OrdinalEncoder` |

# StandardScaler
## What is StandardScaler?
`StandardScaler` transforms each feature so that it has:
- **Mean = 0**
- **Standard Deviation = 1**

This process is called **standardization** or **Z-score normalization**.

## Syntax
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

## Why Do We Need It?
Consider this data:

```
Age:    25, 30, 22, 45    → range: 20 to 60
Salary: 30000, 80000, 45000, 120000  → range: 30k to 120k
```

Without scaling, salary dominates completely because its numbers are 1000x larger.
Distance-based algorithms (KNN, SVM) and gradient-based algorithms think salary matters 1000x more.

After StandardScaler:
```
Age:    -0.8,  0.3, -1.2,  1.7
Salary: -0.9,  0.5, -0.3,  1.7
```

Now both features are on the same scale. The model treats them fairly.

**Models that NEED scaling:**
- Linear Regression, Logistic Regression
- KNN (uses distance)
- SVM (uses distances)
- Neural Networks

**Models that do NOT need scaling:**
- Decision Tree
- Random Forest
- Gradient Boosting

## What Input Does fit() Accept?
```python
X_train = np.array([
    [25, 30000],
    [30, 80000],
    [22, 45000],
    [45, 120000],
])
# 2D array — shape (n_samples, n_features)
```

## What Happens Inside — Step by Step
### fit(X_train):
For each column separately:

1. Computes the **mean** (μ):

`μ = (1/n) × Σ xᵢ`

2. Computes the **standard deviation** (σ):

`σ = √( (1/n) × Σ(xᵢ - μ)² )`

3. Stores μ and σ in `scaler.mean_` and `scaler.scale_`

```
Column 0 (Age):    mean=30.5,   std=8.81
Column 1 (Salary): mean=68750,  std=33...

scaler.mean_  = [30.5,  68750]
scaler.scale_ = [8.81,  33000]
```
### transform(X):
For every value in every column, applies:

`z = (x - μ) / σ`

```
Age 25:    z = (25 - 30.5) / 8.81  = -0.624
Age 30:    z = (30 - 30.5) / 8.81  = -0.057
Salary 30000: z = (30000 - 68750) / 33000 = -1.174
Salary 80000: z = (80000 - 68750) / 33000 =  0.341
```

Result: every column now has mean≈0 and std≈1.

## What Does it Output?
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[25, 30000], [30, 80000], [22, 45000], [45, 120000]])

scaler = StandardScaler()
scaler.fit(X)

print("Mean per column:", scaler.mean_)      # [30.5  68750.]
print("Std per column:", scaler.scale_)      # [8.81  33...]

X_scaled = scaler.transform(X)
print("Scaled data:\n", X_scaled)
# Values centered around 0, spread between roughly -2 and +2

print("Mean after scaling:", X_scaled.mean(axis=0))   # [~0.  ~0.]
print("Std after scaling:", X_scaled.std(axis=0))     # [~1.  ~1.]
```

## Real-life Example
**Scenario:** Predicting whether a customer will churn. Features: age (20–60) and monthly bill ($10–$500).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
age       = np.random.randint(20, 60, 200).reshape(-1, 1)
bill      = np.random.randint(10, 500, 200).reshape(-1, 1)
X         = np.hstack([age, bill])
y         = (bill.flatten() > 250).astype(int)   # churn if bill > 250

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# WITHOUT scaling
model_no_scale = LogisticRegression()
model_no_scale.fit(X_train, y_train)
print("Without scaling:", model_no_scale.score(X_test, y_test))
# WITH scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform in one step
X_test_scaled  = scaler.transform(X_test)         # only transform (use train stats)

model_scaled = LogisticRegression()
model_scaled.fit(X_train_scaled, y_train)
print("With scaling:", model_scaled.score(X_test_scaled, y_test))
# Scaling usually gives equal or better result
```

## inverse_transform — Get Back Original Values
```python
X_original = scaler.inverse_transform(X_scaled)
# Converts z-scores back to original age and salary values
```

# MinMaxScaler
## What is MinMaxScaler?
`MinMaxScaler` scales each feature so all values fall between **0 and 1** (by default).

This is called **normalization**.

## Syntax
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

## What Input Does fit() Accept?
Same as StandardScaler — a 2D numeric array with no NaN values.

```python
X_train = np.array([
    [100],
    [200],
    [300],
    [400],
    [500]
])
```

## What Happens Inside — Step by Step
### fit(X_train):
For each column:
1. Finds the **minimum** value: `min_`
2. Finds the **maximum** value: `max_`
3. Stores them in `scaler.data_min_` and `scaler.data_max_`

```
Column 0: min=100, max=500
scaler.data_min_ = [100.]
scaler.data_max_ = [500.]
```
### transform(X):
For every value applies:

`x_scaled = (x - x_min) / (x_max - x_min)`

```
100 → (100 - 100) / (500 - 100) = 0.0
200 → (200 - 100) / (500 - 100) = 0.25
300 → (300 - 100) / (500 - 100) = 0.5
400 → (400 - 100) / (500 - 100) = 0.75
500 → (500 - 100) / (500 - 100) = 1.0
```

Result: all values are strictly between 0 and 1.

## Custom Range — feature_range Parameter
```python
# Scale to -1 to 1 instead of 0 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
# Scale to 0 to 255 (e.g. for image pixel normalization)
scaler = MinMaxScaler(feature_range=(0, 255))
```

Formula with custom range $(a, b)$:

`x_scaled = a + ( (x - x_min) × (b - a) ) / (x_max - x_min)`

## What Does it Output?
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[100], [200], [300], [400], [500]])

scaler = MinMaxScaler()
scaler.fit(X)

print("Min:", scaler.data_min_)   # [100.]
print("Max:", scaler.data_max_)   # [500.]

X_scaled = scaler.transform(X)
print("Scaled:\n", X_scaled)
# [[0.  ]
#  [0.25]
#  [0.5 ]
#  [0.75]
#  [1.  ]]
```

## Real-life Example
**Scenario:** Image pixel values range from 0 to 255. You want to normalize them to 0–1 for a neural network.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Simulated pixel row data (grayscale image pixels)
pixels = np.array([[0, 128, 255, 64, 192]])
pixels = pixels.reshape(-1, 1)   # shape (5, 1)

scaler = MinMaxScaler()
pixels_normalized = scaler.fit_transform(pixels)

print("Original:", pixels.flatten())
# [  0 128 255  64 192]

print("Normalized:", pixels_normalized.flatten())
# [0.    0.502 1.    0.251 0.753]
```

**Another Scenario:** Combining house age (1–50 years) and price (₹10L–₹2Cr) for KNN model.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([
    [5,  1000000],
    [10, 2000000],
    [30, 5000000],
    [50, 20000000]
])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled:\n", X_scaled)
# All values now 0 to 1 — both features treated equally by KNN
```

## StandardScaler vs MinMaxScaler — When to Use Which?
| Situation | Use |
|-----------|-----|
| Data has outliers (extreme values) | `StandardScaler` — outliers affect MinMax badly |
| You need values strictly 0 to 1 | `MinMaxScaler` |
| Using with Neural Networks | `MinMaxScaler` preferred |
| Using with Linear/Logistic Regression, SVM | `StandardScaler` preferred |
| Normal distributed data | `StandardScaler` |
| Unknown distribution | `MinMaxScaler` is safer |

**Why outliers hurt MinMaxScaler?**

```
Values: [10, 20, 30, 40, 1000]  ← 1000 is an outlier

MinMax: 10 → 0.0,  1000 → 1.0
All other values squished between 0.0 and 0.03!

StandardScaler: handles this better — outlier just becomes a large z-score
```

## Day 2 — Full Summary
| Tool | What it fixes | Key formula | fit() learns |
|------|--------------|-------------|--------------|
| `SimpleImputer` | Missing NaN values | Replace with mean/median/mode | Per-column fill values |
| `LabelEncoder` | String labels → numbers | Alphabetical sorting → index | Class → number mapping |
| `OneHotEncoder` | Categories → binary columns | One column per category | Unique categories per column |
| `StandardScaler` | Features on different scales | $z = (x - \mu) / \sigma$ | Mean and std per column |
| `MinMaxScaler` | Features to 0–1 range | $z = (x - \min) / (\max - \min)$ | Min and max per column |

## Golden Rule for All Preprocessors
```python
# ALWAYS:
preprocessor.fit(X_train)           # learn from training data ONLY
X_train = preprocessor.transform(X_train)
X_test  = preprocessor.transform(X_test)   # apply same transformation
# OR shortcut for training set only:
X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)   # DO NOT fit again on test
```

## Complete Real-World Pipeline — Student Grade Prediction
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Raw messy data
data = {
    "age":       [20, np.nan, 22, 21, np.nan, 23],
    "city":      ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai", "Chennai"],
    "score":     [75, 82, np.nan, 68, 90, 71],
    "result":    ["pass", "pass", "fail", "fail", "pass", "pass"]
}
df = pd.DataFrame(data)
print("Raw data:\n", df)
# Step 1: Separate features and target
X = df[["age", "city", "score"]]
y = df["result"]
# Step 2: Encode target with LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nEncoded target:", y_encoded)   # [0 0 1 1 0 0]  (fail=0, pass=1) — alphabetical
# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
# Step 4: Impute numeric columns
num_imputer = SimpleImputer(strategy="mean")
X_train_age_score = num_imputer.fit_transform(X_train[["age", "score"]])
X_test_age_score  = num_imputer.transform(X_test[["age", "score"]])
# Step 5: One-hot encode city column
ohe = OneHotEncoder(sparse_output=False, drop="first")
X_train_city = ohe.fit_transform(X_train[["city"]])
X_test_city  = ohe.transform(X_test[["city"]])
# Step 6: Combine
X_train_final = np.hstack([X_train_age_score, X_train_city])
X_test_final  = np.hstack([X_test_age_score, X_test_city])
# Step 7: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled  = scaler.transform(X_test_final)
# Step 8: Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("\nScore:", model.score(X_test_scaled, y_test))
```

> Day 2 Complete.
> You now understand all major preprocessing tools — what they fix, how they work internally, and when to use each.
>
> Next: **Day 3 — Regression Models**
> (LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor)