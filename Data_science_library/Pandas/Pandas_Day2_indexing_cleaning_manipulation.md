# 📅 Pandas Day 2 — Indexing, Data Cleaning, Data Manipulation
## Topics Covered
| # | Topic |
|---|-------|
| 1 | Column selection, Row selection |
| 2 | loc[] — label-based indexing |
| 3 | iloc[] — position-based indexing |
| 4 | Conditional filtering |
| 5 | dropna() — removing missing values |
| 6 | fillna() — filling missing values |
| 7 | drop_duplicates() |
| 8 | Renaming columns, changing dtypes (astype) |
| 9 | Adding columns, drop(), sort_values(), sort_index() |

# 1. Column and Row Selection
## Selecting Columns
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":    [22, 25, 21, 28, 23],
    "score":  [85.5, 72.0, 90.3, 68.0, 88.0],
    "city":   ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
})
# Single column → returns a Series
names = df["name"]
print(type(names))     # <class 'pandas.core.series.Series'>
# Single column using dot notation (only when name has no spaces)
names = df.name        # same as df["name"]
# Multiple columns → returns a DataFrame
subset = df[["name", "score"]]
print(type(subset))    # <class 'pandas.core.frame.DataFrame'>

print(subset)
#     name  score
# 0  Alice   85.5
# 1    Bob   72.0
# 2  Carol   90.3
# 3   Dave   68.0
# 4    Eve   88.0
```

## Selecting Rows by Index Position
```python
# Single row by position — returns a Series
print(df.iloc[0])     # first row
print(df.iloc[-1])    # last row
# Slice of rows by position
print(df.iloc[1:3])   # rows at position 1 and 2 (not 3)
```

# 2. loc[] — Label-Based Indexing
## What is loc[]?
`loc[]` selects data using **row labels** (index values) and **column names**.

- Both row and column arguments are **inclusive** on both ends.
- Useful when your index is meaningful (names, dates, IDs).
### Syntax
```python
df.loc[row_label, column_label]
df.loc[row_slice, column_slice]
df.loc[boolean_array, column_label]
```
### What Input Does it Accept?
| Argument | Type | Example |
|----------|------|---------|
| row label | single value | `df.loc[0]` |
| row labels | list | `df.loc[[0, 2, 4]]` |
| row slice | label range | `df.loc[0:2]` |
| boolean mask | Series/array | `df.loc[df["age"] > 22]` |
| column name | single string | `df.loc[0, "name"]` |
| column names | list | `df.loc[0, ["name", "score"]]` |
| column slice | string range | `df.loc[:, "age":"score"]` |
### Examples — Default Integer Index
```python
# Single row
print(df.loc[0])
# name     Alice
# age         22
# score     85.5
# city     Delhi
# Single cell
print(df.loc[0, "score"])    # 85.5
# Multiple rows, single column
print(df.loc[[0, 2, 4], "name"])
# 0    Alice
# 2    Carol
# 4      Eve
# Row slice — loc slice is INCLUSIVE (both ends)
print(df.loc[1:3])           # rows 1, 2, 3
# All rows, column slice (name through score)
print(df.loc[:, "name":"score"])
#     name  age  score
# 0  Alice   22   85.5
# ...
# Modify a value
df.loc[0, "score"] = 91.0
```
### Examples — Named Index
```python
df2 = pd.DataFrame({
    "score": [85.5, 72.0, 90.3, 68.0, 88.0],
    "city":  ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
}, index=["Alice", "Bob", "Carol", "Dave", "Eve"])
# Select by name label
print(df2.loc["Alice"])         # Alice's row
print(df2.loc["Bob":"Carol"])   # Bob and Carol
print(df2.loc[["Alice", "Eve"], "score"])
```

# 3. iloc[] — Position-Based Indexing
## What is iloc[]?
`iloc[]` selects data using **integer positions** — like NumPy indexing.

- Positions are 0-based.
- Slices are **exclusive on the right** (like standard Python).
### Syntax
```python
df.iloc[row_position, col_position]
df.iloc[row_slice, col_slice]
```
### Examples
```python
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":   [22, 25, 21, 28, 23],
    "score": [85.5, 72.0, 90.3, 68.0, 88.0],
    "city":  ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
})
# Single cell by row=0, col=2
print(df.iloc[0, 2])        # 85.5  (first row, third column)
# Entire first row
print(df.iloc[0])
# Rows 1 to 2 (position 1 and 2, not 3)
print(df.iloc[1:3])
# Last row
print(df.iloc[-1])
# First 3 rows, first 2 columns
print(df.iloc[:3, :2])
#     name  age
# 0  Alice   22
# 1    Bob   25
# 2  Carol   21
# Every other row
print(df.iloc[::2])
# Reverse rows
print(df.iloc[::-1])
```

## loc vs iloc — Side-by-Side Comparison
| Feature | loc[] | iloc[] |
|---------|-------|--------|
| Index type | Labels (any type) | Integers only |
| Row selection | By label | By position |
| Slice endpoint | Inclusive | Exclusive |
| Column selection | By name | By position |
| Good for | Named index, readable code | Numerical iteration |

```python
# Same result — different approach
df.loc[0, "score"]    # by label
df.iloc[0, 2]         # by position (score is column 2)
```

# 4. Conditional Filtering
## What is it?
Select rows where a condition is True — like SQL `WHERE`.

```python
# Rows where age > 22
older = df[df["age"] > 22]
print(older)
#   name  age  score     city
# 1  Bob   25   72.0   Mumbai
# 3 Dave   28   68.0  Chennai
# Rows where city is Mumbai
mumbai = df[df["city"] == "Mumbai"]
# Multiple conditions — use & (AND), | (OR), ~ (NOT)
# MUST wrap each condition in parentheses
result = df[(df["age"] > 21) & (df["score"] > 80)]
print(result)
# Either Mumbai OR score above 85
result2 = df[(df["city"] == "Mumbai") | (df["score"] > 85)]
# NOT Delhi
not_delhi = df[~(df["city"] == "Delhi")]
```
### Using isin()
```python
# Rows where city is Delhi OR Mumbai
metro = df[df["city"].isin(["Delhi", "Mumbai"])]
```
### Using str methods
```python
df["name_upper"] = df["name"].str.upper()
starts_with_a = df[df["name"].str.startswith("A")]
contains_al   = df[df["name"].str.contains("al", case=False)]
```
### Using between()
```python
mid_score = df[df["score"].between(70, 90)]  # 70 ≤ score ≤ 90
```

# 5. dropna() — Removing Missing Values
## What is it?
Removes rows (or columns) that contain missing values (NaN).
### Syntax
```python
df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `axis` | `0` | `0` = drop rows, `1` = drop columns |
| `how` | `"any"` | `"any"` = if ANY cell is NaN; `"all"` = only if ALL cells are NaN |
| `thresh` | `None` | Keep row only if it has at least N non-NaN values |
| `subset` | all cols | Only check these columns for NaN |
| `inplace` | `False` | If True, modify df directly; if False, return new df |
### What Happens Inside — Step by Step
1. Pandas checks each row (axis=0) or column (axis=1) for NaN values.
2. If `how="any"`: any NaN in the row → that row is dropped.
3. If `how="all"`: only drop if every value in the row is NaN.
4. If `thresh=N`: drop the row if it has fewer than N non-NaN values.
5. Returns a new DataFrame (original unchanged unless inplace=True).
### Examples
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":  ["Alice", "Bob",    None,   "Dave"],
    "age":   [22,      np.nan,   21,      28],
    "score": [85.5,    72.0,    np.nan,   68.0]
})

print(df)
#     name   age  score
# 0  Alice  22.0   85.5
# 1    Bob   NaN   72.0
# 2   None  21.0    NaN
# 3   Dave  28.0   68.0
# Drop any row with at least one NaN
print(df.dropna())
#     name   age  score
# 0  Alice  22.0   85.5
# 3   Dave  28.0   68.0
# Drop row only if ALL values are NaN
print(df.dropna(how="all"))    # keeps all rows (none are all-NaN)
# Drop row if NaN in "score" column only
print(df.dropna(subset=["score"]))
# Drop columns with any NaN
print(df.dropna(axis=1))       # only "name" col survives... too strict usually
# Keep rows with at least 2 non-NaN values
print(df.dropna(thresh=2))
```

# 6. fillna() — Filling Missing Values
## What is it?
Replaces NaN values with a given value or strategy.
### Syntax
```python
df.fillna(value, method=None, axis=None, inplace=False, limit=None)
```
### Parameters
| Parameter | What it does |
|-----------|--------------|
| `value` | Scalar, dict, or Series to fill with |
| `method` | `"ffill"` (forward fill) or `"bfill"` (backward fill) |
| `inplace` | Modify in-place if True |
| `limit` | Max number of NaN values to fill |
### Examples
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":  ["Alice", "Bob",    None,   "Dave"],
    "age":   [22,      np.nan,   21,      28],
    "score": [85.5,    72.0,    np.nan,   68.0]
})
# Fill all NaN with a single value
print(df.fillna(0))
# Fill specific column with mean
df["score"] = df["score"].fillna(df["score"].mean())
print(df)
# Fill different columns with different values
df.fillna({"name": "Unknown", "age": df["age"].median()}, inplace=True)
print(df)
# Forward fill — fill NaN with the value from the row above
s = pd.Series([1, np.nan, np.nan, 4, np.nan])
print(s.ffill())    # [1, 1, 1, 4, 4]
# Backward fill — fill NaN with the value from the row below
print(s.bfill())    # [1, 4, 4, 4, NaN]
```

# 7. drop_duplicates()
## What is it?
Removes duplicate rows from a DataFrame.
### Syntax
```python
df.drop_duplicates(subset=None, keep="first", inplace=False)
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `subset` | all cols | Check duplicates based on these columns only |
| `keep` | `"first"` | Which duplicate to keep: `"first"`, `"last"`, or `False` (drop all) |
| `inplace` | `False` | Modify in-place |
### Examples
```python
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Alice", "Carol", "Bob"],
    "age":   [22, 25, 22, 21, 25],
    "city":  ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
})
# Remove exact duplicate rows
print(df.drop_duplicates())
#     name  age     city
# 0  Alice   22    Delhi
# 1    Bob   25   Mumbai
# 3  Carol   21  Chennai
# Keep last occurrence of duplicate
print(df.drop_duplicates(keep="last"))
# Drop all occurrences of duplicates (no survivor)
print(df.drop_duplicates(keep=False))
# Duplicate based on "name" only (ignore age/city)
print(df.drop_duplicates(subset=["name"]))
# Check for duplicates (returns boolean Series)
print(df.duplicated())
# 0    False
# 1    False
# 2     True
# 3    False
# 4     True

print(df.duplicated().sum())    # 2 duplicates total
```

# 8. Renaming Columns and Changing dtypes
## df.rename()
```python
df = pd.DataFrame({
    "nm": ["Alice", "Bob"],
    "ag": [22, 25],
    "sc": [85.5, 72.0]
})
# Rename specific columns using a dict
df = df.rename(columns={"nm": "name", "ag": "age", "sc": "score"})
print(df.columns)    # Index(['name', 'age', 'score'], ...)
# Rename rows (index)
df = df.rename(index={0: "row1", 1: "row2"})
# Rename all columns using a function
df.columns = df.columns.str.upper()    # NAME, AGE, SCORE
df.columns = df.columns.str.lower()    # back to lowercase
df.columns = df.columns.str.replace(" ", "_")   # replace spaces with underscore
```

## df.astype() — Changing Column Data Types
### What is it?
Converts a column from one dtype to another.
### Common dtype conversions
| From | To | When |
|------|----|------|
| object (string) | int64 | "22" → 22 |
| object (string) | float64 | "85.5" → 85.5 |
| int64 | float64 | 22 → 22.0 |
| object | category | Low-cardinality string columns |
| object | datetime | Date strings |

```python
import pandas as pd

df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol"],
    "age":   ["22", "25", "21"],       # stored as strings
    "score": ["85.5", "72.0", "90.3"], # stored as strings
    "grade": ["A", "B", "A"]
})

print(df.dtypes)
# name     object
# age      object
# score    object
# grade    object
# Convert age to integer
df["age"] = df["age"].astype(int)
# Convert score to float
df["score"] = df["score"].astype(float)
# Convert grade to category (saves memory for repeated strings)
df["grade"] = df["grade"].astype("category")

print(df.dtypes)
# name      object
# age        int64
# score    float64
# grade   category
# Convert multiple columns at once
df = df.astype({"age": "int64", "score": "float64"})
# Convert to datetime
import pandas as pd
dates = pd.Series(["2023-01-15", "2023-03-22", "2023-07-10"])
dates = pd.to_datetime(dates)
print(dates.dt.year)    # 2023, 2023, 2023
print(dates.dt.month)   # 1, 3, 7
```

# 9. Adding Columns, drop(), sort_values(), sort_index()
## Adding New Columns
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol", "Dave"],
    "math":  [85, 72, 90, 68],
    "sci":   [88, 74, 92, 70]
})
# Direct assignment — element-wise calculation
df["total"]   = df["math"] + df["sci"]
df["average"] = (df["math"] + df["sci"]) / 2
# Conditional column using np.where
df["passed"] = np.where(df["average"] >= 75, "Yes", "No")
# Assign using a constant
df["year"] = 2024
# Multiple new columns at once using .assign()
df = df.assign(
    rank    = df["total"].rank(ascending=False).astype(int),
    scaled  = (df["total"] - df["total"].min()) / (df["total"].max() - df["total"].min())
)

print(df)
```

## df.drop() — Removing Rows or Columns
```python
# Drop a column
df = df.drop(columns=["year"])
# or
df = df.drop("year", axis=1)
# Drop multiple columns
df = df.drop(columns=["total", "rank"])
# Drop a row by index label
df = df.drop(index=0)
# Drop multiple rows
df = df.drop(index=[0, 2])
# Drop in-place
df.drop(columns=["scaled"], inplace=True)
```

## df.sort_values() — Sorting by Column Values
### Syntax
```python
df.sort_values(by, ascending=True, na_position="last", inplace=False, ignore_index=False)
```

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `by` | required | Column name or list of column names |
| `ascending` | `True` | True = A→Z / small→large |
| `na_position` | `"last"` | Where to put NaN rows |
| `inplace` | `False` | Modify in-place |
| `ignore_index` | `False` | If True, reset index after sort |

```python
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol", "Dave"],
    "age":   [22, 25, 21, 28],
    "score": [85.5, 72.0, 90.3, 68.0],
    "city":  ["Delhi", "Mumbai", "Delhi", "Mumbai"]
})
# Sort by score descending (best first)
print(df.sort_values("score", ascending=False))
#    name  age  score    city
# 2 Carol   21   90.3   Delhi
# 0 Alice   22   85.5   Delhi
# 1   Bob   25   72.0  Mumbai
# 3  Dave   28   68.0  Mumbai
# Sort by city ascending, then score descending within each city
print(df.sort_values(by=["city", "score"], ascending=[True, False]))
# Reset index after sorting
df_sorted = df.sort_values("age", ignore_index=True)
print(df_sorted.index)    # 0, 1, 2, 3 (clean new index)
```

## df.sort_index() — Sorting by Row Index
```python
# If the index has been shuffled, restore natural order
df_shuffled = df.sample(frac=1, random_state=42)   # randomize order
print(df_shuffled.index)   # e.g., [2, 0, 3, 1]

df_restored = df_shuffled.sort_index()
print(df_restored.index)   # [0, 1, 2, 3]
# Sort descending (reverse index)
print(df.sort_index(ascending=False))
```

## reset_index()
```python
# After filtering, the index has gaps: [1, 3, 4]
# reset_index() gives a fresh 0, 1, 2, ... index
filtered = df[df["age"] > 22]
print(filtered.index)   # [1, 3]

clean = filtered.reset_index(drop=True)   # drop=True removes old index column
print(clean.index)      # [0, 1]
```

## Complete Day 2 Practice Code
```python
import pandas as pd
import numpy as np
# ── Build dataset ─────────────────────────────────────
np.random.seed(10)
n = 10
df = pd.DataFrame({
    "student":  [f"S{i:02d}" for i in range(1, n+1)],
    "age":      np.random.randint(18, 30, n).astype(float),
    "math":     np.random.randint(40, 100, n).astype(float),
    "english":  np.random.randint(40, 100, n).astype(float),
    "city":     np.random.choice(["Delhi", "Mumbai", "Chennai"], n),
    "year":     ["2023"] * 5 + ["2024"] * 5
})
# Add duplicates
df = pd.concat([df, df.iloc[[2, 5]]], ignore_index=True)
# Add missing values
df.loc[3, "math"]    = np.nan
df.loc[7, "english"] = np.nan
df.loc[9, "age"]     = np.nan

print("=" * 50)
print("ORIGINAL DATA")
print("=" * 50)
print(df.head(12))
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
# ── Clean data ────────────────────────────────────────
print("\n" + "=" * 50)
print("CLEANING")
print("=" * 50)
# Remove duplicates
df = df.drop_duplicates()
print("After drop_duplicates:", len(df), "rows")
# Fill missing math with median
df["math"] = df["math"].fillna(df["math"].median())
# Fill missing age with mean
df["age"] = df["age"].fillna(df["age"].mean())
# Fill missing english with forward fill
df["english"] = df["english"].ffill()
# Fix dtype: year should be int
df["year"] = df["year"].astype(int)

print("Missing after cleaning:\n", df.isnull().sum())
# ── Manipulate data ───────────────────────────────────
print("\n" + "=" * 50)
print("MANIPULATION")
print("=" * 50)
# Add total and grade
df["total"] = df["math"] + df["english"]
df["passed"] = np.where(df["total"] >= 140, "Yes", "No")
# Rename student to student_id
df = df.rename(columns={"student": "student_id"})
# Sort by total descending
df = df.sort_values("total", ascending=False, ignore_index=True)
print(df.head(5))
# ── loc and iloc examples ─────────────────────────────
print("\n" + "=" * 50)
print("INDEXING")
print("=" * 50)
# Top scorer's details
print("Top scorer:\n", df.loc[0, ["student_id", "math", "english", "total"]])
# Students in Mumbai with total > 150
mumbai_top = df[(df["city"] == "Mumbai") & (df["total"] > 150)]
print("\nMumbai students with total > 150:\n", mumbai_top[["student_id", "total"]])
# iloc — first 3 rows, numeric columns only
print("\nFirst 3 rows (num cols):\n", df.iloc[:3, 1:5])
# ── Summary ───────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY TABLE")
print("=" * 50)
print(df[["math", "english", "total"]].describe().round(2))
```

> Day 2 Complete.
> You can now select data precisely with loc/iloc, filter rows with conditions,
> clean missing values and duplicates, change types, add/remove columns, and sort.
>
> Next: **Pandas Day 3 — GroupBy, Merge, Join, Concat**