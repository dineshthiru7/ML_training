# 📅 Pandas Day 1 — Introduction, Series, DataFrame, Reading Data, Inspection
## Topics Covered
| # | Topic |
|---|-------|
| 1 | What is Pandas, why it is used, NumPy vs Pandas |
| 2 | Series — create, index, operations |
| 2 | DataFrame — create from dict/list/CSV, rows and columns |
| 3 | Reading & Writing — read_csv, read_excel, read_json, to_csv, to_excel |
| 4 | Data Inspection — head, tail, info, describe, shape, columns, isnull |

# 1. Introduction to Pandas
## What is Pandas?
Pandas is a Python library for **working with tabular data** — data that has rows and columns,
like a spreadsheet or database table.

It gives you two main data structures:
- **Series** — a labeled 1D array (one column)
- **DataFrame** — a labeled 2D table (many columns)

```
Without Pandas, working with a CSV in Python would require:
  - file.read() to read raw text
  - .split(",") to separate columns
  - loops to process each row
  - manual handling of missing values

With Pandas:
  df = pd.read_csv("file.csv")
  df.head()   ← done!
```

## Why Pandas?
| Task | Without Pandas | With Pandas |
|------|---------------|-------------|
| Load a CSV | 20+ lines of code | `pd.read_csv()` |
| Find missing values | Loop + check | `df.isnull().sum()` |
| Filter rows | List comprehension | `df[df["age"] > 30]` |
| Group and average | Nested loops | `df.groupby("city")["salary"].mean()` |
| Merge two tables | Manual key matching | `pd.merge(df1, df2, on="id")` |

## NumPy vs Pandas
| Feature | NumPy | Pandas |
|---------|-------|--------|
| Data type | Homogeneous (all same) | Mixed (int, float, str, date) |
| Structure | Array (grid of numbers) | Series / DataFrame (labeled) |
| Row/Column names | Integer index only | Custom string labels |
| Missing values | No built-in handling | Built-in NaN handling |
| Primary use | Math, linear algebra, ML internals | Data loading, cleaning, analysis |
| Built on | C | NumPy + C |

**Relationship:** pandas uses NumPy internally. A DataFrame column is a NumPy array under the hood.

## Installing and Importing
```python
# Install (once)
pip install pandas
# Import (every script)
import pandas as pd
import numpy as np    # usually imported together
```

# 2. Series
## What is a Series?
A **Series** is a one-dimensional labeled array.

Think of it as one column from a spreadsheet, with:
- **values** — the actual data
- **index** — the label for each value (default: 0, 1, 2, ...)

## Creating a Series
### From a Python List
```python
import pandas as pd

scores = pd.Series([85, 72, 90, 68, 95])
print(scores)
# 0    85
# 1    72
# 2    90
# 3    68
# 4    95
# dtype: int64
```

The left column is the **index**, the right column is the **value**.
### With a Custom Index
```python
scores = pd.Series([85, 72, 90, 68, 95],
                   index=["Alice", "Bob", "Carol", "Dave", "Eve"])
print(scores)
# Alice    85
# Bob      72
# Carol    90
# Dave     68
# Eve      95
```
### From a Dictionary
```python
data = {"Alice": 85, "Bob": 72, "Carol": 90}
scores = pd.Series(data)
print(scores)
# Alice    85
# Bob      72
# Carol    90
```

## Series Attributes
```python
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])

print(s.values)    # [10 20 30 40]  ← NumPy array
print(s.index)     # Index(['a', 'b', 'c', 'd'], dtype='object')
print(s.dtype)     # int64
print(s.shape)     # (4,)
print(s.size)      # 4
print(len(s))      # 4
print(s.name)      # None  (can be set)
```

## Indexing a Series
```python
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
# By label
print(s["b"])         # 20
print(s[["a", "c"]])  # a=10, c=30
# By position
print(s.iloc[0])      # 10
print(s.iloc[-1])     # 40
# Slicing by label (inclusive on both ends!)
print(s["a":"c"])
# a    10
# b    20
# c    30
# Slicing by position (exclusive on right, like Python)
print(s.iloc[0:2])
# a    10
# b    20
```

## Series Operations
All operations are element-wise and **index-aligned**:

```python
s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])

print(s1 + s2)    # a=11, b=22, c=33
print(s1 * 5)     # a=5, b=10, c=15
print(s1 ** 2)    # a=1, b=4, c=9
# Alignment on index — mismatched indices become NaN
s3 = pd.Series([10, 20, 30], index=["a", "b", "d"])   # d, not c
print(s1 + s3)
# a    11.0
# b    22.0
# c     NaN   ← c not in s3
# d     NaN   ← d not in s1
```

## Useful Series Methods
```python
s = pd.Series([5, 3, 8, 1, 9, 3, 7])

print(s.sum())             # 36
print(s.mean())            # 5.14
print(s.max())             # 9
print(s.min())             # 1
print(s.std())             # 2.85
print(s.median())          # 5.0
print(s.value_counts())    # count of each unique value
print(s.unique())          # [5 3 8 1 9 7] — unique values
print(s.nunique())         # 6 — number of unique values
print(s.sort_values())     # sorted ascending
print(s.describe())        # count, mean, std, min, quartiles, max
```

# 2. DataFrame
## What is a DataFrame?
A **DataFrame** is a 2D labeled data structure — like a spreadsheet with rows and columns.

Each column is a Series. All columns share the same index (row labels).

```
       name  age  score
0     Alice   22   85.5
1       Bob   25   72.0
2     Carol   21   90.3
3      Dave   28   68.0
↑      ↑       ↑    ↑
index  col1   col2  col3
```

## Creating a DataFrame
### From a Dictionary (most common)
```python
import pandas as pd

data = {
    "name":  ["Alice", "Bob", "Carol", "Dave"],
    "age":   [22, 25, 21, 28],
    "score": [85.5, 72.0, 90.3, 68.0]
}
df = pd.DataFrame(data)
print(df)
#     name  age  score
# 0  Alice   22   85.5
# 1    Bob   25   72.0
# 2  Carol   21   90.3
# 3   Dave   28   68.0
```
### From a List of Dictionaries
```python
records = [
    {"name": "Alice", "age": 22, "score": 85.5},
    {"name": "Bob",   "age": 25, "score": 72.0},
    {"name": "Carol", "age": 21, "score": 90.3},
]
df = pd.DataFrame(records)
```
### From a NumPy Array
```python
import numpy as np

arr = np.random.rand(4, 3)
df = pd.DataFrame(arr, columns=["A", "B", "C"])
print(df)
```
### With Custom Row Index
```python
df = pd.DataFrame(data, index=["r1", "r2", "r3", "r4"])
print(df)
#       name  age  score
# r1   Alice   22   85.5
# r2     Bob   25   72.0
```

## DataFrame Attributes
```python
print(df.shape)        # (4, 3) — rows, columns
print(df.columns)      # Index(['name', 'age', 'score'], dtype='object')
print(df.index)        # RangeIndex(start=0, stop=4, step=1)
print(df.dtypes)       # name:object, age:int64, score:float64
print(df.values)       # 2D NumPy array of all values
print(len(df))         # 4 — number of rows
print(df.size)         # 12 — total cells (rows × cols)
print(df.ndim)         # 2
```

# 3. Reading & Writing Data
## pd.read_csv()
### What is it?
Reads a CSV (comma-separated values) file and returns a DataFrame.
This is the most common way to load data in data science.
### Syntax
```python
pd.read_csv(filepath, sep=",", header=0, index_col=None,
            usecols=None, dtype=None, nrows=None, skiprows=None,
            na_values=None, encoding="utf-8")
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `filepath` | required | Path to the CSV file |
| `sep` | `","` | Delimiter between columns |
| `header` | `0` | Row number to use as column names |
| `index_col` | `None` | Column to use as row index |
| `usecols` | all | List of columns to load |
| `dtype` | auto | Force specific dtype per column |
| `nrows` | all | Load only first N rows |
| `skiprows` | none | Skip specific rows |
| `na_values` | standard | Additional strings to treat as NaN |
| `encoding` | `utf-8` | File encoding |
### What Does it Output?
```python
import pandas as pd
# Basic load
df = pd.read_csv("students.csv")
print(df.head())
# Load specific columns only
df = pd.read_csv("students.csv", usecols=["name", "score"])
# Load first 100 rows only
df = pd.read_csv("large_file.csv", nrows=100)
# Use a column as the index
df = pd.read_csv("data.csv", index_col="student_id")
# Custom delimiter (tab-separated)
df = pd.read_csv("data.tsv", sep="\t")
# Treat "NA", "N/A", "-" as missing
df = pd.read_csv("data.csv", na_values=["NA", "N/A", "-", "none"])
```

## pd.read_excel()
```python
# Load first sheet
df = pd.read_excel("report.xlsx")
# Load specific sheet
df = pd.read_excel("report.xlsx", sheet_name="Sales")
# Load by sheet index
df = pd.read_excel("report.xlsx", sheet_name=0)
# Load multiple sheets → returns dict
all_sheets = pd.read_excel("report.xlsx", sheet_name=None)
print(all_sheets.keys())   # dict_keys(['Sheet1', 'Sheet2', ...])
```

## pd.read_json()
```python
# From a JSON file
df = pd.read_json("data.json")
# From a JSON string
import json
json_str = '[{"name":"Alice","age":22},{"name":"Bob","age":25}]'
df = pd.read_json(json_str)
print(df)
#     name  age
# 0  Alice   22
# 1    Bob   25
```

## df.to_csv()
```python
# Save to CSV
df.to_csv("output.csv", index=False)      # index=False → don't write row numbers
# Save with specific separator
df.to_csv("output.tsv", sep="\t", index=False)
# Save only specific columns
df.to_csv("output.csv", columns=["name", "score"], index=False)
```

## df.to_excel()
```python
# Save to Excel
df.to_excel("output.xlsx", index=False, sheet_name="Results")
# Save multiple DataFrames to different sheets
with pd.ExcelWriter("multi_sheet.xlsx") as writer:
    df.to_excel(writer, sheet_name="Students", index=False)
    df2.to_excel(writer, sheet_name="Scores",  index=False)
```

# 4. Data Inspection
## df.head() and df.tail()
### What are they?
- `head(n)` → show first n rows (default 5)
- `tail(n)` → show last n rows (default 5)

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "age":    [22, 25, 21, 28, 23, 30],
    "score":  [85.5, 72.0, 90.3, 68.0, 88.0, 74.5],
    "passed": [True, True, True, False, True, True]
})

print(df.head())     # first 5 rows
print(df.head(3))    # first 3 rows
print(df.tail(2))    # last 2 rows
```

## df.info()
### What is it?
Prints a concise summary of the DataFrame:
- number of rows and columns
- column names and their dtypes
- non-null counts per column (how many values exist)
- memory usage

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   name    6 non-null      object
 1   age     6 non-null      int64
 2   score   6 non-null      float64
 3   passed  6 non-null      bool
dtypes: bool(1), float64(1), int64(1), object(1)
memory usage: 326.0+ bytes
```

This is the first thing to call on any new dataset — it tells you immediately if any column
has missing values (non-null count less than total rows).

## df.describe()
### What is it?
Generates **descriptive statistics** for all numeric columns:
count, mean, std, min, 25th percentile, median (50%), 75th percentile, max.

```python
print(df.describe())
```

```
             age      score
count   6.000000   6.000000
mean   24.833333  79.716667
std     3.311596   8.583219
min    21.000000  68.000000
25%    22.250000  72.750000
50%    24.000000  86.750000
75%    26.750000  88.500000
max    30.000000  90.300000
```

For non-numeric columns:
```python
print(df.describe(include="object"))   # string columns
print(df.describe(include="all"))      # all columns
```

## df.shape, df.columns, df.index
```python
print(df.shape)     # (6, 4)  ← (rows, columns)
print(df.columns)   # Index(['name', 'age', 'score', 'passed'], dtype='object')
print(df.index)     # RangeIndex(start=0, stop=6, step=1)
# Number of rows and columns separately
rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")
# List of column names as Python list
print(list(df.columns))   # ['name', 'age', 'score', 'passed']
```

## Checking Missing Values — isnull() and notnull()
### What are they?
- `isnull()` returns a DataFrame of True/False — True where value is NaN
- `notnull()` returns the opposite

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":  ["Alice", "Bob", None,   "Dave"],
    "age":   [22,      np.nan, 21,    28],
    "score": [85.5,    72.0,  np.nan, 68.0]
})

print(df.isnull())
#     name    age  score
# 0  False  False  False
# 1  False   True  False
# 2   True  False   True
# 3  False  False  False
# Count missing values per column
print(df.isnull().sum())
# name     1
# age      1
# score    1
# Total missing in entire DataFrame
print(df.isnull().sum().sum())   # 3
# Fraction missing per column
print(df.isnull().mean())        # name=0.25, age=0.25, score=0.25
# Which rows have ANY missing value
print(df.isnull().any(axis=1))
# 0    False
# 1     True
# 2     True
# 3    False
```

## Complete Day 1 Practice Code
```python
import pandas as pd
import numpy as np
# ── Create DataFrame ─────────────────────────────────
np.random.seed(42)
n = 8
df = pd.DataFrame({
    "student":  [f"S{i}" for i in range(1, n+1)],
    "age":      np.random.randint(18, 30, n),
    "math":     np.random.randint(50, 100, n).astype(float),
    "science":  np.random.randint(50, 100, n).astype(float),
    "city":     np.random.choice(["Delhi", "Mumbai", "Chennai"], n)
})
# Introduce missing values
df.loc[2, "math"]    = np.nan
df.loc[5, "science"] = np.nan

print("=" * 45)
print("BASIC INSPECTION")
print("=" * 45)
print(df.head())
print("\nShape:", df.shape)
print("Columns:", list(df.columns))

print("\n" + "=" * 45)
print("df.info()")
print("=" * 45)
df.info()

print("\n" + "=" * 45)
print("df.describe()")
print("=" * 45)
print(df.describe())

print("\n" + "=" * 45)
print("MISSING VALUES")
print("=" * 45)
print(df.isnull().sum())
# ── Save and reload ───────────────────────────────────
df.to_csv("students_day1.csv", index=False)
df_loaded = pd.read_csv("students_day1.csv")
print("\nReloaded shape:", df_loaded.shape)
print("Same data:", df.shape == df_loaded.shape)
```

> Day 1 Complete.
> You can now create Series and DataFrames, load data from CSV/Excel/JSON,
> write data back to files, and inspect any dataset's structure and missing values.
>
> Next: **Pandas Day 2 — Indexing, Data Cleaning, Data Manipulation**