# 📅 Pandas Day 3 — GroupBy, Aggregation, Merge, Concat, Join
## Topics Covered
| # | Topic |
|---|-------|
| 1 | groupby() — split-apply-combine pattern |
| 2 | Single aggregations — mean, sum, count, min, max |
| 3 | agg() — multiple aggregations at once |
| 4 | Grouping by multiple columns |
| 5 | Named aggregation |
| 6 | merge() — inner, outer, left, right joins |
| 7 | concat() — stacking DataFrames |
| 8 | join() — index-based joining |
| 9 | Comparison of merge vs concat vs join |

# 1. groupby() — Split-Apply-Combine
## What is it?
`groupby()` is the Pandas way to **group rows by a column's value** and then **apply a
calculation to each group**.

The process has three stages:

```
SPLIT       APPLY         COMBINE
─────       ─────         ───────
DataFrame   function      single
by groups → per group  →  result
```
### Real-world analogy
Imagine a school results table. You want the average score per city.

```
name    city     score
─────   ─────    ─────
Alice   Delhi    85
Bob     Mumbai   72
Carol   Delhi    90
Dave    Mumbai   68
Eve     Delhi    88

Step 1 SPLIT   → Group "Delhi" rows → [85, 90, 88]
                 Group "Mumbai" rows → [72, 68]

Step 2 APPLY   → mean of Delhi  → (85 + 90 + 88) / 3 = 87.67
                 mean of Mumbai → (72 + 68) / 2     = 70.0

Step 3 COMBINE → city     score
                 Delhi    87.67
                 Mumbai   70.0
```

## Syntax
```python
df.groupby(by, axis=0, sort=True, as_index=True, dropna=True)
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `by` | required | Column name or list of columns to group by |
| `sort` | `True` | Sort group keys |
| `as_index` | `True` | Use group keys as index in result |
| `dropna` | `True` | Exclude NaN groups |

## Basic Usage
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "city":   ["Delhi", "Mumbai", "Delhi", "Mumbai", "Delhi", "Chennai"],
    "dept":   ["Math", "Math", "Science", "Science", "Math", "Science"],
    "score":  [85, 72, 90, 68, 88, 74],
    "age":    [22, 25, 21, 28, 23, 24]
})
# Group by city, compute mean score
result = df.groupby("city")["score"].mean()
print(result)
# city
# Chennai    74.000000
# Delhi      87.666667
# Mumbai     70.000000
# Name: score, dtype: float64
# .mean() returns a Series with city as index
# Use .reset_index() to turn it into a flat DataFrame
result = df.groupby("city")["score"].mean().reset_index()
print(result)
#       city      score
# 0  Chennai  74.000000
# 1    Delhi  87.666667
# 2   Mumbai  70.000000
```

# 2. Single Aggregations
```python
grouped = df.groupby("city")["score"]

print(grouped.mean())    # average per city
print(grouped.sum())     # total per city
print(grouped.count())   # number of students per city
print(grouped.min())     # lowest score per city
print(grouped.max())     # highest score per city
print(grouped.std())     # standard deviation per city
print(grouped.median())  # median per city
print(grouped.first())   # first row's value per group
print(grouped.last())    # last row's value per group
# Size vs Count
# .size()  → number of rows in each group (includes NaN)
# .count() → number of non-NaN values
print(df.groupby("city").size())
print(df.groupby("city")["score"].count())
```

## What Does the Output Look Like?
```python
# Group by city — multiple numeric columns
print(df.groupby("city")[["score", "age"]].mean())
#            score    age
# city
# Chennai  74.000  24.0
# Delhi    87.667  22.0
# Mumbai   70.000  26.5
```

# 3. agg() — Multiple Aggregations at Once
## What is it?
`agg()` lets you apply **multiple functions** to one or more columns in a single call.
### Syntax
```python
df.groupby("column")["target"].agg(["func1", "func2", ...])
df.groupby("column").agg({"col1": "func1", "col2": "func2"})
```
### Examples
```python
# Multiple aggregations on one column
result = df.groupby("city")["score"].agg(["mean", "min", "max", "count"])
print(result)
#              mean  min   max  count
# city
# Chennai  74.000   74    74      1
# Delhi    87.667   85    90      3
# Mumbai   70.000   68    72      2
# Different aggregations per column
result = df.groupby("city").agg(
    score_mean = ("score", "mean"),
    score_max  = ("score", "max"),
    age_min    = ("age",   "min")
)
print(result)
#          score_mean  score_max  age_min
# city
# Chennai   74.000000         74       24
# Delhi     87.666667         90       21
# Mumbai    70.000000         72       25
```
### Using Custom Functions in agg()
```python
def score_range(x):
    return x.max() - x.min()

result = df.groupby("city")["score"].agg(
    mean_score  = "mean",
    score_range = score_range
)
print(result)
#          mean_score  score_range
# city
# Chennai   74.000000            0
# Delhi     87.666667            5
# Mumbai    70.000000            4
```

# 4. Grouping by Multiple Columns
## What is it?
You can pass a list of column names to `groupby()` to create groups on the combination
of multiple columns.

```python
result = df.groupby(["city", "dept"])["score"].mean()
print(result)
# city     dept
# Chennai  Science    74.000000
# Delhi    Math       86.500000
#          Science    90.000000
# Mumbai   Math       72.000000
#          Science    68.000000
# This creates a MultiIndex — use reset_index() to flatten
result = result.reset_index()
print(result)
#       city     dept      score
# 0  Chennai  Science  74.000000
# 1    Delhi     Math  86.500000
# 2    Delhi  Science  90.000000
# 3   Mumbai     Math  72.000000
# 4   Mumbai  Science  68.000000
```

## as_index=False — Avoid MultiIndex Automatically
```python
# Cleaner output without calling reset_index
result = df.groupby(["city", "dept"], as_index=False)["score"].mean()
print(result)
```

## transform() — Apply Group Result Back to Original DataFrame
```python
# Add a "city_avg" column — each row gets the average of its city group
df["city_avg"] = df.groupby("city")["score"].transform("mean")
print(df[["name", "city", "score", "city_avg"]])
#     name     city  score   city_avg
# 0  Alice    Delhi     85  87.666667
# 1    Bob   Mumbai     72  70.000000
# 2  Carol    Delhi     90  87.666667
# ...
# Normalize within group: how far each student is from their city's average
df["score_vs_city"] = df["score"] - df.groupby("city")["score"].transform("mean")
```

# 5. merge() — Combining DataFrames Horizontally (Joins)
## What is it?
`merge()` combines two DataFrames using a **common key column** — exactly like SQL JOIN.
### Syntax
```python
pd.merge(left, right, how="inner", on=None,
         left_on=None, right_on=None,
         left_index=False, right_index=False,
         suffixes=("_x", "_y"))
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `left` | required | First DataFrame |
| `right` | required | Second DataFrame |
| `how` | `"inner"` | Join type: `"inner"`, `"left"`, `"right"`, `"outer"` |
| `on` | `None` | Column name(s) that appear in both (use when same name) |
| `left_on` | `None` | Key column in left DataFrame |
| `right_on` | `None` | Key column in right DataFrame |
| `suffixes` | `("_x","_y")` | Suffix for overlapping non-key columns |

## Types of Joins — Visual Explanation
```
Left DF                Right DF
id  name               id  score
──  ────               ──  ─────
1   Alice              1   85
2   Bob                3   90
3   Carol              4   72

INNER join (on="id") — only rows with matching id in BOTH
  id  name   score
   1  Alice     85
   3  Carol     90

LEFT join — all rows from LEFT, matching from RIGHT (NaN if no match)
  id  name   score
   1  Alice     85
   2  Bob       NaN   ← Bob has no score entry
   3  Carol     90

RIGHT join — all rows from RIGHT, matching from LEFT (NaN if no match)
  id  name   score
   1  Alice     85
   3  Carol     90
   4  NaN       72    ← id=4 not in left df

OUTER join — all rows from BOTH (NaN where no match)
  id  name   score
   1  Alice     85
   2  Bob       NaN
   3  Carol     90
   4  NaN       72
```

## merge() Examples
```python
import pandas as pd

students = pd.DataFrame({
    "id":   [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "city": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
})

scores = pd.DataFrame({
    "id":      [1, 3, 4, 6],
    "score":   [85, 90, 68, 77],
    "subject": ["Math", "Math", "Science", "Math"]
})
# INNER join — only students who HAVE a score entry
inner = pd.merge(students, scores, on="id", how="inner")
print(inner)
#    id   name     city  score  subject
# 0   1  Alice    Delhi     85     Math
# 1   3  Carol    Delhi     90     Math
# 2   4   Dave  Chennai     68  Science
# LEFT join — all students, NaN if no score
left = pd.merge(students, scores, on="id", how="left")
print(left)
#    id   name     city  score  subject
# 0   1  Alice    Delhi   85.0     Math
# 1   2    Bob   Mumbai    NaN      NaN
# 2   3  Carol    Delhi   90.0     Math
# 3   4   Dave  Chennai   68.0  Science
# 4   5    Eve   Mumbai    NaN      NaN
# OUTER join — all from both
outer = pd.merge(students, scores, on="id", how="outer")
print(outer)
```

## Merging on Different Column Names
```python
students = pd.DataFrame({"student_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
scores   = pd.DataFrame({"sid":        [1, 2, 4], "score": [85, 72, 90]})
# left column is "student_id", right column is "sid"
result = pd.merge(students, scores, left_on="student_id", right_on="sid", how="left")
print(result)
#    student_id   name   sid  score
# 0           1  Alice   1.0   85.0
# 1           2    Bob   2.0   72.0
# 2           3  Carol   NaN    NaN
```

## Handling Overlapping Non-Key Columns
```python
df1 = pd.DataFrame({"id": [1, 2], "score": [85, 72], "year": [2023, 2023]})
df2 = pd.DataFrame({"id": [1, 2], "score": [88, 75], "year": [2024, 2024]})
# Both have "score" and "year" — Pandas adds suffixes
result = pd.merge(df1, df2, on="id", suffixes=("_before", "_after"))
print(result)
#    id  score_before  year_before  score_after  year_after
# 0   1            85         2023           88        2024
# 1   2            72         2023           75        2024
```

# 6. concat() — Stacking DataFrames
## What is it?
`concat()` **stacks** DataFrames on top of each other (rows) or side by side (columns).
No key column required — it just appends by position or index.
### Syntax
```python
pd.concat(objs, axis=0, ignore_index=False, join="outer", keys=None)
```
### Parameters
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `objs` | required | List of DataFrames to combine |
| `axis` | `0` | `0` = stack rows (vertical), `1` = stack columns (horizontal) |
| `ignore_index` | `False` | Reset index in result |
| `join` | `"outer"` | `"outer"` = keep all columns; `"inner"` = keep only shared columns |
| `keys` | `None` | Add a multi-level index to identify source DataFrame |

## Examples
```python
import pandas as pd

df_2022 = pd.DataFrame({
    "name":  ["Alice", "Bob"],
    "score": [85, 72],
    "year":  [2022, 2022]
})

df_2023 = pd.DataFrame({
    "name":  ["Carol", "Dave", "Eve"],
    "score": [90, 68, 88],
    "year":  [2023, 2023, 2023]
})
# Stack rows vertically
combined = pd.concat([df_2022, df_2023])
print(combined)
#     name  score  year
# 0  Alice     85  2022
# 1    Bob     72  2022
# 0  Carol     90  2023   ← index resets if not ignored
# 1   Dave     68  2023
# 2    Eve     88  2023
# Fix the index
combined = pd.concat([df_2022, df_2023], ignore_index=True)
print(combined.index)   # 0, 1, 2, 3, 4
# Add source label
combined = pd.concat([df_2022, df_2023], keys=["2022", "2023"])
print(combined)
#          name  score  year
# 2022 0  Alice     85  2022
#      1    Bob     72  2022
# 2023 0  Carol     90  2023
#      1   Dave     68  2023
#      2    Eve     88  2023
```

## concat — Outer vs Inner Join on Columns
```python
df_a = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [22, 25]})
df_b = pd.DataFrame({"id": [3, 4], "name": ["Carol", "Dave"], "city": ["Delhi", "Mumbai"]})
# outer — keep all columns (NaN where missing)
outer = pd.concat([df_a, df_b], ignore_index=True)
print(outer)
#    id   name   age     city
# 0   1  Alice  22.0      NaN
# 1   2    Bob  25.0      NaN
# 2   3  Carol   NaN    Delhi
# 3   4   Dave   NaN   Mumbai
# inner — keep only shared columns
inner = pd.concat([df_a, df_b], join="inner", ignore_index=True)
print(inner)
#    id   name
# 0   1  Alice
# 1   2    Bob
# 2   3  Carol
# 3   4   Dave
```

## concat — Horizontal (side by side)
```python
df_scores  = pd.DataFrame({"math": [85, 72, 90], "sci": [88, 74, 92]})
df_info    = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "age": [22, 25, 21]})
# Paste side by side (axis=1)
result = pd.concat([df_info, df_scores], axis=1)
print(result)
#     name  age  math  sci
# 0  Alice   22    85   88
# 1    Bob   25    72   74
# 2  Carol   21    90   92
```

# 7. join() — Index-Based Joining
## What is it?
`df.join()` is a shortcut for merging on the **index** of both DataFrames.
Internally it calls `pd.merge()` with `left_index=True, right_index=True`.

```python
df1 = pd.DataFrame({"score": [85, 72, 90]}, index=["Alice", "Bob", "Carol"])
df2 = pd.DataFrame({"age":   [22, 25, 21]}, index=["Alice", "Bob", "Carol"])

result = df1.join(df2)
print(result)
#        score  age
# Alice     85   22
# Bob       72   25
# Carol     90   21
# Left join (default)
df3 = pd.DataFrame({"city": ["Delhi", "Mumbai"]}, index=["Alice", "Dave"])
result = df1.join(df3, how="left")
print(result)
#        score    city
# Alice     85   Delhi
# Bob       72     NaN
# Carol     90     NaN
```

# 8. Comparison Table
## merge vs concat vs join
| Feature | merge() | concat() | join() |
|---------|---------|----------|--------|
| Combine on | Key column values | Just stacking | Index values |
| Direction | Horizontal | Vertical OR Horizontal | Horizontal |
| SQL equivalent | JOIN | UNION | JOIN on index |
| Key required | Yes (column name) | No | No (index) |
| Multiple DFs | 2 at a time | Any number | 2 at a time |
| Common use | Combine tables with an ID | Stack datasets | Combine after setting index |

## When to use which
| Situation | Use |
|-----------|-----|
| Two tables share an ID column | `pd.merge(df1, df2, on="id")` |
| Same structure, different months/years | `pd.concat([jan, feb, mar])` |
| Two DataFrames with matching row names | `df1.join(df2)` |
| Add columns from a lookup table | `pd.merge(df, lookup, on="key", how="left")` |

## Complete Day 3 Practice Code — Full Pipeline
```python
import pandas as pd
import numpy as np
# ── Build tables ──────────────────────────────────────
np.random.seed(42)
# Students table
students = pd.DataFrame({
    "student_id": range(1, 11),
    "name":  [f"Student_{i}" for i in range(1, 11)],
    "city":  np.random.choice(["Delhi", "Mumbai", "Chennai"], 10),
    "dept":  np.random.choice(["Math", "Science", "Commerce"], 10),
    "year":  np.random.choice([2022, 2023], 10)
})
# Scores table (not all students have scores)
scores = pd.DataFrame({
    "student_id": [1, 2, 3, 5, 6, 7, 8, 10],
    "math":    np.random.randint(50, 100, 8),
    "english": np.random.randint(50, 100, 8)
})

print("=" * 55)
print("STUDENTS TABLE")
print("=" * 55)
print(students)

print("\n" + "=" * 55)
print("SCORES TABLE")
print("=" * 55)
print(scores)
# ── MERGE ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("LEFT JOIN — all students + their scores")
print("=" * 55)
full = pd.merge(students, scores, on="student_id", how="left")
print(full)
print(f"\nStudents without scores: {full['math'].isnull().sum()}")
# Fill missing scores with 0
full["math"]    = full["math"].fillna(0)
full["english"] = full["english"].fillna(0)
full["total"]   = full["math"] + full["english"]
# ── GROUPBY ───────────────────────────────────────────
print("\n" + "=" * 55)
print("GROUPBY — Average total by city and dept")
print("=" * 55)
summary = full.groupby(["city", "dept"], as_index=False).agg(
    avg_total   = ("total",   "mean"),
    max_total   = ("total",   "max"),
    num_students = ("student_id", "count")
)
print(summary.sort_values("avg_total", ascending=False))
# ── CONCAT ────────────────────────────────────────────
print("\n" + "=" * 55)
print("CONCAT — Stack 2022 and 2023 students")
print("=" * 55)
year_2022 = full[full["year"] == 2022].reset_index(drop=True)
year_2023 = full[full["year"] == 2023].reset_index(drop=True)

stacked = pd.concat([year_2022, year_2023], keys=["2022", "2023"])
print(f"Combined rows: {len(stacked)}, 2022: {len(year_2022)}, 2023: {len(year_2023)}")
# ── TRANSFORM ─────────────────────────────────────────
print("\n" + "=" * 55)
print("TRANSFORM — performance vs city average")
print("=" * 55)
full["city_avg"]   = full.groupby("city")["total"].transform("mean")
full["vs_avg"]     = (full["total"] - full["city_avg"]).round(1)
full["above_avg"]  = full["vs_avg"] > 0

print(full[["name", "city", "total", "city_avg", "vs_avg", "above_avg"]].to_string())
# ── FINAL SUMMARY ─────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL CITY SUMMARY")
print("=" * 55)
city_report = full.groupby("city").agg(
    students      = ("student_id", "count"),
    avg_math      = ("math",    "mean"),
    avg_english   = ("english", "mean"),
    avg_total     = ("total",   "mean"),
    top_score     = ("total",   "max")
).round(1)
print(city_report)
# ── SAVE ──────────────────────────────────────────────
full.to_csv("student_report.csv", index=False)
print("\nFull report saved to student_report.csv")
```

## Summary: Full Pandas Curriculum Complete
| Day | Topics | File |
|-----|--------|------|
| Day 1 | Series, DataFrame, read_csv, read_excel, head/info/describe, isnull | Pandas_Day1_intro_series_dataframe_read_inspect.md |
| Day 2 | loc, iloc, conditional filter, dropna, fillna, drop_duplicates, rename, astype, sort | Pandas_Day2_indexing_cleaning_manipulation.md |
| Day 3 | groupby, agg, transform, merge (joins), concat, join | Pandas_Day3_groupby_merge_join.md |

> All 3 Pandas days are now complete.
> You have full coverage from raw data loading → cleaning → manipulation → aggregation → combining tables.