# 📅 NumPy Day 4 — Masking, Performance, File Handling, Advanced Topics
## Topics Covered
| # | Topic |
|---|-------|
| 10 | Masking & Filtering — conditional filtering, replacing values |
| 11 | Performance Concepts — vectorization, avoiding loops, memory |
| 12 | File Handling — save, load, loadtxt, genfromtxt |
| 13 | Advanced Topics — structured arrays, views vs copies, strides |

# 10. Masking & Filtering
## What is a Mask?
A **boolean mask** is an array of True/False values — same shape as your data array.
You use it to select, filter, or modify specific elements based on a condition.

## Conditional Filtering
```python
import numpy as np

scores = np.array([45, 82, 61, 90, 38, 75, 55, 88])
# Create the mask
pass_mask = scores >= 60
print("Mask:        ", pass_mask)
# [False  True  True  True False  True False  True]
# Apply the mask to get filtered values
print("Passing:     ", scores[pass_mask])   # [82 61 90 75 88]
# One-liner
print("Failing:     ", scores[scores < 60])  # [45 38 55]
```

## Boolean Masks on 2D Arrays
```python
mat = np.array([[5, 12, 3],
                [18, 7, 25],
                [9, 14, 6]])
# Mask — elements > 10
mask = mat > 10
print("Mask:\n", mask)
# [[False  True False]
#  [ True False  True]
#  [False  True False]]
# Filter — returns 1D array of matching elements
print("Values > 10:", mat[mask])   # [12 18 25 14]
```

## Replacing Values Based on Condition
Assign a new value directly through the mask:

```python
arr = np.array([10, -5, 30, -8, 15, -2, 40])
# Replace all negatives with 0
arr[arr < 0] = 0
print(arr)    # [10  0 30  0 15  0 40]
# Cap values above 25 at 25
arr[arr > 25] = 25
print(arr)    # [10  0 25  0 15  0 25]
```

## np.where()
`np.where()` is the most powerful conditional operation in NumPy.
### Syntax
```python
np.where(condition, value_if_true, value_if_false)
```
### What Does it Output?
```python
arr = np.array([5, 15, 3, 22, 8, 18])
# Replace: if > 10 → keep as is, else → 0
result = np.where(arr > 10, arr, 0)
print(result)    # [ 0 15  0 22  0 18]
# Replace: pass/fail label
labels = np.where(arr >= 10, "pass", "fail")
print(labels)    # ['fail' 'pass' 'fail' 'pass' 'fail' 'pass']
# Replace with different values each side
result2 = np.where(arr > 10, arr * 2, arr - 1)
print(result2)   # [ 4 30  2 44  7 36]
```
### np.where() with Indices only
```python
arr = np.array([5, 15, 3, 22, 8, 18])
# Where returns indices where condition is True
indices = np.where(arr > 10)
print(indices)         # (array([1, 3, 5]),)
print(arr[indices])    # [15 22 18]
```

## np.select() — Multiple Conditions
```python
arr = np.array([5, 15, 25, 35, 45, 55])

conditions = [
    arr < 10,
    (arr >= 10) & (arr < 30),
    arr >= 30
]
choices = ["low", "medium", "high"]

result = np.select(conditions, choices, default="unknown")
print(result)
# ['low' 'medium' 'medium' 'high' 'high' 'high']
```

## np.isnan() and np.isinf()
```python
arr = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0])

print(np.isnan(arr))    # [False  True False False False False]
print(np.isinf(arr))    # [False False False  True  True False]
print(np.isfinite(arr)) # [ True False  True False False  True]
# Remove NaN values
clean = arr[~np.isnan(arr)]
print(clean)            # [1.  3.  inf -inf  5.]
# Replace NaN with mean
arr_copy = arr.copy()
mean_val = np.nanmean(arr_copy)   # nanmean ignores NaN
arr_copy[np.isnan(arr_copy)] = mean_val
print(arr_copy)
```

# 11. Performance Concepts
## Why is NumPy Faster?
Python is an **interpreted language** — each line is executed one at a time, with overhead.
NumPy operations are implemented in **compiled C** and run directly on CPU — no Python overhead.

```
Python loop over 1M elements: ~100 ms
NumPy vectorized operation:   ~1 ms   ← 100× faster
```

## Vectorization
Vectorization means applying an operation to an **entire array at once** instead of looping.
### Loop vs Vectorized
```python
import numpy as np
import time

n = 1_000_000
a = np.random.rand(n)
b = np.random.rand(n)
# Python loop — slow
result_loop = np.zeros(n)
start = time.time()
for i in range(n):
    result_loop[i] = a[i] + b[i]
print(f"Loop time:    {(time.time()-start)*1000:.1f} ms")
# NumPy vectorized — fast
start = time.time()
result_np = a + b
print(f"NumPy time:   {(time.time()-start)*1000:.1f} ms")
# Verify results are identical
print("Same result:", np.allclose(result_loop, result_np))
```

## Avoiding Loops
### Anti-pattern vs NumPy pattern
```python
# BAD — Python loop
data = np.array([1.5, 2.3, 3.7, 4.1, 5.9])
total = 0
for x in data:
    total += x
# 5 iterations of Python
# GOOD — vectorized
total = data.sum()
# 1 C-level call — much faster
```

```python
# BAD — loop to apply function
result = np.zeros(len(data))
for i, x in enumerate(data):
    result[i] = x ** 2 + 2 * x + 1
# GOOD — vectorized
result = data ** 2 + 2 * data + 1
```

```python
# BAD — loop to normalize each row
mat = np.random.rand(1000, 50)
for i in range(len(mat)):
    mat[i] = (mat[i] - mat[i].mean()) / mat[i].std()
# GOOD — vectorized using axis + broadcasting
mat = np.random.rand(1000, 50)
mat = (mat - mat.mean(axis=1, keepdims=True)) / mat.std(axis=1, keepdims=True)
```

## keepdims Explained
When you reduce along an axis, the dimension disappears by default.
`keepdims=True` keeps it as size 1 — allows broadcasting.

```python
mat = np.array([[1, 2, 3],
                [4, 5, 6]])   # shape (2, 3)

mean_no_keep = mat.mean(axis=1)
print(mean_no_keep.shape)    # (2,)   ← can't broadcast back against (2,3)

mean_keep = mat.mean(axis=1, keepdims=True)
print(mean_keep.shape)       # (2, 1) ← can broadcast against (2,3)
print(mat - mean_keep)       # correctly subtracts row mean from each row
```

## Memory Efficiency
### Views vs Copies — Memory Impact
```python
# View — shares memory (no extra allocation)
arr = np.arange(1_000_000)
view = arr[::2]   # every other element — still a view

import sys
print(sys.getsizeof(arr))    # ~8 MB
print(sys.getsizeof(view))   # very small — just a view descriptor
# Copy — allocates new memory
copy = arr[::2].copy()
print(sys.getsizeof(copy))   # ~4 MB — actual data
```
### Choosing dtype to Save Memory
```python
# Default int64 uses 8 bytes per element
arr64 = np.ones(1_000_000, dtype=np.int64)
print(arr64.nbytes)    # 8,000,000 bytes = 8 MB
# int32 uses 4 bytes — halves memory
arr32 = np.ones(1_000_000, dtype=np.int32)
print(arr32.nbytes)    # 4,000,000 bytes = 4 MB
# float32 vs float64 — important in deep learning (GPU memory)
f64 = np.random.rand(1_000_000).astype(np.float64)
f32 = np.random.rand(1_000_000).astype(np.float32)
print(f64.nbytes)      # 8 MB
print(f32.nbytes)      # 4 MB  ← use float32 for large neural nets
```

# 12. File Handling with NumPy
## np.save() — Save a Single Array
### What is it?
Saves one array to a `.npy` binary file.
Binary format is fast to read/write and preserves dtype and shape exactly.
### What Does it Output?
```python
import numpy as np

data = np.array([[1.5, 2.3, 3.7],
                 [4.1, 5.9, 6.2]])
# Save
np.save("my_data.npy", data)
print("Saved.")
# Load
loaded = np.load("my_data.npy")
print("Loaded:")
print(loaded)
# [[1.5 2.3 3.7]
#  [4.1 5.9 6.2]]
print("dtype:", loaded.dtype)    # float64
print("shape:", loaded.shape)    # (2, 3)
```

## np.savez() — Save Multiple Arrays
### What is it?
Saves multiple arrays into one `.npz` file (a zip archive of `.npy` files).

```python
import numpy as np

X_train = np.random.rand(800, 10)
y_train = np.random.randint(0, 2, 800)
X_test  = np.random.rand(200, 10)
y_test  = np.random.randint(0, 2, 200)
# Save all arrays with names
np.savez("dataset.npz", X_train=X_train, y_train=y_train,
                         X_test=X_test,  y_test=y_test)
print("All arrays saved.")
# Load
data = np.load("dataset.npz")
print("Keys:", list(data.keys()))      # ['X_train', 'y_train', 'X_test', 'y_test']

X_train_loaded = data["X_train"]
y_train_loaded = data["y_train"]
print("X_train shape:", X_train_loaded.shape)   # (800, 10)
```

## np.loadtxt() — Load Text / CSV
### What is it?
Loads data from a plain text file (space-separated, comma-separated, etc.).
Works well for simple numeric data without headers.
### Syntax
```python
np.loadtxt(fname, delimiter=None, skiprows=0, usecols=None, dtype=float)
```

| Argument | Default | What it is |
|----------|---------|------------|
| `fname` | required | File path |
| `delimiter` | whitespace | Column separator |
| `skiprows` | `0` | Skip first N rows (e.g. header) |
| `usecols` | all | Column indices to load |
| `dtype` | `float` | Data type to use |
### What Does it Output?
```python
import numpy as np
# Load a CSV with header row
arr = np.loadtxt("data.csv", delimiter=",", skiprows=1)
print(arr.shape)
print(arr[:3])
# Load only specific columns (indices 0 and 2)
arr = np.loadtxt("data.csv", delimiter=",", skiprows=1, usecols=(0, 2))
```

## np.genfromtxt() — Load with Missing Values
### What is it?
More powerful than `loadtxt` — handles missing values, string columns, and mixed types.
### Syntax
```python
np.genfromtxt(fname, delimiter=",", names=True, dtype=None, filling_values=0)
```

| Argument | What it is |
|----------|------------|
| `names=True` | Use first row as column names |
| `dtype=None` | Auto-detect dtype per column |
| `filling_values` | Value to use for missing entries |
| `skip_header` | Skip N header rows |
### What Does it Output?
```python
import numpy as np
# Load with auto dtype and header names
data = np.genfromtxt("students.csv", delimiter=",", names=True, dtype=None, encoding=None)
print(data.dtype)      # structured: [('name', str), ('age', int), ('score', float)]
print(data["age"])     # access column by name
print(data["score"])
# Load with missing values handled
data = np.genfromtxt("messy.csv", delimiter=",", filling_values=0)
# All missing values replaced with 0
```

## np.savetxt() — Save to Text File
```python
import numpy as np

arr = np.array([[1.5, 2.3, 3.7],
                [4.1, 5.9, 6.2]])
# Save as CSV
np.savetxt("output.csv", arr, delimiter=",", header="col1,col2,col3",
           fmt="%.2f", comments="")
# Output file content:
# col1,col2,col3
# 1.50,2.30,3.70
# 4.10,5.90,6.20
```

# 13. Advanced Topics
## Structured Arrays
A **structured array** stores multiple data types (like a table or database row) in a NumPy array.

```python
import numpy as np
# Define the structure
dtype = np.dtype([
    ("name",  "U20"),     # Unicode string, max 20 chars
    ("age",   np.int32),
    ("score", np.float64)
])
# Create structured array
students = np.array([
    ("Alice", 22, 88.5),
    ("Bob",   25, 75.0),
    ("Carol", 21, 92.3)
], dtype=dtype)

print(students)
print(students["name"])    # ['Alice' 'Bob' 'Carol']
print(students["score"])   # [88.5 75.  92.3]
print(students[0])         # ('Alice', 22, 88.5)
# Sort by score
sorted_students = np.sort(students, order="score")
print(sorted_students["name"])   # ['Bob' 'Alice' 'Carol']
```

## Views vs Copies — In Depth
Understanding views vs copies is critical to avoid accidental data modification.

```
View   → a window into the same memory block (modifying it changes the original)
Copy   → an independent block of memory (safe to modify)
```

| Operation | Returns |
|-----------|---------|
| `arr[a:b]` | View |
| `arr.reshape(...)` | View (usually) |
| `arr.T` | View |
| `arr.ravel()` | View (usually) |
| `arr[arr > 5]` | Copy |
| `arr[[0, 2, 4]]` | Copy |
| `arr.flatten()` | Copy |
| `arr.copy()` | Copy |

```python
arr = np.arange(10)

view = arr[2:6]
view[0] = 999
print(arr)       # [ 0  1 999  3  4  5  6  7  8  9] ← arr changed!

arr2 = np.arange(10)
copy = arr2[2:6].copy()
copy[0] = 999
print(arr2)      # [0 1 2 3 4 5 6 7 8 9] ← arr2 unchanged
# Check if an array is a view
print(view.base is arr)    # True  ← view
print(copy.base is arr2)   # False ← copy (base is None)
```

## Strides
**Strides** describe how many bytes to jump in memory to reach the next element
along each dimension.

```python
arr = np.arange(12).reshape(3, 4)
print(arr.strides)
# (32, 8)
# → to move to next row:    jump 32 bytes (= 4 elements × 8 bytes/element)
# → to move to next column: jump  8 bytes (= 1 element  × 8 bytes/element)
```

Strides are why transpose and slicing return views — they just change stride information,
not the underlying data.

```python
T = arr.T
print(T.strides)    # (8, 32)  ← rows and columns swapped
```

## Advanced Broadcasting Edge Cases
```python
import numpy as np
# Case 1: Two 1D arrays of different sizes → outer product via broadcasting
a = np.array([1, 2, 3])          # shape (3,)
b = np.array([10, 20, 30, 40])   # shape (4,)
# Can't do a + b directly (3,) vs (4,) → incompatible
# But reshape a to column vector:
result = a[:, np.newaxis] + b   # (3,1) + (4,) → (3,4)
print(result)
# [[11 21 31 41]
#  [12 22 32 42]
#  [13 23 33 43]]
# Case 2: np.newaxis adds a dimension
arr = np.array([1, 2, 3])
print(arr.shape)                  # (3,)
print(arr[:, np.newaxis].shape)   # (3, 1)  ← column vector
print(arr[np.newaxis, :].shape)   # (1, 3)  ← row vector
```

## Advanced Linear Algebra
```python
import numpy as np
np.random.seed(42)
# Singular Value Decomposition (SVD) — used in PCA, recommendation systems
A = np.random.rand(5, 3)
U, S, Vt = np.linalg.svd(A)
print("U shape:", U.shape)    # (5, 5)
print("S shape:", S.shape)    # (3,)   ← singular values
print("Vt shape:", Vt.shape)  # (3, 3)
# Reconstruct A from SVD
S_mat = np.zeros(A.shape)
S_mat[:3, :3] = np.diag(S)
A_reconstructed = U @ S_mat @ Vt
print("Reconstruction error:", np.max(np.abs(A - A_reconstructed)))
# ~ 1e-15  ← essentially 0
# Matrix power
A_sq = np.array([[1, 1], [0, 1]])
print("A^5:", np.linalg.matrix_power(A_sq, 5))
# [[1 5]
#  [0 1]]
# Least squares solution (overdetermined system)
# Solves Ax ≈ b when A is not square
A = np.random.rand(10, 3)    # 10 equations, 3 unknowns
b = np.random.rand(10)
x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print("Least squares solution:", x)
```

## Day 4 Full Practice Code
```python
import numpy as np
np.random.seed(42)

print("=== MASKING & FILTERING ===")
data = np.random.randint(0, 100, size=10)
print("Data:", data)
print("Values > 50:", data[data > 50])
print("np.where >50 else 0:", np.where(data > 50, data, 0))

labels = np.select(
    [data < 33, (data >= 33) & (data < 66), data >= 66],
    ["low", "medium", "high"]
)
print("Labels:", labels)

print("\n=== VECTORIZATION SPEED ===")
import time
n = 500_000
a = np.random.rand(n)
b = np.random.rand(n)

t0 = time.time()
c_loop = [a[i] * b[i] for i in range(n)]
print(f"Loop:  {(time.time()-t0)*1000:.1f} ms")

t0 = time.time()
c_np = a * b
print(f"NumPy: {(time.time()-t0)*1000:.1f} ms")

print("\n=== FILE SAVING ===")
arr = np.random.rand(4, 3)
np.save("test_array.npy", arr)
loaded = np.load("test_array.npy")
print("Save/Load successful:", np.allclose(arr, loaded))

np.savez("multi.npz", arr1=arr, arr2=arr * 2)
d = np.load("multi.npz")
print("npz keys:", list(d.keys()))

print("\n=== VIEWS VS COPIES ===")
arr = np.arange(8)
view = arr[2:6]
view[0] = 99
print("arr after modifying view:", arr)   # arr changed

arr2 = np.arange(8)
copy = arr2[2:6].copy()
copy[0] = 99
print("arr2 after modifying copy:", arr2) # arr2 unchanged
```

> Day 4 Complete.
> You now understand boolean masking, conditional replacement, why NumPy is fast,
> how to save and load arrays, and the internal mechanics of views, copies, and strides.
>
> NumPy series complete! All 4 days cover every topic in the curriculum.
> Next: **Pandas Day 1 — Introduction, Series, DataFrame, Reading Data, Inspection**