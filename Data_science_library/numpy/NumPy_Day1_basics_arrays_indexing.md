# 📅 NumPy Day 1 — Basics, Arrays, Indexing & Slicing
## Topics Covered
| # | Topic |
|---|-------|
| 1 | What is NumPy and why it is used |
| 2 | Difference between Python list and NumPy array |
| 3 | Creating arrays — np.array, np.zeros, np.ones, np.arange, np.linspace, np.eye |
| 4 | Array attributes — shape, ndim, size, dtype |
| 5 | Indexing & Slicing — 1D, 2D, Boolean, Fancy |

# 1. What is NumPy?
## What is it?
NumPy (Numerical Python) is a Python library for **fast numerical computation**.

It introduces the **ndarray** (n-dimensional array) — a data structure that stores
elements of the same type in a contiguous block of memory and supports vectorized
mathematical operations without writing loops.

```
Python list:  [1, 2, 3, 4, 5]   → general purpose, slow for math
NumPy array:  np.array([1,2,3])  → optimized for math, 10–100× faster
```
### Why It Is Used
| Reason | Explanation |
|--------|-------------|
| Speed | Operations run in compiled C under the hood, not Python |
| Memory | Stores data in a compact, typed block — less overhead per element |
| Broadcasting | Operate on arrays of different shapes without loops |
| Linear algebra | Built-in matrix math: dot products, inverse, eigenvalues |
| Foundation for ML | pandas, sklearn, TensorFlow, PyTorch all use NumPy arrays internally |

## Installing and Importing
```python
# Install (run once in terminal)
pip install numpy
# Import (do this at the top of every script)
import numpy as np
```

`np` is the standard alias used everywhere in data science.

## Python List vs NumPy Array
| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Type of elements | Can mix types `[1, "a", 3.0]` | All same type |
| Memory | Each element is a separate Python object | Compact, contiguous block |
| Math operations | Must loop manually | Element-wise by default |
| Speed (1M elements) | ~100 ms | ~1 ms |
| Dimensions | Nested lists for 2D | Native multi-dimensional |
### Code Comparison
```python
import numpy as np
import time
# Python list — add 1 to each element
py_list = list(range(1_000_000))
start = time.time()
py_result = [x + 1 for x in py_list]
print(f"List time: {(time.time()-start)*1000:.1f} ms")
# NumPy array — same operation
np_arr = np.arange(1_000_000)
start = time.time()
np_result = np_arr + 1      # no loop needed
print(f"NumPy time: {(time.time()-start)*1000:.1f} ms")
# NumPy is ~50–100× faster
```

# 2. Creating NumPy Arrays
# np.array()
## What is it?
`np.array()` converts a Python list (or nested list) into a NumPy array.
## Syntax
```python
np.array(object, dtype=None)
```
## What Input Does it Accept?
| Argument | Type | What it is |
|----------|------|------------|
| `object` | list, tuple, nested list | The data to convert |
| `dtype` | string or NumPy type | Force a specific data type |
## What Happens Inside
1. Reads the Python list
2. Infers dtype from the data (or uses specified dtype)
3. Allocates a contiguous memory block
4. Copies all values into that block
## What Does it Output?
```python
import numpy as np
# 1D array
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>
# 2D array (matrix)
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
# Force float dtype
c = np.array([1, 2, 3], dtype=float)
print(c)           # [1. 2. 3.]
print(c.dtype)     # float64
# Mixed list — NumPy upcasts all to the broadest type
d = np.array([1, 2.5, 3])
print(d.dtype)     # float64 (int was upcast to float)
```

# np.zeros() and np.ones()
## What are they?
- `np.zeros()` → creates an array filled with **0.0**
- `np.ones()` → creates an array filled with **1.0**

Use them to initialize arrays before filling with actual values.
## Syntax
```python
np.zeros(shape, dtype=float)
np.ones(shape, dtype=float)
```

`shape` can be:
- An integer for 1D: `np.zeros(5)`
- A tuple for multi-dimensional: `np.zeros((3, 4))`
## What Does it Output?
```python
import numpy as np
# 1D
print(np.zeros(5))
# [0. 0. 0. 0. 0.]
# 2D — 3 rows, 4 columns
print(np.zeros((3, 4)))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# Integer dtype
print(np.ones((2, 3), dtype=int))
# [[1 1 1]
#  [1 1 1]]
```

# np.arange()
## What is it?
`np.arange()` creates an array of evenly spaced values within a range.
It is the NumPy equivalent of Python's `range()`.
## Syntax
```python
np.arange(start, stop, step, dtype=None)
```

| Argument | Default | What it is |
|----------|---------|------------|
| `start` | `0` | Starting value (included) |
| `stop` | required | End value (NOT included) |
| `step` | `1` | Spacing between values |
## What Does it Output?
```python
import numpy as np

print(np.arange(5))          # [0 1 2 3 4]
print(np.arange(2, 10))      # [2 3 4 5 6 7 8 9]
print(np.arange(0, 1, 0.2))  # [0.  0.2 0.4 0.6 0.8]
print(np.arange(10, 0, -2))  # [10  8  6  4  2]
```

# np.linspace()
## What is it?
`np.linspace()` creates an array of **N evenly spaced values** between start and stop.

The key difference from `arange`:
- `arange` — you specify the **step size**, number of elements varies
- `linspace` — you specify **how many elements**, step size is calculated automatically
## Syntax
```python
np.linspace(start, stop, num=50, endpoint=True)
```

| Argument | Default | What it is |
|----------|---------|------------|
| `start` | required | First value |
| `stop` | required | Last value |
| `num` | `50` | How many values to generate |
| `endpoint` | `True` | Include stop value in output |
## What Does it Output?
```python
import numpy as np

print(np.linspace(0, 1, 5))
# [0.   0.25 0.5  0.75 1.  ]

print(np.linspace(0, 10, 6))
# [ 0.  2.  4.  6.  8. 10.]
# endpoint=False — stop value excluded
print(np.linspace(0, 1, 5, endpoint=False))
# [0.  0.2 0.4 0.6 0.8]
```
### When to Use linspace
- Plotting smooth curves: `x = np.linspace(-np.pi, np.pi, 500)`
- Generating test inputs for a function
- Creating evenly spaced probability values

# np.eye()
## What is it?
`np.eye()` creates an **identity matrix** — a square matrix with 1s on the diagonal and 0s everywhere else.

Used in linear algebra, matrix operations, and ML math.
## Syntax
```python
np.eye(N, M=None, k=0, dtype=float)
```

| Argument | Default | What it is |
|----------|---------|------------|
| `N` | required | Number of rows |
| `M` | `N` | Number of columns (default = square) |
| `k` | `0` | Diagonal offset (0=main, 1=above, -1=below) |
## What Does it Output?
```python
import numpy as np

print(np.eye(3))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

print(np.eye(3, k=1))   # diagonal shifted up by 1
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]

print(np.eye(3, dtype=int))
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]
```

# 3. Array Attributes
## What are Array Attributes?
After creating an array, you can inspect its structure using built-in attributes.
No parentheses needed — they are properties, not methods.
## All Attributes with Examples
```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.shape)   # (2, 3)  — 2 rows, 3 columns
print(arr.ndim)    # 2       — number of dimensions
print(arr.size)    # 6       — total number of elements
print(arr.dtype)   # int64   — data type of elements
print(arr.itemsize)# 8       — bytes per element (int64 = 8 bytes)
print(arr.nbytes)  # 48      — total bytes = size × itemsize
```
### Shape in Detail
```python
a = np.array([1, 2, 3])
print(a.shape)    # (3,)   ← 1D: tuple with one element

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)    # (2, 3) ← 2D: (rows, columns)

c = np.zeros((2, 3, 4))
print(c.shape)    # (2, 3, 4) ← 3D: (depth, rows, columns)
```
### dtype in Detail
| dtype | Python equivalent | Size | When |
|-------|------------------|------|------|
| `int32` | int | 4 bytes | Integer data, memory saving |
| `int64` | int | 8 bytes | Default integer |
| `float32` | float | 4 bytes | ML models (saves GPU memory) |
| `float64` | float | 8 bytes | Default float |
| `bool` | bool | 1 byte | True/False masks |
| `str_` | str | varies | String arrays |

```python
# Change dtype
arr = np.array([1.7, 2.9, 3.1])
print(arr.astype(int))    # [1 2 3]  — truncates decimals

arr2 = np.array([0, 1, 0, 1])
print(arr2.astype(bool))  # [False  True False  True]
```

# 4. Indexing & Slicing
## 1D Array Indexing
Just like Python lists — zero-indexed. Negative indices count from the end.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
#               0    1   2   3   4   ← positive index
#              -5   -4  -3  -2  -1   ← negative index

print(arr[0])    # 10
print(arr[3])    # 40
print(arr[-1])   # 50  ← last element
print(arr[-2])   # 40
```

## 1D Array Slicing
Syntax: `arr[start:stop:step]`

Rules:
- `start` is **included**
- `stop` is **NOT included**
- Default start=0, stop=end, step=1

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70])

print(arr[1:4])    # [20 30 40]  — indices 1,2,3
print(arr[:3])     # [10 20 30]  — from start to index 3
print(arr[4:])     # [50 60 70]  — from index 4 to end
print(arr[::2])    # [10 30 50 70] — every 2nd element
print(arr[::-1])   # [70 60 50 40 30 20 10] — reversed
```
### Important: Slices Are Views, Not Copies
```python
arr = np.array([10, 20, 30, 40, 50])
sub = arr[1:4]   # this is a VIEW — shares memory with arr
sub[0] = 999
print(arr)       # [10 999 30 40 50] ← arr was modified!
# To get a real copy:
sub_copy = arr[1:4].copy()
sub_copy[0] = 0
print(arr)       # [10 999 30 40 50] ← arr unchanged
```

## 2D Array Indexing (Matrix Style)
Syntax: `arr[row, col]` or `arr[row][col]`

```python
mat = np.array([[1,  2,  3,  4],
                [5,  6,  7,  8],
                [9, 10, 11, 12]])
#               col: 0   1   2   3
# row 0:         1   2   3   4
# row 1:         5   6   7   8
# row 2:         9  10  11  12

print(mat[0, 0])    # 1   ← row 0, col 0
print(mat[1, 2])    # 7   ← row 1, col 2
print(mat[2, -1])   # 12  ← row 2, last column
```

## 2D Array Slicing
```python
mat = np.array([[1,  2,  3,  4],
                [5,  6,  7,  8],
                [9, 10, 11, 12]])
# Select entire row
print(mat[1, :])       # [5 6 7 8]  — row 1, all columns
print(mat[1])          # [5 6 7 8]  — same
# Select entire column
print(mat[:, 2])       # [3 7 11]   — all rows, column 2
# Sub-matrix
print(mat[0:2, 1:3])
# [[2 3]
#  [6 7]]
# Every other row
print(mat[::2, :])
# [[ 1  2  3  4]
#  [ 9 10 11 12]]
```

## Boolean Indexing (Conditional Filtering)
Create a boolean mask (True/False array) and use it to select elements.
### How it works
```python
arr = np.array([10, 25, 3, 47, 8, 30])
# Step 1: create boolean mask
mask = arr > 15
print(mask)     # [False  True False  True False  True]
# Step 2: use mask to filter
print(arr[mask])    # [25 47 30]
# One-liner (most common style)
print(arr[arr > 15])    # [25 47 30]
```
### More Boolean Indexing Examples
```python
import numpy as np

scores = np.array([45, 80, 62, 90, 35, 78, 55])
# Pass (>= 60)
print(scores[scores >= 60])       # [80 62 90 78]
# Between 50 and 80
print(scores[(scores >= 50) & (scores <= 80)])  # [80 62 78 55]
# Fail OR perfect
print(scores[(scores < 60) | (scores == 90)])   # [45 90 35 55]
# NOT condition
print(scores[~(scores >= 60)])    # [45 35 55]
```
### Modify Values Using Boolean Mask
```python
arr = np.array([10, -5, 30, -8, 15, -2])
arr[arr < 0] = 0   # replace negatives with 0
print(arr)         # [10  0 30  0 15  0]
```

## Fancy Indexing
Select elements using an array of indices (not a slice).

```python
arr = np.array([10, 20, 30, 40, 50, 60])
# Select specific indices
idx = [0, 2, 4]
print(arr[idx])    # [10 30 50]
# Or directly
print(arr[[1, 3, 5]])   # [20 40 60]
# Repeated indices allowed
print(arr[[0, 0, 3]])   # [10 10 40]
```
### Fancy Indexing on 2D Arrays
```python
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# Select rows 0 and 2
print(mat[[0, 2]])
# [[1 2 3]
#  [7 8 9]]
# Select specific (row, col) pairs
rows = [0, 1, 2]
cols = [0, 1, 2]
print(mat[rows, cols])    # [1 5 9]  ← diagonal
```
### Fancy vs Slicing
```python
arr = np.array([10, 20, 30, 40, 50])
# Slicing → returns a VIEW (shares memory)
s = arr[1:4]
s[0] = 999
print(arr)    # [10 999 30 40 50]  ← changed!
# Fancy indexing → returns a COPY (independent)
f = arr[[1, 2, 3]]
f[0] = 0
print(arr)    # [10 999 30 40 50]  ← unchanged
```

## Indexing Summary Table
| Method | Syntax | Returns | View or Copy? |
|--------|--------|---------|---------------|
| Integer | `arr[2]` | Single element | N/A |
| Negative | `arr[-1]` | Single element | N/A |
| Slice | `arr[1:4]` | Sub-array | View |
| 2D | `arr[0, 2]` | Single element | N/A |
| 2D slice | `arr[0:2, 1:3]` | Sub-matrix | View |
| Boolean | `arr[arr > 5]` | Filtered array | Copy |
| Fancy | `arr[[0, 2, 4]]` | Selected elements | Copy |

## Day 1 Full Practice Code
```python
import numpy as np

print("=" * 45)
print("CREATING ARRAYS")
print("=" * 45)

a = np.array([10, 20, 30, 40, 50])
b = np.zeros((3, 3))
c = np.ones((2, 4), dtype=int)
d = np.arange(0, 20, 3)
e = np.linspace(0, 1, 6)
f = np.eye(3)

print("array:   ", a)
print("zeros:   \n", b)
print("ones:    \n", c)
print("arange:  ", d)
print("linspace:", e)
print("eye:     \n", f)

print("\n" + "=" * 45)
print("ARRAY ATTRIBUTES")
print("=" * 45)
mat = np.array([[1, 2, 3], [4, 5, 6]])
print("shape:", mat.shape)   # (2, 3)
print("ndim: ", mat.ndim)    # 2
print("size: ", mat.size)    # 6
print("dtype:", mat.dtype)   # int64

print("\n" + "=" * 45)
print("INDEXING & SLICING")
print("=" * 45)
arr = np.array([5, 15, 25, 35, 45, 55])
print("arr[2]        :", arr[2])           # 25
print("arr[1:4]      :", arr[1:4])         # [15 25 35]
print("arr[::-1]     :", arr[::-1])        # reversed
print("arr > 20      :", arr[arr > 20])    # boolean
print("arr[[0,2,4]]  :", arr[[0, 2, 4]])  # fancy
```

> Day 1 Complete.
> You can now create arrays in every major way, inspect their structure,
> and select any subset of data using slicing, boolean masks, and fancy indexing.
>
> Next: **NumPy Day 2 — Array Operations, Manipulation, Statistical Functions**