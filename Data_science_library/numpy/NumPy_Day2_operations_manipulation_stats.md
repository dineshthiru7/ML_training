# 📅 NumPy Day 2 — Array Operations, Manipulation, Statistical Functions
## Topics Covered
| # | Topic |
|---|-------|
| 4 | Arithmetic, element-wise ops, broadcasting, comparison, ufuncs |
| 5 | Reshape, flatten, transpose, concatenate, stack, split |
| 6 | mean, median, std, var, min, max — with axis concept |

# 4. Array Operations
## Arithmetic Operations
NumPy performs arithmetic **element-wise** — no loop needed.

```python
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([2,  4,  5,  8])

print(a + b)     # [12 24 35 48]
print(a - b)     # [ 8 16 25 32]
print(a * b)     # [ 20  80 150 320]
print(a / b)     # [ 5.  5.  6.  5.]
print(a // b)    # [ 5  5  6  5]   ← integer division
print(a % b)     # [0 0 0 0]       ← remainder
print(a ** 2)    # [ 100  400  900 1600]
```
### Scalar Operations
Any scalar (single number) is applied to every element:

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr + 10)   # [11 12 13 14 15]
print(arr * 3)    # [ 3  6  9 12 15]
print(arr / 2)    # [0.5 1.  1.5 2.  2.5]
print(arr ** 2)   # [ 1  4  9 16 25]
```

## Element-wise Operations on 2D Arrays
```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(A + B)
# [[ 6  8]
#  [10 12]]

print(A * B)    # element-wise multiply (NOT matrix multiply)
# [[ 5 12]
#  [21 32]]
```
**Note:** `A * B` is element-wise multiplication.
For matrix multiplication (dot product), use `A @ B` or `np.dot(A, B)` — covered in Day 3.

## Comparison Operations
Return boolean arrays (True/False per element):

```python
arr = np.array([5, 12, 3, 8, 20, 1])

print(arr > 7)       # [False  True False  True  True False]
print(arr == 8)      # [False False False  True False False]
print(arr != 5)      # [False  True  True  True  True  True]
print(arr >= 10)     # [False  True False False  True False]
# Count how many pass the condition
print((arr > 7).sum())   # 3   — number of True values
print((arr > 7).mean())  # 0.5 — fraction that is True
```

## Universal Functions (ufuncs)
ufuncs are **vectorized functions** that operate element-wise on arrays without loops.
They are implemented in compiled C — extremely fast.
### Math ufuncs
```python
arr = np.array([1, 4, 9, 16, 25])

print(np.sqrt(arr))       # [1. 2. 3. 4. 5.]
print(np.square(arr))     # [  1  16  81 256 625]
print(np.abs(np.array([-3, 4, -5])))   # [3 4 5]
print(np.log(arr))        # natural log: [0. 1.386 2.197 2.773 3.219]
print(np.log2(arr))       # log base 2
print(np.log10(arr))      # log base 10
print(np.exp(np.array([0, 1, 2])))     # e^x: [1.     2.718  7.389]
```
### Trigonometric ufuncs
```python
angles = np.linspace(0, np.pi, 5)
print(np.sin(angles))     # [0.  0.707 1.  0.707 0.]
print(np.cos(angles))     # [1.  0.707 0. -0.707 -1.]
```
### Two-array ufuncs
```python
a = np.array([1, 5, 3, 9])
b = np.array([4, 2, 7, 6])

print(np.add(a, b))       # [5  7 10 15]  same as a+b
print(np.multiply(a, b))  # [ 4 10 21 54]
print(np.maximum(a, b))   # [4 5 7 9]  element-wise max
print(np.minimum(a, b))   # [1 2 3 6]  element-wise min
print(np.power(a, 2))     # [ 1 25  9 81]
```
### Rounding ufuncs
```python
arr = np.array([1.2, 2.7, 3.5, -1.8])

print(np.round(arr))      # [ 1.  3.  4. -2.]  round to nearest
print(np.floor(arr))      # [ 1.  2.  3. -2.]  round down
print(np.ceil(arr))       # [ 2.  3.  4. -1.]  round up
print(np.trunc(arr))      # [ 1.  2.  3. -1.]  remove decimal part
```

# 5. Array Manipulation
## reshape()
### What is it?
`reshape()` gives an array a **new shape** without changing its data.
The total number of elements must stay the same.
### Syntax
```python
arr.reshape(new_shape)
np.reshape(arr, new_shape)
```
### What Happens Inside
reshape returns a **view** (not a copy) when possible.
It reinterprets the same memory block with a different shape.

```
Original: 12 elements in a flat row
Reshape (3, 4): same 12 elements, read as 3 rows × 4 cols
```
### What Does it Output?
```python
arr = np.arange(12)          # [0 1 2 3 4 5 6 7 8 9 10 11]
print(arr.shape)             # (12,)

mat = arr.reshape(3, 4)
print(mat)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(mat.shape)             # (3, 4)
# -1 lets NumPy calculate that dimension automatically
print(arr.reshape(2, -1))    # NumPy calculates: 12/2 = 6
# shape → (2, 6)

print(arr.reshape(-1, 3))    # NumPy calculates: 12/3 = 4
# shape → (4, 3)
# 3D
print(arr.reshape(2, 3, 2))
# shape → (2, 3, 2)
```

## Flattening — ravel() and flatten()
### What are they?
Both convert a multi-dimensional array into a 1D array.

| | `ravel()` | `flatten()` |
|--|-----------|-------------|
| Returns | View (if possible) | Always a copy |
| Memory | Efficient | Uses more memory |
| When to use | Just need 1D temporarily | Need independent 1D copy |

```python
mat = np.array([[1, 2, 3],
                [4, 5, 6]])

print(mat.ravel())     # [1 2 3 4 5 6]
print(mat.flatten())   # [1 2 3 4 5 6]
# Proof ravel returns a view
r = mat.ravel()
r[0] = 99
print(mat)             # [[99  2  3] [4  5  6]] ← mat changed!
# Proof flatten returns a copy
f = mat.flatten()
f[0] = 0
print(mat)             # [[99  2  3] [4  5  6]] ← mat unchanged
```

## Transpose — .T
### What is it?
Transpose **flips rows and columns** — what was row becomes column.

For 2D: shape `(m, n)` → `(n, m)`

```python
mat = np.array([[1, 2, 3],
                [4, 5, 6]])
print(mat.shape)       # (2, 3)

T = mat.T
print(T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(T.shape)         # (3, 2)
```

Transpose returns a **view** — very efficient, no data is copied.

## Concatenation — np.concatenate()
### What is it?
`np.concatenate()` joins multiple arrays along an **existing axis**.
### Syntax
```python
np.concatenate((arr1, arr2, ...), axis=0)
```
### What Does it Output?
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])    # shape (2, 2)
b = np.array([[5, 6], [7, 8]])    # shape (2, 2)
# axis=0 → join along rows (stack vertically)
print(np.concatenate((a, b), axis=0))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# shape: (4, 2)
# axis=1 → join along columns (side by side)
print(np.concatenate((a, b), axis=1))
# [[1 2 5 6]
#  [3 4 7 8]]
# shape: (2, 4)
```

## Stacking — vstack, hstack, dstack
### What are they?
Convenience functions that stack arrays along specific axes.

| Function | Full name | Direction | Equivalent |
|----------|-----------|-----------|------------|
| `np.vstack` | Vertical stack | Rows (axis=0) | `concatenate(axis=0)` |
| `np.hstack` | Horizontal stack | Columns (axis=1) | `concatenate(axis=1)` |
| `np.dstack` | Depth stack | 3rd axis | Stacks into 3D |

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.vstack((a, b)))
# [[1 2 3]
#  [4 5 6]]
# shape: (2, 3)

print(np.hstack((a, b)))
# [1 2 3 4 5 6]
# shape: (6,)
# 2D examples
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.vstack((A, B)))    # (4, 2)
print(np.hstack((A, B)))    # (2, 4)
```

## Splitting — np.split()
### What is it?
`np.split()` divides an array into multiple sub-arrays.
### Syntax
```python
np.split(arr, indices_or_sections, axis=0)
np.hsplit(arr, sections)   # split along columns
np.vsplit(arr, sections)   # split along rows
```
### What Does it Output?
```python
arr = np.arange(12)
# Split into 3 equal parts
parts = np.split(arr, 3)
print(parts)
# [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9, 10, 11])]
# Split at specific indices
parts = np.split(arr, [3, 7])   # split before index 3 and 7
print(parts)
# [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9, 10, 11])]
# 2D split
mat = np.arange(16).reshape(4, 4)
top, bottom = np.vsplit(mat, 2)
print("Top:\n",    top)
print("Bottom:\n", bottom)

left, right = np.hsplit(mat, 2)
print("Left:\n",  left)
print("Right:\n", right)
```

# 6. Statistical Functions
## The Axis Concept — VERY IMPORTANT
Before learning stats functions, understand the `axis` parameter.

```
Matrix (2D array):
     col0  col1  col2
row0 [  1,    2,    3 ]
row1 [  4,    5,    6 ]
row2 [  7,    8,    9 ]

axis=0 → operate DOWN the columns (collapse rows into one)
axis=1 → operate ACROSS the rows (collapse columns into one)
```

Visual:

```
axis=0 (collapse rows):          axis=1 (collapse columns):
  ↓ ↓ ↓                            → → →
[ 1  2  3 ]                      [ 1  2  3 ]  →  6
[ 4  5  6 ]  →  [12 15 18]       [ 4  5  6 ]  →  15
[ 7  8  9 ]                      [ 7  8  9 ]  →  24
  (sum of each column)              (sum of each row)
```

## np.mean()
### What is it?
Computes the **arithmetic mean** (average) of array elements.

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
### What Does it Output?
```python
import numpy as np

arr = np.array([2, 4, 6, 8, 10])
print(np.mean(arr))          # 6.0

mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(np.mean(mat))          # 5.0   ← mean of ALL elements
print(np.mean(mat, axis=0))  # [4. 5. 6.]  ← mean of each column
print(np.mean(mat, axis=1))  # [2. 5. 8.]  ← mean of each row
```

## np.median()
### What is it?
Computes the **median** — the middle value when sorted.
More robust to outliers than mean.

```python
arr = np.array([1, 2, 3, 100, 5])
print(np.mean(arr))      # 22.2  ← heavily pulled by 100
print(np.median(arr))    # 3.0   ← not affected by outlier

mat = np.array([[3, 1, 2],
                [6, 4, 5]])
print(np.median(mat))              # 3.5   ← overall
print(np.median(mat, axis=0))     # [4.5 2.5 3.5] ← per column
print(np.median(mat, axis=1))     # [2.  5. ]     ← per row
```

## np.std()
### What is it?
Standard deviation — measures **spread** (how far values are from the mean).

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

Small std = values clustered tightly. Large std = values spread out.

```python
tight  = np.array([9, 10, 11, 10, 10])
spread = np.array([1, 10, 20, 5, 30])

print(np.std(tight))    # 0.632  ← small spread
print(np.std(spread))   # 10.11  ← large spread

mat = np.array([[2, 4, 6], [1, 5, 9]])
print(np.std(mat, axis=0))   # std of each column
print(np.std(mat, axis=1))   # std of each row
```

## np.var()
### What is it?
Variance = std² — measures spread without the square root.

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

```python
arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(np.var(arr))    # 4.0
print(np.std(arr))    # 2.0  ← sqrt(4.0)
```

## np.min() and np.max()
### What are they?
Minimum and maximum values in the array.

```python
arr = np.array([5, 1, 8, 3, 9, 2])

print(np.min(arr))     # 1
print(np.max(arr))     # 9
print(arr.min())       # 1   (method form)
print(arr.max())       # 9
# Position of min/max
print(np.argmin(arr))  # 1   ← index of minimum value
print(np.argmax(arr))  # 4   ← index of maximum value
# With axis
mat = np.array([[3, 1, 4],
                [1, 5, 9],
                [2, 6, 5]])

print(np.min(mat, axis=0))     # [1 1 4]  ← min of each column
print(np.max(mat, axis=1))     # [4 9 6]  ← max of each row
print(np.argmin(mat, axis=0))  # [1 0 0]  ← row index of column min
```

## np.sum() and np.cumsum()
```python
arr = np.array([1, 2, 3, 4, 5])

print(np.sum(arr))        # 15   ← total
print(np.cumsum(arr))     # [ 1  3  6 10 15] ← running total

mat = np.array([[1, 2, 3],
                [4, 5, 6]])

print(np.sum(mat))           # 21   ← sum of all elements
print(np.sum(mat, axis=0))   # [5 7 9]  ← column sums
print(np.sum(mat, axis=1))   # [ 6 15]  ← row sums
```

## np.percentile() and np.quantile()
```python
arr = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

print(np.percentile(arr, 25))   # 13.75  ← Q1
print(np.percentile(arr, 50))   # 27.5   ← median
print(np.percentile(arr, 75))   # 41.25  ← Q3
print(np.percentile(arr, [25, 50, 75]))  # [13.75 27.5  41.25]
```

## All Stats Functions Together
```python
import numpy as np

data = np.array([[82, 75, 90, 68],
                 [55, 92, 78, 85],
                 [91, 63, 74, 88]])

print("Shape:", data.shape)
print("Overall mean:", np.mean(data))
print("Column means:", np.mean(data, axis=0))   # mean of each student feature
print("Row means:   ", np.mean(data, axis=1))   # mean score per student

print("Overall std:", np.std(data))
print("Min per col:", np.min(data, axis=0))
print("Max per col:", np.max(data, axis=0))
print("Percentile 75:", np.percentile(data, 75))
```

## Day 2 Full Practice Code
```python
import numpy as np

print("=== OPERATIONS ===")
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print("a + b:", a + b)
print("b / a:", b / a)
print("sqrt:", np.sqrt(b))
print("a > 2:", a > 2)

print("\n=== RESHAPE & FLATTEN ===")
arr = np.arange(24)
mat = arr.reshape(4, 6)
print("Reshaped (4,6):\n", mat)
print("Flattened:", mat.flatten())
print("Ravel:", mat.ravel())
print("Transpose shape:", mat.T.shape)   # (6, 4)

print("\n=== CONCATENATE & STACK ===")
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print("vstack:\n", np.vstack((x, y)))
print("hstack:\n", np.hstack((x, y)))

print("\n=== STATISTICS ===")
scores = np.array([[85, 72, 91],
                   [60, 88, 74],
                   [95, 68, 82]])
print("Mean per student:", np.mean(scores, axis=1))
print("Mean per subject:", np.mean(scores, axis=0))
print("Top scorer row:", np.argmax(np.mean(scores, axis=1)))
print("Std overall:", np.std(scores))
```

> Day 2 Complete.
> You can now perform all mathematical operations on arrays without loops,
> reshape and reorganize data in any structure, and compute all major statistics.
>
> Next: **NumPy Day 3 — Random Module, Linear Algebra, Broadcasting**