# 📅 NumPy Day 3 — Random Module, Linear Algebra, Broadcasting
## Topics Covered
| # | Topic |
|---|-------|
| 7 | Random Module — rand, randn, randint, seed, sampling |
| 8 | Linear Algebra — dot, @, inv, det, eig, solve |
| 9 | Broadcasting — rules, automatic dimension expansion, ML examples |

# 7. Random Module
## Why Random?
Almost every machine learning workflow needs randomness:
- Generating synthetic datasets for testing
- Weight initialization in neural networks
- Train/test split shuffling
- Stochastic gradient descent sampling
- Simulating real-world noisy data

## np.random.seed()
### What is it?
Sets the **random seed** — makes all subsequent random calls produce the same results every time.

Without a seed: different numbers each run.
With a seed: same numbers every run (reproducible).
### What Does it Output?
```python
import numpy as np
# Without seed — different each run
print(np.random.rand(3))   # e.g. [0.374 0.951 0.732]
print(np.random.rand(3))   # e.g. [0.599 0.156 0.058]  ← different!
# With seed — same every run
np.random.seed(42)
print(np.random.rand(3))   # [0.374 0.951 0.732]  always

np.random.seed(42)
print(np.random.rand(3))   # [0.374 0.951 0.732]  same again
```
**Best practice:** Always set `np.random.seed(42)` at the top of your script for reproducibility.

## np.random.rand()
### What is it?
Generates random floats from a **uniform distribution** in [0.0, 1.0).

Every value between 0 and 1 is equally likely.
### Syntax
```python
np.random.rand(d0, d1, ..., dn)
```

The arguments directly specify the shape — no tuple needed.
### What Does it Output?
```python
import numpy as np
np.random.seed(42)
# 1D — 5 random floats
print(np.random.rand(5))
# [0.3745 0.9507 0.7319 0.5986 0.1560]
# 2D — 3 rows × 4 cols
print(np.random.rand(3, 4))
# [[0.155 0.058 0.866 0.708]
#  [0.020 0.969 0.832 0.212]
#  [0.181 0.183 0.304 0.524]]
# Scale to any range [low, high]
# Formula: low + (high - low) * np.random.rand(shape)
data = 10 + 90 * np.random.rand(5)   # random floats from 10 to 100
print(np.round(data, 1))
```

## np.random.randn()
### What is it?
Generates random floats from a **standard normal distribution** (mean=0, std=1).

Normal distribution: most values near 0, fewer as you go further away.
This is the bell curve.
### Syntax
```python
np.random.randn(d0, d1, ..., dn)
```
### What Does it Output?
```python
import numpy as np
np.random.seed(42)

print(np.random.randn(5))
# [ 0.497  -0.138  0.647  1.523  1.465]
# values mostly between -3 and 3
# Scale to mean=mu, std=sigma
# Formula: mu + sigma * np.random.randn(shape)
heights = 170 + 10 * np.random.randn(1000)   # mean 170 cm, std 10 cm
print(f"Mean:  {heights.mean():.1f}")         # ≈ 170
print(f"Std:   {heights.std():.1f}")          # ≈ 10
print(f"Min:   {heights.min():.1f}")
print(f"Max:   {heights.max():.1f}")
```

## np.random.randint()
### What is it?
Generates random **integers** from a specified range.
### Syntax
```python
np.random.randint(low, high=None, size=None)
```

| Argument | What it is |
|----------|------------|
| `low` | Minimum value (included) |
| `high` | Maximum value (NOT included) |
| `size` | Output shape — int for 1D, tuple for 2D |
### What Does it Output?
```python
import numpy as np
np.random.seed(42)
# Single integer from 0 to 9
print(np.random.randint(10))       # e.g. 6
# Range [5, 20)
print(np.random.randint(5, 20))    # e.g. 11
# Array of 6 integers in [1, 7)  — like dice rolls
print(np.random.randint(1, 7, size=6))    # [4 1 4 1 3 3]
# 2D array
print(np.random.randint(0, 100, size=(3, 4)))
# [[51 92 14 71]
#  [60 20 82 86]
#  [74 74 87 99]]
```

## Random Sampling
### np.random.choice()
Select random elements from an existing array:

```python
import numpy as np
np.random.seed(42)

arr = np.array([10, 20, 30, 40, 50])
# Random selection of 3 elements (with replacement by default)
print(np.random.choice(arr, size=3))
# [20 50 20]  — 20 appears twice (replacement allowed)
# Without replacement
print(np.random.choice(arr, size=3, replace=False))
# [20 30 10]  — all unique
# With custom probabilities
prob = [0.1, 0.1, 0.1, 0.1, 0.6]
print(np.random.choice(arr, size=5, p=prob))
# 50 appears most often due to 60% probability
```
### np.random.shuffle() and np.random.permutation()
```python
arr = np.array([1, 2, 3, 4, 5, 6])
# shuffle — modifies array in place
np.random.shuffle(arr)
print(arr)     # [3 6 1 4 5 2]  ← arr is shuffled
# permutation — returns a new shuffled array, original unchanged
arr2 = np.array([1, 2, 3, 4, 5, 6])
shuffled = np.random.permutation(arr2)
print(shuffled)   # shuffled copy
print(arr2)       # [1 2 3 4 5 6]  ← unchanged
```
### Other Distributions
```python
np.random.seed(42)
# Binomial — n trials, p probability of success
print(np.random.binomial(n=10, p=0.5, size=5))
# e.g. [6 5 4 7 5]  — heads in 10 coin flips
# Poisson — events per interval
print(np.random.poisson(lam=3, size=5))
# e.g. [3 3 2 4 3]
# Exponential — time between events
print(np.random.exponential(scale=1.0, size=5))
# Uniform between low and high
print(np.random.uniform(low=5, high=15, size=5))
```

# 8. Linear Algebra
## Why Linear Algebra in NumPy?
Every machine learning algorithm under the hood is matrix math:
- Linear Regression: $\hat{y} = X\beta$, $\beta = (X^TX)^{-1}X^Ty$
- PCA: eigendecomposition of covariance matrix
- Neural networks: weight matrices multiplied with inputs
- SVM: solving quadratic optimization involving matrix operations

## Matrix Multiplication — np.dot() and @
### What is it?
Matrix multiplication: multiply rows of the first matrix by columns of the second.

For matrices $A$ (m×n) and $B$ (n×p):

`Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ`

Result shape: (m×p)
### What Does it Output?
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])    # shape (2, 2)

B = np.array([[5, 6],
              [7, 8]])    # shape (2, 2)
# Three equivalent ways to do matrix multiplication
print(np.dot(A, B))
print(A @ B)              # preferred modern syntax
print(A.dot(B))
# All produce:
# [[19 22]
#  [43 50]]
# Verify manually:
# C[0,0] = 1*5 + 2*7 = 5+14 = 19  ✓
# C[0,1] = 1*6 + 2*8 = 6+16 = 22  ✓
```
### Element-wise vs Matrix Multiplication
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)      # element-wise: [[ 5 12] [21 32]]
print(A @ B)      # matrix multiply: [[19 22] [43 50]]
```
### Dot Product of 1D Vectors
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 4+10+18 = 32
```

## np.linalg.inv() — Matrix Inverse
### What is it?
Computes the **inverse** of a square matrix $A$, denoted $A^{-1}$.

Property: $A \cdot A^{-1} = I$ (identity matrix)

Used in Linear Regression: $\beta = (X^TX)^{-1}X^Ty$
### What Does it Output?
```python
import numpy as np

A = np.array([[2, 1],
              [5, 3]])

A_inv = np.linalg.inv(A)
print(A_inv)
# [[ 3. -1.]
#  [-5.  2.]]
# Verify: A * A_inv should give identity matrix
print(np.round(A @ A_inv, 10))
# [[1. 0.]
#  [0. 1.]]   ← identity matrix ✓
```

## np.linalg.det() — Determinant
### What is it?
The **determinant** is a scalar value that describes properties of a matrix.

If det = 0 → matrix is **singular** (no inverse exists, system has no unique solution).
If det ≠ 0 → matrix is **invertible**.

For 2×2:

`det([[a, b], [c, d]]) = ad - bc`
### What Does it Output?
```python
import numpy as np

A = np.array([[2, 1],
              [5, 3]])
print(np.linalg.det(A))     # 2*3 - 1*5 = 6-5 = 1.0

B = np.array([[2, 4],
              [1, 2]])       # second row is multiple of first
print(np.linalg.det(B))     # 2*2 - 4*1 = 0.0  ← singular!
```

## np.linalg.eig() — Eigenvalues & Eigenvectors
### What is it?
For a square matrix $A$, eigenvectors $\mathbf{v}$ and eigenvalues $\lambda$ satisfy:

`A × v = λ × v`

The eigenvector does not change direction when multiplied by the matrix — it only scales.
The eigenvalue is the scale factor.

**Use in ML:** PCA uses eigendecomposition of the covariance matrix to find the principal components.
### What Does it Output?
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# [5. 2.]

print("Eigenvectors (columns):")
print(eigenvectors)
# [[ 0.894 -0.707]
#  [ 0.447  0.707]]
# Verify: A @ v = lambda * v
v1 = eigenvectors[:, 0]   # first eigenvector
lam1 = eigenvalues[0]     # first eigenvalue
print("A @ v1:", np.round(A @ v1, 5))
print("lam1 * v1:", np.round(lam1 * v1, 5))
# Both should be equal ✓
```

## np.linalg.solve() — Solving Linear Equations
### What is it?
Solves the system of linear equations $Ax = b$ for $x$.

Mathematically: $x = A^{-1}b$

But `solve()` is faster and more numerically stable than computing the inverse.
### Example
System of equations:
`2x + y = 8`
`x + 3y = 11`

```python
import numpy as np
# Coefficient matrix
A = np.array([[2, 1],
              [1, 3]])
# Right-hand side
b = np.array([8, 11])
# Solve for x
x = np.linalg.solve(A, b)
print("Solution:", x)      # [2.2 3.6]
# Verify: A @ x should equal b
print("Check:", np.round(A @ x, 5))   # [8. 11.] ✓
```

## Other linalg Functions
```python
import numpy as np

A = np.array([[3, 1],
              [1, 3]])
# Matrix rank
print(np.linalg.matrix_rank(A))    # 2  ← full rank
# Norm (magnitude of vector)
v = np.array([3, 4])
print(np.linalg.norm(v))           # 5.0  ← sqrt(9+16)
# Singular Value Decomposition (for PCA, data compression)
U, S, Vt = np.linalg.svd(A)
print("Singular values:", S)
# Trace (sum of diagonal elements)
print(np.trace(A))                 # 6  = 3+3
```

# 9. Broadcasting
## What is Broadcasting?
Broadcasting is NumPy's rule for performing operations on arrays of **different shapes**.

Instead of requiring exact shape match, NumPy automatically "stretches" the smaller array
to match the larger one — without actually copying data.

## The Problem Without Broadcasting
```python
# Without broadcasting, you would need to manually tile:
arr = np.array([[1, 2, 3],
                [4, 5, 6]])    # shape (2, 3)

row = np.array([10, 20, 30])   # shape (3,)
# Manual approach (slow, wasteful):
import numpy as np
row_tiled = np.tile(row, (2, 1))   # make 2 copies of row
result = arr + row_tiled
# [[11 22 33]
#  [14 25 36]]
```

Broadcasting does this automatically:

```python
result = arr + row    # works directly!
# [[11 22 33]
#  [14 25 36]]
```

## Broadcasting Rules
NumPy follows 3 rules when operating on two arrays:

**Rule 1:** If arrays have different number of dimensions, pad the smaller one with 1s on the LEFT.

```
arr  shape: (2, 3)
row  shape:    (3,)   → padded to (1, 3)
```

**Rule 2:** Arrays with size 1 along a dimension are stretched to match the other array.

```
arr  shape: (2, 3)
row  shape: (1, 3)   → stretched to (2, 3)

Now shapes match: (2, 3) + (2, 3) → works!
```

**Rule 3:** If sizes don't match and neither is 1, raise an error.

```
(2, 3) + (3, 2) → ERROR — shapes incompatible
(2, 3) + (2,)   → ERROR — 3 ≠ 2 and neither is 1
```

## Broadcasting Examples
### Scalar + Array
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr + 10)
# [[11 12 13]
#  [14 15 16]]
# scalar 10 is broadcast to shape (2, 3) automatically
```
### 1D Row + 2D Array
```python
arr = np.array([[1, 2, 3],     # (2, 3)
                [4, 5, 6]])
row = np.array([10, 20, 30])   # (3,) → broadcast to (2, 3)

print(arr + row)
# [[11 22 33]
#  [14 25 36]]
```
### Column Vector + 2D Array
```python
arr = np.array([[1, 2, 3],     # (2, 3)
                [4, 5, 6]])

col = np.array([[10],          # (2, 1) → broadcast to (2, 3)
                [20]])

print(arr + col)
# [[11 12 13]   ← row 0 + 10
#  [24 25 26]]  ← row 1 + 20
```
### Row + Column → Full Grid
```python
row = np.array([[0, 1, 2]])     # shape (1, 3)
col = np.array([[0], [10], [20]])  # shape (3, 1)

print(row + col)
# [[ 0  1  2]
#  [10 11 12]
#  [20 21 22]]
# → creates an outer sum grid!
```

## Incompatible Shapes — Error
```python
a = np.array([1, 2, 3])    # shape (3,)
b = np.array([1, 2])       # shape (2,)
# a + b → ERROR: shapes (3,) and (2,) not aligned
```

## Real ML/Data Preprocessing Examples
### Mean Subtraction (Centering)
```python
data = np.array([[85, 72, 90],
                 [60, 88, 74],
                 [95, 63, 82]])   # shape (3, 3)

col_means = np.mean(data, axis=0)  # shape (3,) — mean of each column
print("Column means:", col_means)
# Subtract mean from each column (centering)
centered = data - col_means        # broadcast: (3,3) - (3,) works!
print("Centered:\n", centered)
print("Check means are ~0:", np.mean(centered, axis=0))
```
### Z-score Normalization (StandardScaler equivalent)
```python
data = np.array([[85, 72, 90],
                 [60, 88, 74],
                 [95, 63, 82]], dtype=float)

mu  = np.mean(data, axis=0)   # shape (3,)
sig = np.std(data, axis=0)    # shape (3,)
# Both operations use broadcasting — shape (3,3) op shape (3,)
z_scores = (data - mu) / sig
print("Z-scores:\n", np.round(z_scores, 3))
print("Check mean:", np.round(np.mean(z_scores, axis=0), 5))   # all ~0
print("Check std:", np.round(np.std(z_scores, axis=0), 5))     # all ~1
```
### Adding Bias to Neural Network Output
```python
# In neural networks: output = input @ weights + bias
inputs  = np.random.randn(5, 3)   # 5 samples, 3 features
weights = np.random.randn(3, 4)   # 4 neurons
bias    = np.array([0.1, 0.2, 0.3, 0.4])  # shape (4,)

output = inputs @ weights + bias  # (5,4) + (4,) → broadcast works!
print("Output shape:", output.shape)   # (5, 4)
```

## Broadcasting Summary
| Array 1 | Array 2 | After Broadcast | Works? |
|---------|---------|----------------|--------|
| `(3, 4)` | `scalar` | `(3, 4)` | ✓ |
| `(3, 4)` | `(4,)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(1, 4)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(3, 4)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(3, 2)` | — | ✗ Error |
| `(3, 4)` | `(2, 4)` | — | ✗ Error |

## Day 3 Full Practice Code
```python
import numpy as np
np.random.seed(42)

print("=== RANDOM ===")
print("rand(4):    ", np.random.rand(4))
print("randn(4):   ", np.random.randn(4))
print("randint:    ", np.random.randint(1, 100, size=5))
print("choice:     ", np.random.choice([10, 20, 30, 40, 50], size=3))

print("\n=== LINEAR ALGEBRA ===")
A = np.array([[2, 1], [5, 3]])
b = np.array([8, 11])

print("Matrix A:\n", A)
print("det(A):", np.linalg.det(A))
print("inv(A):\n", np.linalg.inv(A))
print("solve Ax=b:", np.linalg.solve(A, b))
# Matrix multiply
X = np.random.randn(4, 3)
W = np.random.randn(3, 2)
print("X @ W shape:", (X @ W).shape)   # (4, 2)

print("\n=== BROADCASTING ===")
data = np.random.randint(50, 100, (4, 3)).astype(float)
mu   = data.mean(axis=0)
sig  = data.std(axis=0)
norm = (data - mu) / sig
print("Original:\n", data)
print("Normalized:\n", np.round(norm, 2))
```

> Day 3 Complete.
> You now understand random number generation, all major linear algebra operations,
> and the powerful broadcasting mechanism that makes NumPy fast without loops.
>
> Next: **NumPy Day 4 — Masking, Performance, File Handling, Advanced Topics**