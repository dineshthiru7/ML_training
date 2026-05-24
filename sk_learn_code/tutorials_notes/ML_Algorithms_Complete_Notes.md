# Machine Learning Algorithms — Complete Notes

---

## Table of Contents

1. [Linear Regression](#1-linear-regression)
2. [Ridge Regression](#2-ridge-regression)
3. [Logistic Regression](#3-logistic-regression)
4. [Random Forest](#4-random-forest)
5. [Naive Bayes](#5-naive-bayes)
6. [Gradient Boosting](#6-gradient-boosting)
7. [Hierarchical Clustering](#7-hierarchical-clustering)
8. [HDBSCAN](#8-hdbscan)
9. [FP-Growth Algorithm](#9-fp-growth-algorithm)
10. [ECLAT Algorithm](#10-eclat-algorithm)

---

## 1. Linear Regression

### What is Linear Regression?

Linear Regression is one of the simplest and most important machine learning algorithms.

**Used to:**
- Find the relationship between variables
- Predict continuous numerical values
- Draw the best fit line through data points

**Real-world examples:**
- Predict house price based on size
- Predict salary based on years of experience
- Predict sales based on advertising budget
- Predict temperature trends

---

### What Problem Does It Solve?

Linear regression solves **regression problems** — predicting a continuous number.

| Input | Output |
|-------|--------|
| Experience | Salary |
| House size | House price |
| Hours studied | Marks |
| Temperature | Ice cream sales |

**Use linear regression when:**
- Output is numerical
- Relationship is approximately linear

---

### Core Idea

> More experience → higher salary  
> More study hours → higher marks

Linear regression **mathematically represents this pattern** using a straight line.

---

### The Formula

**Simple Linear Regression:**

```
y = mx + b
```

| Symbol | Meaning |
|--------|---------|
| `y` | Predicted output |
| `x` | Input feature |
| `m` | Slope / weight |
| `b` | Intercept / bias |

---

### Understanding `m` and `b`

#### Slope (m)
- Controls steepness
- Tells how much `y` changes when `x` changes
- Example: if `m = 2`, when `x` increases by 1, `y` increases by 2

#### Intercept (b)
- Starting point of line
- When `x = 0`, output becomes `b`

---

### Training Process Step by Step

#### Example Dataset (Marks from Study Hours)

| Hours (x) | Marks (y) |
|-----------|-----------|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

Actual relationship: `y = 2x + 1` (but model must learn this)

---

#### STEP 1 — Initialize Random Values
```
m = 0, b = 0
Equation: y = 0
```

#### STEP 2 — Predict Values

| x | Actual y | Predicted ŷ |
|---|----------|-------------|
| 1 | 3 | 0 |
| 2 | 5 | 0 |
| 3 | 7 | 0 |

#### STEP 3 — Calculate Loss (for one point)

```
Loss = (y - ŷ)²
```

| Point | Loss |
|-------|------|
| (3 - 0)² | 9 |
| (5 - 0)² | 25 |
| (7 - 0)² | 49 |

#### STEP 4 — Calculate Cost Function

```
J(m,b) = (1/n) × Σ(y - ŷ)²
J = (9 + 25 + 49) / 3 = 27.67
```

Huge error → model is bad initially.

#### STEP 5 — Gradient Descent

Update formulas:

```
m = m - α × (∂J/∂m)
b = b - α × (∂J/∂b)
```

Where `α` = learning rate (e.g., 0.1)

---

### Iterations Example

| Iteration | m | b | Cost |
|-----------|---|---|------|
| 0 (start) | 0 | 0 | 27.67 |
| 1 | 1.2 | 0.4 | 5.26 |
| 2 | 1.7 | 0.8 | smaller |
| 3 | 1.95 | 0.98 | smaller |
| Final | 2 | 1 | ≈ 0 |

**Final equation:** `y = 2x + 1` → Perfect predictions!

---

### What Gradient Descent Actually Does

> You are standing on a mountain. Goal: reach the lowest point.

- Gradient tells which direction goes downhill fastest
- If error increases → move opposite direction
- If error decreases → continue
- Eventually you reach minimum cost

---

### Multiple Linear Regression

If multiple inputs exist:

```
y = m1*x1 + m2*x2 + m3*x3 + b
```

---

### Most Important Intuition

> **"Find the best line by continuously adjusting slope and intercept to reduce prediction error."**

---

## 2. Ridge Regression

### What is Ridge Regression?

Ridge Regression is an improved version of linear regression used to **reduce overfitting**.

It adds a **penalty to large coefficient values**.

> "Do not allow the model weights to become too large."

---

### Why Do We Need Ridge Regression?

Normal Linear Regression works well when:
- Data is simple
- Features are not highly correlated
- Noise is low

**Problems happen when:**
- Too many features
- Multicollinearity
- Overfitting
- Noisy data

---

### Problems Ridge Regression Solves

#### 1. Overfitting
- Model memorizes training data instead of learning general pattern
- Example: adding useless features (wall color, owner name length)
- Linear regression creates huge weights to perfectly fit training data
- Result: high training accuracy, poor test accuracy

#### 2. Multicollinearity
- When input features are highly related
- Example: house size in sqft vs house size in square meters (same information)
- Confuses linear regression → weights become unstable

---

### Cost Function Comparison

**Linear Regression:**
```
J(m) = (1/n) × Σ(y - ŷ)²
```
Only focuses on reducing error.

**Ridge Regression:**
```
J(m) = (1/n) × Σ(y - ŷ)² + λ × Σmj²
```

| Part | Meaning |
|------|---------|
| `(1/n) × Σ(y - ŷ)²` | Prediction error |
| `λ × Σmj²` | Regularization penalty (punishes large weights) |

---

### What is Lambda (λ)?

Very important **hyperparameter**.

| Lambda Value | Effect |
|-------------|--------|
| Small | Behaves like normal linear regression, small penalty |
| Large | Strong penalty, coefficients shrink more |

---

### Why Squaring Weights?

| Weight | Squared |
|--------|---------|
| 2 | 4 |
| 10 | 100 |

Large coefficients become very expensive → pushed closer to zero.

> **Important:** Ridge reduces coefficients but **never makes them exactly zero.**

---

### Ridge Regression Training Steps

| Step | Action |
|------|--------|
| 1 | Initialize weights `m1=0, m2=0, b=0` |
| 2 | Predict values |
| 3 | Compute MSE |
| 4 | Add ridge penalty |
| 5 | Gradient descent with extra penalty term |
| 6 | Weights shrink during updates |
| 7 | Repeat until convergence |

---

### Key Difference

| | Linear Regression | Ridge Regression |
|---|---|---|
| Cost | Minimizes only error | Minimizes error + weight size |
| Overfitting | Prone | Reduces |
| Multicollinearity | Unstable | Handles better |
| Coefficients | Can be huge | Shrink |

---

### Bias-Variance Tradeoff

Ridge regression:
- Slightly increases **bias**
- Greatly reduces **variance**
- Usually improves overall prediction

---

### Choosing Lambda

| Lambda | Effect |
|--------|--------|
| 0 | Becomes normal linear regression |
| Very large | All coefficients tiny → underfits |
| Optimal | Balances fitting data and avoiding overfitting |

Best selected using **cross validation**.

---

### Important Limitation

Ridge regression:
- Shrinks coefficients
- But **keeps all features** (even useless ones)

This leads to another method: **Lasso Regression** (makes coefficients exactly zero)

---

### Most Important Intuition

> **"Fit the data well, but avoid overly large coefficients so the model generalizes better."**

---

## 3. Logistic Regression

### What is Logistic Regression?

Logistic Regression is a supervised ML algorithm used for **classification problems**.

Unlike Linear Regression, it predicts:
- Categories / classes
- Probabilities

---

### What Problem Does It Solve?

**Classification problems** — output belongs to a category.

| Input | Output |
|-------|--------|
| Email content | Spam / Not Spam |
| Tumor details | Cancer / No Cancer |
| Student data | Pass / Fail |
| Customer behavior | Buy / Not Buy |

Outputs are usually: `0` or `1`

---

### Why Not Use Linear Regression for Classification?

Linear regression may predict: `1.7`, `-0.5`, `2.3`

These are **invalid probabilities** — output must be between 0 and 1.

---

### Main Idea

Logistic regression:
1. Computes linear equation
2. Converts result into probability using **sigmoid function**

---

### Step 1 — Linear Equation

```
z = mx + b
```
or for multiple features:
```
z = m1*x1 + m2*x2 + b
```

Output `z` can be positive, negative, or very large.

---

### Step 2 — Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))
```

Converts any number to range **0 to 1**.

| z | Sigmoid Output |
|---|----------------|
| -10 | near 0 |
| 0 | 0.5 |
| 10 | near 1 |

---

### Final Prediction Rule

| Probability | Prediction |
|-------------|------------|
| > 0.5 | Class 1 |
| < 0.5 | Class 0 |

---

### Training Example (Pass/Fail)

| Hours Studied | Pass |
|--------------|------|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |

#### STEP 1 — Initialize Weights
```
m = 0, b = 0
z = 0
σ(0) = 0.5  (all predictions = 50%)
```

#### STEP 2 — Compute Loss (Log Loss / Binary Cross Entropy)

```
J = -(1/n) × Σ[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

**Why not MSE?**
- Sigmoid makes optimization non-convex with MSE
- Cross entropy works better with gradient descent

#### STEP 3 — After Training (Iteration 1 example)

```
m = 1, b = -3
z = x - 3
```

For x=1: `z = -2`, `σ(-2) = 0.12` → 12% pass (correct)  
For x=5: `z = 2`, `σ(2) = 0.88` → 88% pass (correct)

---

### Decision Boundary

At probability = 0.5, decision changes.

Here: around 3 study hours.

---

### Why Is It Called "Regression"?

Internally it still computes `mx + b` then applies sigmoid.  
Historically called regression, but actually used for **classification**.

---

### Comparison

| Feature | Linear Regression | Logistic Regression |
|---------|-------------------|---------------------|
| Used for | Regression | Classification |
| Output | Continuous number | Probability/class |
| Equation | mx+b | sigmoid(mx+b) |
| Output range | Any value | 0–1 |
| Loss | MSE | Cross Entropy |

---

### Most Important Intuition

> **"Use a linear equation to estimate the probability that a data point belongs to a particular class."**

---

## 4. Random Forest

### What is Random Forest?

Random Forest is a supervised ML algorithm that combines **many decision trees** together.

- Instead of one tree → uses hundreds or thousands
- Called **Random** + **Forest** (a forest of decision trees)

---

### Why Do We Need Random Forest?

Single Decision Tree problem: **Overfitting**

- Very deep tree learns noise
- Poor performance on new data

Random Forest solves this.

---

### Main Idea

> "Ask many trees and take majority opinion."

**Real-world analogy:** Would you trust 1 doctor or 100 doctors voting together? Group decision is usually better.

---

### Two Core Concepts

#### 1. Bagging (Bootstrap Aggregation)

Random sampling **with replacement** from original dataset.

| Original | Sample |
|----------|--------|
| A, B, C, D | A, C, C, D (B missing, C repeated) |

Each tree trains on different data → becomes slightly different.

#### 2. Random Feature Selection

At each split, tree sees only a **random subset of features**.

- Forces trees to learn differently
- Creates diversity among trees
- Improves ensemble performance

---

### How Random Forest Works

#### STEP 1 — Bootstrap Datasets
Create multiple random sampled datasets from original.

#### STEP 2 — Train Separate Decision Trees
Each dataset trains one tree → every tree becomes different.

#### STEP 3 — Random Feature Selection
At each split, only random subset of features considered.

#### STEP 4 — Prediction (Majority Voting / Averaging)

**Example: New customer prediction**

| Tree | Prediction |
|------|------------|
| Tree 1 | Yes |
| Tree 2 | No |
| Tree 3 | Yes |
| Tree 4 | Yes |
| Tree 5 | Yes |

Vote: Yes=4, No=1 → **Final: YES**

---

### Classification vs Regression

| Type | Method |
|------|--------|
| Classifier | Majority voting |
| Regressor | Average prediction |

**Regression example:** Trees predict 200k, 210k, 190k → Final: (200k+210k+190k)/3 = **200k**

---

### Why Random Forest Reduces Overfitting

- Single tree: high variance, memorizes data
- Random Forest: averages many trees → reduces variance → better generalization

---

### Feature Importance

Random Forest can measure which features are most important.

| Feature | Importance |
|---------|-----------|
| Salary | 40% |
| Credit Score | 35% |
| Age | 20% |
| Married | 5% |

---

### Out-of-Bag (OOB) Samples

Bootstrap sampling leaves some rows unused → used as **validation data** without separate validation set.

---

### Important Hyperparameters

| Hyperparameter | Meaning |
|---------------|---------|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum depth of each tree |
| `max_features` | Random features per split |
| `min_samples_split` | Minimum samples before splitting |

---

### Comparison

| Feature | Decision Tree | Random Forest |
|---------|--------------|---------------|
| Trees | One | Many |
| Overfitting | High | Lower |
| Stability | Low | High |
| Accuracy | Medium | High |
| Interpretability | Easy | Harder |

---

### Most Important Intuition

> **"A large collection of decision trees trained on random data and random features whose combined predictions are more accurate and stable than a single tree."**

---

## 5. Naive Bayes

### What is Naive Bayes?

Naive Bayes is a supervised ML algorithm used for **classification problems**.

Predicts a class using:
- Probability
- Statistics
- **Bayes Theorem**

---

### Why Is It Called "Naive"?

Because it assumes **all features are independent** from each other.

> Example: For spam detection, "free", "offer", "win" are assumed independent — even though in reality they may be related.

Surprisingly, this still works very well.

---

### Common Applications

| Problem | Example |
|---------|---------|
| Spam Detection | Spam / Not Spam |
| Sentiment Analysis | Positive / Negative |
| News Classification | Sports / Politics |
| Medical Diagnosis | Disease / No Disease |

---

### Core Foundation — Bayes Theorem

```
P(A|B) = [P(B|A) × P(A)] / P(B)
```

| Term | Meaning |
|------|---------|
| P(A\|B) | Posterior probability |
| P(B\|A) | Likelihood |
| P(A) | Prior probability |
| P(B) | Evidence probability |

---

### Step-by-Step Example (Play Tennis?)

#### Dataset

| Weather | Windy | Play |
|---------|-------|------|
| Sunny | No | Yes |
| Sunny | Yes | No |
| Rainy | No | Yes |
| Rainy | Yes | No |
| Sunny | No | Yes |
| Rainy | No | Yes |

**New data:** Weather=Sunny, Windy=No → Will person play?

---

#### STEP 1 — Prior Probabilities

```
P(Yes) = 4/6 = 0.67
P(No)  = 2/6 = 0.33
```

#### STEP 2 — Likelihood Probabilities

**For YES class:**
```
P(Sunny|Yes)   = 2/4 = 0.5
P(NoWind|Yes)  = 4/4 = 1.0
```

**For NO class:**
```
P(Sunny|No)    = 1/2 = 0.5
P(NoWind|No)   = 0/2 = 0
```

#### STEP 3 — Apply Naive Bayes Formula

```
Score(Yes) = P(Yes) × P(Sunny|Yes) × P(NoWind|Yes)
           = 0.67 × 0.5 × 1 = 0.335

Score(No)  = P(No) × P(Sunny|No) × P(NoWind|No)
           = 0.33 × 0.5 × 0 = 0
```

#### STEP 4 — Compare

| Class | Score |
|-------|-------|
| YES | 0.335 |
| NO | 0 |

**Final Prediction: PLAY TENNIS ✓**

---

### Zero Probability Problem

Notice: `P(NoWind|No) = 0` made entire probability zero.

**Solution — Laplace Smoothing:**
```
(count + 1) / (total + classes)
```
Prevents zero probabilities. Essential in real-world NLP.

---

### Types of Naive Bayes

| Type | Used For | Example |
|------|----------|---------|
| Gaussian | Continuous numerical data | Height, weight, salary |
| Multinomial | Text classification, word counts | Spam filtering |
| Bernoulli | Binary features | Word exists or not |

---

### Advantages vs Limitations

| Advantages | Limitations |
|-----------|------------|
| Very fast training | Independence assumption unrealistic |
| Works well on small data | Zero probability problem (needs smoothing) |
| Excellent for text | Poor for complex relationships |
| Probabilistic output | |

---

### Comparison

| Feature | Naive Bayes | Logistic Regression |
|---------|------------|---------------------|
| Based on | Probability | Optimization |
| Assumes independence | Yes | No |
| Training speed | Very fast | Moderate |
| Text data | Excellent | Good |
| Correlated features | Poor | Better |

---

### Most Important Intuition

> **"Predict the class that has the highest probability after combining evidence from all features using Bayes theorem."**

---

## 6. Gradient Boosting

### What is Gradient Boosting?

Gradient Boosting is a powerful ensemble algorithm that builds models **sequentially** to correct previous mistakes.

> "Every new model tries to fix the errors made by earlier models."

**Popular implementations:**
- XGBoost
- LightGBM
- CatBoost

---

### Main Idea

Instead of building independent models (like Random Forest):

> "Let the next model focus mainly on previous mistakes."

Models are built one after another, sequentially.

---

### Key Difference

| | Random Forest | Gradient Boosting |
|---|---|---|
| Tree building | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Trees | Independent | Error-correcting |

---

### Real-World Intuition

**Teacher correcting student:**
- Attempt 1: 40/100, teacher identifies mistakes
- Attempt 2: Student focuses on weak areas, score improves
- Attempt 3: Correct remaining mistakes → strong performance

---

### What is Residual Error?

```
Residual = Actual - Predicted
```

The remaining mistake/error.

---

### Step-by-Step Training Example

#### Dataset (House Prices)

| House Size | Price |
|-----------|-------|
| 1 | 100 |
| 2 | 200 |
| 3 | 300 |
| 4 | 400 |

#### STEP 1 — Initial Prediction

Average = (100+200+300+400)/4 = **250**

Predict 250 for every house.

#### STEP 2 — Compute Residuals

| Size | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 1 | 100 | 250 | -150 |
| 2 | 200 | 250 | -50 |
| 3 | 300 | 250 | 50 |
| 4 | 400 | 250 | 150 |

#### STEP 3 — Train New Tree on Residuals

Tree 2 predicts **residual errors** (not original targets).

Learns:
- Small houses → negative residual
- Large houses → positive residual

#### STEP 4 — Update Predictions

```
NewPrediction = OldPrediction + LearningRate × ResidualPrediction
```

Example (Learning Rate = 0.1, Size=1):
```
250 + 0.1 × (-100) = 240
```

#### STEP 5 — Repeat Sequentially

Each tree fixes remaining errors. Residuals become tiny.

#### Final Model

```
F(x) = Tree1 + Tree2 + Tree3 + ...
```

All trees combine **additively**.

---

### Why "Gradient" Boosting?

Uses **Gradient Descent** to minimize loss function step by step using gradients.

### Why "Boosting"?

Weak learners are **boosted sequentially**. Each learner improves previous one.

---

### Weak Learner

Usually shallow decision tree (depth 1-5).

Single small tree = weak predictor  
Many combined = very powerful

---

### Important Hyperparameters

| Hyperparameter | Effect |
|---------------|--------|
| Number of trees | More = more learning, but slower |
| Learning rate | Small = safer learning, better performance |
| Tree depth | Controls each tree's complexity |
| Subsampling | Random subset of data, reduces overfitting |

---

### Overfitting Prevention

- Regularization
- Early stopping
- Smaller trees
- Lower learning rate

---

### Popular Implementations

| Tool | Feature |
|------|---------|
| XGBoost | Regularization, parallel optimization, missing value handling |
| LightGBM | Speed, large datasets |
| CatBoost | Categorical data |

---

### Comparison

| Feature | Random Forest | Gradient Boosting |
|---------|--------------|-------------------|
| Tree building | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Speed | Faster | Slower |
| Overfitting | Lower risk | Higher risk |
| Accuracy | High | Often higher |

---

### Most Important Intuition

> **"Build many small decision trees sequentially where each new tree focuses on correcting the mistakes made by previous trees."**

---

## 7. Hierarchical Clustering

### What is Hierarchical Clustering?

Hierarchical Clustering is an unsupervised ML algorithm that groups similar data points by building a **hierarchy/tree of clusters**.

Unlike K-Means which directly creates K clusters, hierarchical clustering:
- Merges clusters (bottom-up)
- OR splits clusters (top-down)

---

### Why "Hierarchical"?

```
All points
   ↓
Large clusters
   ↓
Smaller clusters
   ↓
Individual points
```

Clusters form in levels = hierarchy.

---

### Advantages Over K-Means

| Advantage | Explanation |
|-----------|-------------|
| No need to choose K initially | Build full tree first, choose clusters later |
| Shows cluster relationships | See which clusters are close |
| Better for small datasets | Discovers natural hierarchy |
| Flexible shapes | K-means prefers spherical, HC more flexible |

---

### Types of Hierarchical Clustering

| Type | Direction | Start |
|------|-----------|-------|
| Agglomerative (most common) | Bottom-up | Each point = separate cluster |
| Divisive | Top-down | All points in one cluster |

---

### Agglomerative Clustering — Step by Step

#### Example Dataset

| Point | Value |
|-------|-------|
| A | 1 |
| B | 2 |
| C | 8 |
| D | 9 |
| E | 25 |

#### STEP 1 — Individual Clusters
```
A   B   C   D   E
```

#### STEP 2 — Compute Distances
| Pair | Distance |
|------|---------|
| A-B | 1 |
| C-D | 1 |
| B-C | 6 |
| D-E | 16 |

#### STEP 3 — Merge Closest → (A,B) and (C,D)

#### STEP 4 — Recompute Distances → (A,B) and (C,D) → merge

#### STEP 5 — Final: (A,B,C,D,E) all in one cluster

---

### Linkage Methods

Defines distance between clusters:

| Method | Distance Calculation | Use |
|--------|---------------------|-----|
| Single | Minimum distance between clusters | Closest points |
| Complete | Maximum distance | Farthest points |
| Average | Average pairwise distance | Balanced |
| Ward | Minimizes variance increase | Very popular |

---

### Dendrogram

A **dendrogram** is a tree diagram showing all merges.

- **Small height merge** = Clusters very similar
- **Large height merge** = Clusters very different

---

### How to Choose Final Clusters?

**Cut dendrogram horizontally.**

- Cut before E merges → 2 clusters: (A,B,C,D) and E
- Cut lower → 3 clusters: (AB), (CD), E

---

### How to Decide Where to Cut?

Look for **large jump in merge distance**.

Example:
| Merge | Distance |
|-------|---------|
| A-B | 1 |
| C-D | 1 |
| AB-CD | 5 |
| ABCD-E | 20 |

Huge jump 5→20 → E is very different → cut before 20.

---

### Distance Metrics

| Metric | Used For |
|--------|---------|
| Euclidean | Most common |
| Manhattan | Grid distance |
| Cosine | Text data |

---

### Comparison

| Feature | K-Means | Hierarchical |
|---------|---------|--------------|
| Need K initially | Yes | No |
| Output | Flat clusters | Cluster hierarchy |
| Speed | Faster | Slower |
| Large datasets | Better | Poor |
| Interpretability | Medium | High |

---

### Most Important Intuition

> **"Build a tree of clusters by repeatedly merging the closest groups, then choose final clusters by cutting the tree at an appropriate level."**

---

## 8. HDBSCAN

### What is HDBSCAN?

**Hierarchical Density-Based Spatial Clustering of Applications with Noise**

Advanced version of DBSCAN that combines:
- DBSCAN
- Hierarchical Clustering

---

### Why Was HDBSCAN Created?

DBSCAN's major problem: **struggles when clusters have different densities.**

- Small epsilon → sparse cluster breaks apart
- Large epsilon → dense clusters merge incorrectly

---

### Main Idea

> "Analyze clustering across many density levels and keep only the most stable clusters."

No single fixed epsilon needed.

---

### Core Concepts

#### 1. Core Distance

Distance to a point's **kth nearest neighbor** (k = min_samples).

| Region | Core Distance |
|--------|--------------|
| Dense | Small (neighbors nearby) |
| Sparse | Large (neighbors far) |

#### 2. Mutual Reachability Distance (MRD)

```
MRD(a,b) = max(core(a), core(b), distance(a,b))
```

**Why?** Prevents weak sparse connections from being treated as real clusters.

Example:
```
distance(a,b) = 2
core(a) = 5, core(b) = 3
MRD = max(5, 3, 2) = 5
```

Even though actual distance is small, sparse density increases effective distance.

#### 3. Minimum Spanning Tree (MST)

Connects all points using minimum total edge weight.

Reveals natural density structure of data.

#### 4. Cluster Hierarchy

Gradually remove large-distance edges → cluster splits.

As density requirement increases:
- Sparse connections break first
- Dense clusters survive longer

#### 5. Cluster Stability

**Stable Cluster** = Exists consistently across many density thresholds → likely real cluster.

**Unstable Cluster** = Appears briefly then disappears → likely noise.

---

### Step-by-Step Process

| Step | Action |
|------|--------|
| 1 | Compute core distances |
| 2 | Compute mutual reachability distances |
| 3 | Build weighted graph |
| 4 | Create minimum spanning tree |
| 5 | Build cluster hierarchy |
| 6 | Analyze cluster stability |
| 7 | Select stable clusters |
| 8 | Label remaining points as noise |

---

### Important Hyperparameters

| Hyperparameter | Meaning |
|---------------|---------|
| `min_cluster_size` | Minimum cluster size (smaller = noise) |
| `min_samples` | Higher = more noise detection, stricter density |

---

### Comparison

| Feature | DBSCAN | HDBSCAN |
|---------|--------|---------|
| Fixed epsilon | Yes | No |
| Varying density | Poor | Excellent |
| Automatic cluster selection | No | Yes |
| Hierarchy | No | Yes |
| Noise handling | Good | Better |

| Feature | K-Means | HDBSCAN |
|---------|---------|---------|
| Need K | Yes | No |
| Arbitrary shapes | Poor | Excellent |
| Outlier detection | Poor | Excellent |
| Varying density | Poor | Excellent |

---

### Most Important Intuition

> **"Build a hierarchy of density-based clusters across multiple density levels and keep only the most stable meaningful clusters while identifying noise automatically."**

---

## 9. FP-Growth Algorithm

### What is FP-Growth?

**Frequent Pattern Growth**

Used for:
- Frequent itemset mining
- Association rule mining

**Goal:** Find items frequently bought/occurring together.

Same goal as Apriori but **much faster** and more efficient.

---

### Why FP-Growth?

Apriori problem: generates huge number of candidate itemsets → very slow.

**FP-Growth solution:** Compresses database into special tree structure called **FP-Tree**.

---

### Main Idea

> "Store transactions compactly in a tree and mine frequent patterns directly from tree."

**Key advantages:**
- No candidate generation
- Fewer database scans (usually only 2)
- Faster on large dense datasets

---

### Important Concepts

| Term | Meaning |
|------|---------|
| FP-Tree | Compressed transaction tree |
| Header Table | Links to similar items |
| Conditional Pattern Base | Prefix paths for an item |
| Conditional FP-Tree | Smaller subtree for mining |

---

### Step-by-Step Example

#### Dataset

| T ID | Items |
|------|-------|
| T1 | Milk, Bread, Butter |
| T2 | Bread, Butter |
| T3 | Milk, Bread |
| T4 | Milk, Butter |
| T5 | Milk, Bread, Butter |

`min_support = 3`

#### STEP 1 — Count Frequencies

| Item | Count |
|------|-------|
| Milk | 4 |
| Bread | 4 |
| Butter | 4 |

All survive threshold.

#### STEP 2 — Sort Items by Frequency

Order: Milk > Bread > Butter

#### STEP 3 — Build FP-Tree (Insert transactions one by one)

**Final FP-Tree:**
```
ROOT
 ├── Milk:4
 │     ├── Bread:3
 │     │      └── Butter:2
 │     │
 │     └── Butter:1
 │
 └── Bread:1
       └── Butter:1
```

**Core idea:** Common prefixes share branches → efficient compression.

#### STEP 4 — Build Header Table

Stores item frequencies + links to same items in tree.

#### STEP 5 — Mine Patterns (Bottom-up for Butter)

**Conditional Pattern Base for Butter:**

| Prefix Path | Count |
|-------------|-------|
| Milk, Bread | 2 |
| Bread | 1 |
| Milk | 1 |

**Frequent patterns with Butter:**
- Butter
- Milk, Butter
- Bread, Butter
- Milk, Bread, Butter

#### Final Frequent Itemsets
- Milk, Bread, Butter
- Milk, Bread
- Milk, Butter
- Bread, Butter
- Milk, Bread, Butter

---

### FP-Growth vs Apriori

| Feature | Apriori | FP-Growth |
|---------|---------|-----------|
| Candidate generation | Yes | No |
| Database scans | Many | Few (usually 2) |
| Speed | Slower | Faster |
| Memory efficiency | Lower | Higher |
| Large datasets | Poorer | Better |

---

### Most Important Intuition

> **"Compress transaction data into a frequent-pattern tree and recursively mine frequent itemsets without generating huge candidate combinations."**

---

## 10. ECLAT Algorithm

### What is ECLAT?

**Equivalence Class Clustering and bottom-up Lattice Traversal**

Used for:
- Frequent itemset mining
- Association rule mining

Same goal as Apriori/FP-Growth but uses **vertical database format** — the key idea.

---

### Main Idea

**Instead of:** Transaction → Items (horizontal format)

**ECLAT stores:** Item → Transactions (vertical format)

This is called **TID sets** (Transaction ID sets).

---

### Horizontal vs Vertical Format

**Horizontal (normal):**
| Transaction | Items |
|-------------|-------|
| T1 | Milk, Bread |
| T2 | Bread, Butter |

**Vertical (ECLAT):**
| Item | TID Set |
|------|---------|
| Milk | T1, T3, T4, T5 |
| Bread | T1, T2, T3, T5 |

---

### Why Vertical Format Powerful?

Support calculation becomes **simple intersection operation**.

```
Support(A,B) = |TID(A) ∩ TID(B)|
```

---

### Step-by-Step Example

#### Same Dataset as FP-Growth

#### STEP 1 — Convert to Vertical Format

```
Milk   → {T1, T3, T4, T5}
Bread  → {T1, T2, T3, T5}
Butter → {T1, T2, T4, T5}
```

#### STEP 2 — Support = Size of TID Set

All have support = 4. All frequent.

#### STEP 3 — Generate 2-Itemsets via Intersection

**(Milk, Bread):**
```
{T1,T3,T4,T5} ∩ {T1,T2,T3,T5} = {T1,T3,T5}
Support = 3 ✓ (Frequent)
```

**(Milk, Butter):**
```
{T1,T3,T4,T5} ∩ {T1,T2,T4,T5} = {T1,T4,T5}
Support = 3 ✓ (Frequent)
```

**(Bread, Butter):**
```
{T1,T2,T3,T5} ∩ {T1,T2,T4,T5} = {T1,T2,T5}
Support = 3 ✓ (Frequent)
```

#### STEP 4 — Generate 3-Itemsets

**(Milk, Bread, Butter):**
```
{T1,T3,T5} ∩ {T1,T2,T4,T5} = {T1,T5}
Support = 2 ✗ (Below threshold of 3)
```

#### Final Frequent Itemsets

**1-Itemsets:** Milk, Bread, Butter

**2-Itemsets:** (Milk,Bread), (Milk,Butter), (Bread,Butter)

**3-Itemsets:** None

---

### Depth First Search (DFS)

ECLAT uses **DFS** (not BFS like Apriori).

```
Apriori (BFS):  All 1-items → All 2-items → All 3-items
ECLAT (DFS):    Milk → Milk,Bread → Milk,Bread,Butter → Milk,Butter
```

DFS benefits:
- Reduces memory usage
- Efficient recursion

---

### Comparison

| Feature | Apriori | ECLAT |
|---------|---------|-------|
| Database format | Horizontal | Vertical |
| Main operation | Candidate generation | TID intersection |
| Search style | BFS | DFS |
| Database scans | Many | Few |
| Speed | Slower | Faster |

| Feature | FP-Growth | ECLAT |
|---------|-----------|-------|
| Main structure | FP-tree | TID sets |
| Compression | Tree | Vertical format |
| Mining style | Recursive tree | Set intersections |

---

### Advantages vs Limitations

| Advantages | Limitations |
|-----------|------------|
| Faster than Apriori | Large TID sets can consume memory |
| Fewer database scans | Sparse datasets harder |
| Elegant support calculation | Recursive complexity |
| DFS memory efficiency | Harder than Apriori to implement |

---

### Most Important Intuition

> **"Store items with their transaction IDs and find frequent itemsets efficiently using recursive set intersections."**

---

## Quick Reference Summary

| Algorithm | Type | Problem Solved | Key Idea |
|-----------|------|---------------|----------|
| Linear Regression | Supervised | Predict continuous values | Best fit line using gradient descent |
| Ridge Regression | Supervised | Overfitting, multicollinearity | Penalize large weights with λ |
| Logistic Regression | Supervised | Binary classification | Sigmoid transforms linear output to probability |
| Random Forest | Supervised | Classification/Regression | Many trees → voting/averaging |
| Naive Bayes | Supervised | Classification | Bayes theorem with feature independence |
| Gradient Boosting | Supervised | Classification/Regression | Sequential trees correct previous errors |
| Hierarchical Clustering | Unsupervised | Grouping | Merge closest clusters, cut dendrogram |
| HDBSCAN | Unsupervised | Density-based clustering | Stable clusters across density levels |
| FP-Growth | Unsupervised | Frequent itemset mining | FP-tree compression, no candidates |
| ECLAT | Unsupervised | Frequent itemset mining | Vertical TID sets, intersections |

---

## Interconnected Concepts

| Topic | Connected Concept |
|-------|------------------|
| Linear Regression | Gradient Descent |
| Logistic Regression | Classification |
| Decision Trees | Random Forest, Gradient Boosting |
| K-Means | Distance Metrics |
| DBSCAN/HDBSCAN | Density-based clustering |
| Apriori/FP-Growth/ECLAT | Frequent pattern mining |

---

## Next Learning Steps

### Model Evaluation Metrics

**Classification:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

**Regression:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Other Important Topics
- Bias vs Variance Tradeoff
- Feature Engineering
- Dimensionality Reduction (PCA, t-SNE)
- Ensemble Methods (Stacking, Blending)
- Cross Validation
- Deep Learning Basics

---

*Notes created for deep conceptual understanding of ML algorithms.*
