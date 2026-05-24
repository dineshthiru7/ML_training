# Day 8 — Clustering (Unsupervised Learning)

## What is Clustering?
Clustering is an **unsupervised learning** technique that groups similar data points together without using labelled output (no y). The algorithm finds hidden structure or natural groupings in the data on its own.

**Supervised vs Unsupervised**
| | Supervised | Unsupervised |
|---|---|---|
| Labels | Yes (y provided) | No labels |
| Goal | Predict output | Discover structure |
| Example | Predict house price | Group customers by behaviour |

**When to use clustering**
- Customer segmentation (group users by spending habits)
- Document grouping (news topic detection)
- Image compression (group similar pixel colours)
- Anomaly detection (points that don't fit any cluster)

---

## 1. KMeans Clustering

### What is it?
KMeans divides data into **K clusters** by minimising the distance from each point to its cluster centre (centroid). It is the most commonly used clustering algorithm.

### Internal Working
**Step 1** — Choose K (number of clusters).
**Step 2** — Randomly place K centroids.
**Step 3** — Assign every point to the nearest centroid.
**Step 4** — Recompute centroid as the mean of all assigned points.
**Step 5** — Repeat Steps 3–4 until centroids stop moving (convergence).

**Distance formula (each point to centroid):**
`d(x, c) = √( Σ(xⱼ - cⱼ)² )`

**New centroid formula:**
`cₖ = (1/|Cₖ|) × Σ x   (mean of all points in cluster k)`

**Objective — minimise WCSS (Within-Cluster Sum of Squares):**
`WCSS = Σₖ Σ(x ∈ Cₖ) ||x - cₖ||²`

### Syntax
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
labels = model.labels_            # cluster assignment for each point
centers = model.cluster_centers_  # centroid coordinates
wcss = model.inertia_             # WCSS value
```

### Key Parameters
| Parameter | Meaning | Default |
|---|---|---|
| `n_clusters` | Number of clusters K | 8 |
| `init` | Centroid initialisation method (`'k-means++'` recommended) | `'k-means++'` |
| `n_init` | Number of random initialisations to try | 10 |
| `max_iter` | Max iterations per run | 300 |
| `random_state` | Reproducibility seed | None |

**`k-means++` init** — smarter centroid initialisation that spreads initial centroids far apart, leading to better and faster convergence.

### Attributes
| Attribute | Description |
|---|---|
| `labels_` | Cluster index (0 to K-1) for each training sample |
| `cluster_centers_` | Array of centroid coordinates |
| `inertia_` | Final WCSS score (lower = tighter clusters) |
| `n_iter_` | Number of iterations run |

### Elbow Method — Choosing K
The **Elbow Method** plots WCSS vs number of clusters. WCSS drops steeply at first, then flattens. The "elbow" point is the best K.

```python
import matplotlib.pyplot as plt

wcss_values = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss_values.append(km.inertia_)

plt.plot(range(1, 11), wcss_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

### Limitations
- Must specify K in advance.
- Sensitive to outliers (they can pull centroids).
- Assumes clusters are spherical and similar in size.
- Result can vary with random initialisation (use `n_init` > 1).

---

## 2. DBSCAN (Density-Based Spatial Clustering)

### What is it?
DBSCAN groups points that are **densely packed together** and marks points in sparse areas as **noise/outliers**. It does not require specifying K in advance.

### Internal Working
**Core concepts:**
- **eps (ε)** — maximum radius around a point to search for neighbours.
- **min_samples** — minimum number of points required within ε to form a dense region.
- **Core point** — has ≥ min_samples neighbours within ε.
- **Border point** — within ε of a core point but not itself a core point.
- **Noise point** — not reachable from any core point → labelled as -1.

**Algorithm:**
1. Pick any unvisited point.
2. If it has ≥ min_samples neighbours within ε → it is a core point, start a new cluster.
3. Expand the cluster by adding all reachable core/border points.
4. Points not reachable from any core point are labelled noise (-1).

### Syntax
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
model.fit(X)
labels = model.labels_   # -1 = noise, 0,1,2... = cluster IDs
```

### Key Parameters
| Parameter | Meaning | Default |
|---|---|---|
| `eps` | Neighbourhood radius | 0.5 |
| `min_samples` | Min points for a core point | 5 |
| `metric` | Distance metric | `'euclidean'` |

### Advantages over KMeans
- Finds arbitrary shaped clusters (not just spheres).
- Automatically detects number of clusters.
- Handles noise and outliers explicitly (label = -1).

### Limitations
- Choosing eps and min_samples requires tuning.
- Struggles when clusters have very different densities.

---

## 3. Agglomerative Clustering (Hierarchical)

### What is it?
Agglomerative clustering is a **bottom-up** hierarchical approach. Every point starts as its own cluster and clusters are iteratively merged based on distance, until one big cluster remains or a stopping condition is reached.

### Internal Working
1. Start: each of N points is its own cluster.
2. Find the two closest clusters.
3. Merge them into one cluster.
4. Repeat until you have the desired number of clusters.

**Linkage criteria** — how distance between clusters is measured:
- **ward** — minimises variance within clusters (most common).
- **complete** — max distance between any two points in two clusters.
- **average** — average distance between all pairs of points.
- **single** — min distance between any two points.

### Dendrogram
A dendrogram is a tree diagram that shows the order of merges. The height at which clusters are merged indicates dissimilarity. Cut the dendrogram at a horizontal line to choose the number of clusters.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrogram')
plt.show()
```

### Syntax
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
```

### Key Parameters
| Parameter | Meaning | Default |
|---|---|---|
| `n_clusters` | Number of final clusters | 2 |
| `linkage` | Merge strategy (`'ward'`, `'complete'`, `'average'`, `'single'`) | `'ward'` |
| `metric` | Distance metric | `'euclidean'` |

---

## 4. Cluster Evaluation Metrics

### Silhouette Score
Measures how similar a point is to its own cluster vs neighbouring clusters. Ranges from **-1 to 1** (1 = perfect clustering, 0 = overlapping clusters, -1 = wrong cluster).

`s(i) = (b - a) / max(a, b)`
Where:
- `a` = average distance from point i to all other points in its cluster (intra-cluster distance).
- `b` = average distance from point i to all points in the nearest other cluster (inter-cluster distance).

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

**Rule of thumb:**
| Score | Interpretation |
|---|---|
| > 0.7 | Strong clusters |
| 0.5–0.7 | Reasonable |
| 0.25–0.5 | Weak |
| < 0.25 | No structure |

### Davies-Bouldin Index
Lower is better. Measures average ratio of within-cluster scatter to between-cluster separation.

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {score:.3f}")
```

### Inertia / WCSS (KMeans only)
Lower WCSS means tighter, more compact clusters. Used in the Elbow Method but not comparable across different datasets.

---

## 5. Preprocessing Before Clustering

Clustering uses distance measures so **feature scaling is essential**. Without scaling, features with large numeric ranges dominate.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X_scaled)
```

---

## 6. Full Example — Customer Segmentation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Sample data: annual income and spending score
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 20, 60, 61, 62, 63, 100, 101, 102],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 60, 4, 49, 10, 77, 3, 50]
}
df = pd.DataFrame(data)
X = df[['Annual_Income', 'Spending_Score']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
wcss = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 8), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()

# Fit final model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

# Evaluate
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")

df['Cluster'] = labels
print(df)
```

---

## 7. Algorithm Comparison

| Feature | KMeans | DBSCAN | Agglomerative |
|---|---|---|---|
| Specify K? | Yes | No | Yes |
| Cluster shape | Spherical | Any | Any |
| Handles outliers? | No | Yes (label = -1) | Limited |
| Scalability | Fast, large data | Medium | Slow on large data |
| Best for | General use, compact clusters | Irregular shapes, noise | Hierarchical view needed |

---

## Summary
- **KMeans** — fast, simple, need to specify K, use Elbow Method to find K.
- **DBSCAN** — density-based, finds arbitrary shapes, automatically detects K, handles noise.
- **Agglomerative** — hierarchical, dendrogram shows merge history, use ward linkage by default.
- Always **scale features** before clustering.
- Use **Silhouette Score** to evaluate cluster quality.
