# KNN Linear Oversampling

This repository implements a custom oversampling algorithm based on
K-Nearest Neighbours (KNN) and linear interpolation.

---

## Problem Definition

Imbalanced datasets negatively affect classification performance,
especially for minority classes.

---

## Oversampling Algorithm

1. Randomly select a minority class sample p  
2. Find k nearest neighbours of p within the minority class  
3. Randomly select one neighbour  
4. Generate a synthetic sample using linear interpolation  

---

## Oversampling Implementation

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_linear_oversampling(X, y, minority_label, k=5):
    X_min = X[y == minority_label]
    X_maj = X[y != minority_label]

    n_min = len(X_min)
    n_maj = len(X_maj)
    samples_to_generate = n_maj - n_min

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_min)

    synthetic_samples = []

    for _ in range(samples_to_generate):
        idx = np.random.randint(0, n_min)
        p = X_min[idx]

        neighbors = knn.kneighbors(p.reshape(1, -1),
                                   return_distance=False)[0]
        n = X_min[np.random.choice(neighbors)]

        lam = np.random.rand()
        s = p + lam * (n - p)

        synthetic_samples.append(s)

    X_syn = np.array(synthetic_samples)
    y_syn = np.full(len(X_syn), minority_label)

    return np.vstack((X, X_syn)), np.hstack((y, y_syn))
