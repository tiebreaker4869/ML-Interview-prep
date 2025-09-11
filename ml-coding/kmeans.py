import numpy as np
import random
from typing import Optional
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_cluster: int, eps=1e-8, max_iter=1000):
        self.n_cluster = n_cluster
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def _init_centers(self, X: np.ndarray):
        n = X.shape[0]
        if not (1 <= self.n_cluster <= n):
            raise ValueError("n_cluster must be in [1, n_samples].")
        indices = random.sample(range(n), self.n_cluster)
        # ensure float dtype to avoid integer truncation
        self.cluster_centers_ = X[indices, :].astype(float, copy=True)

    def _compute_distance(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        # X: (n, d), C: (k, d) -> (n, k), squared Euclidean distances
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)  # ensure float math
        self._init_centers(X)

        for _ in range(self.max_iter):
            # E-step: assign
            distance = self._compute_distance(X, self.cluster_centers_)
            labels = distance.argmin(axis=1)

            # M-step: update
            new_centers = np.zeros((self.n_cluster, X.shape[1]), dtype=float)
            for c in range(self.n_cluster):
                pts = X[labels == c]
                if pts.size > 0:
                    new_centers[c] = pts.mean(axis=0)
                else:
                    # handle empty cluster by reseeding to a random point
                    new_centers[c] = X[random.randrange(X.shape[0])]

            shift = np.linalg.norm(self.cluster_centers_ - new_centers)
            self.cluster_centers_ = new_centers  # make sure centers are updated

            if shift < self.eps:
                break

        # final labels w.r.t. final centers
        self.labels_ = self._compute_distance(X, self.cluster_centers_).argmin(axis=1)
        return self

    def predict(self, X: np.ndarray):
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        distance = self._compute_distance(X, self.cluster_centers_)
        return distance.argmin(axis=1)

def main():
    n_per = 200
    c1 = np.random.randn(n_per, 2) * 0.6 + np.array([0.0, 0.0])
    c2 = np.random.randn(n_per, 2) * 0.6 + np.array([8.0, 8.0])
    c3 = np.random.randn(n_per, 2) * 0.6 + np.array([0.0, 10.0])
    X = np.vstack([c1, c2, c3]).astype(float)

    km = KMeans(n_cluster=3, eps=1e-6, max_iter=300)
    km.fit(X)

    centers = np.round(km.cluster_centers_, 3)
    counts = np.bincount(km.labels_, minlength=3)
    inertia = float(((X - centers[km.labels_]) ** 2).sum())

    print("Cluster centers:\n", centers)
    print("Counts per cluster:", counts)
    print("Inertia (sum of squared distances):", round(inertia, 3))
    print("First 10 labels:", km.labels_[:10])

    X_test = np.array([[0, 0], [8, 8], [0, 10], [4, 4]], dtype=float)
    preds = km.predict(X_test)
    print("Test points:\n", X_test)
    print("Predicted labels:", preds)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, s=12)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker='x', s=200)
    plt.title("K-Means Clusters")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()