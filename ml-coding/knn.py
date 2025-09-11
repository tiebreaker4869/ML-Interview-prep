import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X = X_train
        self.y = y_train
    def _predict_one(self, x: np.ndarray):
        x = x.reshape(1, -1)
        distances = np.linalg.norm(self.X - x, axis=1) # (n_train, )
        indices = distances.argsort()[:self.n_neighbors]
        votes = self.y[indices]
        y = Counter(votes).most_common(1)[0][0]
        return y
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.array([self._predict_one(x) for x in X])
        return y_pred

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # two blobs
    X0 = rng.normal([0, 0], 1.0, size=(150, 2))
    X1 = rng.normal([3, 3], 1.0, size=(150, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*150 + [1]*150)

    # shuffle & split
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    X_tr, y_tr = X[:240], y[:240]
    X_te, y_te = X[240:], y[240:]

    knn = KNNClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    acc = (y_pred == y_te).mean()
    print("Accuracy:", round(float(acc), 3))