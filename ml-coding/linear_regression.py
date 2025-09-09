import numpy as np

class LinearRegression:
    def __init__(self, n_features: int, lr=1e-2, max_iter=1000, lamb=0.0, eps=1e-8):
        self.n_features = n_features
        self.lr = lr
        self.max_iter = max_iter
        self.lamb = lamb
        self.eps = eps
        self.w = None
        self.b = None
    def _mse_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lamb: float) -> float:
        # loss = 1/2m * (Xw + b - y)^T (Xw + b - y) + 1/2m * lamb * wTw
        err = (X @ w + b - y)
        m = X.shape[0]
        return float((err.T @ err + lamb * w.T @ w)/(2 * m))
    def fit(self, X: np.ndarray, y: np.ndarray):
        # dl/dw = (XT(Xw + b - y) + w * lamb) / m
        # dl/db = (1T (Xw + b - y)) / m
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.w = np.random.randn(self.n_features, 1)
        self.b = np.random.randn()
        m = X.shape[0]
        for _ in range(self.max_iter):
            old_loss = self._mse_loss(X, y, self.w, self.b, self.lamb)
            err = X @ self.w + self.b - y
            grad_w = (X.T @ err + self.w * self.lamb) / m
            grad_b = np.sum(err) / m
            self.w, self.b = self.w - self.lr * grad_w, self.b - self.lr * grad_b
            new_loss = self._mse_loss(X, y, self.w, self.b, self.lamb)
            if abs(new_loss - old_loss) < self.eps:
                break
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        m = X.shape[0]
        ones = np.ones((m, 1), dtype=X.dtype)
        X = np.concatenate((X, ones), axis=1)
        # Xw = y
        w = np.linalg.pinv(X.T @ X) @ X.T  @ y
        n = X.shape[1]
        self.w = w[:n-1, 0].reshape(-1, 1)
        self.b = w[n-1, 0]
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def main():
    np.random.seed(42)
    X = np.random.rand(100, 1)         # (100, 1)
    y = 3 * X[:, 0] + 2 + 0.1 * np.random.randn(100)
    y = y.reshape(-1, 1)

    # ====== gradient descent ======
    model_gd = LinearRegression(n_features=1, lr=0.1, max_iter=5000, lamb=0.0)
    model_gd.fit(X, y)
    print("Gradient Descent: w =", model_gd.w.ravel(), "b =", model_gd.b)

    # ====== Normal Equation ======
    model_ne = LinearRegression(n_features=1)
    model_ne.fit_normal_equation(X, y)
    print("Normal Equation:  w =", model_ne.w.ravel(), "b =", model_ne.b)

    X_test = np.array([[0.0], [0.5], [1.0]])
    y_pred_gd = model_gd.predict(X_test)
    y_pred_ne = model_ne.predict(X_test)
    print("Test X:", X_test.ravel())
    print("Pred (GD):", y_pred_gd.ravel())
    print("Pred (NE):", y_pred_ne.ravel())

if __name__ == "__main__":
    main()