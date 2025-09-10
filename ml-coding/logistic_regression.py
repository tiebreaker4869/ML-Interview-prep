import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, n_features: int, lr=1e-2, max_iter=1000, eps=1e-8, lamb=0.0):
        self.n_features = n_features
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.lamb = lamb
    
    def _ce_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
        """
        loss = -sum(y_i * log(sigmoid(wTx_i + b)) + (1 - y_i) * log(1 - sigmoid(wTx_i + b)))  + lambda * wTw / 2m
        """
        z = X @ w + b # (m, 1)
        h = sigmoid(z)
        m = X.shape[0]
        loss = - (y.T  @ np.log(h) + (1 - y).T @ np.log(1 - h)) / m + w.T @ w * self.lamb/ m * 0.5
        
        loss = float(loss)
        return loss
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.random.randn(self.n_features, 1)
        self.b = np.random.randn()
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        m = X.shape[0]
        for _ in range(self.max_iter):
            old_loss = self._ce_loss(X, y, self.w, self.b)
            z = X @ self.w + self.b
            p = sigmoid(z)
            err = p - y
            grad_w = (X.T @ err + self.w * self.lamb) / m
            grad_b = np.sum(err) / m
            self.w, self.b = self.w - self.lr * grad_w, self.b - self.lr * grad_b
            new_loss = self._ce_loss(X, y, self.w, self.b)
            if abs(new_loss - old_loss) < self.eps:
                break
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        p = sigmoid(z)
        return p
    
    def predict(self, X: np.ndarray, thresh=0.5) -> np.ndarray:
        p = self.predict_prob(X)
        y = (p >= thresh).astype(int)
        return y



def main():
    # 1) Generate a toy binary classification dataset
    np.random.seed(42)
    m, d = 500, 2
    X = np.random.randn(m, d)

    # Ground-truth parameters to synthesize labels
    w_true = np.array([[1.5], [-2.0]])
    b_true = -0.25
    logits = X @ w_true + b_true
    probs  = 1.0 / (1.0 + np.exp(-logits))

    # Sample noisy labels from Bernoulli(probs)
    y = (np.random.rand(m, 1) < probs).astype(float)

    # 2) Train/test split
    idx = np.random.permutation(m)
    split = int(0.8 * m)
    tr, te = idx[:split], idx[split:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # 3) Train model (slight L2 helps when data is near separable)
    model = LogisticRegression(n_features=d, lr=0.1, max_iter=5000, eps=1e-9, lamb=1e-3)
    model.fit(X_tr, y_tr)

    # 4) Evaluate on test set
    y_pred = model.predict(X_te)                 # (n_test, 1) in {0,1}
    acc = float(np.mean(y_pred == y_te))

    # Optional: compute test loss using the class's loss function
    test_loss = model._ce_loss(X_te, y_te, model.w, model.b)

    # 5) Pretty print results
    np.set_printoptions(precision=3, suppress=True)
    print("=== Ground truth ===")
    print("w_true:", w_true.ravel())
    print("b_true:", float(b_true))

    print("\n=== Learned parameters ===")
    print("w_hat :", model.w.ravel())
    print("b_hat :", float(model.b))

    print("\n=== Test metrics ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"CE loss : {test_loss:.4f}")

    # Show a few probabilities/predictions
    proba_sample = model.predict_prob(X_te[:5]).ravel()
    print("\nSample probs (first 5):", np.round(proba_sample, 3))
    print("Sample preds (first 5):", y_pred[:5].ravel())
    print("Sample labels(first 5):", y_te[:5].ravel())

if __name__ == "__main__":
    main()