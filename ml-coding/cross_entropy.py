import numpy as np

def cross_entropy(y: np.ndarray, y_pred: np.ndarray):
    # y: onehot-vec, (B, N), y_pred: (B, N)
    ce = np.sum(-y * np.log2(y_pred), axis=-1)
    return ce.mean()
    