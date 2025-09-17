import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode='constant')
    H, W = x.shape
    H_k, W_k = kernel.shape
    H_o, W_o = (H - H_k) // stride + 1, (W - W_k) // stride + 1
    out = np.zeros((H_o, W_o), dtype=x.dtype)
    for i in range(H_o):
        for j in range(W_o):
            region = x[i * stride: i * stride + H_k, j * stride : j * stride + W_k]
            out[i, j] = np.sum(region * kernel)
    return out


if __name__ == "__main__":
    np.set_printoptions(linewidth=120, suppress=True)

    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    k = np.array([
        [1, 0],
        [0,-1]
    ], dtype=float)

    print("Input x:\n", x)
    print("Kernel k:\n", k)

    y0 = conv2d(x, k, padding=0, stride=1)
    print("\nNo padding, stride=1 -> shape", y0.shape)
    print(y0)

    y1 = conv2d(x, k, padding=1, stride=1)
    print("\nPadding=1, stride=1 -> shape", y1.shape)
    print(y1)

    y2 = conv2d(x, k, padding=1, stride=2)
    print("\nPadding=1, stride=2 -> shape", y2.shape)
    print(y2)

    delta = np.zeros((3,3), dtype=float)
    delta[1,1] = 1.0
    y_same = conv2d(x, delta, padding=1, stride=1)
    print("\nDelta kernel test (should equal input):", np.allclose(y_same, x))