import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = X.mean(dim=dims, keepdim=True)
        var  = X.var(dim=dims, keepdim=True, unbiased=False)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            shape = (1,) * (X.dim() - len(self.normalized_shape)) + self.normalized_shape
            return X_hat * self.weight.view(shape) + self.bias.view(shape)
        else:
            return X_hat

def main():
    bs, s, d = 128, 128, 768
    X = torch.randn(bs, s, d, device="cpu")
    my_ln = LayerNorm(d)
    out = my_ln(X)
    print(out.shape)  # torch.Size([128, 128, 768])

if __name__ == "__main__":
    main()
