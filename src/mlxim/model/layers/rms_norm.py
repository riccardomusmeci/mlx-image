import mlx.core as mx
import mlx.nn as nn

# TODO: Delete RMSNorm if the internal rms norm works

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(shape=(dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.square(x).mean(axis=-1, keepdims=True) + self.eps) * self.weight

    def reset_parameters(self):
        self.weight = mx.ones_like(self.weight)


if __name__ == "__main__":
    rms_norm = RMSNorm(dim=1024)
    x = mx.random.uniform(shape=(1, 50000, 1024))
    print(rms_norm(x).shape)
