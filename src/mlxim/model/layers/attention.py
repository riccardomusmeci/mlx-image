from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .functional import scaled_dot_product_attention
from .rope_position_encoding import rope_apply


class LinearKMaskedBias(nn.Linear):
    """Linear layer with masked bias.
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to use bias.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_features = self.weight.shape[0]
        assert self.out_features % 3 == 0
        self.in_features = self.weight.shape[1]
        if "bias" in self:
            d = self.out_features // 3
            ones = mx.ones((d,), dtype=self.bias.dtype)
            zeros = mx.zeros((d,), dtype=self.bias.dtype)
            self.bias_mask = mx.concat([ones, zeros, ones], axis=0)
            self.freeze(keys="bias_mask", strict=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input array.
        Returns:
            Output array.
        """
        if "bias" in self:
            masked_bias = self["bias"] * self["bias_mask"]
            return mx.addmm(masked_bias, x, self["weight"].T)
        return x @ self["weight"].T



class RoPESelfAttention(nn.Module):
    """RoPE Self-Attention layer.
    Args:
        dim: Dimension of the input features.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias for QKV projection.
        proj_bias: Whether to use bias for output projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        mask_k_bias: Whether to use masked bias for K projection.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
      super().__init__()
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = head_dim**-0.5

      linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
      self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj = nn.Linear(dim, dim, bias=proj_bias)
      self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: mx.array, k: mx.array, rope: mx.array | Tuple[mx.array, mx.array]) -> Tuple[mx.array, mx.array]:
      """
      Args:
          q: Query array.
          k: Key array.
          rope: RoPE embeddings.
      Returns:
          Tuple of query and key arrays with RoPE applied.
      """
      # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
      q_dtype = q.dtype
      k_dtype = k.dtype
      sin, cos = rope
      rope_dtype = sin.dtype

      q = q.astype(rope_dtype)
      k = k.astype(rope_dtype)
      N = q.shape[-2]
      prefix = N - sin.shape[-2]
      assert prefix >= 0

      q_prefix = q[:, :, :prefix, :]
      q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
      q = mx.concat([q_prefix, q], axis=-2)  # [B, head, N, D//head]

      k_prefix = k[:, :, :prefix, :]
      k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
      k = mx.concat([k_prefix, k], axis=-2)  # [B, head, N, D//head]

      return q.astype(q_dtype), k.astype(k_dtype)


    def __call__(self, x: mx.array, attn_bias: mx.array | None = None, rope: List[mx.array] | None = None) -> mx.array:
      """
      Compute RoPE Self-Attention.
      Args:
          x: Input array.
          attn_bias: Attention bias.
          rope: RoPE embeddings.
      Returns:
          Output array.
      """
      qkv = self.qkv(x)
      attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
      x = self.proj(attn_v)
      x = self.proj_drop(x)
      return x

    def compute_attention(self, qkv: mx.array, attn_bias: mx.array | None = None, rope: Tuple[mx.array, mx.array] | None = None) -> mx.array:
      """
      Compute RoPE Self-Attention.
      Args:
          qkv: QKV array.
          attn_bias: Attention bias.
          rope: RoPE embeddings.
      Returns:
          Output array.
      """
      assert attn_bias is None
      B, N, _ = qkv.shape
      C = self.qkv.weight.shape[1]

      qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
      q = qkv[:, :, 0]
      k = qkv[:, :, 1]
      v = qkv[:, :, 2]

      q, k, v = [mx.swapaxes(t, 1, 2) for t in [q, k, v]]
      if rope is not None:
          q, k = self.apply_rope(q, k, rope)
      x = scaled_dot_product_attention(q, k, v, self.scale)
      x = mx.swapaxes(x, 1, 2)
      return x.reshape([B, N, C])


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention layer.
    Args:
        dim: Dimension of the input features.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias for QKV projection.
        proj_bias: Whether to use bias for output projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
      super().__init__()
      self.dim = dim
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = head_dim**-0.5

      self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
      self.attn_drop = attn_drop
      self.proj = nn.Linear(dim, dim, bias=proj_bias)
      self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
      # Keep MLX's default parameter initialization; this method is
      # retained only for API compatibility with the original code.
      _ = init_attn_std
      _ = init_proj_std
      _ = factor

    def __call__(self, x: mx.array, is_causal: bool = True) -> mx.array:
      """
      Compute Causal Self-Attention.
      Args:
          x: Input array.
          is_causal: Whether to use causal attention.
      Returns:
          Output array.
      """
      B, N, C = x.shape
      qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
      q = qkv[:, :, 0]
      k = qkv[:, :, 1]
      v = qkv[:, :, 2]

      q, k, v = [mx.swapaxes(t, 1, 2) for t in [q, k, v]]
      x = scaled_dot_product_attention(
            q, k, v, self.scale, attn_mask=None, dropout_p=self.attn_drop if self.training else 0.0, is_causal=is_causal
      )
      x = mx.swapaxes(x, 1, 2).reshape(B, N, C)
      x = self.proj_drop(self.proj(x))
      return x


if __name__ == "__main__":
  attn = RoPESelfAttention(dim=1024, num_heads=16)
  x = mx.random.uniform(shape=(1, 1024, 1024))
  print(attn(x).shape)

  attn = CausalSelfAttention(dim=1024, num_heads=16)
  attn.init_weights()
  x = mx.random.uniform(shape=(1, 1024, 1024))
  print(attn(x, is_causal=True).shape)
