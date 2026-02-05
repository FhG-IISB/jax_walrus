"""
Spatial Attention: JAX/Flax translation of walrus.models.spatial_blocks.full_attention.FullAttention

Fused FF + Q/K/V projection, SwiGLU MLP, axial 3D RoPE via lucidrains-style
RotaryEmbedding, QK-Norm, and scaled dot-product attention.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from walrus_jax.normalization import RMSGroupNorm
from walrus_jax.rope import LRRotaryEmbedding, apply_rotary_emb


class SwiGLU(nn.Module):
    """SwiGLU activation: splits input in half and applies SiLU gating."""

    @nn.compact
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * x


class FullAttention(nn.Module):
    """
    Spatial attention block with fused FF/QKV, SwiGLU, RoPE, and QK-norm.

    Operates on ``(B, C, H, W, D)`` tensors. The fused projection produces
    feedforward, query, key, and value in a single linear layer. Attention
    is computed over the flattened ``H*W*D`` spatial tokens.
    """

    hidden_dim: int = 768
    mlp_dim: int = 0  # 0 defaults to hidden_dim * 4
    num_heads: int = 12
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    max_d: int = 3

    @nn.compact
    def __call__(self, x, bcs=None, return_att=False):
        """
        Args:
            x: (B, C, H, W, D)

        Returns:
            x: (B, C, H, W, D), att_maps: []
        """
        mlp_dim = self.mlp_dim if self.mlp_dim > 0 else self.hidden_dim * 4
        B, C, H, W, D = x.shape

        residual = x

        # Pre-norm
        x = RMSGroupNorm(
            num_groups=self.num_heads,
            num_channels=self.hidden_dim,
            name="norm1",
        )(x)

        x = rearrange(x, "b c h w d -> b h w d c")

        # Fused FF + Q + K + V projection
        fused_dims = (mlp_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)
        total_dim = sum(fused_dims)
        fused_ff_qkv = nn.Dense(total_dim, name="fused_ff_qkv")(x)
        ff, q, k, v = jnp.split(
            fused_ff_qkv,
            [fused_dims[0], fused_dims[0] + fused_dims[1], fused_dims[0] + fused_dims[1] + fused_dims[2]],
            axis=-1,
        )

        # Split into heads
        head_dim = self.hidden_dim // self.num_heads
        q = rearrange(q, "b h w d (he c) -> b he h w d c", he=self.num_heads)
        k = rearrange(k, "b h w d (he c) -> b he h w d c", he=self.num_heads)
        v = rearrange(v, "b h w d (he c) -> b he h w d c", he=self.num_heads)

        # QK-norm
        q = nn.LayerNorm(name="q_norm")(q)
        k = nn.LayerNorm(name="k_norm")(k)

        # Axial 3D RoPE
        rope = LRRotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=256.0,
            name="rotary_emb",
        )
        pos_emb = rope.get_axial_freqs(H, W, D)

        q = apply_rotary_emb(pos_emb, q)
        k = apply_rotary_emb(pos_emb, k)

        # Flatten spatial dims for attention
        q = rearrange(q, "b he h w d c -> b he (h w d) c")
        k = rearrange(k, "b he h w d c -> b he (h w d) c")
        v = rearrange(v, "b he h w d c -> b he (h w d) c")

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("bhsc, bhtc -> bhst", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        att = jnp.einsum("bhst, bhtc -> bhsc", attn_weights, v)

        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)

        # Output projections
        attn_out = nn.Dense(self.hidden_dim, use_bias=False, name="attn_out")(att)
        ff_out = nn.Dense(self.hidden_dim, name="ff_out")(SwiGLU()(ff))

        x = attn_out + ff_out
        x = rearrange(x, "b h w d c -> b c h w d") + residual

        return x, []
