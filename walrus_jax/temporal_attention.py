"""
Temporal Attention: JAX/Flax translation of
walrus.models.temporal_blocks.axial_time_attention.AxialTimeAttention

Temporal attention with 1x1 conv input/output heads, T5-style relative
position bias (or rotary), QK-norm, and optional causal masking. Attention
is applied independently at each spatial location along the time axis.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from walrus_jax.normalization import RMSGroupNorm
from walrus_jax.rope import (
    SimpleRotaryEmbedding,
    apply_rotary_pos_emb_simple,
    RelativePositionBias,
)


def _conv3d_1x1(x, weight, bias=None):
    """
    1x1 3D convolution: matmul along the channel dimension.

    Args:
        x: (N, C_in, D1, D2, D3)
        weight: (C_out, C_in, 1, 1, 1)

    Returns:
        (N, C_out, D1, D2, D3)
    """
    w = weight[:, :, 0, 0, 0]  # (C_out, C_in)
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    out = x_t @ w.T
    if bias is not None:
        out = out + bias
    return jnp.transpose(out, (0, 4, 1, 2, 3))


class AxialTimeAttention(nn.Module):
    """
    Temporal attention applied independently at each spatial location.

    Input shape: ``(T, B, C, H, W, D)``
    Attention is computed over the T dimension for each ``(h, w, d)`` location.
    """

    hidden_dim: int = 768
    num_heads: int = 12
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    bias_type: str = "rel"  # "rel", "rotary", or "none"
    causal_in_time: bool = False

    @nn.compact
    def __call__(self, x, return_att=False):
        """
        Args:
            x: (T, B, C, H, W, D)

        Returns:
            x: (T, B, C, H, W, D), att_maps: []
        """
        T, B, C, H, W, D = x.shape
        residual = x

        x = rearrange(x, "t b c h w d -> (t b) c h w d")

        x = RMSGroupNorm(
            num_groups=self.num_heads,
            num_channels=self.hidden_dim,
            name="norm1",
        )(x)

        # 1x1 conv to produce Q, K, V
        input_head_weight = self.param(
            "input_head_weight",
            nn.initializers.lecun_normal(),
            (3 * self.hidden_dim, self.hidden_dim, 1, 1, 1),
        )
        input_head_bias = self.param(
            "input_head_bias",
            nn.initializers.zeros_init(),
            (3 * self.hidden_dim,),
        )

        x = _conv3d_1x1(x, input_head_weight, input_head_bias)

        # Reshape: (T*B, 3C, H, W, D) -> (B*H*W*D, heads, T, head_dim)
        x = rearrange(
            x, "(t b) (he c) h w d -> (b h w d) he t c", t=T, he=self.num_heads
        )
        head_dim = self.hidden_dim // self.num_heads
        q, k, v = jnp.split(x, 3, axis=-1)

        # QK norm
        q = nn.LayerNorm(name="qnorm")(q)
        k = nn.LayerNorm(name="knorm")(k)

        # Position encoding
        rel_pos_bias = None
        if self.bias_type == "rotary":
            rotary_emb = SimpleRotaryEmbedding(dim=head_dim, name="rotary_emb")
            positions = rotary_emb(T)
            q = apply_rotary_pos_emb_simple(positions, q)
            k = apply_rotary_pos_emb_simple(positions, k)
        elif self.bias_type == "rel":
            rel_pos_bias_mod = RelativePositionBias(
                bidirectional=not self.causal_in_time,
                n_heads=self.num_heads,
                name="rel_pos_bias",
            )
            rel_pos_bias = rel_pos_bias_mod(T, T)

        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("bhsc, bhtc -> bhst", q, k) * scale

        if rel_pos_bias is not None:
            attn_weights = attn_weights + rel_pos_bias

        if self.causal_in_time and self.bias_type != "rel":
            mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
            attn_weights = jnp.where(mask[None, None, :, :], float("-inf"), attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        att_out = jnp.einsum("bhst, bhtc -> bhsc", attn_weights, v)

        att_out = rearrange(att_out, "(b h w d) he t c -> (t b) (he c) h w d", h=H, w=W, d=D)

        # 1x1 output conv
        output_head_weight = self.param(
            "output_head_weight",
            nn.initializers.lecun_normal(),
            (self.hidden_dim, self.hidden_dim, 1, 1, 1),
        )
        output_head_bias = self.param(
            "output_head_bias",
            nn.initializers.zeros_init(),
            (self.hidden_dim,),
        )

        x = _conv3d_1x1(att_out, output_head_weight, output_head_bias)
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)

        return x + residual, []
