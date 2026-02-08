"""
Rotary Position Embeddings (RoPE) for JAX.

1-to-1 translation of the lucidrains-style RotaryEmbedding used in Walrus's
FullAttention spatial block (lr_rope_temporary.py).

Also includes the simpler RotaryEmbedding and apply_rotary_pos_emb from
position_biases.py used in AxialTimeAttention.
"""

from math import pi

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def rotate_half_lr(x):
    """Lucidrains-style rotate_half for the full RoPE."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x1 = x1[..., 0]
    x2 = x2[..., 0]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2, freqs_seq_dim=None):
    """Apply rotary embeddings to tensor t using precomputed freqs."""
    if t.ndim == 3 or freqs_seq_dim is not None:
        if freqs_seq_dim is None:
            freqs_seq_dim = 0
        seq_len = t.shape[seq_dim]
        freqs = jax.lax.dynamic_slice_in_dim(
            freqs, freqs.shape[freqs_seq_dim] - seq_len, seq_len, axis=freqs_seq_dim
        )

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    t_transformed = (t_middle * jnp.cos(freqs) * scale) + (
        rotate_half_lr(t_middle) * jnp.sin(freqs) * scale
    )

    return jnp.concatenate((t_left, t_transformed, t_right), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Full RotaryEmbedding (lucidrains-style, used in FullAttention spatial block)
# ──────────────────────────────────────────────────────────────────────────────

class LRRotaryEmbedding(nn.Module):
    """
    Lucidrains-style RotaryEmbedding used in the FullAttention spatial block.

    Matches the PyTorch RotaryEmbedding from lr_rope_temporary.py.
    The ``freqs`` buffer is a 1-D learned/fixed frequency vector and the
    ``get_axial_freqs`` method produces a broadcastable ND frequency grid.
    """

    dim: int
    freqs_for: str = "pixel"
    theta: float = 10000.0
    max_freq: float = 256.0
    num_freqs: int = 1
    theta_rescale_factor: float = 1.0

    def setup(self):
        dim = self.dim
        theta = self.theta * self.theta_rescale_factor ** (dim / (dim - 2)) if dim > 2 else self.theta

        if self.freqs_for == "lang":
            freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        elif self.freqs_for == "pixel":
            freqs = jnp.linspace(1.0, self.max_freq / 2, dim // 2) * pi
        elif self.freqs_for == "constant":
            freqs = jnp.ones(self.num_freqs)
        else:
            raise ValueError(f"Unknown freqs_for: {self.freqs_for}")

        self.freqs = self.variable("params", "freqs", lambda: freqs)

    def __call__(self, t, seq_len=None, offset=0, index=0):
        """Compute frequencies for positions t."""
        freqs = self.freqs.value
        freqs = jnp.einsum("..., f -> ... f", t.astype(freqs.dtype), freqs)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        return freqs

    def get_axial_freqs(self, *dims):
        """Compute broadcastable axial frequencies for ND spatial grids."""
        all_freqs = []
        for ind, dim_size in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = jnp.linspace(-1.0, 1.0, dim_size)
            else:
                pos = jnp.arange(dim_size, dtype=jnp.float32)
            freqs = self(pos, seq_len=dim_size, index=ind)
            shape = [1] * len(dims)
            shape[ind] = dim_size
            freqs = freqs.reshape(*shape, -1)
            all_freqs.append(freqs)

        all_freqs = jnp.broadcast_arrays(*all_freqs)
        return jnp.concatenate(all_freqs, axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Simple RotaryEmbedding (position_biases.py style, used in AxialTimeAttention)
# ──────────────────────────────────────────────────────────────────────────────

def rotate_half_simple(x):
    """Simple rotate_half from position_biases.py."""
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_simple(pos, t):
    """Apply rotary position embeddings (simple version from position_biases.py)."""
    return (t * jnp.cos(pos)) + (rotate_half_simple(t) * jnp.sin(pos))


class SimpleRotaryEmbedding(nn.Module):
    """Simple rotary embedding used in AxialTimeAttention (position_biases.py style)."""

    dim: int

    def setup(self):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self.inv_freq = self.variable("params", "inv_freq", lambda: inv_freq)

    def __call__(self, max_seq_len):
        """Returns frequencies of shape (max_seq_len, dim)."""
        seq = jnp.arange(max_seq_len, dtype=self.inv_freq.value.dtype)
        freqs = jnp.einsum("i , j -> i j", seq, self.inv_freq.value)
        return jnp.concatenate((freqs, freqs), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Relative Position Bias (T5-style, used in AxialTimeAttention)
# ──────────────────────────────────────────────────────────────────────────────

class RelativePositionBias(nn.Module):
    """T5-style relative position bias, matching position_biases.py."""

    bidirectional: bool = True
    num_buckets: int = 32
    max_distance: int = 128
    n_heads: int = 2

    @nn.compact
    def __call__(self, qlen, klen, bc=0):
        relative_attention_bias = nn.Embed(
            num_embeddings=self.num_buckets,
            features=self.n_heads,
            name="relative_attention_bias",
        )

        context_position = jnp.arange(qlen)[:, None]
        memory_position = jnp.arange(klen)[None, :]
        relative_position = memory_position - context_position

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        values = relative_attention_bias(rp_bucket)  # (qlen, klen, n_heads)
        values = jnp.transpose(values, (2, 0, 1))[None, ...]  # (1, n_heads, qlen, klen)

        if not self.bidirectional:
            mask = relative_position > 0
            values = jnp.where(mask[None, None, :, :], float("-inf"), values)

        return values

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=32):
        import math

        ret = jnp.zeros_like(relative_position, dtype=jnp.int32)
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret = ret + (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)

        ret = ret + jnp.where(is_small, n, val_if_large)
        return ret
