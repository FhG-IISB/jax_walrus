"""
RMSGroupNorm: JAX/Flax translation of walrus.models.shared_utils.normalization.RMSGroupNorm

Applies RMS normalization per group (no mean subtraction), then applies a learned
per-channel scale. Uses the same (B, C, *spatial) layout as the PyTorch original
for 1-to-1 weight mapping.
"""

import jax.numpy as jnp
import flax.linen as nn


class RMSGroupNorm(nn.Module):
    """RMS version of GroupNorm (no bias, no mean centering)."""

    num_groups: int
    num_channels: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, *spatial)
        spatial_dims = x.shape[2:]
        B = x.shape[0]

        # Reshape into groups: (B, G, C//G, *spatial)
        grouped = x.reshape(B, self.num_groups, -1, *spatial_dims)

        # RMS norm over spatial dims within each group (dims 3+)
        norm_axes = tuple(range(3, grouped.ndim))
        rms = jnp.sqrt(jnp.mean(grouped ** 2, axis=norm_axes, keepdims=True) + self.eps)
        grouped = grouped / rms

        # Reshape back: (B, C, *spatial)
        out = grouped.reshape(B, self.num_channels, *spatial_dims)

        # Learnable per-channel scale (maps to PyTorch's self.weight)
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.num_channels,),
        )
        # Broadcast weight over batch and spatial dims
        indexing = (slice(None),) + (None,) * len(spatial_dims)
        return out * weight[indexing]
