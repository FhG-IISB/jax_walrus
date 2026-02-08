"""
Processor Block: JAX/Flax translation of
walrus.models.spatiotemporal_blocks.space_time_split.SpaceTimeSplitBlock

Composes temporal attention -> spatial attention -> channel mixing (identity).
Each block is one layer in the isotropic processor stack.
"""

import flax.linen as nn
from einops import rearrange

from walrus_jax.spatial_attention import FullAttention
from walrus_jax.temporal_attention import AxialTimeAttention


class SpaceTimeSplitBlock(nn.Module):
    """
    One processor block: temporal mixing -> spatial mixing -> channel mixing.

    Input/Output: ``(T, B, C, H, W, D)``
    """

    hidden_dim: int = 768
    num_heads: int = 12
    mlp_dim: int = 0  # 0 = hidden_dim * 4
    drop_path: float = 0.0
    causal_in_time: bool = False
    bias_type: str = "rel"

    @nn.compact
    def __call__(self, x, bcs=None, return_att=False, deterministic: bool = True):
        """
        Args:
            x: (T, B, C, H, W, D)
            bcs: boundary conditions
            deterministic: if False, enable stochastic depth (drop_path)

        Returns:
            x: (T, B, C, H, W, D), att_maps: []
        """
        T, B, C, H, W, D = x.shape

        # 1) Temporal mixing
        x, t_att = AxialTimeAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            drop_path=self.drop_path,
            bias_type=self.bias_type,
            causal_in_time=self.causal_in_time,
            name="time_mixing",
        )(x, return_att=return_att, deterministic=deterministic)

        # 2) Spatial mixing — operates on (T*B, C, H, W, D)
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        x, s_att = FullAttention(
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            drop_path=self.drop_path,
            name="space_mixing",
        )(x, bcs=bcs, return_att=return_att, deterministic=deterministic)
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)

        # 3) Channel mixing (identity in default Walrus config)

        return x, t_att + s_att
