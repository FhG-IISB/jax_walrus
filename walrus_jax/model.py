"""
IsotropicModel: JAX/Flax translation of walrus.models.isotropic_model.IsotropicModel

Top-level model that composes encoder -> processor blocks -> decoder.
Maintains exact 1-to-1 weight compatibility with the PyTorch model, including
both 2D and 3D encoder/decoder variants (embed_2/embed_3, debed_2/debed_3)
and the encoder_dummy parameter.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from walrus_jax.encoder import AdaptiveDVstrideEncoder, SpaceBagAdaptiveDVstrideEncoder
from walrus_jax.decoder import AdaptiveDVstrideDecoder
from walrus_jax.processor import SpaceTimeSplitBlock


class IsotropicModel(nn.Module):
    """
    Isotropic model: encoder -> N processor blocks -> decoder.

    Weight layout mirrors the PyTorch model exactly:

    - ``embed_2`` / ``embed_3``: encoder variants for 2D / 3D data
    - ``debed_2`` / ``debed_3``: decoder variants for 2D / 3D data
    - ``encoder_dummy``: unused parameter kept for weight compatibility
    - ``blocks_0`` .. ``blocks_{N-1}``: processor blocks

    Both 2D and 3D variants use the same class with ``spatial_dims=3``
    (2D data is padded with a singleton dim). At runtime, ``dim_key``
    selects which variant to use based on the input's non-singleton
    spatial dimensions.
    """

    hidden_dim: int = 768
    intermediate_dim: int = 192
    n_states: int = 4
    processor_blocks: int = 12
    groups: int = 16
    num_heads: int = 12
    mlp_dim: int = 0
    max_d: int = 3
    causal_in_time: bool = False
    drop_path: float = 0.05
    bias_type: str = "rel"
    base_kernel_size: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4), (8, 4))
    use_spacebag: bool = True
    use_silu: bool = True
    include_d: Tuple[int, ...] = (2, 3)
    encoder_groups: int = 16

    def _make_encoder(self, name: str, field_indices, x_flat, stride1, stride2):
        """Create and call an encoder (SpaceBag or plain) with the given name."""
        if self.use_spacebag and field_indices is not None:
            return SpaceBagAdaptiveDVstrideEncoder(
                input_dim=self.n_states,
                inner_dim=self.intermediate_dim,
                output_dim=self.hidden_dim,
                base_kernel_size=self.base_kernel_size,
                groups=self.encoder_groups,
                spatial_dims=self.max_d,
                extra_dims=3,
                use_silu=self.use_silu,
                name=name,
            )(x_flat, field_indices, stride1, stride2)
        else:
            return AdaptiveDVstrideEncoder(
                input_dim=self.n_states,
                inner_dim=self.intermediate_dim,
                output_dim=self.hidden_dim,
                base_kernel_size=self.base_kernel_size,
                groups=self.encoder_groups,
                spatial_dims=self.max_d,
                use_silu=self.use_silu,
                name=name,
            )(x_flat, stride1, stride2)

    def _make_decoder(self, name: str, x_flat, state_labels, bcs_flat, stride1, stride2):
        """Create and call a decoder with the given name."""
        return AdaptiveDVstrideDecoder(
            input_dim=self.hidden_dim,
            inner_dim=self.intermediate_dim,
            output_dim=self.n_states,
            base_kernel_size=self.base_kernel_size,
            groups=self.encoder_groups,
            spatial_dims=self.max_d,
            use_silu=self.use_silu,
            name=name,
        )(x_flat, state_labels, bcs_flat, stride1, stride2)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        state_labels: jnp.ndarray,
        bcs: list,
        stride1: Tuple[int, ...] = (4, 4, 4),
        stride2: Tuple[int, ...] = (4, 4, 4),
        field_indices: Optional[jnp.ndarray] = None,
        dim_key: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Forward pass (inference mode).

        Args:
            x: (T, B, C, H, W, D) — always 3 spatial dims (pad 2D data with size-1 dim)
            state_labels: (C_out,) output channel indices
            bcs: boundary conditions, list of (n_dims, 2)
            stride1: encoder first-layer stride per spatial dim
            stride2: encoder second-layer stride per spatial dim
            field_indices: for SpaceBag encoder, which input channels to select
            dim_key: which encoder/decoder variant (2 or 3). Auto-detected if None.

        Returns:
            (1, B, C_out, H, W, D) for non-causal, (T, B, C_out, H, W, D) for causal
        """
        T, B, C = x.shape[:3]

        # Unused param, present for weight compatibility
        _ = self.param("encoder_dummy", nn.initializers.ones, (1,))

        # Determine variant
        if dim_key is None:
            spatial_shape = x.shape[3:]
            dim_key = sum(int(s != 1) for s in spatial_shape)
        enc_name = f"embed_{dim_key}"
        dec_name = f"debed_{dim_key}"

        # Encode
        x_flat = rearrange(x, "T B ... -> (T B) ...")
        x_enc = self._make_encoder(enc_name, field_indices, x_flat, stride1, stride2)
        x_enc = rearrange(x_enc, "(T B) ... -> T B ...", T=T)

        # Process
        dp = jnp.linspace(0, self.drop_path, self.processor_blocks)
        x_proc = x_enc
        for i in range(self.processor_blocks):
            x_proc, _ = SpaceTimeSplitBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                drop_path=float(dp[i]),
                causal_in_time=self.causal_in_time,
                bias_type=self.bias_type,
                name=f"blocks_{i}",
            )(x_proc, bcs=bcs, return_att=False)

        # Non-causal: only decode last time step
        if not self.causal_in_time:
            x_proc = x_proc[-1:]

        # Decode (strides are reversed)
        T_out = x_proc.shape[0]
        x_dec = rearrange(x_proc, "T B ... -> (T B) ...")
        bcs_flat = bcs[0] if isinstance(bcs, tuple) else bcs
        x_dec = self._make_decoder(dec_name, x_dec, state_labels, bcs_flat, stride2, stride1)
        x_dec = rearrange(x_dec, "(T B) ... -> T B ...", T=T_out)

        return x_dec
