"""
Encoder: JAX/Flax translation of walrus.models.encoders.vstride_encoder

Variable-stride 3D convolution encoder with adaptive handling of singleton
spatial dimensions. Includes both the plain encoder and the SpaceBag variant
that implements sparse channel selection via index-based weight subsampling.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from walrus_jax.normalization import RMSGroupNorm


def _conv3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    """
    Manual 3D convolution matching PyTorch Conv3d behavior.

    Args:
        x: (N, C_in, D1, D2, D3)
        weight: (C_out, C_in, k1, k2, k3)
        bias: (C_out,) or None
        stride: per-spatial-dim strides
        padding: per-spatial-dim padding

    Returns:
        (N, C_out, D1', D2', D3')
    """
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC
    w_t = jnp.transpose(weight, (2, 3, 4, 1, 0))  # kD kH kW Cin Cout

    out = jax.lax.conv_general_dilated(
        x_t,
        w_t,
        window_strides=stride,
        padding=[(p, p) for p in padding],
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    if bias is not None:
        out = out + bias[None, None, None, None, :]
    return jnp.transpose(out, (0, 4, 1, 2, 3))


def _conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    """
    Manual 3D transposed convolution matching PyTorch ConvTranspose3d.

    Implements transposed convolution via zero-insertion upsampling followed
    by a regular convolution with a spatially flipped kernel, exactly matching
    PyTorch's ``F.conv_transpose3d`` semantics.

    Args:
        x: (N, C_in, D1, D2, D3)
        weight: (C_in, C_out, k1, k2, k3) — PyTorch ConvTranspose weight layout
        bias: (C_out,) or None
        stride: per-spatial-dim strides
        padding: per-spatial-dim padding

    Returns:
        (N, C_out, D1', D2', D3')
    """
    N, C_in, D, H, W = x.shape
    sD, sH, sW = stride
    kD, kH, kW = weight.shape[2:]

    # Step 1: Upsample by inserting (stride-1) zeros between pixels
    if any(s > 1 for s in stride):
        D_up = D + (D - 1) * (sD - 1)
        H_up = H + (H - 1) * (sH - 1)
        W_up = W + (W - 1) * (sW - 1)
        x_up = jnp.zeros((N, C_in, D_up, H_up, W_up), dtype=x.dtype)
        x_up = x_up.at[:, :, ::sD, ::sH, ::sW].set(x)
        x = x_up

    # Step 2: Flip kernel spatially (critical for matching PyTorch)
    weight = weight[:, :, ::-1, ::-1, ::-1]

    # Step 3: Regular conv with adjusted padding
    eff_padding = [(k - 1 - p, k - 1 - p) for k, p in zip([kD, kH, kW], padding)]

    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_t = jnp.transpose(weight, (2, 3, 4, 1, 0))  # (kD, kH, kW, C_in, C_out)

    out = jax.lax.conv_general_dilated(
        x_t,
        w_t,
        window_strides=(1, 1, 1),
        padding=eff_padding,
        dimension_numbers=("NDHWC", "DHWOI", "NDHWC"),
    )

    if bias is not None:
        out = out + bias[None, None, None, None, :]

    return jnp.transpose(out, (0, 4, 1, 2, 3))


class AdaptiveDVstrideEncoder(nn.Module):
    """
    Variable-stride 3D encoder with adaptive convolutions.

    Two conv layers (input -> inner -> output), each followed by RMSGroupNorm
    and activation. Singleton spatial dimensions are handled by summing the
    kernel over those axes and setting stride to 1.
    """

    input_dim: int = 768
    inner_dim: int = 192
    output_dim: int = 768
    base_kernel_size: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4), (8, 4))
    groups: int = 12
    spatial_dims: int = 3
    use_silu: bool = True

    @nn.compact
    def __call__(self, x, stride1, stride2):
        """
        Args:
            x: (TB, C, H, W, D)
            stride1: tuple of 3 ints — stride for first conv
            stride2: tuple of 3 ints — stride for second conv

        Returns:
            (TB, C_out, H', W', D')
        """
        base_kernel1 = tuple(self.base_kernel_size[i][0] for i in range(self.spatial_dims))
        base_kernel2 = tuple(self.base_kernel_size[i][1] for i in range(self.spatial_dims))

        # First conv: input_dim -> inner_dim
        proj1_weight = self.param(
            "proj1_weight",
            nn.initializers.lecun_normal(),
            (self.inner_dim, self.input_dim, *base_kernel1),
        )

        w1 = proj1_weight
        s1 = list(stride1)
        spatial_shape = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape), start=1):
            if dim_size == 1:
                w1 = jnp.sum(w1, axis=-i, keepdims=True)
                s1[-i] = 1

        x = _conv3d(x, w1, bias=None, stride=tuple(s1), padding=(0, 0, 0))
        x = RMSGroupNorm(num_groups=self.groups, num_channels=self.inner_dim, name="norm1")(x)
        x = jax.nn.silu(x) if self.use_silu else jax.nn.gelu(x)

        # Second conv: inner_dim -> output_dim
        proj2_weight = self.param(
            "proj2_weight",
            nn.initializers.lecun_normal(),
            (self.output_dim, self.inner_dim, *base_kernel2),
        )

        w2 = proj2_weight
        s2 = list(stride2)
        spatial_shape2 = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape2), start=1):
            if dim_size == 1:
                w2 = jnp.sum(w2, axis=-i, keepdims=True)
                s2[-i] = 1

        x = _conv3d(x, w2, bias=None, stride=tuple(s2), padding=(0, 0, 0))
        x = RMSGroupNorm(num_groups=self.groups, num_channels=self.output_dim, name="norm2")(x)
        x = jax.nn.silu(x) if self.use_silu else jax.nn.gelu(x)

        return x


class SpaceBagAdaptiveDVstrideEncoder(nn.Module):
    """
    SpaceBag variant of the encoder.

    The first conv weight is subsampled by ``field_indices``, implementing a
    sparse embedding bag in the first layer. Non-BC channels are scaled by
    ``sqrt(total_fields / selected_fields)`` to preserve magnitude.

    Note: The PyTorch code scales ``weight[:, :-2]`` (not ``:-extra_dims``),
    matching the original implementation exactly.
    """

    input_dim: int = 768
    inner_dim: int = 192
    output_dim: int = 768
    base_kernel_size: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4), (8, 4))
    groups: int = 12
    spatial_dims: int = 3
    extra_dims: int = 3
    use_silu: bool = True

    @nn.compact
    def __call__(self, x, field_indices, stride1, stride2):
        """
        Args:
            x: (TB, C_selected, H, W, D)
            field_indices: integer indices selecting channels from the full weight
            stride1, stride2: tuple of 3 ints

        Returns:
            (TB, C_out, H', W', D')
        """
        base_kernel1 = tuple(self.base_kernel_size[i][0] for i in range(self.spatial_dims))
        base_kernel2 = tuple(self.base_kernel_size[i][1] for i in range(self.spatial_dims))

        # Full proj1 weight: (inner_dim, full_input_dim, k1, k2, k3)
        proj1_weight = self.param(
            "proj1_weight",
            nn.initializers.lecun_normal(),
            (self.inner_dim, self.input_dim, *base_kernel1),
        )

        # Subsample input channels and apply SpaceBag scaling
        w1 = proj1_weight[:, field_indices]
        scale_factor = (
            (proj1_weight.shape[1] - self.extra_dims)
            / (w1.shape[1] - self.extra_dims)
        ) ** 0.5

        # Match PyTorch: scale all except last 2 channels (:-2, not :-extra_dims)
        w1 = jnp.concatenate([
            w1[:, :-2] * scale_factor,
            w1[:, -2:]
        ], axis=1)

        s1 = list(stride1)
        spatial_shape = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape), start=1):
            if dim_size == 1:
                w1 = jnp.sum(w1, axis=-i, keepdims=True)
                s1[-i] = 1

        x = _conv3d(x, w1, bias=None, stride=tuple(s1), padding=(0, 0, 0))
        x = RMSGroupNorm(num_groups=self.groups, num_channels=self.inner_dim, name="norm1")(x)
        x = jax.nn.silu(x) if self.use_silu else jax.nn.gelu(x)

        # Second conv (same as plain encoder)
        proj2_weight = self.param(
            "proj2_weight",
            nn.initializers.lecun_normal(),
            (self.output_dim, self.inner_dim, *base_kernel2),
        )

        w2 = proj2_weight
        s2 = list(stride2)
        spatial_shape2 = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape2), start=1):
            if dim_size == 1:
                w2 = jnp.sum(w2, axis=-i, keepdims=True)
                s2[-i] = 1

        x = _conv3d(x, w2, bias=None, stride=tuple(s2), padding=(0, 0, 0))
        x = RMSGroupNorm(num_groups=self.groups, num_channels=self.output_dim, name="norm2")(x)
        x = jax.nn.silu(x) if self.use_silu else jax.nn.gelu(x)

        return x
