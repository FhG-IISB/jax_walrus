"""
Decoder: JAX/Flax translation of walrus.models.decoders.vstride_decoder.AdaptiveDVstrideDecoder

Transposed convolution decoder with adaptive strides and periodic boundary
condition handling. Uses circular padding before transposed conv and cropping
after for periodic spatial dimensions.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax_walrus.normalization import RMSGroupNorm
from jax_walrus.encoder import _conv_transpose3d


# Boundary condition value matching the_well.data.datasets.BoundaryCondition.PERIODIC
BC_PERIODIC = 2


class AdaptiveDVstrideDecoder(nn.Module):
    """
    Variable-stride 3D transposed conv decoder.

    Two transposed conv layers (hidden -> inner -> output), with adaptive
    handling of singleton spatial dims, periodic boundary padding, and
    output channel selection via ``state_labels``.
    """

    input_dim: int = 768
    inner_dim: int = 192
    output_dim: int = 4
    base_kernel_size: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4), (8, 4))
    groups: int = 12
    spatial_dims: int = 3
    use_silu: bool = True

    @nn.compact
    def __call__(self, x, state_labels, bcs, stride1, stride2):
        """
        Args:
            x: (TB, C_in, H, W, D)
            state_labels: integer indices for output channel selection
            bcs: boundary conditions list, shape (n_dims, 2)
            stride1: tuple of 3 ints
            stride2: tuple of 3 ints

        Returns:
            (TB, C_out_selected, H', W', D')
        """
        # Decoder reverses kernel order: first uses kernel2 then kernel1
        base_kernel1 = tuple(
            self.base_kernel_size[i][1] for i in range(self.spatial_dims)
        )
        base_kernel2 = tuple(
            self.base_kernel_size[i][0] for i in range(self.spatial_dims)
        )

        # --- First transposed conv: hidden_dim -> inner_dim ---
        proj1_weight = self.param(
            "proj1_weight",
            nn.initializers.lecun_normal(),
            (self.input_dim, self.inner_dim, *base_kernel1),
        )

        x = self._adaptive_conv_transpose(
            x,
            bcs,
            proj1_weight,
            bias=None,
            stride=stride1,
            padding=(0,) * self.spatial_dims,
        )

        x = RMSGroupNorm(
            num_groups=self.groups, num_channels=self.inner_dim, name="norm1"
        )(x)
        x = jax.nn.silu(x) if self.use_silu else jax.nn.gelu(x)

        # --- Second transposed conv: inner_dim -> output_dim ---
        proj2_weight = self.param(
            "proj2_weight",
            nn.initializers.lecun_normal(),
            (self.inner_dim, self.output_dim, *base_kernel2),
        )
        proj2_bias = self.param(
            "proj2_bias",
            nn.initializers.zeros_init(),
            (self.output_dim,),
        )

        # Select output channels by state_labels
        w2 = proj2_weight[:, state_labels]
        b2 = proj2_bias[state_labels]

        x = self._adaptive_conv_transpose(
            x,
            bcs,
            w2,
            bias=b2,
            stride=stride2,
            padding=(0,) * self.spatial_dims,
        )

        return x

    def _adaptive_conv_transpose(self, x, bcs, weight, bias, stride, padding):
        """
        Transposed conv with adaptive singleton handling and periodic BC padding.

        For each spatial dimension:
        - Singleton dims: average the kernel and set stride to 1
        - Periodic dims: circular-pad by ``(k-s)//s`` before conv,
          then crop by ``k-s`` from both sides after conv
        - Non-periodic dims: no pre-padding or cropping
        """
        spatial_shape = x.shape[2:]
        stride = list(stride)
        padding = list(padding)

        periodic_padding_list = []
        padding_out_list = []

        bcs_padded = (
            list(bcs) if len(bcs) >= self.spatial_dims else list(bcs) + [[2, 2]]
        )

        # Iterate reversed spatial dims matching PyTorch order
        for i in range(1, self.spatial_dims + 1):
            dim_idx = self.spatial_dims - i
            dim_size = spatial_shape[dim_idx]
            weight_dim_idx = -(i)

            if dim_size == 1:
                weight = jnp.mean(weight, axis=weight_dim_idx, keepdims=True)
                stride[-i] = 1
                padding[-i] = 0
                periodic_padding_list.extend([0, 0])
                padding_out_list.extend([0, 0])
            else:
                k = weight.shape[weight_dim_idx]
                s = stride[-i]
                pad_in = (k - s) // s
                pad_out = k - s

                bc_val = int(bcs_padded[-i][0])

                if bc_val == BC_PERIODIC:
                    periodic_padding_list.extend([pad_in, pad_in])
                    padding_out_list.extend([-pad_out, -pad_out])
                else:
                    periodic_padding_list.extend([0, 0])
                    padding_out_list.extend([0, 0])

        # Reverse to (H, W, D) order
        periodic_padding_list = periodic_padding_list[::-1]
        padding_out_list = padding_out_list[::-1]

        # Apply circular padding before conv
        if any(p > 0 for p in periodic_padding_list):
            pad_pairs = [(0, 0), (0, 0)]  # batch, channel
            for ii in range(0, len(periodic_padding_list), 2):
                pad_pairs.append(
                    (periodic_padding_list[ii], periodic_padding_list[ii + 1])
                )
            x = jnp.pad(x, pad_pairs, mode="wrap")

        # Transposed conv
        x = _conv_transpose3d(
            x, weight, bias=bias, stride=tuple(stride), padding=tuple(padding)
        )

        # Apply output cropping
        if any(p < 0 for p in padding_out_list):
            slices = [slice(None), slice(None)]  # batch, channel
            for ii in range(0, len(padding_out_list), 2):
                left_crop = -padding_out_list[ii]
                right_crop = -padding_out_list[ii + 1]
                if left_crop > 0 or right_crop > 0:
                    dim_size = x.shape[2 + ii // 2]
                    slices.append(
                        slice(
                            left_crop if left_crop > 0 else None,
                            (dim_size - right_crop) if right_crop > 0 else None,
                        )
                    )
                else:
                    slices.append(slice(None))
            x = x[tuple(slices)]

        return x
