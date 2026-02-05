"""
Component-level equivalence tests: verify JAX and PyTorch implementations
produce identical outputs given the same weights and inputs.

Tests each component individually with random weights:
1. RMSGroupNorm
2. AdaptiveDVstrideEncoder
3. AdaptiveDVstrideDecoder
4. FullAttention (spatial block)
5. AxialTimeAttention (temporal block)

Requires:
- PyTorch walrus source on sys.path (set WALRUS_ROOT env var or default to ../walrus)
- the_well package (or mocked automatically)
"""

import sys
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

# ── Resolve walrus source path ──
WALRUS_ROOT = os.environ.get("WALRUS_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "walrus"))
sys.path.insert(0, WALRUS_ROOT)

# ── Mock the_well to avoid hard dependency ──
import types

_well_mod = types.ModuleType("the_well")
_data_mod = types.ModuleType("the_well.data")
_ds_mod = types.ModuleType("the_well.data.datasets")


class _BoundaryConditionEnum:
    _map = {"PERIODIC": type("BC", (), {"value": 2})(), "OPEN": type("BC", (), {"value": 0})()}
    def __getitem__(self, key):
        return self._map[key]


_ds_mod.BoundaryCondition = _BoundaryConditionEnum()
_well_mod.data = _data_mod
_data_mod.datasets = _ds_mod
sys.modules.setdefault("the_well", _well_mod)
sys.modules.setdefault("the_well.data", _data_mod)
sys.modules.setdefault("the_well.data.datasets", _ds_mod)


# ── Import PyTorch implementations ──
from walrus.models.shared_utils.normalization import RMSGroupNorm as TorchRMSGroupNorm
from walrus.models.encoders.vstride_encoder import AdaptiveDVstrideEncoder as TorchEncoder
from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder as TorchDecoder
from walrus.models.spatial_blocks.full_attention import FullAttention as TorchFullAttention
from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention as TorchAxialTime

# ── Import JAX implementations ──
from walrus_jax.normalization import RMSGroupNorm as JaxRMSGroupNorm
from walrus_jax.encoder import AdaptiveDVstrideEncoder as JaxEncoder
from walrus_jax.decoder import AdaptiveDVstrideDecoder as JaxDecoder
from walrus_jax.spatial_attention import FullAttention as JaxFullAttention
from walrus_jax.temporal_attention import AxialTimeAttention as JaxAxialTime
from walrus_jax.convert_weights import torch_to_numpy


np.random.seed(42)
torch.manual_seed(42)


def assert_close(name, torch_out, jax_out, atol=1e-4, rtol=1e-4):
    """Check that PyTorch and JAX outputs match."""
    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)
    max_diff = np.max(np.abs(t_np - j_np))
    mean_diff = np.mean(np.abs(t_np - j_np))
    print(f"  [{name}] shape: torch={t_np.shape} jax={j_np.shape} | max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}")
    if not np.allclose(t_np, j_np, atol=atol, rtol=rtol):
        print(f"  [FAIL] MISMATCH (atol={atol}, rtol={rtol})")
        return False
    print(f"  [PASS] MATCH")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: RMSGroupNorm
# ══════════════════════════════════════════════════════════════════════════════

def test_rmsgroupnorm():
    print("\n" + "=" * 60)
    print("Test 1: RMSGroupNorm")
    print("=" * 60)

    num_groups, num_channels = 12, 768
    x_np = np.random.randn(2, num_channels, 4, 4, 4).astype(np.float32)

    torch_norm = TorchRMSGroupNorm(num_groups, num_channels)
    torch_norm.eval()
    weight_np = torch_to_numpy(torch_norm.weight)
    with torch.no_grad():
        torch_out = torch_norm(torch.from_numpy(x_np))

    jax_norm = JaxRMSGroupNorm(num_groups=num_groups, num_channels=num_channels)
    jax_params = {"params": {"weight": weight_np}}
    jax_out = jax_norm.apply(jax_params, jnp.array(x_np))

    return assert_close("RMSGroupNorm", torch_out, jax_out)


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: AdaptiveDVstrideEncoder
# ══════════════════════════════════════════════════════════════════════════════

def test_encoder():
    print("\n" + "=" * 60)
    print("Test 2: AdaptiveDVstrideEncoder")
    print("=" * 60)

    n_states, inner_dim, output_dim, groups = 7, 48, 96, 12
    bks = ((4, 4), (4, 4), (4, 4))
    T, B = 2, 1
    x_np = np.random.randn(T, B, n_states, 32, 32, 32).astype(np.float32)

    torch_enc = TorchEncoder(
        kernel_scales_seq=((4, 4),), base_kernel_size3d=bks,
        input_dim=n_states, inner_dim=inner_dim, output_dim=output_dim,
        spatial_dims=3, groups=groups, variable_downsample=True,
        variable_deterministic_ds=True, learned_pad=True,
        norm_layer=TorchRMSGroupNorm, activation=nn.GELU,
    )
    torch_enc.eval()

    proj1_w = torch_to_numpy(torch_enc.proj1.weight)
    norm1_w = torch_to_numpy(torch_enc.norm1.weight)
    proj2_w = torch_to_numpy(torch_enc.proj2.weight)
    norm2_w = torch_to_numpy(torch_enc.norm2.weight)

    with torch.no_grad():
        torch_out, _ = torch_enc(torch.from_numpy(x_np), random_kernel=((4, 4), (4, 4), (4, 4)))

    x_flat = x_np.reshape(T * B, n_states, 32, 32, 32)
    jax_enc = JaxEncoder(
        input_dim=n_states, inner_dim=inner_dim, output_dim=output_dim,
        base_kernel_size=bks, groups=groups, spatial_dims=3, use_silu=False,
    )
    jax_params = {"params": {
        "proj1_weight": proj1_w, "norm1": {"weight": norm1_w},
        "proj2_weight": proj2_w, "norm2": {"weight": norm2_w},
    }}
    jax_out = jax_enc.apply(jax_params, jnp.array(x_flat), (4, 4, 4), (4, 4, 4))
    jax_out_reshaped = jax_out.reshape(T, B, *jax_out.shape[1:])

    return assert_close("Encoder", torch_out, jax_out_reshaped, atol=1e-3, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: AdaptiveDVstrideDecoder
# ══════════════════════════════════════════════════════════════════════════════

def test_decoder():
    print("\n" + "=" * 60)
    print("Test 3: AdaptiveDVstrideDecoder")
    print("=" * 60)

    input_dim, inner_dim, output_dim, groups = 96, 48, 4, 12
    bks = ((4, 4), (4, 4), (4, 4))
    T, B = 1, 1
    x_np = np.random.randn(T, B, input_dim, 2, 2, 2).astype(np.float32)
    state_labels_np = np.array([0, 1, 2, 3])
    bcs = [[2, 2], [2, 2], [2, 2]]

    torch_dec = TorchDecoder(
        base_kernel_size3d=bks, input_dim=input_dim, inner_dim=inner_dim,
        output_dim=output_dim, spatial_dims=3, groups=groups, learned_pad=True,
        norm_layer=TorchRMSGroupNorm, activation=nn.GELU,
    )
    torch_dec.eval()

    proj1_w = torch_to_numpy(torch_dec.proj1.weight)
    norm1_w = torch_to_numpy(torch_dec.norm1.weight)
    proj2_w = torch_to_numpy(torch_dec.proj2.weight)
    proj2_b = torch_to_numpy(torch_dec.proj2.bias)

    with torch.no_grad():
        torch_out = torch_dec(
            torch.from_numpy(x_np), torch.tensor(state_labels_np),
            bcs=bcs, stage_info={"random_kernel": ((4, 4), (4, 4), (4, 4))},
        )

    debed_kernel = tuple((b, a) for (a, b) in ((4, 4), (4, 4), (4, 4)))
    x_flat = x_np.reshape(T * B, input_dim, 2, 2, 2)
    jax_dec = JaxDecoder(
        input_dim=input_dim, inner_dim=inner_dim, output_dim=output_dim,
        base_kernel_size=bks, groups=groups, spatial_dims=3, use_silu=False,
    )
    jax_params = {"params": {
        "proj1_weight": proj1_w, "norm1": {"weight": norm1_w},
        "proj2_weight": proj2_w, "proj2_bias": proj2_b,
    }}
    dec_stride1 = tuple(debed_kernel[i][0] for i in range(3))
    dec_stride2 = tuple(debed_kernel[i][1] for i in range(3))
    jax_out = jax_dec.apply(
        jax_params, jnp.array(x_flat), jnp.array(state_labels_np),
        bcs, dec_stride1, dec_stride2,
    )
    jax_out_reshaped = jax_out.reshape(T, B, *jax_out.shape[1:])

    return assert_close("Decoder", torch_out, jax_out_reshaped, atol=1e-3, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: FullAttention (spatial block)
# ══════════════════════════════════════════════════════════════════════════════

def test_full_attention():
    print("\n" + "=" * 60)
    print("Test 4: FullAttention (spatial block)")
    print("=" * 60)

    hidden_dim, num_heads = 96, 4
    mlp_dim = hidden_dim * 4
    B, H, W, D = 1, 2, 2, 2
    x_np = np.random.randn(B, hidden_dim, H, W, D).astype(np.float32)

    torch_attn = TorchFullAttention(hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_heads=num_heads, drop_path=0.0)
    torch_attn.eval()

    with torch.no_grad():
        torch_out, _ = torch_attn(torch.from_numpy(x_np), [[2, 2], [2, 2], [2, 2]])

    jax_attn = JaxFullAttention(hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_heads=num_heads)
    jax_params = {"params": {
        "norm1": {"weight": torch_to_numpy(torch_attn.norm1.weight)},
        "fused_ff_qkv": {"kernel": torch_to_numpy(torch_attn.fused_ff_qkv.weight).T,
                         "bias": torch_to_numpy(torch_attn.fused_ff_qkv.bias)},
        "q_norm": {"scale": torch_to_numpy(torch_attn.q_norm.weight),
                   "bias": torch_to_numpy(torch_attn.q_norm.bias)},
        "k_norm": {"scale": torch_to_numpy(torch_attn.k_norm.weight),
                   "bias": torch_to_numpy(torch_attn.k_norm.bias)},
        "rotary_emb": {"freqs": torch_to_numpy(torch_attn.rotary_emb.freqs)},
        "attn_out": {"kernel": torch_to_numpy(torch_attn.attn_out.weight).T},
        "ff_out": {"kernel": torch_to_numpy(torch_attn.ff_out.weight).T,
                   "bias": torch_to_numpy(torch_attn.ff_out.bias)},
    }}
    jax_out, _ = jax_attn.apply(jax_params, jnp.array(x_np))

    return assert_close("FullAttention", torch_out, jax_out, atol=1e-3, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: AxialTimeAttention (temporal block)
# ══════════════════════════════════════════════════════════════════════════════

def test_axial_time_attention():
    print("\n" + "=" * 60)
    print("Test 5: AxialTimeAttention (temporal block)")
    print("=" * 60)

    hidden_dim, num_heads = 96, 4
    T, B, H, W, D = 3, 1, 2, 2, 2
    x_np = np.random.randn(T, B, hidden_dim, H, W, D).astype(np.float32)

    torch_time = TorchAxialTime(hidden_dim=hidden_dim, num_heads=num_heads, drop_path=0.0,
                                bias_type="rel", causal_in_time=False)
    torch_time.eval()

    with torch.no_grad():
        torch_out, _ = torch_time(torch.from_numpy(x_np))

    jax_time = JaxAxialTime(hidden_dim=hidden_dim, num_heads=num_heads,
                            bias_type="rel", causal_in_time=False)
    jax_params = {"params": {
        "norm1": {"weight": torch_to_numpy(torch_time.norm1.weight)},
        "input_head_weight": torch_to_numpy(torch_time.input_head.weight),
        "input_head_bias": torch_to_numpy(torch_time.input_head.bias),
        "output_head_weight": torch_to_numpy(torch_time.output_head.weight),
        "output_head_bias": torch_to_numpy(torch_time.output_head.bias),
        "qnorm": {"scale": torch_to_numpy(torch_time.qnorm.weight),
                  "bias": torch_to_numpy(torch_time.qnorm.bias)},
        "knorm": {"scale": torch_to_numpy(torch_time.knorm.weight),
                  "bias": torch_to_numpy(torch_time.knorm.bias)},
        "rel_pos_bias": {
            "relative_attention_bias": {"embedding": torch_to_numpy(torch_time.rel_pos_bias.relative_attention_bias.weight)},
        },
    }}
    jax_out, _ = jax_time.apply(jax_params, jnp.array(x_np))

    return assert_close("AxialTimeAttention", torch_out, jax_out, atol=1e-3, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {
        "RMSGroupNorm": test_rmsgroupnorm(),
        "Encoder": test_encoder(),
        "Decoder": test_decoder(),
        "FullAttention": test_full_attention(),
        "AxialTimeAttention": test_axial_time_attention(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} -- {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! JAX implementation matches PyTorch.")
    else:
        print("\nSome tests failed. See details above.")
        sys.exit(1)
