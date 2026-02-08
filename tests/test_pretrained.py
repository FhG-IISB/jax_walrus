"""
Pretrained weight equivalence test: verify that the JAX model with converted
weights produces the same output as the PyTorch model with original weights.

Loads both checkpoints, runs the core encoder->processor->decoder pipeline
on identical random inputs, and compares outputs.

The pretrained model uses SpaceBag encoding, so we use the SpaceBag path
on both sides.  PyTorch's ``_encoder_forward`` appends ``[2, 0, 1]`` to the
state_labels before passing as field_indices to the SpaceBag encoder (those
three extra dims correspond to embed.3 weights for dx/dy/dz).

Usage:
    python tests/test_pretrained.py

Requires:
    - PyTorch walrus.pt at WALRUS_PT (default: ../walrus/walrus.pt)
    - JAX walrus_jax.msgpack at WALRUS_JAX (default: ../walrus/walrus_jax.msgpack)
    - PyTorch walrus source on sys.path (set WALRUS_ROOT or default to ../walrus)
"""

import sys
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

# ── Resolve paths ──
WALRUS_ROOT = os.environ.get(
    "WALRUS_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "walrus"),
)
WALRUS_PT = os.environ.get(
    "WALRUS_PT",
    os.path.join(WALRUS_ROOT, "walrus.pt"),
)
WALRUS_JAX_MSGPACK = os.environ.get(
    "WALRUS_JAX",
    os.path.join(WALRUS_ROOT, "walrus_jax.msgpack"),
)
sys.path.insert(0, WALRUS_ROOT)

# ── Mock the_well ──
import types

_well_mod = types.ModuleType("the_well")
_data_mod = types.ModuleType("the_well.data")
_ds_mod = types.ModuleType("the_well.data.datasets")


class _BoundaryConditionEnum:
    _map = {
        "PERIODIC": type("BC", (), {"value": 2})(),
        "OPEN": type("BC", (), {"value": 0})(),
    }
    def __getitem__(self, key):
        return self._map[key]


_ds_mod.BoundaryCondition = _BoundaryConditionEnum()
_well_mod.data = _data_mod
_data_mod.datasets = _ds_mod
sys.modules.setdefault("the_well", _well_mod)
sys.modules.setdefault("the_well.data", _data_mod)
sys.modules.setdefault("the_well.data.datasets", _ds_mod)

# ── Imports ──
from walrus.models.encoders.vstride_encoder import (
    SpaceBagAdaptiveDVstrideEncoder as TorchSpaceBagEncoder,
)
from walrus.models.decoders.vstride_decoder import (
    AdaptiveDVstrideDecoder as TorchAdaptiveDecoder,
)
from walrus.models.spatiotemporal_blocks.space_time_split import (
    SpaceTimeSplitBlock as TorchBlock,
)
from walrus.models.spatial_blocks.full_attention import (
    FullAttention as TorchFullAttn,
)
from walrus.models.temporal_blocks.axial_time_attention import (
    AxialTimeAttention as TorchAxialTime,
)
from walrus.models.shared_utils.normalization import (
    RMSGroupNorm as TorchRMSGroupNorm,
)
from walrus_jax.model import IsotropicModel as JaxIsotropicModel
from walrus_jax.convert_weights import (
    load_pytorch_state_dict,
    torch_to_numpy,
)

from einops import rearrange

np.random.seed(0)
torch.manual_seed(0)


def assert_close(name, torch_out, jax_out, atol=5e-3, rtol=5e-3):
    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)
    max_diff = np.max(np.abs(t_np - j_np))
    mean_diff = np.mean(np.abs(t_np - j_np))
    print(f"  [{name}] shape: torch={t_np.shape} jax={j_np.shape}")
    print(f"           max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}")
    if not np.allclose(t_np, j_np, atol=atol, rtol=rtol):
        print(f"  [FAIL] MISMATCH (atol={atol}, rtol={rtol})")
        return False
    print(f"  [PASS] MATCH")
    return True


def main():
    # ═══════════════════════════════════════════════════════════════════
    # Pretrained model config (from extended_config.yaml)
    # ═══════════════════════════════════════════════════════════════════
    hidden_dim = 1408
    intermediate_dim = 352
    n_states = 67  # total input channels (from embed.3.proj1.weight shape)
    processor_blocks = 40
    num_heads = 16
    groups = 16
    bks_3d = ((8, 4), (8, 4), (8, 4))
    causal = True
    bias_type = "rel"
    extra_dims = 3  # SpaceBag appends 3 dims (dx, dy, dz)

    # We use a small 3D input: 32×32×32, stride (8,8,8)+(4,4,4) -> 1×1×1 latent
    T, B = 2, 1
    H = W = D = 32
    stride1 = (8, 8, 8)
    stride2 = (4, 4, 4)
    random_kernel = ((8, 4), (8, 4), (8, 4))
    bcs = [[2, 2], [2, 2], [2, 2]]  # periodic BCs

    # We pick 4 output fields; the SpaceBag encoder also needs field_indices
    # matching PyTorch's _encoder_forward, which appends [2, 0, 1] to
    # state_labels before passing to the SpaceBag encoder
    n_out_states = 4
    state_labels_np = np.arange(n_out_states, dtype=np.int64)
    # field_indices for the SpaceBag encoder (state_labels + [2,0,1])
    field_indices_np = np.concatenate(
        [state_labels_np, np.array([2, 0, 1], dtype=np.int64)]
    )

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Load PyTorch checkpoint
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Loading PyTorch checkpoint...")
    print("=" * 60)
    t0 = time.time()
    sd = load_pytorch_state_dict(WALRUS_PT)
    print(f"  Loaded {len(sd)} parameters in {time.time() - t0:.1f}s")

    # Verify key shapes
    assert sd["embed.3.proj1.weight"].shape == (intermediate_dim, n_states, 8, 8, 8)
    assert sd["embed.3.proj2.weight"].shape == (hidden_dim, intermediate_dim, 4, 4, 4)
    encoder_dummy_val = sd["encoder_dummy"].numpy()
    print(f"  encoder_dummy = {encoder_dummy_val}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Build PyTorch components and load weights
    # ═══════════════════════════════════════════════════════════════════
    print("\nBuilding PyTorch model components...")

    # The pretrained model uses SpaceBag encoder
    torch_enc = TorchSpaceBagEncoder(
        kernel_scales_seq=((4, 4),),
        base_kernel_size3d=bks_3d,
        input_dim=n_states, inner_dim=intermediate_dim, output_dim=hidden_dim,
        spatial_dims=3, groups=groups, extra_dims=extra_dims,
        variable_downsample=True, variable_deterministic_ds=True, learned_pad=True,
        norm_layer=TorchRMSGroupNorm, activation=nn.SiLU,
    )
    torch_enc.eval()
    enc_sd = {
        "proj1.weight": sd["embed.3.proj1.weight"],
        "norm1.weight": sd["embed.3.norm1.weight"],
        "proj2.weight": sd["embed.3.proj2.weight"],
        "norm2.weight": sd["embed.3.norm2.weight"],
    }
    torch_enc.load_state_dict(enc_sd)

    torch_dec = TorchAdaptiveDecoder(
        base_kernel_size3d=bks_3d,
        input_dim=hidden_dim, inner_dim=intermediate_dim, output_dim=n_states,
        spatial_dims=3, groups=groups, learned_pad=True,
        norm_layer=TorchRMSGroupNorm, activation=nn.SiLU,
    )
    torch_dec.eval()
    dec_sd = {
        "proj1.weight": sd["debed.3.proj1.weight"],
        "norm1.weight": sd["debed.3.norm1.weight"],
        "proj2.weight": sd["debed.3.proj2.weight"],
        "proj2.bias": sd["debed.3.proj2.bias"],
    }
    torch_dec.load_state_dict(dec_sd)

    torch_blocks = nn.ModuleList()
    for i in range(processor_blocks):
        blk = TorchBlock(
            space_mixing=partial(TorchFullAttn, num_heads=num_heads, mlp_dim=None),
            time_mixing=partial(TorchAxialTime, num_heads=num_heads, bias_type=bias_type),
            channel_mixing=partial(nn.Identity),
            hidden_dim=hidden_dim, drop_path=0.0,
            causal_in_time=causal, norm_layer=TorchRMSGroupNorm,
        )
        blk.eval()

        blk_sd = {}
        prefix = f"blocks.{i}"
        for k, v in sd.items():
            if k.startswith(prefix + "."):
                blk_sd[k[len(prefix) + 1:]] = v
        blk.load_state_dict(blk_sd)
        torch_blocks.append(blk)

    print(f"  Built SpaceBag encoder + {processor_blocks} blocks + decoder")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Load JAX checkpoint
    # ═══════════════════════════════════════════════════════════════════
    print("\nLoading JAX checkpoint...")
    t0 = time.time()
    from flax.serialization import from_bytes

    # Create a target structure for deserialization
    from walrus_jax.convert_weights import convert_pytorch_to_jax_params
    target = convert_pytorch_to_jax_params(sd, processor_blocks=40, dim_keys=[2, 3])
    # Convert to jax arrays for target structure
    def to_jax_arrays(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = to_jax_arrays(v)
            else:
                result[k] = jnp.array(v)
        return result

    target_jnp = {"params": to_jax_arrays(target["params"])}

    with open(WALRUS_JAX_MSGPACK, "rb") as f:
        jax_params = from_bytes(target_jnp, f.read())
    print(f"  Loaded JAX params in {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Sanity check — compare a few weights directly
    # ═══════════════════════════════════════════════════════════════════
    print("\nSanity checking weight equivalence...")
    checks = [
        ("encoder_dummy", sd["encoder_dummy"].numpy(), np.array(jax_params["params"]["encoder_dummy"])),
        ("embed_3.proj1_weight", sd["embed.3.proj1.weight"].numpy(), np.array(jax_params["params"]["embed_3"]["proj1_weight"])),
        ("blocks_0.space_mixing.fused_ff_qkv.kernel", sd["blocks.0.space_mixing.fused_ff_qkv.weight"].numpy().T, np.array(jax_params["params"]["blocks_0"]["space_mixing"]["fused_ff_qkv"]["kernel"])),
        ("blocks_39.time_mixing.norm1.weight", sd["blocks.39.time_mixing.norm1.weight"].numpy(), np.array(jax_params["params"]["blocks_39"]["time_mixing"]["norm1"]["weight"])),
    ]
    all_weights_match = True
    for name, pt_w, jax_w in checks:
        diff = np.max(np.abs(pt_w - jax_w))
        status = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"  {status} {name}: max_diff={diff:.2e}")
        if diff >= 1e-6:
            all_weights_match = False

    if not all_weights_match:
        print("\n  [FAIL] Weight mismatch detected!")
        sys.exit(1)
    print("  All sampled weights match!")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Create input
    # ═══════════════════════════════════════════════════════════════════
    # Input: only the channels selected by field_indices
    n_input_ch = len(field_indices_np)  # = n_out_states + extra_dims = 7
    x_np = np.random.randn(T, B, n_input_ch, H, W, D).astype(np.float32) * 0.01

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: PyTorch forward (manual pipeline, no jitter)
    # ═══════════════════════════════════════════════════════════════════
    print("\nRunning PyTorch forward pass...")
    t0 = time.time()
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)

        # Multiply by encoder_dummy (matching PyTorch _encoder_forward)
        x_pt = x_pt * torch.from_numpy(encoder_dummy_val)

        # SpaceBag encode: pass field_indices
        x_pt, _ = torch_enc(
            x_pt, torch.from_numpy(field_indices_np),
            random_kernel=random_kernel,
        )
        print(f"  Encoded: {x_pt.shape} in {time.time() - t0:.1f}s")

        # Process through all blocks
        t1 = time.time()
        for ii, blk in enumerate(torch_blocks):
            x_pt, _ = blk(x_pt, (bcs,), return_att=False)
            if (ii + 1) % 10 == 0:
                print(f"  Block {ii + 1}/{processor_blocks} done ({time.time() - t1:.1f}s)")

        # Causal: keep all timesteps
        if not causal:
            x_pt = x_pt[-1:]
        T_out = x_pt.shape[0]

        # Decode
        t2 = time.time()
        torch_out = torch_dec(
            x_pt,
            torch.from_numpy(state_labels_np),
            bcs,
            stage_info={"random_kernel": random_kernel},
        )
        print(f"  Decoded: {torch_out.shape} in {time.time() - t2:.1f}s")

    pt_time = time.time() - t0
    print(f"  PyTorch total: {pt_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: JAX forward
    # ═══════════════════════════════════════════════════════════════════
    print("\nRunning JAX forward pass...")
    t0 = time.time()

    jax_model = JaxIsotropicModel(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        n_states=n_states,
        processor_blocks=processor_blocks,
        groups=groups,
        num_heads=num_heads,
        mlp_dim=0,
        max_d=3,
        causal_in_time=causal,
        drop_path=0.0,
        bias_type=bias_type,
        base_kernel_size=bks_3d,
        use_spacebag=True,
        use_silu=True,
        include_d=(2, 3),
        encoder_groups=groups,
        learned_pad=False,
        jitter_patches=False,
    )

    # ── Print model parameter tree ──
    import flax.linen as fnn
    def _print_param_tree(d, prefix=""):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                print(f"{prefix}{k}/")
                _print_param_tree(v, prefix + "  ")
            else:
                print(f"{prefix}{k}: {v.shape} ({v.dtype})")
    print("\n" + "=" * 60)
    print("JAX IsotropicModel Parameter Tree")
    print("=" * 60)
    _print_param_tree(jax_params["params"])
    total_params = sum(int(np.prod(v.shape)) for v in jax.tree.leaves(jax_params["params"]))
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {total_params * 4 / 1e6:.1f} MB (float32)")
    print("=" * 60 + "\n")

    jax_out = jax_model.apply(
        jax_params,
        jnp.array(x_np),
        jnp.array(state_labels_np),
        bcs,
        stride1=stride1,
        stride2=stride2,
        field_indices=jnp.array(field_indices_np),
        dim_key=3,
    )

    jax_time = time.time() - t0
    print(f"  JAX output: {jax_out.shape} in {jax_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 8: Compare
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PRETRAINED WEIGHT COMPARISON")
    print("=" * 60)
    passed = assert_close(
        "Pretrained FullModel", torch_out, jax_out, atol=5e-2, rtol=5e-2
    )

    if passed:
        print("\n✓ Pretrained JAX model matches PyTorch model!")
    else:
        print("\n✗ Pretrained models diverge. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
