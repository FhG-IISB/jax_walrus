"""
Convert pretrained Walrus PyTorch weights to JAX/Flax msgpack format.

Usage:
    python convert_pretrained.py --input walrus.pt --output walrus_jax.msgpack

Loads the PyTorch checkpoint, maps all parameters to the JAX/Flax param tree,
validates the mapping is complete, and saves as a msgpack file that can be
loaded with ``flax.serialization.from_bytes()``.
"""

import argparse
import os
import sys

import numpy as np

from walrus_jax.convert_weights import (
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
)


def count_params(d, prefix=""):
    """Count leaf arrays in a nested dict."""
    count = 0
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            count += count_params(v, path)
        else:
            count += 1
    return count


def flatten_params(d, prefix=""):
    """Flatten nested dict to list of (path, array) tuples."""
    result = []
    for k, v in sorted(d.items()):
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.extend(flatten_params(v, path))
        else:
            result.append((path, v))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert Walrus PyTorch weights to JAX msgpack"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output msgpack path (default: <input>_jax.msgpack)",
    )
    parser.add_argument(
        "--processor-blocks", type=int, default=40,
        help="Number of processor blocks in the model (default: 40)",
    )
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + "_jax.msgpack"

    # ── Load PyTorch checkpoint ──
    print(f"Loading PyTorch checkpoint: {args.input}")
    sd = load_pytorch_state_dict(args.input)
    print(f"  PyTorch state_dict: {len(sd)} parameters")

    # ── Convert to JAX param tree ──
    print("Converting to JAX parameter tree...")
    jax_params = convert_pytorch_to_jax_params(
        sd,
        processor_blocks=args.processor_blocks,
        dim_keys=[2, 3],
    )

    # ── Validate mapping ──
    jax_flat = flatten_params(jax_params["params"])
    n_jax = len(jax_flat)

    pt_keys_mapped = set()
    for d in [2, 3]:
        for suffix in ["proj1.weight", "norm1.weight", "proj2.weight", "norm2.weight"]:
            k = f"embed.{d}.{suffix}"
            if k in sd:
                pt_keys_mapped.add(k)
        for suffix in ["proj1.weight", "norm1.weight", "proj2.weight", "proj2.bias"]:
            k = f"debed.{d}.{suffix}"
            if k in sd:
                pt_keys_mapped.add(k)

    if "encoder_dummy" in sd:
        pt_keys_mapped.add("encoder_dummy")

    for i in range(args.processor_blocks):
        for k in sd:
            if k.startswith(f"blocks.{i}."):
                pt_keys_mapped.add(k)

    unmapped = set(sd.keys()) - pt_keys_mapped
    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} PyTorch keys not mapped:")
        for k in sorted(unmapped):
            print(f"    {k}: {tuple(sd[k].shape)}")

    print(f"\n  JAX params: {n_jax} leaf arrays")
    print(f"  PyTorch mapped: {len(pt_keys_mapped)}/{len(sd)} keys")

    total_elements = sum(np.prod(v.shape) for _, v in jax_flat)
    print(f"  Total parameters: {total_elements:,}")

    # ── Summary ──
    print("\n  Parameter groups:")
    groups = {}
    for path, arr in jax_flat:
        group = path.split(".")[0]
        if group not in groups:
            groups[group] = {"count": 0, "elements": 0}
        groups[group]["count"] += 1
        groups[group]["elements"] += int(np.prod(arr.shape))
    for g in sorted(groups):
        print(f"    {g}: {groups[g]['count']} arrays, {groups[g]['elements']:,} params")

    # ── Serialize to msgpack ──
    import jax.numpy as jnp
    from flax.serialization import to_bytes, from_bytes

    def to_jax_arrays(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = to_jax_arrays(v)
            else:
                result[k] = jnp.array(v)
        return result

    jax_params_jnp = {"params": to_jax_arrays(jax_params["params"])}

    print(f"\nSaving to: {args.output}")
    serialized = to_bytes(jax_params_jnp)
    with open(args.output, "wb") as f:
        f.write(serialized)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # ── Verify roundtrip ──
    print("Verifying saved file...")
    with open(args.output, "rb") as f:
        loaded_bytes = f.read()

    loaded = from_bytes(jax_params_jnp, loaded_bytes)
    loaded_flat = flatten_params(loaded["params"])
    assert len(loaded_flat) == n_jax, f"Mismatch: saved {n_jax}, loaded {len(loaded_flat)}"

    max_diff = 0.0
    for (orig_path, orig_arr), (load_path, load_arr) in zip(jax_flat, loaded_flat):
        assert orig_path == load_path, f"Path mismatch: {orig_path} vs {load_path}"
        diff = np.max(np.abs(np.array(orig_arr) - np.array(load_arr)))
        max_diff = max(max_diff, diff)

    print(f"  Verification: {n_jax} params loaded, max roundtrip diff = {max_diff:.2e}")
    print("\nDone!")


if __name__ == "__main__":
    main()
