"""
Weight conversion utility: PyTorch -> JAX/Flax parameter mapping.

Maps every key in the PyTorch ``IsotropicModel`` state_dict to the
corresponding leaf in the Flax parameter tree, applying transpositions
where needed (e.g. ``nn.Linear.weight`` -> ``nn.Dense.kernel``).
"""

from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to numpy, handling GPU tensors."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint and extract the model state dict.

    Handles the Walrus checkpoint format: ``{'app': {'model': state_dict}}``.
    """
    assert torch is not None, "PyTorch is required to load checkpoints"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "app" in ckpt and "model" in ckpt["app"]:
            return ckpt["app"]["model"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def convert_pytorch_to_jax_params(
    pytorch_state_dict: Dict[str, Any],
    processor_blocks: int = 40,
    dim_keys: list = None,
) -> Dict[str, Any]:
    """
    Convert a PyTorch IsotropicModel state_dict to a Flax parameter dict.

    Mapping rules:

    - ``embed.{d}.*`` -> ``embed_{d}.*`` (SpaceBagAdaptiveDVstrideEncoder)
    - ``debed.{d}.*`` -> ``debed_{d}.*`` (AdaptiveDVstrideDecoder)
    - ``blocks.{i}.space_mixing.*`` -> ``blocks_{i}.space_mixing.*`` (FullAttention)
    - ``blocks.{i}.time_mixing.*`` -> ``blocks_{i}.time_mixing.*`` (AxialTimeAttention)
    - ``nn.Linear.weight`` -> ``.kernel`` (transposed)
    - ``nn.LayerNorm.weight`` -> ``.scale``
    - ``nn.Embedding.weight`` -> ``.embedding``
    - Conv weights are stored as-is (no transpose needed)

    Returns:
        ``{'params': nested_dict}`` suitable for ``model.apply()``.
    """
    if dim_keys is None:
        dim_keys = [2, 3]

    sd = {k: torch_to_numpy(v) for k, v in pytorch_state_dict.items()}
    params = {}

    def _set(path, value):
        keys = path.split(".")
        d = params
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    # ── encoder_dummy ──
    if "encoder_dummy" in sd:
        _set("encoder_dummy", sd["encoder_dummy"])

    # ── Encoder (embed.{d}) ──
    for d in dim_keys:
        enc_prefix = f"embed.{d}"
        jax_prefix = f"embed_{d}"
        if f"{enc_prefix}.proj1.weight" not in sd:
            continue
        _set(f"{jax_prefix}.proj1_weight", sd[f"{enc_prefix}.proj1.weight"])
        _set(f"{jax_prefix}.norm1.weight", sd[f"{enc_prefix}.norm1.weight"])
        _set(f"{jax_prefix}.proj2_weight", sd[f"{enc_prefix}.proj2.weight"])
        _set(f"{jax_prefix}.norm2.weight", sd[f"{enc_prefix}.norm2.weight"])

    # ── Decoder (debed.{d}) ──
    for d in dim_keys:
        dec_prefix = f"debed.{d}"
        jax_prefix = f"debed_{d}"
        if f"{dec_prefix}.proj1.weight" not in sd:
            continue
        _set(f"{jax_prefix}.proj1_weight", sd[f"{dec_prefix}.proj1.weight"])
        _set(f"{jax_prefix}.norm1.weight", sd[f"{dec_prefix}.norm1.weight"])
        _set(f"{jax_prefix}.proj2_weight", sd[f"{dec_prefix}.proj2.weight"])
        _set(f"{jax_prefix}.proj2_bias", sd[f"{dec_prefix}.proj2.bias"])

    # ── Processor blocks ──
    for i in range(processor_blocks):
        blk_prefix = f"blocks.{i}"
        jax_blk = f"blocks_{i}"

        # ── Space mixing (FullAttention) ──
        sm_prefix = f"{blk_prefix}.space_mixing"
        jax_sm = f"{jax_blk}.space_mixing"

        _set(f"{jax_sm}.norm1.weight", sd[f"{sm_prefix}.norm1.weight"])
        _set(f"{jax_sm}.fused_ff_qkv.kernel", sd[f"{sm_prefix}.fused_ff_qkv.weight"].T)
        _set(f"{jax_sm}.fused_ff_qkv.bias", sd[f"{sm_prefix}.fused_ff_qkv.bias"])
        _set(f"{jax_sm}.q_norm.scale", sd[f"{sm_prefix}.q_norm.weight"])
        _set(f"{jax_sm}.q_norm.bias", sd[f"{sm_prefix}.q_norm.bias"])
        _set(f"{jax_sm}.k_norm.scale", sd[f"{sm_prefix}.k_norm.weight"])
        _set(f"{jax_sm}.k_norm.bias", sd[f"{sm_prefix}.k_norm.bias"])

        rope_key = f"{sm_prefix}.rotary_emb.freqs"
        if rope_key in sd:
            _set(f"{jax_sm}.rotary_emb.freqs", sd[rope_key])

        _set(f"{jax_sm}.attn_out.kernel", sd[f"{sm_prefix}.attn_out.weight"].T)
        _set(f"{jax_sm}.ff_out.kernel", sd[f"{sm_prefix}.ff_out.weight"].T)
        _set(f"{jax_sm}.ff_out.bias", sd[f"{sm_prefix}.ff_out.bias"])

        # ── Time mixing (AxialTimeAttention) ──
        tm_prefix = f"{blk_prefix}.time_mixing"
        jax_tm = f"{jax_blk}.time_mixing"

        _set(f"{jax_tm}.norm1.weight", sd[f"{tm_prefix}.norm1.weight"])
        _set(f"{jax_tm}.input_head_weight", sd[f"{tm_prefix}.input_head.weight"])
        _set(f"{jax_tm}.input_head_bias", sd[f"{tm_prefix}.input_head.bias"])
        _set(f"{jax_tm}.output_head_weight", sd[f"{tm_prefix}.output_head.weight"])
        _set(f"{jax_tm}.output_head_bias", sd[f"{tm_prefix}.output_head.bias"])
        _set(f"{jax_tm}.qnorm.scale", sd[f"{tm_prefix}.qnorm.weight"])
        _set(f"{jax_tm}.qnorm.bias", sd[f"{tm_prefix}.qnorm.bias"])
        _set(f"{jax_tm}.knorm.scale", sd[f"{tm_prefix}.knorm.weight"])
        _set(f"{jax_tm}.knorm.bias", sd[f"{tm_prefix}.knorm.bias"])

        rel_key = f"{tm_prefix}.rel_pos_bias.relative_attention_bias.weight"
        if rel_key in sd:
            _set(
                f"{jax_tm}.rel_pos_bias.relative_attention_bias.embedding",
                sd[rel_key],
            )

        rot_key = f"{tm_prefix}.rotary_emb.inv_freq"
        if rot_key in sd:
            _set(f"{jax_tm}.rotary_emb.inv_freq", sd[rot_key])

    return {"params": params}
