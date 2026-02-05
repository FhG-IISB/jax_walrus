"""
Walrus-JAX: JAX/Flax translation of the Walrus PDE foundation model.

A 1-to-1 translation of the Walrus model architecture from PyTorch to JAX/Flax,
maintaining exact weight compatibility for pretrained checkpoint conversion.
"""

from walrus_jax.model import IsotropicModel

__all__ = ["IsotropicModel"]
