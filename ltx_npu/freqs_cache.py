"""RoPE frequency caching for the DiT sampling loop.

Within a single generate() call, the video resolution is fixed, so the
RoPE frequencies computed by `precompute_freqs_cis` are identical across
all denoising steps. This module caches the result on the preprocessor
instance after the first computation and returns it directly on subsequent
calls, avoiding 12 redundant calculations per generate (8 Stage-1 + 4 Stage-2).

Usage:
    install_freqs_cache(preprocessor)   # before sampling loop
    clear_freqs_cache(preprocessor)     # after generate returns
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CACHE_ATTR = "_freqs_cache"
_ORIGINAL_FN_ATTR = "_original_prepare_positional_embeddings"


def install_freqs_cache(preprocessor: Any) -> None:
    """Monkey-patch _prepare_positional_embeddings to cache (cos, sin) results.

    The cache key includes all arguments that affect RoPE values.  In
    multimodal models video/audio preprocessors may share shapes while using
    different position grids or max_pos/head settings.
    """
    if hasattr(preprocessor, _ORIGINAL_FN_ATTR):
        return

    original_fn = preprocessor._prepare_positional_embeddings

    setattr(preprocessor, _ORIGINAL_FN_ATTR, original_fn)
    setattr(preprocessor, _CACHE_ATTR, None)

    def _cached_prepare_positional_embeddings(
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.float()
        cache = getattr(preprocessor, _CACHE_ATTR, None)
        cache_key = (
            tuple(positions.shape),
            str(positions.device),
            str(positions.dtype),
            float(positions.detach().float().sum().cpu().item()),
            float(positions.detach().float().abs().sum().cpu().item()),
            inner_dim,
            tuple(max_pos),
            bool(use_middle_indices_grid),
            num_attention_heads,
            str(x_dtype),
            preprocessor.rope_type,
        )

        if cache is not None and cache[0] == cache_key:
            return cache[1]

        result = original_fn(
            positions=positions,
            inner_dim=inner_dim,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            x_dtype=x_dtype,
        )

        setattr(preprocessor, _CACHE_ATTR, (cache_key, result))
        logger.debug("Cached RoPE freqs for key %s", cache_key)
        return result

    import types
    preprocessor._prepare_positional_embeddings = types.MethodType(
        lambda self, **kwargs: _cached_prepare_positional_embeddings(**kwargs),
        preprocessor,
    )
    logger.info("Installed RoPE freqs cache on %s", type(preprocessor).__name__)


def clear_freqs_cache(preprocessor: Any) -> None:
    """Clear cached RoPE frequencies and restore original method."""
    if hasattr(preprocessor, _CACHE_ATTR):
        setattr(preprocessor, _CACHE_ATTR, None)
        logger.debug("Cleared RoPE freqs cache")


def install_freqs_cache_on_model(model: torch.nn.Module) -> int:
    """Find all TransformerArgsPreprocessor instances and install caching.

    Returns the number of preprocessors patched.
    """
    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, "_prepare_positional_embeddings") and hasattr(module, "rope_type"):
            install_freqs_cache(module)
            patched += 1
    return patched


def clear_freqs_cache_on_model(model: torch.nn.Module) -> None:
    """Clear all RoPE frequency caches on the model."""
    for name, module in model.named_modules():
        if hasattr(module, _CACHE_ATTR):
            clear_freqs_cache(module)
