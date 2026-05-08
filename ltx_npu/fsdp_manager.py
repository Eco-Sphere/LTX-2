"""FSDP wrapper for LTX-2 Transformer inference on Ascend NPU.

Shards the 22B Transformer across N ranks using PyTorch FSDP FULL_SHARD.
Each rank holds 1/N of the parameters permanently. During forward, each
BasicAVTransformerBlock AllGathers its full parameters, computes, then
frees the gathered copy — eliminating the gpu_model build/free/cleanup cycle.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _collect_fsdp_ignored_modules(model):
    """FSDP ignored_modules for audio branch counter-evidence testing.

    LTX_FSDP_REPLICATE_AUDIO=block0_attn2 : only block0.audio_attn2 replicated
    LTX_FSDP_REPLICATE_AUDIO=attn2        : all audio_attn2 replicated
    LTX_FSDP_REPLICATE_AUDIO=audio         : audio_attn1/audio_attn2/audio_ff replicated
    LTX_FSDP_REPLICATE_AUDIO=audio_a2v     : audio plus A2V cross-attn replicated
    LTX_FSDP_REPLICATE_AUDIO=audio_v2a     : audio plus V2A cross-attn replicated
    LTX_FSDP_REPLICATE_AUDIO=audio_av      : audio plus both A/V cross-attn replicated
    """
    mode = os.getenv("LTX_FSDP_REPLICATE_AUDIO", "0").strip().lower()
    if mode in ("", "0", "false", "off", "none"):
        return []

    ignored, seen, names = [], set(), []
    for name, module in model.named_modules():
        lname = name.lower()
        hit = False
        if mode == "block0_attn2":
            hit = lname.endswith("transformer_blocks.0.audio_attn2")
        elif mode == "attn2":
            hit = re.search(r"transformer_blocks\.\d+\.audio_attn2$", lname) is not None
        elif mode in ("audio", "audio_a2v", "audio_v2a", "audio_av"):
            hit = (
                re.search(r"transformer_blocks\.\d+\.audio_attn1$", lname) is not None
                or re.search(r"transformer_blocks\.\d+\.audio_attn2$", lname) is not None
                or re.search(r"transformer_blocks\.\d+\.audio_ff$", lname) is not None
                or (
                    mode in ("audio_a2v", "audio_av")
                    and re.search(r"transformer_blocks\.\d+\.audio_to_video_attn$", lname) is not None
                )
                or (
                    mode in ("audio_v2a", "audio_av")
                    and (
                        re.search(r"transformer_blocks\.\d+\.video_to_audio_attn$", lname) is not None
                    )
                )
            )
        else:
            raise ValueError(f"Unknown LTX_FSDP_REPLICATE_AUDIO={mode}")
        if hit and id(module) not in seen:
            ignored.append(module); seen.add(id(module)); names.append(name)

    logger.info("LTX_FSDP_REPLICATE_AUDIO=%s ignored_modules=%d", mode, len(ignored))
    for n in names[:10]:
        logger.info("  ignored: %s", n)
    return ignored


def shard_transformer(
    model: torch.nn.Module,
    device_id: torch.device,
    process_group: ProcessGroup | None = None,
) -> FSDP:
    """Wrap *model* with FSDP FULL_SHARD, sharding per BasicAVTransformerBlock.

    Args:
        model: The full X0Model (with velocity_model containing transformer_blocks).
               Must already have Ulysses SP hooks injected before calling this.
        device_id: Target NPU device (e.g. torch.device("npu:0")).
        process_group: Distributed process group for FSDP communication.
                       Defaults to the world group.

    Returns:
        FSDP-wrapped model with parameters sharded across ranks.
    """
    from torch.distributed.fsdp import MixedPrecision
    from ltx_core.model.transformer.model import BasicAVTransformerBlock

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BasicAVTransformerBlock},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_root_forward_inputs=False,
    )

    ignored_modules = _collect_fsdp_ignored_modules(model)
    for module in ignored_modules:
        module.to(device=device_id, dtype=torch.bfloat16)

    fsdp_model = FSDP(
        model.to(dtype=torch.bfloat16),
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        process_group=process_group,
        use_orig_params=True,
        mixed_precision=mp_policy,
        ignored_modules=ignored_modules if ignored_modules else None,
        forward_prefetch=os.getenv("LTX_FSDP_FORWARD_PREFETCH", "1") == "1",
        limit_all_gathers=os.getenv("LTX_FSDP_LIMIT_ALL_GATHERS", "1") == "1",
    )

    num_params = sum(p.numel() for p in fsdp_model.parameters())
    world_size = dist.get_world_size(process_group) if process_group else dist.get_world_size()
    shard_size_gib = num_params * 2 / (1024**3 * world_size)  # bf16
    logger.info(
        "FSDP wrapped transformer: %d params, %d ranks, ~%.1f GiB/rank (bf16)",
        num_params, world_size, shard_size_gib,
    )

    return fsdp_model
