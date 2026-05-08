"""Ulysses Sequence Parallelism: AllToAll-based attention + sequence split.

Optimized version:
- env-controlled all_to_all_single fast path
- fused QKV all_to_all option to reduce 3 collectives to 1
- V2A cross-attention AllGather for audio-quality parity
- HCCL-compatible fallback to list-based all_to_all
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_USE_A2A_SINGLE = os.getenv("LTX_UA2A_USE_A2A_SINGLE", "1") == "1"
_FUSE_QKV_A2A = os.getenv("LTX_UA2A_FUSE_QKV", "1") == "1"
_DISABLE_V2A_GATHER = os.getenv("LTX_UA2A_DISABLE_V2A_GATHER", "0") == "1"
_DISABLE_AUDIO_SPECIAL = os.getenv("LTX_UA2A_DISABLE_AUDIO_SPECIAL", "0") == "1"
_ENABLE_AUDIO_HP = os.getenv("LTX_AUDIO_HEAD_PARALLEL", "0") == "1"
_DISABLE_A2V = os.getenv("LTX_UA2A_DISABLE_A2V", "1") == "1"
_DISABLE_V2A = os.getenv("LTX_UA2A_DISABLE_V2A", "1") == "1"
_ENABLE_V2A_OVERLAP = os.getenv("LTX_V2A_OVERLAP_GATHER", "0") == "1"


def _npu_synchronize():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


# ---------------------------------------------------------------------------
# AllToAll (optimized with all_to_all_single fast path)
# ---------------------------------------------------------------------------

def _a2a_list_fallback(input, scatter_idx, gather_idx, group):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    shape = list(input.shape)
    assert shape[scatter_idx] % world_size == 0, (
        f"dim {scatter_idx} size {shape[scatter_idx]} not divisible by world_size {world_size}"
    )
    input_chunks = [c.contiguous() for c in input.chunk(world_size, dim=scatter_idx)]
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks, group=group)
    return torch.cat(output_chunks, dim=gather_idx)


def all_to_all_4d(input, scatter_idx, gather_idx, group):
    """AllToAll on a 4D tensor: scatter on one dim, gather on another."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input

    shape = list(input.shape)
    assert len(shape) == 4, f"expected 4D tensor, got {tuple(shape)}"
    assert shape[scatter_idx] % world_size == 0, (
        f"dim {scatter_idx} size {shape[scatter_idx]} not divisible by world_size {world_size}"
    )

    if not _USE_A2A_SINGLE:
        return _a2a_list_fallback(input, scatter_idx, gather_idx, group)

    try:
        remain = [i for i in range(4) if i not in (scatter_idx, gather_idx)]
        perm = [scatter_idx, gather_idx, *remain]
        x = input.permute(perm).contiguous()  # [S, G, R0, R1]
        s_dim, g_dim, r0, r1 = x.shape
        s_local = s_dim // world_size

        send = x.view(world_size, s_local, g_dim, r0, r1).reshape(world_size, -1).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)

        y = recv.view(world_size, s_local, g_dim, r0, r1)
        y = y.permute(1, 0, 2, 3, 4).contiguous().reshape(s_local, world_size * g_dim, r0, r1)

        inv = [0] * 4
        for i, a in enumerate(perm):
            inv[a] = i
        return y.permute(inv).contiguous()
    except Exception:
        return _a2a_list_fallback(input, scatter_idx, gather_idx, group)


# ---------------------------------------------------------------------------
# UlyssesAttention
# ---------------------------------------------------------------------------

class UlyssesAttention:
    """Wraps attention with pre/post AllToAll for correct full-sequence attention."""

    def __init__(self, original_fn, sp_group):
        self.original_fn = original_fn
        self.sp_group = sp_group
        self.world_size = dist.get_world_size(sp_group)

    def __call__(self, q, k, v, heads, mask=None):
        if self.world_size <= 1:
            return self.original_fn(q, k, v, heads, mask)

        b, local_t, hd = q.shape
        dim_head = hd // heads
        q4 = q.view(b, local_t, heads, dim_head)
        k4 = k.view(b, local_t, heads, dim_head)
        v4 = v.view(b, local_t, heads, dim_head)

        if _FUSE_QKV_A2A:
            qkv4 = torch.cat([q4, k4, v4], dim=0)
            qkv4 = all_to_all_4d(qkv4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            q4, k4, v4 = torch.chunk(qkv4, 3, dim=0)
        else:
            q4 = all_to_all_4d(q4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            k4 = all_to_all_4d(k4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            v4 = all_to_all_4d(v4, scatter_idx=2, gather_idx=1, group=self.sp_group)

        local_heads = heads // self.world_size
        gathered_t = q4.shape[1]

        out = self.original_fn(
            q4.reshape(b, gathered_t, local_heads * dim_head),
            k4.reshape(b, gathered_t, local_heads * dim_head),
            v4.reshape(b, gathered_t, local_heads * dim_head),
            local_heads, mask,
        )
        out4 = out.view(b, gathered_t, local_heads, dim_head)
        out4 = all_to_all_4d(out4, scatter_idx=1, gather_idx=2, group=self.sp_group)
        return out4.reshape(b, local_t, heads * dim_head).contiguous()


# ---------------------------------------------------------------------------
# AudioHeadParallelAttention
# ---------------------------------------------------------------------------

class AudioHeadParallelAttention:
    """Audio attention: full sequence per rank, split heads, all_gather output.

    Unlike Ulysses (which splits tokens by sequence), this keeps the full
    audio sequence on every rank and parallelizes across attention heads.
    Avoids audio token boundary artifacts at token16/78.
    """

    def __init__(self, original_fn, sp_group):
        self.original_fn = original_fn
        self.sp_group = sp_group
        self.world_size = dist.get_world_size(sp_group)

    def __call__(self, q, k, v, heads, mask=None):
        if self.world_size <= 1:
            return self.original_fn(q, k, v, heads, mask)

        b, t, hd = q.shape

        if heads % self.world_size != 0:
            raise ValueError(
                f"AudioHeadParallelAttention requires heads divisible by world_size, "
                f"got heads={heads}, world_size={self.world_size}"
            )

        dim_head = hd // heads
        local_heads = heads // self.world_size
        rank = dist.get_rank(self.sp_group)

        q4 = q.view(b, t, heads, dim_head)
        k4 = k.view(b, t, heads, dim_head)
        v4 = v.view(b, t, heads, dim_head)

        hs = rank * local_heads
        he = (rank + 1) * local_heads

        q_local = q4[:, :, hs:he, :].contiguous()
        k_local = k4[:, :, hs:he, :].contiguous()
        v_local = v4[:, :, hs:he, :].contiguous()

        out_local = self.original_fn(
            q_local.reshape(b, t, local_heads * dim_head),
            k_local.reshape(b, t, local_heads * dim_head),
            v_local.reshape(b, t, local_heads * dim_head),
            local_heads,
            mask,
        )

        out4_local = out_local.view(b, t, local_heads, dim_head).contiguous()

        gathered = [torch.empty_like(out4_local) for _ in range(self.world_size)]
        dist.all_gather(gathered, out4_local, group=self.sp_group)
        _npu_synchronize()

        out4 = torch.cat(gathered, dim=2).contiguous()
        return out4.reshape(b, t, heads * dim_head).contiguous()


# ---------------------------------------------------------------------------
# Attention injection
# ---------------------------------------------------------------------------

def _is_audio_self(name: str) -> bool:
    return "audio_attn1" in name.lower()

def _is_cross_av(name: str) -> bool:
    n = name.lower()
    return ("video_to_audio" in n or "audio_to_video" in n
            or "v2a" in n or "a2v" in n)

def _is_video_self(name: str) -> bool:
    n = name.lower()
    return "attn1" in n and "audio_attn1" not in n and "audio_attn2" not in n

def inject_ulysses_attention(model, sp_group):
    """
    Plan B:
    - video self-attn: Ulysses sequence parallel
    - audio_attn1: keep original full-head attention on every rank
    - audio_attn2 / A2V / V2A: skip for now
    """
    injected_video = 0
    injected_audio_hp = 0
    skipped_audio = 0
    skipped_cross = 0

    for name, module in model.named_modules():
        if not hasattr(module, "attention_function"):
            continue
        if not hasattr(module, "heads") or not hasattr(module, "dim_head"):
            continue

        inner = module.heads * module.dim_head

        if _is_audio_self(name):
            if _ENABLE_AUDIO_HP:
                module.attention_function = AudioHeadParallelAttention(
                    module.attention_function, sp_group)
                injected_audio_hp += 1
            else:
                skipped_audio += 1
            continue

        if _is_cross_av(name):
            skipped_cross += 1
            continue

        if _is_video_self(name) and inner == 4096:
            module.attention_function = UlyssesAttention(
                module.attention_function, sp_group)
            injected_video += 1

    logger.info(
        "inject_ulysses_attention: video=%d audio_hp=%d skipped_audio=%d skipped_cross=%d",
        injected_video, injected_audio_hp, skipped_audio, skipped_cross,
    )

# ---------------------------------------------------------------------------
# V2A cross-attention injection (audio-quality fix)
# ---------------------------------------------------------------------------

class _V2AAttentionGather:
    """AllGather video K/V across ranks before V2A cross-attention."""

    def __init__(self, original_fn, sp_group):
        self._orig = original_fn
        self._sp = sp_group
        self._w = dist.get_world_size(sp_group)

    def __call__(self, q, k, v, heads, mask=None):
        if self._w <= 1:
            return self._orig(q, k, v, heads, mask)
        kv = torch.stack([k, v], dim=0).contiguous()
        kv = _gather_tensor_along_seq_kv(kv, self._sp)
        k, v = kv[0], kv[1]
        return self._orig(q, k, v, heads, mask)


def _gather_tensor_along_seq_kv(tensor, group):
    """AllGather along sequence dim (dim=2) for stacked [2, B, T, D] tensor."""
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    _npu_synchronize()
    return torch.cat(gathered, dim=2).contiguous()


def inject_ulysses_v2a_cross_attention(model, sp_group):
    if _DISABLE_V2A_GATHER:
        logger.info("Skipping V2A context-gather hook because LTX_UA2A_DISABLE_V2A_GATHER=1")
        return 0
    """Gather video context on V2A module via forward_pre_hook.

    Uses register_forward_pre_hook to intercept the `context` kwarg
    before video_to_audio_attn.forward runs, AllGathering it across ranks.
    This avoids patching forward (FSDP conflicts) or attention_function
    (post-projection, doesn't fix A2V's influence).
    """
    world_size = dist.get_world_size(sp_group)
    if world_size <= 1:
        return 0
    blocks = getattr(model, "transformer_blocks", getattr(model, "blocks", []))
    n_patched = 0
    for block in blocks:
        if hasattr(block, "video_to_audio_attn"):
            v2a = block.video_to_audio_attn
            _install_v2a_context_gather(v2a, sp_group)
            n_patched += 1
    logger.info("Hooked %d V2A modules with context-gather pre-hook", n_patched)
    return n_patched


def _install_v2a_context_gather(v2a_module, sp_group):
    """Register pre-hook on V2A Attention module to AllGather context and k_pe."""
    world_size = dist.get_world_size(sp_group)

    def _gather_rope_hook(pe):
        if pe is None:
            return None
        cos, sin = pe
        if cos.ndim == 4:
            gathered_cos = [torch.zeros_like(cos) for _ in range(world_size)]
            gathered_sin = [torch.zeros_like(sin) for _ in range(world_size)]
            dist.all_gather(gathered_cos, cos.contiguous(), group=sp_group)
            dist.all_gather(gathered_sin, sin.contiguous(), group=sp_group)
            _npu_synchronize()
            return (torch.cat(gathered_cos, dim=2).contiguous(),
                    torch.cat(gathered_sin, dim=2).contiguous())
        return (_gather_tensor_along_seq(cos, sp_group),
                _gather_tensor_along_seq(sin, sp_group))

    def _pre_hook(module, args, kwargs):
        kwargs = dict(kwargs)  # mutable
        if getattr(module, "_ltx_v2a_skip_gather_once", False):
            module._ltx_v2a_skip_gather_once = False
            return args, kwargs
        if kwargs.get('context') is not None:
            kwargs['context'] = _gather_tensor_along_seq(kwargs['context'], sp_group)
        if kwargs.get('k_pe') is not None:
            kwargs['k_pe'] = _gather_rope_hook(kwargs['k_pe'])
        return args, kwargs

    v2a_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    v2a_module._ltx_v2a_sp_group = sp_group


class _V2AAttentionGather:
    """(Deprecated) AllGather video K/V — kept for backward compat.

    Replaced by _install_v2a_context_gather which gathers at the
    forward_pre_hook level (pre-projection), not at attention_function level.
    Actual fix now uses register_forward_pre_hook above.
    """
    pass


# ---------------------------------------------------------------------------
# Sequence parallel hooks
# ---------------------------------------------------------------------------

def _split_tensor_along_seq(tensor, rank, world_size):
    seq_len = tensor.shape[1]
    chunk_size = seq_len // world_size
    return tensor[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()


def _gather_tensor_along_seq(tensor, group):
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=1)


class _AsyncSeqGather:
    def __init__(self, tensor, group, dim):
        self._dim = dim
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            self._work = None
            self._parts = None
            self._result = tensor
        else:
            self._parts = [torch.zeros_like(tensor) for _ in range(world_size)]
            self._work = dist.all_gather(self._parts, tensor.contiguous(), group=group, async_op=True)
            self._result = None

    def wait(self):
        if self._result is not None:
            return self._result
        self._work.wait()
        self._result = torch.cat(self._parts, dim=self._dim).contiguous()
        return self._result


class _AsyncV2AGather:
    def __init__(self, module, context, k_pe, group):
        self._module = module
        self._context = _AsyncSeqGather(context, group, dim=1) if context is not None else None
        self._k_pe = None
        if k_pe is not None:
            cos, sin = k_pe
            dim = 2 if cos.ndim == 4 else 1
            self._k_pe = (_AsyncSeqGather(cos, group, dim=dim), _AsyncSeqGather(sin, group, dim=dim))

    def wait(self):
        context = self._context.wait() if self._context is not None else None
        k_pe = None
        if self._k_pe is not None:
            k_pe = (self._k_pe[0].wait(), self._k_pe[1].wait())
        self._module._ltx_v2a_skip_gather_once = True
        return context, k_pe


def start_v2a_context_gather(v2a_module, context, k_pe):
    if not _ENABLE_V2A_OVERLAP:
        return None
    group = getattr(v2a_module, "_ltx_v2a_sp_group", None)
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return None
    return _AsyncV2AGather(v2a_module, context, k_pe, group)


def _split_rope(pe, rank, world_size):
    if pe is None:
        return None
    cos, sin = pe
    if cos.ndim == 4:
        seq_len = cos.shape[2]
        chunk = seq_len // world_size
        start = rank * chunk
        end = start + chunk
        return (cos[:, :, start:end].contiguous(), sin[:, :, start:end].contiguous())
    return (_split_tensor_along_seq(cos, rank, world_size),
            _split_tensor_along_seq(sin, rank, world_size))


def _pad_rope(pe, pad_len):
    if pe is None or pad_len == 0:
        return pe
    cos, sin = pe
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad_len))
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad_len))
    return (cos, sin)


def _pad_seq_dim(t, pad_len):
    if t is None or pad_len == 0:
        return t
    if t.shape[1] <= 1:
        return t
    return torch.nn.functional.pad(t, (0, 0, 0, pad_len))


def _split_seq_dim(t, rank, world_size):
    if t is None:
        return t
    if t.shape[1] <= 1:
        return t
    return _split_tensor_along_seq(t, rank, world_size)


def _pad_transformer_args(args, pad_len):
    return replace(
        args,
        x=torch.nn.functional.pad(args.x, (0, 0, 0, pad_len)),
        positional_embeddings=_pad_rope(args.positional_embeddings, pad_len),
        cross_positional_embeddings=_pad_rope(args.cross_positional_embeddings, pad_len),
        timesteps=_pad_seq_dim(args.timesteps, pad_len),
        embedded_timestep=_pad_seq_dim(args.embedded_timestep, pad_len),
    )


def _split_transformer_args(args, rank, world_size):
    return replace(
        args,
        x=_split_tensor_along_seq(args.x, rank, world_size),
        positional_embeddings=_split_rope(args.positional_embeddings, rank, world_size),
        cross_positional_embeddings=_split_rope(args.cross_positional_embeddings, rank, world_size),
        timesteps=_split_seq_dim(args.timesteps, rank, world_size),
        embedded_timestep=_split_seq_dim(args.embedded_timestep, rank, world_size),
        self_attention_mask=None,
    )


def install_sequence_parallel_hooks(model, sp_group):
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)
    if world_size <= 1:
        return

    orig = model._process_transformer_blocks

    def parallel_process_blocks(video, audio, perturbations):
        original_vseq_len = 0
        if video is not None:
            original_vseq_len = video.x.shape[1]

        if video is not None:
            seq_len = video.x.shape[1]
            if seq_len % world_size != 0:
                pad_len = world_size - (seq_len % world_size)
                video = _pad_transformer_args(video, pad_len)
            video = _split_transformer_args(video, rank, world_size)

        globals()["GLOBAL_AUDIO_LEN"] = 0

        video_out, audio_out = orig(video, audio, perturbations)

        if video_out is not None:
            gathered_x = _gather_tensor_along_seq(video_out.x, sp_group)
            gathered_et = video_out.embedded_timestep
            if gathered_et is not None and gathered_et.shape[1] > 1:
                gathered_et = _gather_tensor_along_seq(gathered_et, sp_group)
            if original_vseq_len > 0 and gathered_x.shape[1] != original_vseq_len:
                gathered_x = gathered_x[:, :original_vseq_len]
                if gathered_et is not None and gathered_et.shape[1] > 1:
                    gathered_et = gathered_et[:, :original_vseq_len]
            video_out = replace(video_out, x=gathered_x, embedded_timestep=gathered_et)

        return video_out, audio_out

    model._process_transformer_blocks = parallel_process_blocks
    logger.info(
        "Installed sequence-parallel hooks on LTXModel (rank=%d, world_size=%d)",
        rank, world_size,
    )
