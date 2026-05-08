"""VAE spatial patch parallel: distributes VAE decode across NPUs by H×W slicing.

Each rank processes a spatial patch of the latent. Convolution boundary padding
is replaced with real data from neighboring ranks via P2P communication.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _all_gather_patches(local_x: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Gather same-shaped local patches with the fastest available collective."""
    if world_size <= 1:
        return [local_x]

    try:
        if hasattr(dist, "all_gather_into_tensor"):
            out = torch.empty(
                (world_size, *local_x.shape),
                dtype=local_x.dtype,
                device=local_x.device,
            )
            dist.all_gather_into_tensor(out, local_x.contiguous())
            return list(out.unbind(0))
    except Exception:
        pass

    gathered = [torch.zeros_like(local_x) for _ in range(world_size)]
    dist.all_gather(gathered, local_x.contiguous())
    return gathered


def _exchange_peer_tensors(send_to: int | None, send_tensor: torch.Tensor | None, recv_from: int | None, recv_tensor: torch.Tensor | None) -> None:
    """Non-blocking peer exchange to avoid serialized send/recv handshakes."""
    ops = []
    if recv_from is not None and recv_tensor is not None:
        ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_from))
    if send_to is not None and send_tensor is not None:
        ops.append(dist.P2POp(dist.isend, send_tensor, send_to))
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()



def _compute_grid(world_size: int) -> tuple[int, int]:
    """Compute (h_split, w_split) grid for a given world_size."""
    import os

    if world_size <= 1:
        return 1, 1

    grid_env = os.getenv("LTX_VAE_GRID", "").strip().lower()
    if grid_env:
        if "x" not in grid_env:
            raise ValueError(f"LTX_VAE_GRID must look like HxW, got: {grid_env}")
        h, w = [int(x) for x in grid_env.split("x", 1)]
        if h <= 0 or w <= 0 or h * w != world_size:
            raise ValueError(
                f"LTX_VAE_GRID={grid_env} invalid for world_size={world_size}; need h*w == world_size"
            )
        return h, w

    h_env = os.getenv("LTX_VAE_H_SPLIT", "").strip()
    w_env = os.getenv("LTX_VAE_W_SPLIT", "").strip()
    if h_env and w_env:
        h = int(h_env)
        w = int(w_env)
        if h <= 0 or w <= 0 or h * w != world_size:
            raise ValueError(
                f"LTX_VAE_H_SPLIT={h}, LTX_VAE_W_SPLIT={w} invalid for world_size={world_size}; need h*w == world_size"
            )
        return h, w

    # default heuristic: keep current behavior
    w = int(math.sqrt(world_size))
    while world_size % w != 0:
        w -= 1
    h = world_size // w
    return h, w

class VAEParallelContext:
    """Manages spatial patch distribution for parallel VAE decoding."""

    def __init__(self, world_size: int, rank: int, device: torch.device):
        self.world_size = world_size
        self.rank = rank
        self.device = device

        self.h_split, self.w_split = _compute_grid(world_size)
        self.h_rank = rank // self.w_split
        self.w_rank = rank % self.w_split

        self._row_group = None
        self._col_group = None

        if world_size > 1 and dist.is_initialized():
            # row group: ranks in the same row (same h_rank)
            for h in range(self.h_split):
                ranks = [h * self.w_split + w for w in range(self.w_split)]
                g = dist.new_group(ranks)
                if h == self.h_rank:
                    self._row_group = g

            # col group: ranks in the same column (same w_rank)
            for w in range(self.w_split):
                ranks = [h * self.w_split + w for h in range(self.h_split)]
                g = dist.new_group(ranks)
                if w == self.w_rank:
                    self._col_group = g

    def patch(self, x: torch.Tensor) -> torch.Tensor:
        """Split a global (B, C, T, H, W) tensor into the local spatial patch."""
        if self.world_size <= 1:
            return x
        B, C, T, H, W = x.shape
        h_chunk = H // self.h_split
        w_chunk = W // self.w_split
        h_start = self.h_rank * h_chunk
        w_start = self.w_rank * w_chunk
        return x[:, :, :, h_start:h_start + h_chunk, w_start:w_start + w_chunk].contiguous()

    def dispatch(self, local_x: torch.Tensor) -> torch.Tensor:
        """Gather local patches from all ranks into a full tensor via all_gather."""
        if self.world_size <= 1:
            return local_x

        gathered = _all_gather_patches(local_x, self.world_size)

        B, C, T, local_H, local_W = local_x.shape
        full_H = local_H * self.h_split
        full_W = local_W * self.w_split
        result = torch.zeros(B, C, T, full_H, full_W, dtype=local_x.dtype, device=local_x.device)

        for r, patch in enumerate(gathered):
            h_idx = r // self.w_split
            w_idx = r % self.w_split
            h_start = h_idx * local_H
            w_start = w_idx * local_W
            result[:, :, :, h_start:h_start + local_H, w_start:w_start + local_W] = patch

        return result

    def exchange_rows(self, data: torch.Tensor, pad_h: int) -> torch.Tensor:
        """Exchange boundary rows with vertical neighbors (top/bottom)."""
        if self.world_size <= 1 or pad_h == 0:
            return F.pad(data, (0, 0, pad_h, pad_h, 0, 0), mode="constant", value=0)

        B, C, T, H, W = data.shape
        top_pad = torch.zeros(B, C, T, pad_h, W, dtype=data.dtype, device=data.device)
        bottom_pad = torch.zeros(B, C, T, pad_h, W, dtype=data.dtype, device=data.device)

        top_neighbor = self.h_rank - 1
        bottom_neighbor = self.h_rank + 1

        my_top_rows = data[:, :, :, :pad_h, :].contiguous()
        my_bottom_rows = data[:, :, :, -pad_h:, :].contiguous()

        if self.h_split > 1:
            top_src = top_neighbor * self.w_split + self.w_rank if top_neighbor >= 0 else None
            bottom_dst = bottom_neighbor * self.w_split + self.w_rank if bottom_neighbor < self.h_split else None
            _exchange_peer_tensors(bottom_dst, my_bottom_rows, top_src, top_pad)

            top_dst = top_neighbor * self.w_split + self.w_rank if top_neighbor >= 0 else None
            bottom_src = bottom_neighbor * self.w_split + self.w_rank if bottom_neighbor < self.h_split else None
            _exchange_peer_tensors(top_dst, my_top_rows, bottom_src, bottom_pad)

        return torch.cat([top_pad, data, bottom_pad], dim=3)

    def exchange_columns(self, data: torch.Tensor, pad_w: int) -> torch.Tensor:
        """Exchange boundary columns with horizontal neighbors (left/right)."""
        if self.world_size <= 1 or pad_w == 0:
            return F.pad(data, (pad_w, pad_w, 0, 0, 0, 0), mode="constant", value=0)

        B, C, T, H, W = data.shape
        left_pad = torch.zeros(B, C, T, H, pad_w, dtype=data.dtype, device=data.device)
        right_pad = torch.zeros(B, C, T, H, pad_w, dtype=data.dtype, device=data.device)

        left_neighbor = self.w_rank - 1
        right_neighbor = self.w_rank + 1

        my_left_cols = data[:, :, :, :, :pad_w].contiguous()
        my_right_cols = data[:, :, :, :, -pad_w:].contiguous()

        if self.w_split > 1:
            right_dst = self.h_rank * self.w_split + right_neighbor if right_neighbor < self.w_split else None
            left_src = self.h_rank * self.w_split + left_neighbor if left_neighbor >= 0 else None
            _exchange_peer_tensors(right_dst, my_right_cols, left_src, left_pad)

            left_dst = self.h_rank * self.w_split + left_neighbor if left_neighbor >= 0 else None
            right_src = self.h_rank * self.w_split + right_neighbor if right_neighbor < self.w_split else None
            _exchange_peer_tensors(left_dst, my_left_cols, right_src, right_pad)

        return torch.cat([left_pad, data, right_pad], dim=4)


@contextmanager
def vae_parallel_decode(decoder, vae_ctx: VAEParallelContext):
    """Context manager that monkey-patches F.conv3d and F.pad for distributed VAE decode.

    Usage:
        with vae_parallel_decode(decoder, vae_ctx):
            result = decoder.decode_video(latent, tiling_config, generator)
    """
    if vae_ctx.world_size <= 1:
        yield
        return

    original_conv3d = F.conv3d
    original_pad = F.pad
    original_forward = decoder.forward

    def wrapped_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(padding, (list, tuple)) and len(padding) == 3:
            _, pad_h, pad_w = padding
            if pad_h > 0 or pad_w > 0:
                input = vae_ctx.exchange_rows(input, pad_h)
                input = vae_ctx.exchange_columns(input, pad_w)
                padding = (padding[0], 0, 0)
        return original_conv3d(input, weight, bias, stride, padding, dilation, groups)

    def wrapped_pad(input, pad, mode="constant", value=0.0):
        if len(pad) == 6:
            w_left, w_right, h_top, h_bottom, t_front, t_back = pad
            if (h_top > 0 or h_bottom > 0 or w_left > 0 or w_right > 0):
                # Only apply H/W padding on global edges
                if vae_ctx.h_rank > 0:
                    h_top = 0
                if vae_ctx.h_rank < vae_ctx.h_split - 1:
                    h_bottom = 0
                if vae_ctx.w_rank > 0:
                    w_left = 0
                if vae_ctx.w_rank < vae_ctx.w_split - 1:
                    w_right = 0
            pad = (w_left, w_right, h_top, h_bottom, t_front, t_back)
        return original_pad(input, pad, mode, value)

    def wrapped_forward(sample, timestep=None, generator=None):
        local_sample = vae_ctx.patch(sample)
        result = original_forward(local_sample, timestep, generator)
        return vae_ctx.dispatch(result)

    F.conv3d = wrapped_conv3d
    F.pad = wrapped_pad
    decoder.forward = wrapped_forward

    try:
        yield
    finally:
        F.conv3d = original_conv3d
        F.pad = original_pad
        decoder.forward = original_forward
