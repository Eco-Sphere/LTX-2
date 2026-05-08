"""Unified device abstraction layer for NPU/GPU/CPU."""

from __future__ import annotations

import abc
import os
import subprocess

import torch
import torch.distributed as dist


class DeviceContext(abc.ABC):
    """Abstract base for device-specific runtime operations."""

    _instance: DeviceContext | None = None

    def __init__(self, local_rank: int = 0, rank: int = 0, world_size: int = 1):
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size

    @abc.abstractmethod
    def init_runtime(self) -> None: ...

    @abc.abstractmethod
    def init_distributed(self) -> None: ...

    @abc.abstractmethod
    def get_device(self) -> torch.device: ...

    @abc.abstractmethod
    def synchronize(self) -> None: ...

    @abc.abstractmethod
    def get_comm_backend(self) -> str: ...

    @property
    @abc.abstractmethod
    def device_type(self) -> str: ...

    def cleanup_processes(self) -> None:
        """Kill residual processes on local devices. Override per-backend."""

    # --- factory / singleton ---

    @classmethod
    def create(cls, local_rank: int | None = None, rank: int | None = None,
               world_size: int | None = None) -> DeviceContext:
        lr = local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", "0"))
        r = rank if rank is not None else int(os.environ.get("RANK", "0"))
        ws = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))

        if _npu_available():
            ctx = NPUDeviceContext(lr, r, ws)
        elif torch.cuda.is_available():
            ctx = CUDADeviceContext(lr, r, ws)
        else:
            ctx = CPUDeviceContext(lr, r, ws)

        ctx.init_runtime()
        cls._instance = ctx
        return ctx

    @classmethod
    def current(cls) -> DeviceContext:
        if cls._instance is None:
            cls._instance = cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def _npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except AttributeError:
        return False


class NPUDeviceContext(DeviceContext):
    @property
    def device_type(self) -> str:
        return "npu"

    def init_runtime(self) -> None:
        import ltx_npu  # noqa: F401 — triggers runtime config

    def init_distributed(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl", rank=self.rank, world_size=self.world_size)
        torch.npu.set_device(self.local_rank)

    def get_device(self) -> torch.device:
        return torch.device(f"npu:{self.local_rank}")

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def get_comm_backend(self) -> str:
        return "hccl"

    def cleanup_processes(self) -> None:
        try:
            subprocess.run(
                ["bash", os.path.join(os.path.dirname(__file__), "..", "examples", "scripts", "clean_npu.sh")],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass


class CUDADeviceContext(DeviceContext):
    """Stub — interface only; full implementation deferred to future iteration."""

    @property
    def device_type(self) -> str:
        return "cuda"

    def init_runtime(self) -> None:
        pass

    def init_distributed(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.local_rank)

    def get_device(self) -> torch.device:
        return torch.device(f"cuda:{self.local_rank}")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def get_comm_backend(self) -> str:
        return "nccl"


class CPUDeviceContext(DeviceContext):
    """Fallback for environments without accelerators (testing only)."""

    @property
    def device_type(self) -> str:
        return "cpu"

    def init_runtime(self) -> None:
        pass

    def init_distributed(self) -> None:
        if not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)

    def get_device(self) -> torch.device:
        return torch.device("cpu")

    def synchronize(self) -> None:
        pass

    def get_comm_backend(self) -> str:
        return "gloo"
