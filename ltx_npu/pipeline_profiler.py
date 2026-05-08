"""PipelineProfiler: fine-grained timing for LTX Pipeline operations.

Provides two capabilities:
1. PipelineProfiler — records per-operation timing with device synchronization,
   outputs a table with name / elapsed / percentage.
2. install_hooks() — monkey-patches gpu_model and cleanup_memory to automatically
   record model lifecycle timing into a profiler instance.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltx_npu.device_context import DeviceContext


class PipelineProfiler:
    """Records per-operation timing with device synchronization.

    Usage::

        profiler = PipelineProfiler(ctx)
        with profiler.record("encode_prompt"):
            encode(prompt)
        print(profiler.report())
    """

    def __init__(self, ctx: DeviceContext | None = None):
        self._ctx = ctx
        self._records: list[tuple[str, float]] = []
        self._aggregate_records: dict[str, tuple[float, int]] = {}
        self._model_counts: dict[str, int] = {}
        self._last_gpu_model_exit: float | None = None
        self._gpu_model_depth: int = 0

    def sync(self) -> None:
        """Synchronize the device to ensure accurate timing boundaries."""
        if self._ctx is not None:
            self._ctx.synchronize()

    @contextmanager
    def record(self, name: str):
        """Context manager that records elapsed wall time of the enclosed block."""
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        self._records.append((name, time.perf_counter() - t0))

    def add(self, name: str, elapsed: float) -> None:
        """Add a pre-measured timing entry."""
        self._records.append((name, elapsed))

    def add_aggregate(self, name: str, elapsed: float) -> None:
        """Accumulate timing under a shared label."""
        total, count = self._aggregate_records.get(name, (0.0, 0))
        self._aggregate_records[name] = (total + elapsed, count + 1)

    def next_label(self, base: str) -> str:
        """Return a unique label: 'Foo' first time, 'Foo#2' second time, etc."""
        self._model_counts[base] = self._model_counts.get(base, 0) + 1
        n = self._model_counts[base]
        return base if n == 1 else f"{base}#{n}"

    def total(self) -> float:
        return sum(t for _, t in self._records)

    def report(self) -> str:
        if not self._records:
            if not self._aggregate_records:
                return "[PipelineProfiler] No records."

        total = self.total()
        name_w = max(max(len(n) for n, _ in self._records), len("Operation")) + 2
        time_w = 12
        pct_w = 8

        def row(name: str, elapsed: float, pct_str: str) -> str:
            return f"│ {name:<{name_w - 2}} │{elapsed:>{time_w - 2}.3f}  │{pct_str:>{pct_w - 2}}  │"

        lines = [
            "\n[PipelineProfiler] Detailed Timing Report",
            f"┌{'─' * name_w}┬{'─' * time_w}┬{'─' * pct_w}┐",
            f"│{'Operation'.center(name_w)}│{'Time (s)'.center(time_w)}│{'%'.center(pct_w)}│",
            f"├{'─' * name_w}┼{'─' * time_w}┼{'─' * pct_w}┤",
        ]
        for name, elapsed in self._records:
            pct = f"{elapsed / total * 100:.1f}" if total > 0 else "0.0"
            lines.append(row(name, elapsed, pct))
        if self._aggregate_records:
            lines.append(f"├{'─' * name_w}┼{'─' * time_w}┼{'─' * pct_w}┤")
            lines.append(row("Aggregates", 0.0, ""))
            agg_total = sum(total for total, _ in self._aggregate_records.values())
            for name, (agg_elapsed, count) in sorted(
                self._aggregate_records.items(), key=lambda item: item[1][0], reverse=True
            ):
                avg = agg_elapsed / max(1, count)
                pct = f"{agg_elapsed / agg_total * 100:.1f}" if agg_total > 0 else "0.0"
                lines.append(f"│ {name:<{name_w - 2}} │{agg_elapsed:>{time_w - 2}.3f}  │{pct:>{pct_w - 2}}  │")
                lines.append(f"│ {'avg='+format(avg, '.3f')+', n='+str(count):<{name_w - 2}} │{'':>{time_w - 2}}  │{'':>{pct_w - 2}}  │")
        lines.append(f"├{'─' * name_w}┼{'─' * time_w}┼{'─' * pct_w}┤")
        lines.append(row("TOTAL", total, "100.0"))
        lines.append(f"└{'─' * name_w}┴{'─' * time_w}┴{'─' * pct_w}┘")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all records and counters for a fresh profiling run."""
        self._records.clear()
        self._aggregate_records.clear()
        self._model_counts.clear()
        self._last_gpu_model_exit = time.perf_counter()
        self._gpu_model_depth = 0


def install_hooks(profiler: PipelineProfiler) -> None:
    """Monkey-patch gpu_model and cleanup_memory to record timing into *profiler*.

    Patches are applied to both the defining modules and the blocks module
    (which imports them at module level — Python import binding requires
    patching both references).
    """
    from contextlib import contextmanager as _cm

    import ltx_pipelines.utils.blocks as blocks_mod
    import ltx_pipelines.utils.gpu_model as gpu_model_mod
    import ltx_pipelines.utils.helpers as helpers_mod

    original_cleanup = helpers_mod.cleanup_memory

    @_cm
    def profiled_gpu_model(model):
        label = profiler.next_label(type(model).__name__)
        profiler.sync()
        t_enter = time.perf_counter()

        if profiler._gpu_model_depth == 0 and profiler._last_gpu_model_exit is not None:
            gap = t_enter - profiler._last_gpu_model_exit
            if gap > 0.001:
                profiler.add(f"build {label}", gap)

        profiler._gpu_model_depth += 1
        try:
            yield model
        finally:
            profiler._gpu_model_depth -= 1
            # sync() replaces the original torch.cuda.synchronize() before to("meta")
            profiler.sync()
            profiler.add(f"use {label}", time.perf_counter() - t_enter)

            t1 = time.perf_counter()
            model.to("meta")
            original_cleanup()
            t_free_end = time.perf_counter()
            if profiler._gpu_model_depth == 0:
                profiler._last_gpu_model_exit = t_free_end
            profiler.add(f"free {label}", t_free_end - t1)

    def profiled_cleanup():
        profiler.sync()
        t0 = time.perf_counter()
        original_cleanup()
        profiler.add("cleanup_memory", time.perf_counter() - t0)

    gpu_model_mod.gpu_model = profiled_gpu_model
    helpers_mod.cleanup_memory = profiled_cleanup
    blocks_mod.gpu_model = profiled_gpu_model
    blocks_mod.cleanup_memory = profiled_cleanup
