# AGENTS.md — LTX-2 Ascend NPU

Ascend NPU adaptation of LTX-2.3 for multi-card inference on Ascend 910B3. The production path supports 8-card text-to-video generation with model-generated audio.

## Quick Commands

```bash
# Environment
conda activate ltx2_npu

# Install local packages
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-trainer
pip install -e .

# Clean residual NPU processes for this repo
KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh && sleep 5

# 8-card example; edit model paths before running
bash examples/scripts/run_8npu_distilled.sh

# Distributed tests must use torchrun
torchrun --nproc_per_node=N -m pytest testcase/ -v
```

## Main Entry Point

`run_distilled.py` is the production inference entry point. It contains small runtime dtype patches for Ascend Conv/Vocoder paths and must run before model loading.

Use `README_NPU.md` as the user-facing setup and inference guide. Avoid adding personal absolute paths to docs or examples; use `/PATH/TO/...` placeholders or environment variables.

## NPU Adaptation Module

`ltx_npu/` contains the Ascend adaptation layer. Keep production code focused and avoid committing one-off debug dump utilities unless they are documented and maintained.

Key files:

- `pipeline_wrapper.py` — wraps `DistilledPipeline` with FSDP, Ulysses SP, VAE parallel, resident text/audio, timing, and optional profiler.
- `fsdp_manager.py` — FSDP FULL_SHARD for the DiT transformer. Keep `cast_root_forward_inputs=False`; it prevents audio RoPE precision regressions.
- `ulysses_attn.py` — AllToAll-based Ulysses sequence parallel attention for video self-attention.
- `vae_parallel.py` — spatial H x W patch parallel video VAE decode with boundary exchange.
- `fused_ops.py` — Ascend fused RMSNorm and attention backend dispatch.
- `freqs_cache.py` — RoPE frequency cache; keeps positions in fp32.
- `device_context.py` / `parallel_config.py` — device and distributed setup.

## Production Environment

Recommended defaults for 8-card inference:

| Variable | Recommended | Purpose |
|----------|-------------|---------|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Visible NPUs |
| `HCCL_CONNECT_TIMEOUT` | `300` | HCCL startup tolerance |
| `ALGO` | `1` | Laser Attention backend when available |
| `FUSED_RMSNORM` | `1` | Ascend fused RMSNorm |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | Reduce allocator fragmentation |
| `LTX_ENABLE_AUDIO_ON_NPU` | `1` | Enable model-generated audio |
| `LTX_DISABLE_TQDM` | `1` | Reduce host stdout overhead |
| `LTX_PIPELINE_PROFILER` | unset / `0` | Keep profiler off for production timing |

Experimental audio flags are not production defaults:

- `LTX_AUDIO_HEAD_PARALLEL=1` was slower in validation.
- `LTX_AUDIO_ALL_RANKS=1` repeats audio decode on all ranks and is not true parallelism.
- `LTX_AUDIO_DISABLE_BWE=1` changes the audio quality path and was not faster in validation.

## Documentation And Examples

- Keep the main NPU guide in `README_NPU.md`.
- Keep runnable shell examples in `examples/scripts/`.
- Do not place shell launchers in the repository root.
- Do not hard-code local usernames or machine-specific model paths.

## Testing Notes

- All distributed tests in `testcase/` should run via `torchrun --nproc_per_node=N`.
- `conftest.py` initializes a distributed process group based on available hardware.
- Syntax sanity check for edited Python files: `python -m py_compile <files>`.

## Language

Use Chinese for user-facing discussion in this workspace.
