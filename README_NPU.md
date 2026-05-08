# Ascend NPU Inference Guide

中文版本见 [README_NPU_CN.md](README_NPU_CN.md)。

This guide explains how to run LTX-2.3 Distilled inference on Ascend 910B3 NPUs from a clean environment. It is written for users who are new to this repository and want to run text-to-video generation with model-generated audio.

## 1. Hardware And Software

Recommended hardware:

| Item | Requirement |
|------|-------------|
| NPU | 8 x Ascend 910B3, 64 GiB HBM per card |
| Driver/CANN | Version compatible with torch_npu 2.8.0 |
| Python | 3.10 |
| PyTorch | torch 2.8.0 |
| torch_npu | torch_npu 2.8.0 |
| Distributed backend | HCCL |

The 8-card example below assumes all 8 cards are available on a single node.

## 2. Prepare Python Environment

Create and activate a Python environment. Conda is shown as an example:

```bash
conda create -n ltx2_npu python=3.10 -y
conda activate ltx2_npu
```

Install PyTorch and torch_npu according to your Ascend/CANN release. For the validated environment:

```bash
pip install torch==2.8.0 torch_npu==2.8.0
```

Install repository dependencies:

```bash
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-trainer
pip install -e .
```

If your environment uses `uv`, you can also run:

```bash
uv sync --frozen
```

## 3. Install MindIE-SD

MindIE-SD provides optimized Ascend operators used by this adaptation, including attention and normalization acceleration paths.

Install the package that matches your CANN and Python environment. In many internal Ascend environments this is provided by a wheel package:

```bash
pip install /PATH/TO/mindiesd*.whl
```

After installation, verify that Python can import it:

```bash
python - <<'PY'
import mindiesd
print('mindiesd import ok')
PY
```

If MindIE-SD is not installed, the code falls back to PyTorch attention paths where possible, but the best performance numbers require MindIE-SD.

## 4. Download Model Weights

Prepare the following files or directories. Replace the paths with your local model locations:

| Argument | Example path |
|----------|--------------|
| `--distilled-checkpoint-path` | `/PATH/TO/ltx-2.3-22b-dev.safetensors` |
| `--lora` | `/PATH/TO/ltx-2.3-22b-distilled-lora-384.safetensors 0.8` |
| `--spatial-upsampler-path` | `/PATH/TO/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` |
| `--gemma-root` | `/PATH/TO/gemma-3-12b-it-qat-q4_0-unquantized` |

Do not put model weights inside the git repository.

## 5. Set Runtime Environment

Run from the repository root:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PWD/packages/ltx-core/src:$PWD/packages/ltx-pipelines/src:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=300
export ALGO=1
export FUSED_RMSNORM=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LTX_ENABLE_AUDIO_ON_NPU=1
export LTX_DISABLE_TQDM=1
export LTX_ENABLE_RESIDENT_UPSAMPLER=1
export LTX_STAGE2_VIDEO_ONLY=1
export LTX_FSDP_FORWARD_PREFETCH=1
export LTX_FSDP_LIMIT_ALL_GATHERS=1
export LTX_FSDP_REPLICATE_AUDIO=audio

unset LTX_PIPELINE_PROFILER
unset LTX_AUDIO_HEAD_PARALLEL
unset LTX_AUDIO_ALL_RANKS
```

Meaning of the most important variables:

| Variable | Recommended | Purpose |
|----------|-------------|---------|
| `ALGO` | `1` | Use the Laser Attention backend when available |
| `FUSED_RMSNORM` | `1` | Use Ascend fused RMSNorm |
| `LTX_ENABLE_AUDIO_ON_NPU` | `1` | Enable model-generated audio on NPU |
| `LTX_DISABLE_TQDM` | `1` | Disable progress bars for best host-side performance |
| `LTX_ENABLE_RESIDENT_UPSAMPLER` | `1` | Keep the spatial upsampler resident on NPU |
| `LTX_STAGE2_VIDEO_ONLY` | `1` | Run stage 2 as video-only and keep stage 1 audio latent |
| `LTX_FSDP_FORWARD_PREFETCH` | `1` | Prefetch the next FSDP block parameters during DiT forward |
| `LTX_FSDP_LIMIT_ALL_GATHERS` | `1` | Keep FSDP all-gather scheduling bounded; disabling this was slower in validation |
| `LTX_FSDP_REPLICATE_AUDIO` | `audio` | Replicate audio self/cross-text/MLP submodules to reduce FSDP all-gather traffic |
| `LTX_PIPELINE_PROFILER` | unset or `0` | Keep profiler off for production timing |

## 6. Clean Residual NPU Processes

Before running a large inference job, clear residual processes from previous failed runs:

```bash
KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh && sleep 5
```

The script only kills processes that belong to the current repository path and waits if other users' processes are still using the devices.

## 7. Run 8-Card Text-To-Video + Audio Inference

Replace all `/PATH/TO/...` values with your local model paths:

```bash
torchrun --nproc_per_node=8 --master_port=29501 run_distilled.py \
  --distilled-checkpoint-path /PATH/TO/ltx-2.3-22b-dev.safetensors \
  --lora /PATH/TO/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /PATH/TO/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /PATH/TO/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A beautiful sunset over the ocean." \
  --seed 42 \
  --num-frames 121 \
  --height 1024 \
  --width 1536 \
  --ulysses-degree 8 \
  --vae-parallel \
  --output-path output/ltx_8npu_audio.mp4
```

You can also edit and run the example script:

```bash
bash examples/scripts/run_8npu_distilled.sh
```

## 8. Expected Result

The output is an MP4 file containing H.264 video and AAC audio.

Validated benchmark configuration:

| Configuration | Total Inference |
|---------------|-----------------|
| 8-card video-only | 16.51s |
| 8-card video + resident full-BWE audio + no tqdm | 16.61s |
| Optimized 8-card video + audio, recommended env above | ~9.73s |

Additional experimental FSDP audio replication modes are available for profiling:

| `LTX_FSDP_REPLICATE_AUDIO` | Result |
|--------------------------------|--------|
| `audio` | Recommended stable setting, ~9.73s |
| `audio_a2v` | Similar to `audio` in validation |
| `audio_v2a` | Single run reached 9.50s, but rerun regressed to ~10.01s |
| `audio_av` | Slower in validation due to extra replicated parameters |

`Total Inference` measures model inference after warmup. It does not include the final MP4 encoding/muxing time.

Latest Ascend `msprof` data on the optimized configuration shows that DiT is still the dominant cost. Communication is led by FSDP `hcom_allGather_`, followed by Ulysses `hcom_alltoall_`; broadcast is not the main bottleneck.

You can inspect the output stream with:

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,sample_rate,channels,duration \
  -of default=noprint_wrappers=1 output/ltx_8npu_audio.mp4
```

Expected audio stream: AAC, 48 kHz, 2 channels.

## 9. Architecture Summary

The Ascend NPU adaptation is implemented in `ltx_npu/`:

| Module | Purpose |
|--------|---------|
| `pipeline_wrapper.py` | Wraps the original pipeline with FSDP, Ulysses SP, VAE parallel, and resident audio |
| `fsdp_manager.py` | Applies FSDP FULL_SHARD to the DiT transformer |
| `ulysses_attn.py` | Implements AllToAll-based Ulysses sequence parallel attention |
| `vae_parallel.py` | Splits video VAE decode by spatial H x W patches |
| `fused_ops.py` | Applies Ascend fused RMSNorm and NPU attention dispatch |
| `freqs_cache.py` | Caches RoPE frequencies and keeps positions in fp32 |

Key precision fix:

```python
MixedPrecision(..., cast_root_forward_inputs=False)
positions = positions.float()
```

This prevents `audio.positions` from being cast to bf16 before RoPE computation, which fixes multi-card model-generated audio artifacts.

## 10. Troubleshooting

| Symptom | Check |
|---------|-------|
| HCCL initialization fails | Run `KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh`, then retry with a new `--master_port` |
| Performance is slower than expected | Ensure `LTX_PIPELINE_PROFILER`, `LTX_AUDIO_HEAD_PARALLEL`, and `LTX_AUDIO_ALL_RANKS` are unset; set `LTX_DISABLE_TQDM=1` |
| Audio is missing | Ensure `LTX_ENABLE_AUDIO_ON_NPU=1` and do not pass `--audio-path` unless using external audio |
| dtype mismatch errors on NPU | Keep the runtime dtype patches in `run_distilled.py` and `ltx_npu/pipeline_wrapper.py` enabled |
| MindIE-SD import fails | Reinstall the wheel matching your CANN, Python, and torch_npu versions |

## 11. Optional Profiling

For detailed module timing, enable the pipeline profiler:

```bash
export LTX_PIPELINE_PROFILER=1
```

Do not use profiler-on results as production latency numbers because profiler hooks add synchronization overhead.
