# Ascend NPU 推理指南

本文档说明如何从零开始在 Ascend 910B3 NPU 上运行 LTX-2.3 Distilled 推理。目标读者是第一次使用本仓库的新手，按照本文步骤可以完成文本生成视频并输出模型自生成音频。

英文版本见 [README_NPU.md](README_NPU.md)。

## 1. 硬件和软件要求

推荐硬件：

| 项目 | 要求 |
|------|------|
| NPU | 8 x Ascend 910B3，每卡 64 GiB HBM |
| Driver/CANN | 与 torch_npu 2.8.0 兼容的版本 |
| Python | 3.10 |
| PyTorch | torch 2.8.0 |
| torch_npu | torch_npu 2.8.0 |
| 分布式后端 | HCCL |

下文的 8 卡示例默认在单机 8 卡环境中运行。

## 2. 准备 Python 环境

创建并激活 Python 环境。下面以 Conda 为例：

```bash
conda create -n ltx2_npu python=3.10 -y
conda activate ltx2_npu
```

根据你的 Ascend/CANN 版本安装 PyTorch 和 torch_npu。已验证环境如下：

```bash
pip install torch==2.8.0 torch_npu==2.8.0
```

安装仓库内的本地包：

```bash
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-trainer
pip install -e .
```

如果你的环境使用 `uv`，也可以运行：

```bash
uv sync --frozen
```

## 3. 安装 MindIE-SD

MindIE-SD 提供本适配使用的 Ascend 优化算子，包括 attention 和 normalization 等加速路径。

请安装与当前 CANN、Python 和 torch_npu 版本匹配的 MindIE-SD 包。在很多 Ascend 环境中，它通常以 wheel 包提供：

```bash
pip install /PATH/TO/mindiesd*.whl
```

安装后验证是否可以导入：

```bash
python - <<'PY'
import mindiesd
print('mindiesd import ok')
PY
```

如果没有安装 MindIE-SD，代码会尽量回退到 PyTorch attention 路径，但最佳性能数据需要 MindIE-SD。

## 4. 准备模型权重

准备以下文件或目录。请把示例路径替换成你本机的模型路径：

| 参数 | 示例路径 |
|------|----------|
| `--distilled-checkpoint-path` | `/PATH/TO/ltx-2.3-22b-dev.safetensors` |
| `--lora` | `/PATH/TO/ltx-2.3-22b-distilled-lora-384.safetensors 0.8` |
| `--spatial-upsampler-path` | `/PATH/TO/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` |
| `--gemma-root` | `/PATH/TO/gemma-3-12b-it-qat-q4_0-unquantized` |

不要把模型权重放进 git 仓库。

## 5. 设置运行环境变量

在仓库根目录运行：

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

关键变量说明：

| 变量 | 推荐值 | 作用 |
|------|--------|------|
| `ALGO` | `1` | 使用 Laser Attention 后端 |
| `FUSED_RMSNORM` | `1` | 使用 Ascend 融合 RMSNorm |
| `LTX_ENABLE_AUDIO_ON_NPU` | `1` | 开启模型自生成音频 |
| `LTX_DISABLE_TQDM` | `1` | 关闭进度条，降低 host 侧 stdout 开销 |
| `LTX_ENABLE_RESIDENT_UPSAMPLER` | `1` | 将 spatial upsampler 常驻在 NPU 上 |
| `LTX_STAGE2_VIDEO_ONLY` | `1` | stage2 只细化 video，audio 复用 stage1 latent |
| `LTX_FSDP_FORWARD_PREFETCH` | `1` | DiT forward 时预取下一层 FSDP 参数 |
| `LTX_FSDP_LIMIT_ALL_GATHERS` | `1` | 保持 FSDP all-gather 调度受限；验证中关闭会变慢 |
| `LTX_FSDP_REPLICATE_AUDIO` | `audio` | 复制 audio self/text-cross/MLP 子模块，减少 FSDP all-gather 通信 |
| `LTX_PIPELINE_PROFILER` | unset 或 `0` | 正式性能测试时关闭 profiler |

## 6. 清理残留 NPU 进程

大模型推理前建议清理上次失败运行留下的残留进程：

```bash
KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh && sleep 5
```

该脚本只会清理属于当前仓库路径的进程；如果检测到其他用户或其他项目占用 NPU，会等待而不是强杀。

## 7. 运行 8 卡文本生成视频和音频

请把所有 `/PATH/TO/...` 替换成你的本地模型路径：

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

也可以编辑并运行示例脚本：

```bash
bash examples/scripts/run_8npu_distilled.sh
```

## 8. 预期输出和性能

输出文件是一个 MP4，包含 H.264 视频和 AAC 音频。

已验证 benchmark 配置：

| 配置 | Total Inference |
|------|-----------------|
| 8 卡 video-only | 16.51s |
| 8 卡视频 + resident full-BWE 音频 + no tqdm | 16.61s |
| 推荐环境变量下的优化版 8 卡视频 + 音频 | 约 9.73s |

额外提供以下 FSDP audio 复制实验模式，主要用于 profiling：

| `LTX_FSDP_REPLICATE_AUDIO` | 验证结果 |
|--------------------------------|----------|
| `audio` | 推荐稳定配置，约 9.73s |
| `audio_a2v` | 与 `audio` 基本持平 |
| `audio_v2a` | 单次达到 9.50s，但复跑回到约 10.01s，暂不作为推荐配置 |
| `audio_av` | 复制参数更多，验证中更慢 |

`Total Inference` 表示 warmup 后的模型推理耗时，不包含最后 MP4 编码和 mux 时间。

最新优化配置的 Ascend `msprof` 显示，DiT 仍是主要耗时；通信热点主要是 FSDP `hcom_allGather_`，其次是 Ulysses `hcom_alltoall_`，broadcast 已不是主要瓶颈。

可以用下面命令检查输出音视频流：

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,sample_rate,channels,duration \
  -of default=noprint_wrappers=1 output/ltx_8npu_audio.mp4
```

预期音频流：AAC，48 kHz，2 声道。

## 9. 架构简介

Ascend NPU 适配主要实现在 `ltx_npu/` 目录：

| 模块 | 作用 |
|------|------|
| `pipeline_wrapper.py` | 包装原始 pipeline，接入 FSDP、Ulysses SP、VAE parallel 和 resident audio |
| `fsdp_manager.py` | 对 DiT Transformer 应用 FSDP FULL_SHARD |
| `ulysses_attn.py` | 实现基于 AllToAll 的 Ulysses 序列并行 attention |
| `vae_parallel.py` | 按空间 H x W patch 切分视频 VAE decode |
| `fused_ops.py` | 接入 Ascend 融合 RMSNorm 和 NPU attention dispatch |
| `freqs_cache.py` | 缓存 RoPE 频率，并保证 positions 使用 fp32 |

关键精度修复：

```python
MixedPrecision(..., cast_root_forward_inputs=False)
positions = positions.float()
```

这可以避免 `audio.positions` 在 RoPE 计算前被 cast 成 bf16，从而修复多卡模型自生成音频异常。

## 10. 常见问题

| 现象 | 排查方式 |
|------|----------|
| HCCL 初始化失败 | 运行 `KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh`，然后换一个 `--master_port` 重试 |
| 性能明显慢于预期 | 确认 `LTX_PIPELINE_PROFILER`、`LTX_AUDIO_HEAD_PARALLEL`、`LTX_AUDIO_ALL_RANKS` 未开启，并设置 `LTX_DISABLE_TQDM=1` |
| 输出没有音频 | 确认 `LTX_ENABLE_AUDIO_ON_NPU=1`，并且没有传入 `--audio-path` 覆盖模型自生成音频 |
| NPU dtype mismatch | 保留 `run_distilled.py` 和 `ltx_npu/pipeline_wrapper.py` 中的 dtype 对齐 patch |
| MindIE-SD 无法导入 | 重新安装与 CANN、Python、torch_npu 匹配的 MindIE-SD wheel |

## 11. 可选 Profiler

如果需要拆解模块级耗时，可以开启 pipeline profiler：

```bash
export LTX_PIPELINE_PROFILER=1
```

注意：profiler 会加入同步和 hook 开销，不要把 profiler-on 结果作为正式生产性能数据。
