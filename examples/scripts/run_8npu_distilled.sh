#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${DISTILLED_CHECKPOINT_PATH:=/PATH/TO/ltx-2.3-22b-dev.safetensors}"
: "${DISTILLED_LORA_PATH:=/PATH/TO/ltx-2.3-22b-distilled-lora-384.safetensors}"
: "${SPATIAL_UPSAMPLER_PATH:=/PATH/TO/ltx-2.3-spatial-upscaler-x2-1.0.safetensors}"
: "${GEMMA_ROOT:=/PATH/TO/gemma-3-12b-it-qat-q4_0-unquantized}"
: "${OUTPUT_PATH:=output/ltx_8npu_audio.mp4}"
: "${MASTER_PORT:=29501}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PROJECT_ROOT/packages/ltx-core/src:$PROJECT_ROOT/packages/ltx-pipelines/src:${PYTHONPATH:-}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-300}"
export ALGO="${ALGO:-1}"
export FUSED_RMSNORM="${FUSED_RMSNORM:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export LTX_ENABLE_AUDIO_ON_NPU="${LTX_ENABLE_AUDIO_ON_NPU:-1}"
export LTX_DISABLE_TQDM="${LTX_DISABLE_TQDM:-1}"

unset LTX_PIPELINE_PROFILER
unset LTX_AUDIO_HEAD_PARALLEL
unset LTX_AUDIO_ALL_RANKS

KILL_TORCHRUN=1 bash examples/scripts/clean_npu.sh
sleep 5

torchrun --nproc_per_node=8 --master_port="$MASTER_PORT" run_distilled.py \
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH" \
  --lora "$DISTILLED_LORA_PATH" 0.8 \
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "${PROMPT:-A beautiful sunset over the ocean.}" \
  --seed "${SEED:-42}" \
  --num-frames "${NUM_FRAMES:-121}" \
  --height "${HEIGHT:-1024}" \
  --width "${WIDTH:-1536}" \
  --ulysses-degree 8 \
  --vae-parallel \
  --output-path "$OUTPUT_PATH"
