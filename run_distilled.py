
# LTX_RANK0_ONLY_OUTPUT_ARGV_PATCH
# Only rank0 writes the requested final output path.
# Non-main ranks rewrite --output-path in sys.argv before argparse parses it.
import os as _ltx_rank_os
import sys as _ltx_rank_sys

_ltx_rank = int(_ltx_rank_os.getenv("RANK", _ltx_rank_os.getenv("LOCAL_RANK", "0")))

if _ltx_rank != 0 and "--output-path" in _ltx_rank_sys.argv:
    _i = _ltx_rank_sys.argv.index("--output-path")
    if _i + 1 < len(_ltx_rank_sys.argv):
        _orig_out = _ltx_rank_sys.argv[_i + 1]
        _out_dir = _ltx_rank_os.path.dirname(_orig_out) or "."
        _out_base = _ltx_rank_os.path.basename(_orig_out)
        _ltx_rank_sys.argv[_i + 1] = _ltx_rank_os.path.join(
            _out_dir,
            f".rank{_ltx_rank}_{_out_base}",
        )

import torch
import torch.nn as nn
import torch.nn.functional as F

if not hasattr(F, "_ltx_npu_conv_patched"):
    _orig_F_conv1d = F.conv1d

    def _patched_F_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if torch.is_tensor(input) and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_conv1d(input, weight, bias, stride, padding, dilation, groups)

    F.conv1d = _patched_F_conv1d

    _orig_F_conv2d = F.conv2d

    def _patched_F_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if torch.is_tensor(input) and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_conv2d(input, weight, bias, stride, padding, dilation, groups)

    F.conv2d = _patched_F_conv2d

    _orig_nn_conv1d = nn.Conv1d.forward

    def _patched_nn_conv1d(self, input):
        if torch.is_tensor(input) and input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_nn_conv1d(self, input)

    nn.Conv1d.forward = _patched_nn_conv1d

    _orig_nn_conv2d = nn.Conv2d.forward

    def _patched_nn_conv2d(self, input):
        if torch.is_tensor(input) and input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_nn_conv2d(self, input)

    nn.Conv2d.forward = _patched_nn_conv2d
    F._ltx_npu_conv_patched = True

from ltx_core.model.audio_vae.vocoder import Vocoder

if not hasattr(Vocoder, "_fp32_patched"):
    _orig_vocoder_forward = Vocoder.forward

    def _patched_vocoder_forward(self, mel_spec):
        if _ltx_rank_os.getenv("LTX_VOCODER_FORCE_FP32", "0") == "1":
            if next(self.parameters()).dtype != torch.float32:
                self.to(torch.float32)
        else:
            try:
                p = next(self.parameters())
            except StopIteration:
                p = None
            if p is not None and torch.is_tensor(mel_spec):
                if mel_spec.device != p.device:
                    mel_spec = mel_spec.to(p.device)
                if mel_spec.is_floating_point() and mel_spec.dtype != p.dtype:
                    mel_spec = mel_spec.to(p.dtype)
        return _orig_vocoder_forward(self, mel_spec)

    Vocoder.forward = _patched_vocoder_forward
    Vocoder._fp32_patched = True
"""Multi-card DistilledPipeline entry point for Ascend NPU.

Usage:
    # Single-card (equivalent to original DistilledPipeline)
    python run_distilled.py --distilled-checkpoint-path ... --prompt "..."

    # Multi-card with Ulysses SP
    torchrun --nproc_per_node=4 run_distilled.py --ulysses-degree 4 --prompt "..."

    # Multi-card with Ulysses SP + VAE Parallel
    torchrun --nproc_per_node=8 run_distilled.py --ulysses-degree 8 --vae-parallel --prompt "..."
"""

import logging
import os

import torch

import ltx_npu  # noqa: F401 — NPU runtime init

from ltx_npu.device_context import DeviceContext
from ltx_npu.parallel_config import ParallelConfig


def build_arg_parser():
    from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, detect_checkpoint_path
    from ltx_pipelines.utils.constants import detect_params

    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)

    parser.add_argument(
        "--ulysses-degree", type=int, default=1,
        help="Ulysses Sequence Parallelism degree. Must divide 32 (num_attention_heads). "
             "Use with torchrun --nproc_per_node=N where N equals this value.",
    )
    parser.add_argument(
        "--vae-parallel", action="store_true", default=False,
        help="Enable VAE spatial patch parallel decoding across all ranks.",
    )
    parser.add_argument(
        "--no-warmup", action="store_true", default=False,
        help="Skip warmup inference (for debugging; timing will include NPU compilation overhead).",
    )
    parser.add_argument(
        "--audio-path", type=str, default=None,
        help="Path to external WAV audio file. When provided, this audio is muxed "
             "directly and bypasses model audio generation.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    ctx = DeviceContext.create()
    if "RANK" in os.environ:
        ctx.init_distributed()

    parser = build_arg_parser()
    args = parser.parse_args()
    pcfg = ParallelConfig.from_args(args)

    if pcfg.is_main:
        logger.info("Device: %s | World size: %d | Ulysses degree: %d | VAE parallel: %s",
                     ctx.device_type, pcfg.world_size, pcfg.ulysses_degree, pcfg.vae_parallel)

    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    from ltx_npu.pipeline_wrapper import ParallelDistilledPipeline

    pipeline = ParallelDistilledPipeline(
        ctx=ctx,
        pcfg=pcfg,
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=getattr(args,"compile",False),
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    if os.environ.get("LTX_NPU_PROFILE") == "1":
        from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        _prof_dir = os.environ.get("LTX_NPU_PROFILE_DIR", "./prof_data")
        os.makedirs(_prof_dir, exist_ok=True)
        _prof_name = os.environ.get("LTX_NPU_PROFILE_NAME", "npu_trace")
        _prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(f"{_prof_dir}/{_prof_name}"),
        )
        _prof.start()
        logger.info("NPU profiler started, output: %s/%s", _prof_dir, _prof_name)

    result = pipeline.run_with_timing(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        streaming_prefetch_count=args.streaming_prefetch_count,
        warmup=not args.no_warmup,
    )

    if os.environ.get("LTX_NPU_PROFILE") == "1":
        _prof.stop()
        logger.info("NPU profiler stopped")

    if pcfg.is_main and result is not None:
        video, audio = result
        if args.audio_path:
            from scipy.io import wavfile
            sr, waveform_np = wavfile.read(args.audio_path)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[:, None]
            waveform = torch.from_numpy(waveform_np.T.copy()).unsqueeze(0).float() / 32768.0
            from ltx_core.types import Audio
            audio = Audio(waveform=waveform, sampling_rate=sr)
            logger.info("Using external audio: %s (%dHz, %.1fs)",
                        args.audio_path, sr, waveform.shape[-1] / sr)
        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=video_chunks_number,
        )
        logger.info("Video saved to %s", args.output_path)


if __name__ == "__main__":
    main()
