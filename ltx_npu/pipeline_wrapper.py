"""Multi-card wrapper for DistilledPipeline with Ulysses SP and VAE Parallel."""

from __future__ import annotations
import os

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ltx_core.model.upsampler import upsample_video
from ltx_pipelines.utils.helpers import generate_enhanced_prompt

from ltx_npu.device_context import DeviceContext
from ltx_npu.parallel_config import ParallelConfig
from ltx_npu.pipeline_profiler import PipelineProfiler, install_hooks
from ltx_npu.timing import TimingReport
from ltx_npu.freqs_cache import clear_freqs_cache_on_model, install_freqs_cache_on_model
from ltx_npu.fused_ops import get_config, inject_npu_attention, patch_layernorm
from ltx_npu.ulysses_attn import inject_ulysses_attention, install_sequence_parallel_hooks, inject_ulysses_v2a_cross_attention

if TYPE_CHECKING:
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.registry import Registry
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.types import Audio
    from ltx_pipelines.utils.args import ImageConditioningInput

logger = logging.getLogger(__name__)


_ACTIVE_PIPELINE_PROFILER: PipelineProfiler | _NullPipelineProfiler | None = None


def get_active_profiler():
    return _ACTIVE_PIPELINE_PROFILER


class _NullPipelineProfiler:
    """No-op profiler used for production timing without sync hooks."""

    _last_gpu_model_exit = None
    _gpu_model_depth = 0

    def sync(self) -> None:
        pass

    def add(self, name: str, elapsed: float) -> None:
        pass

    def add_aggregate(self, name: str, elapsed: float) -> None:
        pass

    def next_label(self, base: str) -> str:
        return base

    def reset(self) -> None:
        pass

    def report(self) -> str:
        return ""


class _SilentAudioDecoder:
    """Drop-in replacement that returns None audio without touching the vocoder."""

    def __call__(self, latent):
        return None


class _InputAlignModuleProxy(torch.nn.Module):
    """Wrap a module and align floating tensor inputs to its parameter dtype/device."""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def _first_param(self):
        try:
            return next(self._module.parameters())
        except Exception:
            return None

    def _cast_tree(self, obj, *, dtype=None, device=None):
        if torch.is_tensor(obj):
            out = obj
            if device is not None and out.device != device:
                out = out.to(device)
            if dtype is not None and out.is_floating_point() and out.dtype != dtype:
                out = out.to(dtype=dtype)
            return out
        if isinstance(obj, (list, tuple)):
            items = [self._cast_tree(x, dtype=dtype, device=device) for x in obj]
            return type(obj)(items)
        if isinstance(obj, dict):
            return {k: self._cast_tree(v, dtype=dtype, device=device) for k, v in obj.items()}
        return obj

    def __call__(self, *args, **kwargs):
        p = self._first_param()
        dtype = p.dtype if p is not None else None
        device = p.device if p is not None else None
        args2 = self._cast_tree(args, dtype=dtype, device=device)
        kwargs2 = self._cast_tree(kwargs, dtype=dtype, device=device)
        return self._module(*args2, **kwargs2)


_LTX_VAE_AUDIO_PATCHED = False


def _patch_global_vae_decode_audio():
    """Patch audio decode path in ltx_npu only.

    Match the Ascend branch behavior without editing ltx-core:
    - decoder input follows decoder dtype/device
    - vocoder and decoded_audio use fp16 on NPU
    - output waveform returns fp32 for encode_video
    """
    global _LTX_VAE_AUDIO_PATCHED
    if _LTX_VAE_AUDIO_PATCHED:
        return

    try:
        import ltx_pipelines.utils.blocks as _blocks
        from ltx_core.types import Audio
    except Exception:
        return

    orig = getattr(_blocks, "vae_decode_audio", None)
    if orig is None:
        return
    if getattr(orig, "_ltx_npu_fp16_patched", False):
        _LTX_VAE_AUDIO_PATCHED = True
        return

    def _first_param(module):
        try:
            return next(module.parameters())
        except Exception:
            return None

    def _cast_tensor(x, *, device=None, dtype=None):
        if not torch.is_tensor(x):
            return x
        if device is not None and x.device != device:
            x = x.to(device)
        if dtype is not None and x.is_floating_point() and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x

    def _patched_vae_decode_audio(latent, decoder, vocoder, *args, **kwargs):
        dec = decoder
        voc = vocoder

        dp = _first_param(dec)
        if dp is not None:
            latent = _cast_tensor(latent, device=dp.device, dtype=dp.dtype)

        decoded_audio = dec(latent)

        if torch.is_tensor(decoded_audio) and decoded_audio.device.type == "npu":
            voc = voc.to(device=decoded_audio.device, dtype=torch.float16).eval()
            decoded_audio = decoded_audio.to(dtype=torch.float16)

        waveform = voc(decoded_audio).squeeze(0).float()
        return Audio(waveform=waveform, sampling_rate=voc.output_sampling_rate)

    _patched_vae_decode_audio._ltx_npu_fp16_patched = True
    _patched_vae_decode_audio._ltx_npu_patched = True
    _blocks.vae_decode_audio = _patched_vae_decode_audio
    _LTX_VAE_AUDIO_PATCHED = True


class _AudioDecoderProxy:
    """NPU-only audio fix in ltx_npu.

    - Patch decoder input alignment.
    - Monkeypatch outer vocoder.forward directly so it no longer hard-casts mel_spec
      to float32 before calling the inner vocoder model.
    """

    def __init__(self, decoder):
        self._decoder = decoder
        self._patched = False
        self._patch_nested_modules()

    def __getattr__(self, name):
        return getattr(self._decoder, name)

    def _patch_nested_modules(self):
        if self._patched:
            return

        _patch_global_vae_decode_audio()

        # decoder input align
        if hasattr(self._decoder, "decoder") and not isinstance(self._decoder.decoder, _InputAlignModuleProxy):
            self._decoder.decoder = _InputAlignModuleProxy(self._decoder.decoder)

        # patch outer vocoder.forward directly
        if hasattr(self._decoder, "vocoder"):
            voc = self._decoder.vocoder
            if not getattr(voc, "_ltx_npu_forward_patched", False):
                import types

                def _patched_forward(this, mel_spec, *args, **kwargs):
                    inner = getattr(this, "vocoder", None)
                    if inner is None:
                        raise RuntimeError("Patched vocoder missing inner module 'vocoder'")
                    try:
                        p = next(inner.parameters())
                        target_dtype = p.dtype
                        target_device = p.device
                    except Exception:
                        target_dtype = mel_spec.dtype if torch.is_tensor(mel_spec) else None
                        target_device = mel_spec.device if torch.is_tensor(mel_spec) else None
                    x = mel_spec
                    if torch.is_tensor(x):
                        if target_device is not None and x.device != target_device:
                            x = x.to(target_device)
                        if target_dtype is not None and x.is_floating_point() and x.dtype != target_dtype:
                            x = x.to(dtype=target_dtype)
                    return inner(x)

                voc.forward = types.MethodType(_patched_forward, voc)
                voc._ltx_npu_forward_patched = True

        self._patched = True

    def __call__(self, *args, **kwargs):
        self._patch_nested_modules()
        import os
        import torch
        import logging
        if os.getenv("LTX_AUDIO_FORCE_CPU", "0") == "1":
            logging.getLogger(__name__).info("🚀 FULL BWE AUDIO DECODE ON CPU (MAX QUALITY) 🚀")
            
            # 1. 终极修复：解除 NPU 分支对 BWE (高频扩展) 的物理阉割
            if hasattr(self._decoder, "vocoder"):
                voc = self._decoder.vocoder
                if getattr(voc, "_ltx_npu_forward_patched", False):
                    try:
                        del voc.forward
                        voc._ltx_npu_forward_patched = False
                        logging.getLogger(__name__).info("✅ BWE restored! High frequencies will be generated.")
                    except Exception:
                        pass

            # 2. 关键修复：AudioDecoder.__call__ 内部 rebuild 时读取的是
            #    self._device / self._dtype，必须临时改为 CPU/fp32 才生效
            old_device = getattr(self._decoder, '_device', None)
            old_dtype = getattr(self._decoder, '_dtype', None)
            self._decoder._device = torch.device("cpu")
            self._decoder._dtype = torch.float32
            
            # 3. 转换输入 Tensor
            args = tuple(a.to(device="cpu", dtype=torch.float32) if torch.is_tensor(a) else a for a in args)
            kwargs = {k: v.to(device="cpu", dtype=torch.float32) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            
            try:
                with torch.no_grad():
                    return self._decoder(*args, **kwargs)
            finally:
                if old_device is not None:
                    self._decoder._device = old_device
                if old_dtype is not None:
                    self._decoder._dtype = old_dtype

        # NPU path: run audio decode on NPU directly
        with torch.no_grad():
            return self._decoder(*args, **kwargs)


class _ResidentParallelVideoDecoder:
    """Callable proxy that keeps the decoder resident and applies VAE parallel decode."""

    def __init__(self, block, decoder, vae_ctx, profiler):
        self._block = block
        self._decoder = decoder
        self._vae_ctx = vae_ctx
        self._profiler = profiler

    def __getattr__(self, name):
        return getattr(self._block, name)

    def __call__(self, latent, tiling_config=None, generator=None):
        label = self._profiler.next_label("VideoDecoder")
        self._profiler.sync()
        t0 = time.perf_counter()
        try:
            from ltx_core.model.video_vae.tiling import TilingConfig
            from ltx_npu.vae_parallel import vae_parallel_decode

            effective_tiling = tiling_config
            if tiling_config is not None and getattr(tiling_config, "spatial_config", None) is not None:
                effective_tiling = TilingConfig(
                    spatial_config=None,
                    temporal_config=getattr(tiling_config, "temporal_config", None),
                )
                logger.info("VAE parallel enabled: disable spatial tiled_decode and keep temporal tiling only")

            with vae_parallel_decode(self._decoder, self._vae_ctx):
                chunks = list(self._decoder.decode_video(latent, effective_tiling, generator))
            # 🚀 物理隔离锁：确保视频并行解码的 HCCL 通信彻底结束，绝不干扰后续音频
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier() 
            return iter(chunks)
        finally:
            self._profiler.sync()
            self._profiler.add(f"use {label}", time.perf_counter() - t0)
            self._profiler._last_gpu_model_exit = time.perf_counter()


class ParallelDistilledPipeline:
    """Wraps DistilledPipeline with Ulysses SP for multi-card DiT inference.

    Each rank loads the full model and runs the full pipeline. The only modification
    is that self-attention in the DiT transformer is wrapped with Ulysses AllToAll,
    so each rank computes attention on a subset of heads while seeing the full sequence.

    For a proper sequence-split implementation (where each rank processes only local_T
    tokens through ALL layers), the TransformerArgs need to be split/gathered around
    the transformer blocks. This is deferred to a future optimization.
    """

    def __init__(
        self,
        ctx: DeviceContext,
        pcfg: ParallelConfig,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.ctx = ctx
        self.pcfg = pcfg
        self.timer = TimingReport(ctx)
        self.profile_enabled = os.getenv("LTX_PIPELINE_PROFILER", "0") == "1"
        self.profiler = PipelineProfiler(ctx) if self.profile_enabled else _NullPipelineProfiler()
        self._inject_done = False

        if self.profile_enabled:
            install_hooks(self.profiler)

        from ltx_pipelines.distilled import DistilledPipeline

        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=distilled_checkpoint_path,
            gemma_root=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            device=ctx.get_device(),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )

        if pcfg.is_parallel:
            self._build_fsdp_transformer()
            self._hook_fsdp_transformer_ctx()
            if os.getenv("LTX_ENABLE_RESIDENT_TEXT_ENCODER", "1") == "1":
                self._hook_resident_text_encoder()
            if os.getenv("LTX_ENABLE_RESIDENT_EMBEDDINGS_PROCESSOR", "1") == "1":
                self._hook_resident_embeddings_processor()
            if os.getenv("LTX_BROADCAST_PROMPT_ENCODER", "1") == "1":
                self._hook_prompt_encoder_broadcast()
            if os.getenv("LTX_ENABLE_RESIDENT_IMAGE_ENCODER", "0") == "1":
                self._hook_resident_image_encoder()
            if os.getenv("LTX_ENABLE_RESIDENT_UPSAMPLER", "0") == "1":
                self._hook_resident_upsampler()

        self._hook_empty_image_conditioner_fastpath()

        self._hook_denoising_loop_timing()

        if pcfg.vae_parallel and pcfg.is_parallel:
            self._hook_vae_parallel()

        self._patched_pipeline_call = self._pipeline.__call__
        self._fix_audio_vocoder_dtype()
        if os.getenv("LTX_ENABLE_RESIDENT_AUDIO", "1") == "1":
            self._hook_resident_audio()

    @staticmethod
    def _fix_builder_model_path(stage) -> None:
        """Fix model_path when it's a directory instead of a safetensors file.

        safetensors >= 0.7.0 requires a file path, not a directory.
        When model_path is a directory, find the distilled safetensors file.
        """
        import os
        from dataclasses import replace as dc_replace

        builder = stage._transformer_builder
        path = builder.model_path
        if isinstance(path, str) and os.path.isdir(path):
            candidates = [f for f in os.listdir(path) if f.endswith(".safetensors") and "distilled" in f and "lora" not in f]
            if candidates:
                resolved = os.path.join(path, candidates[0])
                stage._transformer_builder = dc_replace(builder, model_path=resolved)
                logger.info("Fixed model_path: %s → %s", path, resolved)

    def _fix_audio_vocoder_dtype(self) -> None:
        """Keep audio only on main rank; non-main ranks must not decode audio."""
        if self.ctx.device_type == "npu":
            enable_audio = os.getenv("LTX_ENABLE_AUDIO_ON_NPU", "0") == "1"
            audio_all_ranks = os.getenv("LTX_AUDIO_ALL_RANKS", "0") == "1"
            rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
            is_main_rank = rank == 0

            if enable_audio and (is_main_rank or audio_all_ranks):
                if audio_all_ranks and not is_main_rank:
                    logger.info("Audio enabled on NPU: non-main rank keeps audio decoder (all-ranks experiment).")
                else:
                    logger.info("Audio enabled on NPU: main rank keeps audio decoder.")
                self._pipeline.audio_decoder = _AudioDecoderProxy(self._pipeline.audio_decoder)
            elif enable_audio:
                logger.info("Audio enabled on NPU: non-main rank uses silent stub.")
                self._pipeline.audio_decoder = _SilentAudioDecoder()
            else:
                logger.info("Replacing audio decoder with silent stub (NPU vocoder dtype workaround).")
                self._pipeline.audio_decoder = _SilentAudioDecoder()

        self._patched_pipeline_call = self._pipeline.__call__

    def _build_fsdp_transformer(self) -> None:
        """Build transformer on CPU, inject Ulysses SP, wrap with FSDP.

        The FSDP model is stored as self._fsdp_transformer and reused for
        every generate call — no build/free/cleanup cycle during inference.
        """
        from ltx_npu.fsdp_manager import shard_transformer

        stage = self._pipeline.stage
        self._fix_builder_model_path(stage)
        logger.info("Building transformer on CPU for FSDP sharding...")
        torch.npu.set_device(self.ctx.get_device())
        model = stage._build_transformer(device=torch.device("cpu"))

        fused_cfg = get_config()

        velocity_model = model.velocity_model

        if fused_cfg.algo != 0 or fused_cfg.fast_layernorm:
            if fused_cfg.algo != 0:
                n_attn = inject_npu_attention(velocity_model)
                logger.info("Injected NPUAttention (ALGO=%d) into %d modules", fused_cfg.algo, n_attn)
            if fused_cfg.fast_layernorm:
                n_ln = patch_layernorm(velocity_model)
                logger.info("Patched %d LayerNorm modules with fast_layernorm", n_ln)

        if self.pcfg.sp_group is not None:
            inject_ulysses_attention(velocity_model, self.pcfg.sp_group)
            install_sequence_parallel_hooks(velocity_model, self.pcfg.sp_group)
            if os.getenv("LTX_ENABLE_V2A", "0") == "1":
                inject_ulysses_v2a_cross_attention(velocity_model, self.pcfg.sp_group)
            logger.info(
                "Installed e2e sequence parallel (degree=%d) before FSDP wrap",
                self.pcfg.ulysses_degree,
            )

        # Barrier: ensure all ranks finish building before FSDP all_gather
        if dist.is_initialized():
            dist.barrier()

        self._fsdp_transformer = shard_transformer(
            model,
            device_id=self.ctx.get_device(),
        )
        self._fsdp_transformer.eval()

        n_cached = install_freqs_cache_on_model(self._fsdp_transformer)
        if n_cached:
            logger.info("Installed RoPE freqs cache on %d preprocessors", n_cached)

        logger.info("FSDP transformer ready on %s", self.ctx.get_device())

    def _hook_fsdp_transformer_ctx(self) -> None:
        """Replace DiffusionStage._transformer_ctx to yield the FSDP model directly.

        Bypasses gpu_model entirely — no build/to_meta/cleanup_memory.
        The FSDP model stays sharded on device across all generate calls.
        """
        fsdp_transformer = self._fsdp_transformer
        profiler = self.profiler

        @contextmanager
        def fsdp_ctx(streaming_prefetch_count=None, **kwargs):
            label = profiler.next_label("X0Model")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                yield fsdp_transformer
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        self._pipeline.stage._transformer_ctx = fsdp_ctx

    def _hook_resident_text_encoder(self) -> None:
        """Pre-build Gemma text encoder and keep it resident on device.

        Eliminates the ~6s per-generate disk I/O for Gemma loading.
        The model stays on device across all generate calls.
        """
        prompt_encoder = self._pipeline.prompt_encoder
        if not self.pcfg.is_main:
            logger.info("Resident text encoder skipped on non-main rank.")
            return
        device = self.ctx.get_device()
        dtype = prompt_encoder._dtype

        logger.info("Building Gemma text encoder on device (resident mode)...")
        self._resident_text_encoder = (
            prompt_encoder._text_encoder_builder
            .build(device=device, dtype=dtype)
            .eval()
        )
        logger.info("Gemma text encoder resident on %s", device)

        cached_encoder = self._resident_text_encoder
        profiler = self.profiler

        @contextmanager
        def resident_text_encoder_ctx(streaming_prefetch_count=None):
            label = profiler.next_label("GemmaTextEncoder")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                yield cached_encoder
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        prompt_encoder._text_encoder_ctx = resident_text_encoder_ctx


    def _hook_resident_embeddings_processor(self) -> None:
        """Pre-build embeddings processor and keep it resident on device."""
        prompt_encoder = self._pipeline.prompt_encoder
        if not self.pcfg.is_main:
            logger.info("Resident embeddings processor skipped on non-main rank.")
            return
        device = self.ctx.get_device()
        dtype = prompt_encoder._dtype

        logger.info("Building embeddings processor on device (resident mode)...")
        self._resident_embeddings_processor = (
            prompt_encoder._embeddings_processor_builder
            .build(device=device, dtype=dtype)
            .to(device)
            .eval()
        )
        logger.info("Embeddings processor resident on %s", device)

        cached_processor = self._resident_embeddings_processor
        profiler = self.profiler

        def patched_prompt_encoder_call(
            self_pe,
            prompts: list[str],
            *,
            enhance_first_prompt: bool = False,
            enhance_prompt_image: str | None = None,
            enhance_prompt_seed: int = 42,
            streaming_prefetch_count: int | None = None,
        ):
            with self_pe._text_encoder_ctx(streaming_prefetch_count) as text_encoder:
                if enhance_first_prompt:
                    prompts = list(prompts)
                    prompts[0] = generate_enhanced_prompt(
                        text_encoder,
                        prompts[0],
                        enhance_prompt_image,
                        seed=enhance_prompt_seed,
                    )
                raw_outputs = [text_encoder.encode(p) for p in prompts]

            label = profiler.next_label("EmbeddingsProcessor")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                return [cached_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        prompt_encoder.__class__.__call__ = patched_prompt_encoder_call

    def _hook_prompt_encoder_broadcast(self) -> None:
        """Encode prompts once on rank0 and broadcast the embeddings outputs."""
        if not dist.is_initialized():
            return

        prompt_encoder = self._pipeline.prompt_encoder
        device = self.ctx.get_device()
        is_main = self.pcfg.is_main
        original_call = prompt_encoder.__class__.__call__

        def _dtype_from_name(name: str) -> torch.dtype:
            return getattr(torch, name.split(".")[-1])

        def _broadcast_outputs(outputs):
            meta = [
                {
                    "video_shape": tuple(out.video_encoding.shape),
                    "video_dtype": str(out.video_encoding.dtype),
                    "audio_present": out.audio_encoding is not None,
                    "audio_shape": tuple(out.audio_encoding.shape) if out.audio_encoding is not None else None,
                    "audio_dtype": str(out.audio_encoding.dtype) if out.audio_encoding is not None else None,
                    "mask_shape": tuple(out.attention_mask.shape),
                    "mask_dtype": str(out.attention_mask.dtype),
                }
                for out in outputs
            ]
            payload = [meta]
            dist.broadcast_object_list(payload, src=0)
            for out in outputs:
                dist.broadcast(out.video_encoding, src=0)
                if out.audio_encoding is not None:
                    dist.broadcast(out.audio_encoding, src=0)
                dist.broadcast(out.attention_mask, src=0)
            return outputs

        def _receive_outputs():
            payload = [None]
            dist.broadcast_object_list(payload, src=0)
            outputs = []
            for item in payload[0]:
                video_encoding = torch.empty(item["video_shape"], device=device, dtype=_dtype_from_name(item["video_dtype"]))
                audio_encoding = None
                if item["audio_present"]:
                    audio_encoding = torch.empty(item["audio_shape"], device=device, dtype=_dtype_from_name(item["audio_dtype"]))
                attention_mask = torch.empty(item["mask_shape"], device=device, dtype=_dtype_from_name(item["mask_dtype"]))
                outputs.append((video_encoding, audio_encoding, attention_mask))
            for video_encoding, audio_encoding, attention_mask in outputs:
                dist.broadcast(video_encoding, src=0)
                if audio_encoding is not None:
                    dist.broadcast(audio_encoding, src=0)
                dist.broadcast(attention_mask, src=0)
            from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
            return [EmbeddingsProcessorOutput(v, a, m) for v, a, m in outputs]

        def broadcasted_prompt_encoder_call(
            self_pe,
            prompts: list[str],
            *,
            enhance_first_prompt: bool = False,
            enhance_prompt_image: str | None = None,
            enhance_prompt_seed: int = 42,
            streaming_prefetch_count: int | None = None,
        ):
            if is_main:
                outputs = original_call(
                    self_pe,
                    prompts,
                    enhance_first_prompt=enhance_first_prompt,
                    enhance_prompt_image=enhance_prompt_image,
                    enhance_prompt_seed=enhance_prompt_seed,
                    streaming_prefetch_count=streaming_prefetch_count,
                )
                return _broadcast_outputs(outputs)
            return _receive_outputs()

        prompt_encoder.__class__.__call__ = broadcasted_prompt_encoder_call

    def _hook_resident_image_encoder(self) -> None:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Replacing VideoEncoder with a Hollow Proxy (keeps stats, 0 NPU memory)...")
        
        # 1. 在 CPU 上建一个真实的 Encoder，拿到它所有的属性（比如 per_channel_statistics）
        real_encoder = self._pipeline.image_conditioner._build_encoder()
        
        # 2. 造一个壳，拦截所有往 NPU 搬运的动作，同时代理所有的属性访问
        class HollowEncoderProxy:
            def __init__(self, real):
                self._real = real
            def __getattr__(self, name):
                return getattr(self._real, name)
            def to(self, *args, **kwargs): return self  # 拦截！绝不占 NPU 显存！
            def eval(self): return self
            def __call__(self, *args, **kwargs): return None # 拦截！防止意外前向传播
            
        self._resident_video_encoder = HollowEncoderProxy(real_encoder)
        cached_encoder = self._resident_video_encoder
        profiler = self.profiler

        def patched_image_conditioner_call(self_ic, fn):
            label = profiler.next_label("VideoEncoder_Hollow")
            profiler.sync()
            import time
            t0 = time.perf_counter()
            try:
                return fn(cached_encoder)
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        self._pipeline.image_conditioner.__class__.__call__ = patched_image_conditioner_call

    def _hook_empty_image_conditioner_fastpath(self) -> None:
        """Skip VideoEncoder build/free when no image conditionings are provided."""
        image_conditioner = self._pipeline.image_conditioner
        profiler = self.profiler

        def patched_image_conditioner_call(self_ic, fn):
            closure = getattr(fn, "__closure__", None) or ()
            for cell in closure:
                try:
                    if cell.cell_contents == []:
                        profiler.add("skip empty VideoEncoder", 0.0)
                        return []
                except Exception:
                    continue
            from ltx_pipelines.utils.gpu_model import gpu_model

            with gpu_model(self_ic._build_encoder()) as encoder:
                return fn(encoder)

        image_conditioner.__class__.__call__ = patched_image_conditioner_call
    def _hook_resident_upsampler(self) -> None:
        """Pre-build spatial upsampler and keep only encoder statistics resident."""
        upsampler_block = self._pipeline.upsampler
        device = self.ctx.get_device()
        dtype = upsampler_block._dtype

        if not hasattr(self, "_resident_video_encoder"):
            real_encoder = upsampler_block._encoder_builder.build(device=torch.device("cpu"), dtype=dtype).eval()

            class HollowEncoderProxy:
                def __init__(self, real):
                    self._real = real

                def __getattr__(self, name):
                    return getattr(self._real, name)

                def to(self, *args, **kwargs):
                    return self

                def eval(self):
                    return self

                def __call__(self, *args, **kwargs):
                    raise RuntimeError("Hollow encoder is statistics-only")

            self._resident_video_encoder = HollowEncoderProxy(real_encoder)

        logger.info("Building spatial upsampler on device (resident mode)...")
        self._resident_spatial_upsampler = (
            upsampler_block._upsampler_builder
            .build(device=device, dtype=dtype)
            .to(device)
            .eval()
        )
        logger.info("Spatial upsampler resident on %s", device)

        cached_encoder = self._resident_video_encoder
        cached_upsampler = self._resident_spatial_upsampler
        profiler = self.profiler

        def patched_upsampler_call(self_up, latent: torch.Tensor) -> torch.Tensor:
            label = profiler.next_label("LatentUpsampler")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                return upsample_video(
                    latent=latent,
                    video_encoder=cached_encoder,
                    upsampler=cached_upsampler,
                )
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        upsampler_block.__class__.__call__ = patched_upsampler_call





    def _hook_resident_audio(self) -> None:
        """Resident audio on NPU.

        Safe mode:
        - only rank0 decodes audio
        - optionally bypass BWE to avoid Conv2DTranspose OOM / stutter
        Env:
          LTX_AUDIO_RESIDENT_DTYPE=fp16|bf16|fp32
          LTX_AUDIO_DISABLE_BWE=1  # faster, lower-quality path
        """
        import os
        import time
        import torch
        import torch.nn.functional as F

        # 非主 rank 不需要真的解音频，否则 8 张卡都会各自跑一遍 audio decoder/vocoder
        if hasattr(self, "pcfg") and not self.pcfg.is_main:
            logger.info("Resident audio skipped on non-main rank.")
            self._pipeline.audio_decoder = _SilentAudioDecoder()
            self._patched_pipeline_call = self._pipeline.__call__
            return

        audio_dtype_env = os.getenv("LTX_AUDIO_RESIDENT_DTYPE", "fp16").lower().strip()
        if audio_dtype_env in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif audio_dtype_env in ("fp32", "float32"):
            dtype = torch.float32
        else:
            dtype = torch.float16

        disable_bwe = os.getenv("LTX_AUDIO_DISABLE_BWE", "0") == "1"

        logger.info(
            "Building resident audio on device, dtype=%s, disable_bwe=%s",
            dtype,
            disable_bwe,
        )

        audio_block = self._pipeline.audio_decoder
        real_block = audio_block._decoder if hasattr(audio_block, "_decoder") else audio_block

        if not hasattr(real_block, "_decoder_builder") or not hasattr(real_block, "_vocoder_builder"):
            logger.warning("Resident audio skipped: audio block has no decoder/vocoder builders")
            return

        device = self.ctx.get_device()

        dec_mod = (
            real_block._decoder_builder
            .build(device=device, dtype=dtype)
            .to(device)
            .to(dtype)
            .eval()
        )

        voc_mod = (
            real_block._vocoder_builder
            .build(device=device, dtype=dtype)
            .to(device)
            .to(dtype)
            .eval()
        )

        profiler = self.profiler

        class _TimedModule:
            def __init__(self, mod, label):
                self.mod = mod
                self.label = label

            def __getattr__(self, name):
                return getattr(self.mod, name)

            def _first_param(self):
                try:
                    return next(self.mod.parameters())
                except Exception:
                    return None

            def _cast_tree(self, obj, *, device=None, dtype=None):
                if torch.is_tensor(obj):
                    out = obj
                    if device is not None and out.device != device:
                        out = out.to(device)
                    if dtype is not None and out.is_floating_point() and out.dtype != dtype:
                        out = out.to(dtype=dtype)
                    return out
                if isinstance(obj, (list, tuple)):
                    return type(obj)(self._cast_tree(x, device=device, dtype=dtype) for x in obj)
                if isinstance(obj, dict):
                    return {k: self._cast_tree(v, device=device, dtype=dtype) for k, v in obj.items()}
                return obj

            def __call__(self, *args, **kwargs):
                p = self._first_param()
                target_device = p.device if p is not None else device
                target_dtype = p.dtype if p is not None else dtype

                args = self._cast_tree(args, device=target_device, dtype=target_dtype)
                kwargs = self._cast_tree(kwargs, device=target_device, dtype=target_dtype)

                profiler.sync()
                t0 = time.perf_counter()
                try:
                    return self.mod(*args, **kwargs)
                finally:
                    profiler.sync()
                    profiler.add(f"use {self.label}", time.perf_counter() - t0)
                    profiler._last_gpu_model_exit = time.perf_counter()

        wrapped_dec = _TimedModule(dec_mod, f"AudioDecoder_{str(dtype).replace('torch.', '').upper()}")

        def _run_base_vocoder(voc, mel_spec):
            """Run only base vocoder, bypass BWE generator."""
            profiler.sync()
            t0 = time.perf_counter()
            try:
                mel_spec = mel_spec.to(device=device, dtype=dtype)
                audio = voc.vocoder(mel_spec)
                return audio
            finally:
                profiler.sync()
                profiler.add(f"use BaseVocoder_{str(dtype).replace('torch.', '').upper()}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        def _run_vocoder_with_bwe(voc, mel_spec):
            """Full VocoderWithBWE path without mel_spec.float()."""
            profiler.sync()
            t0 = time.perf_counter()
            try:
                mel_spec = mel_spec.to(device=device, dtype=dtype)

                x = voc.vocoder(mel_spec)

                _, _, length_low_rate = x.shape
                output_length = length_low_rate * voc.output_sampling_rate // voc.input_sampling_rate

                remainder = length_low_rate % voc.hop_length
                if remainder != 0:
                    x = F.pad(x, (0, voc.hop_length - remainder))

                mel = voc._compute_mel(x).to(device=device, dtype=dtype)
                mel_for_bwe = mel.transpose(2, 3).to(device=device, dtype=dtype)

                residual = voc.bwe_generator(mel_for_bwe)
                skip = voc.resampler(x)

                if skip.dtype != residual.dtype:
                    skip = skip.to(dtype=residual.dtype)
                if skip.device != residual.device:
                    skip = skip.to(residual.device)

                return torch.clamp(residual + skip, -1, 1)[..., :output_length]
            finally:
                profiler.sync()
                profiler.add(f"use VocoderWithBWE_{str(dtype).replace('torch.', '').upper()}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        class _ResidentAudioBlock:
            def __call__(self, latent):
                from ltx_core.types import Audio

                with torch.no_grad():
                    if torch.is_tensor(latent):
                        latent = latent.to(device=device, dtype=dtype)

                    decoded_audio = wrapped_dec(latent)

                    if torch.is_tensor(decoded_audio):
                        decoded_audio = decoded_audio.to(device=device, dtype=dtype)

                    if disable_bwe:
                        waveform = _run_base_vocoder(voc_mod, decoded_audio).squeeze(0).float()
                        sampling_rate = getattr(voc_mod, "input_sampling_rate", voc_mod.output_sampling_rate)
                    else:
                        waveform = _run_vocoder_with_bwe(voc_mod, decoded_audio).squeeze(0).float()
                        sampling_rate = voc_mod.output_sampling_rate

                    return Audio(waveform=waveform, sampling_rate=sampling_rate)

        self._pipeline.audio_decoder = _ResidentAudioBlock()
        self._patched_pipeline_call = self._pipeline.__call__
        logger.info("Resident audio installed. dtype=%s, disable_bwe=%s", dtype, disable_bwe)

    def _hook_denoising_loop_timing(self) -> None:
        """Monkey-patch euler_denoising_loop in both samplers AND blocks modules."""
        import time
        import ltx_pipelines.utils.samplers as samplers_mod
        import ltx_pipelines.utils.blocks as blocks_mod

        ctx = self.ctx
        pcfg = self.pcfg
        original_loop = samplers_mod.euler_denoising_loop

        def timed_euler_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
            import torch.distributed as dist
            # Keep initial noise identical across ranks before sequence-parallel denoising.
            if dist.is_initialized() and dist.get_world_size() > 1:
                if video_state is not None and getattr(video_state, "latent", None) is not None:
                    dist.broadcast(video_state.latent, src=0)
                if audio_state is not None and getattr(audio_state, "latent", None) is not None:
                    dist.broadcast(audio_state.latent, src=0)

            if ctx is not None:
                ctx.synchronize()
            if dist.is_initialized():
                dist.barrier()
            t0 = time.perf_counter()
            
            result = original_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser)

            if ctx is not None:
                ctx.synchronize()
            if dist.is_initialized():
                dist.barrier()
            elapsed = time.perf_counter() - t0
            n_steps = len(sigmas) - 1
            if pcfg.is_main:
                logger.info("[DiT Loop] %d steps in %.3fs (%.3fs/step)", n_steps, elapsed, elapsed / max(1, n_steps))
            return result

        samplers_mod.euler_denoising_loop = timed_euler_loop
        blocks_mod.euler_denoising_loop = timed_euler_loop

    def _hook_vae_parallel(self) -> None:
        """Keep decoder resident and replace the block with a callable VAE-parallel proxy."""
        from ltx_npu.vae_parallel import VAEParallelContext

        vae_ctx = VAEParallelContext(
            world_size=self.pcfg.world_size,
            rank=self.pcfg.rank,
            device=self.ctx.get_device(),
        )

        video_decoder_block = self._pipeline.video_decoder
        logger.info("Building video decoder on device (resident VAE parallel mode)...")
        self._resident_video_decoder = (
            video_decoder_block._decoder_builder
            .build(device=video_decoder_block._device, dtype=video_decoder_block._dtype)
            .to(video_decoder_block._device)
            .eval()
        )

        self._pipeline.video_decoder = _ResidentParallelVideoDecoder(
            block=video_decoder_block,
            decoder=self._resident_video_decoder,
            vae_ctx=vae_ctx,
            profiler=self.profiler,
        )
        logger.info(
            "Hooked resident VAE parallel decode (grid=%dx%d) on %s",
            vae_ctx.h_split,
            vae_ctx.w_split,
            video_decoder_block._device,
        )

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio] | None:
        video, audio = self._patched_pipeline_call(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if hasattr(self, "_fsdp_transformer"):
            clear_freqs_cache_on_model(self._fsdp_transformer)

        if self.pcfg.is_main:
            return video, audio
        return None

    def run_with_timing(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        warmup: bool = True,
    ) -> tuple[Iterator[torch.Tensor], Audio] | None:
        """Run inference with optional warmup and timing report."""

        call_kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images,
            tiling_config=tiling_config, enhance_prompt=enhance_prompt,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if warmup:
            logger.info("Running warmup inference (not timed)...")
            result = self._patched_pipeline_call(**call_kwargs)
            if result is not None:
                # Consume video iterator to ensure full execution
                for _ in result[0]:
                    pass
            if dist.is_initialized():
                dist.barrier()
            logger.info("Warmup complete.")

        self.timer = TimingReport(self.ctx)
        self.profiler.reset()
        global _ACTIVE_PIPELINE_PROFILER
        prev_profiler = _ACTIVE_PIPELINE_PROFILER
        _ACTIVE_PIPELINE_PROFILER = self.profiler
        try:
            with self.timer.stage("Total Inference"):
                result = self._patched_pipeline_call(**call_kwargs)
                if self.pcfg.is_main and result is not None:
                    video_chunks = list(result[0])
                    audio = result[1]
        finally:
            _ACTIVE_PIPELINE_PROFILER = prev_profiler

        if self.pcfg.is_main:
            print(self.timer.report())
            if self.profile_enabled:
                print(self.profiler.report())
            return iter(video_chunks), audio
        return None
