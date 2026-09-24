"""Shared staged MLX generation engine for direct, interactive, and batch use."""

from __future__ import annotations

import gc
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config import Config, ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.image_util import ImageUtil
from mlx_image.types import Failure, Job, Result, Summary, validate_job

MODEL_ID = "mlx-community/Qwen-Image-2.1-MLX-4bit"


@dataclass
class _StageState:
    job: Job
    embeds_path: Path
    latents_path: Path
    compute_seconds: float = 0.0
    ready: bool = True


def build_text_encoder_mapping() -> list[WeightTarget]:
    targets = [
        WeightTarget(to_pattern="embed_tokens.weight", from_pattern=["language_model.model.embed_tokens.weight"]),
        WeightTarget(to_pattern="embed_tokens.scales", from_pattern=["language_model.model.embed_tokens.scales"]),
        WeightTarget(to_pattern="embed_tokens.biases", from_pattern=["language_model.model.embed_tokens.biases"]),
        WeightTarget(to_pattern="norm.weight", from_pattern=["language_model.model.norm.weight"]),
    ]
    for layer in range(36):
        src_prefix = f"language_model.model.layers.{layer}"
        dst_prefix = f"layers.{layer}"
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ):
            targets.append(WeightTarget(to_pattern=f"{dst_prefix}.{name}", from_pattern=[f"{src_prefix}.{name}"]))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for suffix in ("weight", "scales", "biases"):
                name = f"self_attn.{proj}.{suffix}"
                targets.append(WeightTarget(to_pattern=f"{dst_prefix}.{name}", from_pattern=[f"{src_prefix}.{name}"]))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                name = f"mlp.{proj}.{suffix}"
                targets.append(WeightTarget(to_pattern=f"{dst_prefix}.{name}", from_pattern=[f"{src_prefix}.{name}"]))
    return targets


def _snapshot(model_path: Path | None) -> Path:
    snapshot = model_path.expanduser().resolve() if model_path else Path(snapshot_download(repo_id=MODEL_ID))
    for component in ("text_encoder", "transformer", "vae"):
        if not (snapshot / component / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing {component} model weights")
    return snapshot


def _load_text_encoder(snapshot: Path):
    raw_te_weights = mx.load(str(snapshot / "text_encoder" / "model.safetensors"))
    te = Qwen21TextEncoder(
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
        head_dim=128,
        rms_norm_eps=1e-6,
    )
    nn.quantize(te, group_size=64, bits=4, mode="affine")
    mapped_te_dict = WeightMapper.apply_mapping(raw_te_weights, build_text_encoder_mapping())
    te.update(mapped_te_dict, strict=True)
    mx.eval(te.parameters())
    del raw_te_weights, mapped_te_dict
    gc.collect()
    return te


def _load_transformer(snapshot: Path):
    tr_data = mx.load(str(snapshot / "transformer" / "model.safetensors"))
    tr_items = []
    for k, v in tr_data.items():
        target_k = k
        if k.startswith("modulation.0."):
            target_k = k.replace("modulation.0.", "modulation.layers.1.")
        elif k.startswith("time_text_embed.linear_"):
            target_k = k.replace("time_text_embed.linear_", "time_text_embed.timestep_embedder.linear_")
        tr_items.append((target_k, v))
    del tr_data
    tr = Qwen21Transformer()
    nn.quantize(tr, group_size=64, bits=4)
    tr.update(tree_unflatten(tr_items), strict=True)
    del tr_items
    gc.collect()
    mx.eval(tr.parameters())
    return tr


def _load_vae(snapshot: Path):
    vae_data = mx.load(str(snapshot / "vae" / "model.safetensors"))
    vae = Qwen21VAE()
    vae_params = dict(tree_flatten(vae.parameters()))
    vae_items = []
    for k, v in vae_data.items():
        target_k = k
        if target_k.endswith(".gamma"):
            target_k = target_k[:-6] + ".weight"
            if v.ndim == 4:
                v = v.squeeze()
        elif target_k.endswith(".beta"):
            target_k = target_k[:-5] + ".bias"
            if v.ndim == 4:
                v = v.squeeze()
        target_k = target_k.replace(".downsampler.resample.1.", ".downsampler.conv.")
        target_k = target_k.replace(".upsampler.resample.1.", ".upsampler.conv.")
        if target_k not in vae_params:
            for suffix in (".weight", ".bias"):
                if target_k.endswith(suffix):
                    alt = target_k[:-len(suffix)] + ".conv" + suffix
                    if alt in vae_params:
                        target_k = alt
                        break
        if target_k in vae_params and v.shape == vae_params[target_k].shape:
            vae_items.append((target_k, v))
    del vae_data
    vae.update(tree_unflatten(vae_items), strict=False)
    del vae_items
    gc.collect()
    mx.eval(vae.parameters())
    return vae


def _denoise(job: Job, tr, embeds, mask, model_config: ModelConfig):
    config = Config(
        width=job.width,
        height=job.height,
        guidance=job.guidance,
        scheduler="linear",
        model_config=model_config,
        num_inference_steps=job.steps,
    )
    latents = LatentCreator.create_for_txt2img_or_img2img(
        seed=job.seed,
        width=config.width,
        height=config.height,
        img2img=Img2Img(
            vae=None,
            latent_creator=Qwen21LatentCreator,
            sigmas=config.scheduler.sigmas,
            init_time_step=config.init_time_step,
            image_path=config.image_path,
            tiling_config=None,
        ),
    ).astype(ModelConfig.precision)
    mx.eval(latents)
    for t in config.time_steps:
        latents_scaled = config.scheduler.scale_model_input(latents, t)
        noise = tr(
            t=t,
            config=config,
            hidden_states=latents_scaled,
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=mask,
        )
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        mx.eval(latents)
    return latents


def _save_png(decoded, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".mlx-image-{uuid.uuid4().hex}.png"
    try:
        ImageUtil.to_pil(decoded).save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run_jobs(
    jobs: Sequence[Job],
    *,
    model_path: Path | None = None,
    on_complete: Callable[[Result], None] | None = None,
    on_failure: Callable[[Failure], None] | None = None,
    progress: bool = True,
) -> Summary:
    """Run sequential jobs with one load of each heavy model and disk-spilled intermediates."""
    summary = Summary(total=len(jobs))
    started = time.monotonic()
    states: list[_StageState] = []

    def fail(index: int, stage: str, exc: Exception) -> None:
        failure = Failure(index, f"{stage}: {type(exc).__name__}")
        summary.failed.append(failure)
        if on_failure:
            on_failure(failure)

    for job in jobs:
        try:
            validate_job(job)
        except Exception as exc:
            fail(job.index, "validation", exc)

    invalid = {f.index for f in summary.failed}
    valid = [job for job in jobs if job.index not in invalid]
    if not valid:
        summary.elapsed_seconds = time.monotonic() - started
        return summary

    mx.reset_peak_memory()
    with tempfile.TemporaryDirectory(prefix="mlx-image-") as temporary_dir:
        spool = Path(temporary_dir)
        states = [_StageState(job, spool / f"{job.index}-embeds.npz", spool / f"{job.index}-latents.npz") for job in valid]
        stage = "model setup"
        try:
            snapshot = _snapshot(model_path)
            model_config = ModelConfig.qwen_image_21()

            stage = "text encoder"
            if progress:
                print("Loading text encoder once...")
            te = _load_text_encoder(snapshot)
            qwen = QwenImage21.__new__(QwenImage21)
            super(QwenImage21, qwen).__init__()
            Qwen21Initializer._init_config(qwen, model_config)
            Qwen21Initializer._init_tokenizers(qwen, str(snapshot))
            try:
                for state in states:
                    embeds = mask = None
                    try:
                        began = time.monotonic()
                        embeds, mask = Qwen21PromptEncoder.encode_prompt(
                            prompt=state.job.prompt,
                            prompt_cache=qwen.prompt_cache,
                            tokenizer=qwen.tokenizers["qwen21"],
                            text_encoder=te,
                        )
                        mx.eval(embeds, mask)
                        mx.savez(str(state.embeds_path), embeds=embeds, mask=mask)
                        state.compute_seconds += time.monotonic() - began
                    except Exception as exc:
                        state.ready = False
                        fail(state.job.index, "prompt encoding", exc)
                    finally:
                        qwen.prompt_cache.clear()
                        # Keep no frame-local reference to model outputs between jobs.
                        del embeds, mask
                        gc.collect()
                        mx.clear_cache()
            finally:
                del te, qwen
                gc.collect()
                mx.clear_cache()

            stage = "transformer"
            if progress:
                print("Loading 4-bit transformer once...")
            tr = _load_transformer(snapshot)
            try:
                for state in states:
                    if not state.ready:
                        continue
                    data = latents = None
                    try:
                        if progress:
                            print(f"Denoising job {state.job.index}/{summary.total} ({state.job.steps} steps, seed {state.job.seed})...")
                        began = time.monotonic()
                        data = mx.load(str(state.embeds_path))
                        latents = _denoise(state.job, tr, data["embeds"], data["mask"], model_config)
                        mx.savez(str(state.latents_path), latents=latents)
                        state.compute_seconds += time.monotonic() - began
                    except Exception as exc:
                        state.ready = False
                        fail(state.job.index, "denoising", exc)
                    finally:
                        state.embeds_path.unlink(missing_ok=True)
                        del data, latents
                        gc.collect()
                        mx.clear_cache()
            finally:
                del tr
                gc.collect()
                mx.clear_cache()

            stage = "VAE"
            if progress:
                print("Loading VAE once...")
            vae = _load_vae(snapshot)
            try:
                for state in states:
                    if not state.ready:
                        continue
                    data = unpacked = decoded = None
                    try:
                        began = time.monotonic()
                        data = mx.load(str(state.latents_path))
                        unpacked = Qwen21LatentCreator.unpack_latents(
                            latents=data["latents"], height=state.job.height, width=state.job.width
                        )
                        decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=None)
                        mx.eval(decoded)
                        _save_png(decoded, state.job.output)
                        state.compute_seconds += time.monotonic() - began
                        result = Result(
                            job=state.job,
                            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                            elapsed_seconds=state.compute_seconds,
                            peak_metal_gb=mx.get_peak_memory() / (1024**3),
                        )
                        summary.completed.append(result)
                        if on_complete:
                            on_complete(result)
                    except Exception as exc:
                        state.ready = False
                        fail(state.job.index, "VAE or save", exc)
                    finally:
                        state.latents_path.unlink(missing_ok=True)
                        del data, unpacked, decoded
                        gc.collect()
                        mx.clear_cache()
            finally:
                del vae
                gc.collect()
                mx.clear_cache()
        except KeyboardInterrupt:
            summary.interrupted = True
        except Exception as exc:
            for state in states:
                if state.ready and all(result.job.index != state.job.index for result in summary.completed):
                    state.ready = False
                    fail(state.job.index, stage, exc)
        finally:
            gc.collect()
            mx.clear_cache()
    summary.elapsed_seconds = time.monotonic() - started
    return summary
