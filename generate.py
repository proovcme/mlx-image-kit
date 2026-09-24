#!/usr/bin/env python3
"""
CLI Image Generator for Qwen-Image-2.1 using Native MLX 4-bit Pipeline
"""

import argparse
import gc
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from huggingface_hub import snapshot_download

from mflux.models.common.config import ModelConfig, Config
from mflux.models.common.latent_creator.latent_creator import LatentCreator, Img2Img
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.image_util import ImageUtil


def get_swap_info() -> Dict[str, float]:
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        m = re.search(r"total\s*=\s*([\d\.]+)M\s+used\s*=\s*([\d\.]+)M\s+free\s*=\s*([\d\.]+)M", out)
        if m:
            return {"total_mb": float(m.group(1)), "used_mb": float(m.group(2)), "free_mb": float(m.group(3))}
    except Exception:
        pass
    return {"total_mb": 0.0, "used_mb": 0.0, "free_mb": 0.0}


def build_text_encoder_mapping() -> List[WeightTarget]:
    targets = [
        WeightTarget(to_pattern="embed_tokens.weight", from_pattern=["language_model.model.embed_tokens.weight"]),
        WeightTarget(to_pattern="embed_tokens.scales", from_pattern=["language_model.model.embed_tokens.scales"]),
        WeightTarget(to_pattern="embed_tokens.biases", from_pattern=["language_model.model.embed_tokens.biases"]),
        WeightTarget(to_pattern="norm.weight", from_pattern=["language_model.model.norm.weight"]),
    ]
    for layer in range(36):
        src_prefix = f"language_model.model.layers.{layer}"
        dst_prefix = f"layers.{layer}"

        targets.append(
            WeightTarget(
                to_pattern=f"{dst_prefix}.input_layernorm.weight",
                from_pattern=[f"{src_prefix}.input_layernorm.weight"],
            )
        )
        targets.append(
            WeightTarget(
                to_pattern=f"{dst_prefix}.post_attention_layernorm.weight",
                from_pattern=[f"{src_prefix}.post_attention_layernorm.weight"],
            )
        )
        targets.append(
            WeightTarget(
                to_pattern=f"{dst_prefix}.self_attn.q_norm.weight",
                from_pattern=[f"{src_prefix}.self_attn.q_norm.weight"],
            )
        )
        targets.append(
            WeightTarget(
                to_pattern=f"{dst_prefix}.self_attn.k_norm.weight",
                from_pattern=[f"{src_prefix}.self_attn.k_norm.weight"],
            )
        )

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            for suffix in ["weight", "scales", "biases"]:
                targets.append(
                    WeightTarget(
                        to_pattern=f"{dst_prefix}.self_attn.{proj}.{suffix}",
                        from_pattern=[f"{src_prefix}.self_attn.{proj}.{suffix}"],
                    )
                )

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for suffix in ["weight", "scales", "biases"]:
                targets.append(
                    WeightTarget(
                        to_pattern=f"{dst_prefix}.mlp.{proj}.{suffix}",
                        from_pattern=[f"{src_prefix}.mlp.{proj}.{suffix}"],
                    )
                )
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images locally using Qwen-Image-2.1 MLX 4-bit")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output PNG filepath (default: output.png)")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels (default: 1024)")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--model-path", type=Path, help="Local model snapshot directory; otherwise use the Hugging Face cache/download")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()

    print("=" * 80)
    print("QWEN-IMAGE-2.1 MLX 4-BIT: IMAGE GENERATION")
    print("=" * 80)
    print("Output: PNG file")
    print(f"Resolution:  {args.width}x{args.height} | Steps: {args.steps} | Seed: {args.seed} | Guidance: {args.guidance}")

    t_global_start = time.time()

    # 1. Baseline Environment
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    swap_init = get_swap_info()
    print(f"\nInitial Swap used: {swap_init['used_mb']:.1f} MB (Total: {swap_init['total_mb']:.1f} MB)")

    model_id = "mlx-community/Qwen-Image-2.1-MLX-4bit"
    snapshot = args.model_path.expanduser().resolve() if args.model_path else Path(snapshot_download(repo_id=model_id))
    for component in ("text_encoder", "transformer", "vae"):
        if not (snapshot / component / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {component}/model.safetensors in the model snapshot")

    # 2. Text Encoder: Load, Encode & Free
    print("\n[Phase 1/4] Text Encoder: Loading native Q4 weights...")
    t0_te = time.time()
    te_dir = snapshot / "text_encoder"
    raw_te_weights = mx.load(str(te_dir / "model.safetensors"))

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
    te_mapping = build_text_encoder_mapping()
    mapped_te_dict = WeightMapper.apply_mapping(raw_te_weights, te_mapping)
    te.update(mapped_te_dict, strict=True)
    mx.eval(te.parameters())
    del raw_te_weights, mapped_te_dict
    gc.collect()

    print("Encoding prompt...")
    model_config = ModelConfig.qwen_image_21()
    qwen = QwenImage21.__new__(QwenImage21)
    super(QwenImage21, qwen).__init__()
    Qwen21Initializer._init_config(qwen, model_config)
    Qwen21Initializer._init_tokenizers(qwen, str(snapshot))

    t0_enc = time.time()
    prompt_embeds, prompt_mask = Qwen21PromptEncoder.encode_prompt(
        prompt=args.prompt,
        prompt_cache=qwen.prompt_cache,
        tokenizer=qwen.tokenizers["qwen21"],
        text_encoder=te,
    )
    mx.eval(prompt_embeds, prompt_mask)
    t_prompt_encode = time.time() - t0_enc
    print(f"Prompt encoding completed in: {t_prompt_encode:.2f} s")
    print(f"Prompt embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

    # Immediately unload text encoder
    print("Unloading text encoder to release unified memory...")
    del te, qwen
    gc.collect()
    mx.clear_cache()
    print(f"Active Metal memory after text encoder release: {mx.get_active_memory() / (1024**3):.3f} GB")

    # 3. Transformer Denoising Loop
    print("\n[Phase 2/4] Transformer: Loading native 4-bit weights...")
    t0_tr = time.time()
    tr_path = snapshot / "transformer" / "model.safetensors"
    tr_data = mx.load(str(tr_path))
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
    print(f"Transformer loaded and evaluated in: {time.time() - t0_tr:.2f} s")

    print(f"\nInitializing diffusion latents (seed={args.seed}, {args.width}x{args.height})...")
    config = Config(
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        scheduler="linear",
        model_config=model_config,
        num_inference_steps=args.steps,
    )

    latents = LatentCreator.create_for_txt2img_or_img2img(
        seed=args.seed,
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

    print(f"\nStarting {args.steps}-step Denoising Loop...")
    t0_denoise = time.time()
    step_times = []

    for step_idx, t in enumerate(config.time_steps):
        step_num = step_idx + 1
        t_step_start = time.time()

        latents_scaled = config.scheduler.scale_model_input(latents, t)
        noise = tr(
            t=t,
            config=config,
            hidden_states=latents_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
        )
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        mx.eval(latents)
        step_dur = time.time() - t_step_start
        step_times.append(step_dur)

        active_gb = mx.get_active_memory() / (1024**3)
        peak_gb = mx.get_peak_memory() / (1024**3)
        cur_swap = get_swap_info()
        print(f"Step {step_num:2d}/{args.steps}: {step_dur:5.2f} s | Active Metal: {active_gb:.2f} GB | Peak Metal: {peak_gb:.2f} GB | Swap: {cur_swap['used_mb']:.0f} MB")

    t_denoise = time.time() - t0_denoise
    avg_step = (t_denoise / args.steps) if args.steps > 0 else 0.0
    print(f"\n{args.steps}-step Denoising finished in: {t_denoise:.2f} s (Average: {avg_step:.2f} s/step)")

    # Unload transformer to keep memory minimal for VAE
    print("Unloading transformer...")
    del tr
    gc.collect()
    mx.clear_cache()

    # 4. VAE Decode Phase
    print("\n[Phase 3/4] VAE: Loading weights & decoding latents...")
    t0_vae = time.time()
    vae_path = snapshot / "vae" / "model.safetensors"
    vae_data = mx.load(str(vae_path))
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
            for suffix in [".weight", ".bias"]:
                if target_k.endswith(suffix):
                    alt = target_k[:-len(suffix)] + ".conv" + suffix
                    if alt in vae_params:
                        target_k = alt
                        break
        if target_k in vae_params:
            if v.shape == vae_params[target_k].shape:
                vae_items.append((target_k, v))
    del vae_data

    vae.update(tree_unflatten(vae_items), strict=False)
    del vae_items
    gc.collect()
    mx.eval(vae.parameters())

    print("Unpacking latents and decoding to RGB...")
    unpacked_latents = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
    decoded = VAEUtil.decode(vae=vae, latent=unpacked_latents, tiling_config=None)
    mx.eval(decoded)
    t_vae_decode = time.time() - t0_vae
    print(f"VAE decode completed in: {t_vae_decode:.2f} s")

    # 5. Save PNG
    print("\n[Phase 4/4] Saving image...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = ImageUtil.to_pil(decoded)
    pil_image.save(str(output_path))
    print("Image saved")

    # Final metrics
    t_total = time.time() - t_global_start
    peak_metal_gb = mx.get_peak_memory() / (1024**3)
    swap_final = get_swap_info()
    swap_delta_mb = swap_final["used_mb"] - swap_init["used_mb"]
    steady_state_avg = (sum(step_times[1:]) / (len(step_times) - 1)) if len(step_times) > 1 else (step_times[0] if step_times else 0.0)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print("Output:                 PNG file")
    print(f"Image Resolution:       {pil_image.size[0]}x{pil_image.size[1]}")
    print(f"Prompt encoding time:   {t_prompt_encode:.2f} s")
    print(f"Denoising time:         {t_denoise:.2f} s ({args.steps} steps, steady-state avg: {steady_state_avg:.2f} s/step)")
    print(f"VAE decode time:        {t_vae_decode:.2f} s")
    print(f"Total time:             {t_total:.2f} s ({t_total / 60:.2f} min)")
    print(f"Peak Metal memory:      {peak_metal_gb:.2f} GB")
    print(f"Swap used:              {swap_final['used_mb']:.1f} MB (Delta: {swap_delta_mb:+.1f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
