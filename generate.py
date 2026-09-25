#!/usr/bin/env python3
"""Direct, script-friendly image generation using the shared staged MLX engine."""

import argparse
from pathlib import Path

from mlx_image.types import Job


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image locally with MLX")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output PNG path (default: output.png)")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels (default: 1024)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--cache", choices=("off", "balanced"), default="off", help="Denoising cache mode (default: off)")
    parser.add_argument("--model-path", type=Path, help="Local model snapshot directory; otherwise use the Hugging Face cache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from mlx_image.engine import run_jobs

    job = Job(
        index=1,
        prompt=args.prompt,
        output=Path(args.output).expanduser().resolve(),
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        cache_mode=args.cache,
    )
    kwargs = {"model_path": args.model_path}
    if args.cache != "off":
        from mlx_image.cli import _cache_config

        kwargs["cache_config"] = _cache_config(args.cache)
    print("GENERATE")
    print(f"{job.width}×{job.height} · {job.steps} steps · seed {job.seed} · guidance {job.guidance} · cache {job.cache_mode}")
    summary = run_jobs([job], **kwargs)
    if summary.completed:
        result = summary.completed[0]
        print(f"Saved: {result.job.output}")
        print(f"Elapsed: {summary.elapsed_seconds:.2f} s | Seed: {result.job.seed} | Peak Metal: {result.peak_metal_gb:.2f} GB")
        return 0
    if summary.interrupted:
        print("Generation interrupted")
        return 130
    for failure in summary.failed:
        print(f"Generation failed: {failure.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
