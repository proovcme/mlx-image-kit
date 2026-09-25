#!/usr/bin/env python3
"""Direct, script-friendly image generation using the shared staged MLX engine."""

import sys
import traceback
from pathlib import Path

from mlx_image.types import Job, validate_job
from mlx_image.presentation import FriendlyArgumentParser


def parse_args(argv=None):
    parser = FriendlyArgumentParser(description="Generate an image locally with MLX")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output PNG path (default: output.png)")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels (default: 1024)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--cache", choices=("off", "balanced"), default="off", help="Denoising cache mode (default: off)")
    parser.add_argument("--model-path", type=Path, help="Local model snapshot directory; otherwise use the Hugging Face cache")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quiet", action="store_true", help="Print only the output path on success")
    modes.add_argument("--verbose", action="store_true", help="Show diagnostic details")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    from mlx_image.engine import run_jobs
    from mlx_image.presentation import report_failure, run_presented

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
    try:
        validate_job(job)
    except ValueError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    cache_config = None
    if args.cache != "off":
        from mlx_image.cli import _cache_config

        cache_config = _cache_config(args.cache)
    mode = "quiet" if args.quiet else "verbose" if args.verbose else "normal"
    try:
        summary = run_presented(run_jobs, [job], model_path=args.model_path, cache_config=cache_config, mode=mode)
    except KeyboardInterrupt:
        print("\nGeneration interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"✗ generation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    if summary.completed:
        return 0
    if summary.interrupted:
        print("\nGeneration interrupted.", file=sys.stderr)
        return 130
    for failure in summary.failed:
        report_failure(failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
