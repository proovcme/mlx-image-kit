"""One terminal presentation layer for direct, interactive, and batch runs."""

from __future__ import annotations

import contextlib
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence, TextIO

from mlx_image.types import Failure, Job, Result, Summary


class FriendlyArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, f"✗ {message}\n")


def _duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    return f"{seconds:.1f} s"


def _version() -> str:
    # Package metadata is the only version source.
    from mlx_image.cli import _version as package_version

    return package_version()


class Presenter:
    def __init__(self, jobs: Sequence[Job], *, mode: str = "normal", batch: bool = False,
                 stream: TextIO | None = None, clock: Callable[[], float] = time.monotonic):
        self.jobs = jobs
        self.mode = mode
        self.batch = batch
        self.stream = stream or sys.stdout
        self.clock = clock
        self.tty = bool(self.stream.isatty())
        self.started = clock()
        self.stage_started: dict[str, float] = {}
        self.timings: dict[int, dict[str, float]] = {}
        self.counts: dict[int, tuple[int, int]] = {}
        self.saved: list[Path] = []
        self._line_open = False
        self._last_reported: dict[int, int] = {}
        self._current_job = 0

    def _print(self, line: str = "") -> None:
        if self.mode == "quiet":
            return
        if self._line_open:
            self.stream.write("\n")
            self._line_open = False
        print(line, file=self.stream, flush=True)

    def begin(self, *, model_path: Path | None = None) -> None:
        if self.mode == "quiet":
            return
        if self.batch:
            first = self.jobs[0]
            self._print(f"MLX Image {_version()} · BATCH")
            self._print(f"{len(self.jobs)} images · {first.width}×{first.height} · {first.steps} steps · cache {first.cache_mode}")
        else:
            job = self.jobs[0]
            self._print(f"MLX Image {_version()}")
            self._print(f"Qwen-Image 2.1 · MLX Q4 · {job.cache_mode}")
            self._print()
            self._print(f"{job.width}×{job.height} · {job.steps} steps · seed {job.seed} · guidance {job.guidance}")
        if self.mode == "verbose":
            self._print(f"Model source: {model_path if model_path else 'Hugging Face cache'}")
        self._print("\nPreparing")
        self._print("  model          preparing")
        self.stage_started["model"] = self.clock()

    def event(self, name: str, index: int | None) -> None:
        now = self.clock()
        if name == "model_ready":
            self._print(f"  ✓ model        ready ({_duration(now - self.stage_started.get('model', now))})")
            self.stage_started["text_encoder"] = now
        elif name == "text_encoder_ready":
            self._print(f"  ✓ text encoder {_duration(now - self.stage_started.get('text_encoder', now))}")
        elif name == "prompt_encoded":
            if self.batch:
                done = sum("prompt_encoding" in values for values in self.timings.values())
                self._print(f"  encoded       {done}/{len(self.jobs)}") if done == len(self.jobs) else None
            else:
                seconds = self.timings.get(index, {}).get("prompt_encoding")
                self._print(f"  ✓ prompt       {_duration(seconds)}" if seconds is not None else "  ✓ prompt       ready")
        elif name == "transformer_ready":
            self._print("\nGenerating")
        elif name == "denoising_start":
            self._current_job = index or 0
            self.stage_started["denoising"] = now
            if self.batch:
                position = next((n for n, job in enumerate(self.jobs, 1) if job.index == index), index)
                self._print(f"  image {position}/{len(self.jobs)}")
        elif name == "vae_ready":
            self._print("\nDecoding")
        elif name == "decoding_start" and self.mode == "verbose":
            self._print(f"  decoding job {index}")
        if self.mode == "verbose" and name in {"transformer_ready", "vae_ready"}:
            self._print(f"  loader        {name.replace('_ready', '')} ready")

    def timing(self, index: int, name: str, seconds: float) -> None:
        self.timings.setdefault(index, {})[name] = seconds

    def forward(self, index: int, forward: bool) -> None:
        computed, skipped = self.counts.get(index, (0, 0))
        if forward:
            computed += 1
        else:
            skipped += 1
        self.counts[index] = (computed, skipped)

    def step(self, index: int, step: int, total: int) -> None:
        if self.mode == "quiet":
            return
        job = next(job for job in self.jobs if job.index == index)
        computed, skipped = self.counts.get(index, (step, 0))
        elapsed = _duration(self.clock() - self.stage_started.get("denoising", self.clock()))
        stats = f" · computed {computed} · skipped {skipped}" if job.cache_mode == "balanced" else ""
        if self.tty:
            filled = round(20 * step / total)
            line = f"  {'█' * filled}{'░' * (20 - filled)}  {step}/{total} · elapsed {elapsed}{stats}"
            self.stream.write("\r" + line + "  ")
            self.stream.flush()
            self._line_open = step != total
            if step == total:
                self.stream.write("\n")
        elif step == total or step >= self._last_reported.get(index, 0) + max(1, total // 4):
            self._last_reported[index] = step
            self._print(f"  Denoising {step}/{total} · elapsed {elapsed}{stats}")

    def complete(self, result: Result) -> None:
        self.saved.append(result.job.output)
        if self.mode == "quiet":
            print(result.job.output, file=self.stream, flush=True)
            return
        if self.batch:
            self._print(f"  saved         {len(self.saved)}/{len(self.jobs)}")
            return
        vae_seconds = self.timings.get(result.job.index, {}).get("vae_decode")
        if vae_seconds is not None:
            self._print(f"  ✓ VAE          {_duration(vae_seconds)}")

    def finish(self, summary: Summary, *, elapsed: float | None = None) -> None:
        if self.mode == "quiet":
            return
        if self._line_open:
            self._print()
        wall = elapsed if elapsed is not None else summary.elapsed_seconds
        if self.batch:
            self._print(f"\nCompleted  {len(summary.completed)}")
            self._print(f"Failed     {len(summary.failed)}")
            self._print(f"Elapsed    {_duration(wall)}")
            peaks = [r.peak_metal_gb for r in summary.completed if math.isfinite(r.peak_metal_gb) and r.peak_metal_gb > 0]
            if peaks:
                self._print(f"Peak Metal {max(peaks):.2f} GB")
            return
        if not summary.completed:
            return
        result = summary.completed[0]
        times = self.timings.get(result.job.index, {})
        self._print(f"\n✓ Saved  {result.job.output}")
        self._print(f"  total       {_duration(wall)}")
        for key, label in (("prompt_encoding", "prompt"), ("denoising", "denoising"), ("vae_decode", "VAE")):
            if key in times:
                self._print(f"  {label:<11} {_duration(times[key])}")
        if result.job.cache_mode == "balanced":
            computed, skipped = self.counts.get(result.job.index, (result.job.steps, 0))
            self._print(f"  forwards    {computed}/{result.job.steps}")
            self._print(f"  skipped     {skipped}/{result.job.steps}")
        if math.isfinite(result.peak_metal_gb) and result.peak_metal_gb > 0:
            self._print(f"  peak Metal  {result.peak_metal_gb:.2f} GB")


def run_presented(
    runner: Callable,
    jobs: Sequence[Job],
    *,
    model_path: Path | None = None,
    cache_config=None,
    mode: str = "normal",
    batch: bool = False,
    on_complete: Callable[[Result], None] | None = None,
    extra_failures: Sequence[Failure] = (),
) -> Summary:
    presenter = Presenter(jobs, mode=mode, batch=batch)
    presenter.begin(model_path=model_path)

    def complete(result: Result) -> None:
        if on_complete:
            on_complete(result)
        presenter.complete(result)

    # Official hub switch is scoped; exceptions propagate out of this context.
    if mode == "verbose":
        quiet_hub = contextlib.nullcontext()
    else:
        from huggingface_hub.utils import disable_progress_bars

        quiet_hub = disable_progress_bars()
    with quiet_hub:
        kwargs = {
            "model_path": model_path,
            "on_complete": complete,
            "on_stage_event": presenter.event,
            "on_stage_timing": presenter.timing,
            "on_forward": presenter.forward if any(job.cache_mode == "balanced" for job in jobs) else None,
            "on_step": presenter.step,
            "progress": False,
            "show_library_progress": mode == "verbose",
        }
        if cache_config is not None:
            kwargs["cache_config"] = cache_config
        summary = runner(jobs, **kwargs)
    saved = set(presenter.saved)
    for result in summary.completed:
        if result.job.output not in saved:
            complete(result)
            saved.add(result.job.output)
    displayed = Summary(
        total=summary.total + len(extra_failures),
        completed=summary.completed,
        failed=list(extra_failures) + summary.failed,
        elapsed_seconds=summary.elapsed_seconds,
        interrupted=summary.interrupted,
    )
    presenter.finish(displayed)
    return summary


def report_failure(failure: Failure, *, stream: TextIO | None = None) -> None:
    print(f"✗ generation failed: {failure.message}", file=stream or sys.stderr)
