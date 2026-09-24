"""Small interactive and batch CLI around the shared staged engine."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from mlx_image.types import Failure, Job, Result, Summary, validate_job

MAX_SEED = 0xFFFFFFFF
PRESETS = {"portrait": (768, 1152), "landscape": (1152, 768), "square": (1024, 1024)}


def _run_jobs(*args, **kwargs) -> Summary:
    from mlx_image.engine import run_jobs

    return run_jobs(*args, **kwargs)


def _seed(value: str | int) -> int | None:
    if value == "random":
        return None
    if type(value) is not int and (not isinstance(value, str) or not value.isdecimal()):
        raise ValueError("seed must be an integer or random")
    seed = int(value)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be between 0 and 4294967295")
    return seed


def _actual_seed(value: int | None, variation: int = 0) -> int:
    return secrets.randbits(32) if value is None else (value + variation) & MAX_SEED


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise ValueError("size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("width and height must be positive multiples of 16")
    return width, height


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _unique_output(directory: Path, reserved: set[Path], requested: str | None = None) -> Path:
    if requested is None:
        base = directory / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    else:
        supplied = Path(requested).expanduser()
        if supplied.suffix.lower() != ".png":
            raise ValueError("output must be a PNG path")
        base = supplied if supplied.is_absolute() else directory / supplied
    base = base.resolve()
    candidate = base
    suffix = 2
    while candidate in reserved or candidate.exists():
        candidate = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
        suffix += 1
    reserved.add(candidate)
    return candidate


class History:
    """Private local JSONL with atomic replacement after each completed image."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path.cwd() / ".history" / "history.jsonl"

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, result: Result) -> None:
        record = {
            "timestamp": result.timestamp,
            "prompt": result.job.prompt,
            "output": str(result.job.output),
            "width": result.job.width,
            "height": result.job.height,
            "steps": result.job.steps,
            "seed": result.job.seed,
            "guidance": result.job.guidance,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "peak_metal_gb": round(result.peak_metal_gb, 3),
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        previous = self.path.read_bytes() if self.path.exists() else b""
        # Parse existing lines before replacement, so a corrupt file is never extended silently.
        for line in previous.splitlines():
            if line.strip():
                json.loads(line)
        payload = previous + (b"" if not previous or previous.endswith(b"\n") else b"\n")
        payload += (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def ensure(self, results: list[Result]) -> None:
        known_outputs = {record.get("output") for record in self.read()}
        for result in results:
            output = str(result.job.output)
            if output not in known_outputs:
                self.append(result)
                known_outputs.add(output)


@dataclass
class Settings:
    width: int = 1152
    height: int = 768
    steps: int = 20
    seed: int | None = None
    guidance: float = 1.0


class InteractiveSession:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        output_dir: Path | None = None,
        history: History | None = None,
        runner: Callable | None = None,
        opener: Callable = subprocess.run,
    ):
        self.model_path = model_path
        self.output_dir = output_dir or Path.cwd() / "outputs"
        self.history = history or History()
        self.settings = Settings()
        self.runner = runner or _run_jobs
        self.opener = opener
        self.reserved: set[Path] = set()

    def status(self) -> None:
        s = self.settings
        print(f"{s.width}x{s.height} | {s.steps} steps | seed {s.seed if s.seed is not None else 'random'} | guidance {s.guidance}")

    def _last(self) -> dict | None:
        try:
            records = self.history.read()
        except (OSError, ValueError):
            print("Local history could not be read")
            return None
        return records[-1] if records else None

    def _show_record(self, record: dict) -> None:
        print(f"{record['timestamp']} | {record['width']}x{record['height']} | {record['steps']} steps | seed {record['seed']} | guidance {record['guidance']}")
        print(f"Output: {record['output']}")

    def _record(self, result: Result) -> None:
        try:
            self.history.append(result)
        except (OSError, ValueError):
            print("Local history could not be saved")

    def generate(self, prompt: str, *, repeat: dict | None = None) -> Summary:
        if repeat:
            width, height = int(repeat["width"]), int(repeat["height"])
            steps, seed, guidance = int(repeat["steps"]), int(repeat["seed"]), float(repeat["guidance"])
        else:
            s = self.settings
            width, height = s.width, s.height
            steps, seed, guidance = s.steps, _actual_seed(s.seed), s.guidance
        output = _unique_output(self.output_dir, self.reserved)
        job = Job(1, prompt, output, width, height, steps, seed, guidance)
        print(f"Generating {width}x{height} | {steps} steps | seed {seed} | guidance {guidance}")
        summary = self.runner([job], model_path=self.model_path, on_complete=self._record)
        try:
            self.history.ensure(summary.completed)
        except (OSError, ValueError):
            print("Local history could not be saved")
        if summary.completed:
            result = summary.completed[0]
            print(f"Saved: {os.path.relpath(result.job.output)}")
            print(f"Elapsed: {summary.elapsed_seconds:.2f} s | Seed: {seed} | Peak Metal: {result.peak_metal_gb:.2f} GB")
        elif summary.interrupted:
            print("Generation interrupted")
        else:
            for failure in summary.failed:
                print(f"Generation failed: {failure.message}")
        return summary

    def handle(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            self.generate(line)
            return True
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        name = command[1:].lower()
        try:
            if name in PRESETS and not argument:
                self.settings.width, self.settings.height = PRESETS[name]
                self.status()
            elif name == "size":
                self.settings.width, self.settings.height = _size(argument)
                self.status()
            elif name == "steps":
                value = _positive_int(argument)
                self.settings.steps = value
                self.status()
            elif name == "seed":
                self.settings.seed = _seed(argument)
                self.status()
            elif name == "guidance":
                value = float(argument)
                if not 0 <= value < float("inf"):
                    raise ValueError("guidance must be finite and nonnegative")
                self.settings.guidance = value
                self.status()
            elif name == "status" and not argument:
                self.status()
            elif name == "last" and not argument:
                record = self._last()
                if record:
                    self._show_record(record)
                else:
                    print("No completed generation")
            elif name == "history" and not argument:
                records = self.history.read()
                if not records:
                    print("No completed generation")
                for record in records[-10:]:
                    self._show_record(record)
            elif name == "repeat" and not argument:
                record = self._last()
                if record:
                    self.generate(record["prompt"], repeat=record)
                else:
                    print("No completed generation")
            elif name == "open" and not argument:
                record = self._last()
                if record and Path(record["output"]).is_file():
                    self.opener(["open", record["output"]], check=False)
                else:
                    print("No completed PNG to open")
            elif name == "help" and not argument:
                print("/portrait /landscape /square /size WIDTHxHEIGHT /steps N /seed N|random /guidance X")
                print("/status /last /repeat /history /open /help /quit")
            elif name == "quit" and not argument:
                return False
            else:
                print("Unknown command; use /help")
        except (ValueError, argparse.ArgumentTypeError, OSError):
            print("Invalid command value")
        return True

    def run(self) -> int:
        print("MLX IMAGE")
        print("Qwen-Image 2.1 · 4-bit")
        self.status()
        print("Type a prompt or /help")
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("\nInput cancelled")
                continue
            if not self.handle(line):
                return 0


def _batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-image batch", description="Sequential staged generation from TXT or JSONL")
    parser.add_argument("file", type=Path, help="TXT (one prompt per line) or JSONL job file")
    preset = parser.add_mutually_exclusive_group()
    for name in PRESETS:
        preset.add_argument(f"--{name}", action="store_true")
    preset.add_argument("--size", help="WIDTHxHEIGHT")
    parser.add_argument("--steps", type=_positive_int, default=20)
    parser.add_argument("--seed", default="random", help="Integer or random")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--count", type=_positive_int, default=1, help="Images per prompt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, help="Local model snapshot directory")
    return parser


def parse_batch_jobs(args: argparse.Namespace) -> tuple[list[Job], list[Failure]]:
    if args.file.suffix.lower() not in (".txt", ".jsonl"):
        raise ValueError("input must be .txt or .jsonl")
    if not 0 <= args.guidance < float("inf"):
        raise ValueError("guidance must be finite and nonnegative")
    base_seed = _seed(args.seed)
    if args.size:
        base_width, base_height = _size(args.size)
    else:
        preset = next((name for name in PRESETS if getattr(args, name)), "landscape")
        base_width, base_height = PRESETS[preset]
    jobs: list[Job] = []
    failures: list[Failure] = []
    reserved: set[Path] = set()
    next_index = 1
    allowed = {"prompt", "output", "width", "height", "steps", "seed", "guidance"}
    for line_no, line in enumerate(args.file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            if args.file.suffix.lower() == ".txt":
                fields = {"prompt": line.strip()}
            else:
                try:
                    fields = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSON") from exc
                if not isinstance(fields, dict) or set(fields) - allowed:
                    raise ValueError("unsupported job fields")
            prompt = fields.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("missing prompt")
            width = fields.get("width", base_width)
            height = fields.get("height", base_height)
            steps = fields.get("steps", args.steps)
            guidance = fields.get("guidance", args.guidance)
            if any(type(value) is not int for value in (width, height, steps)) or type(guidance) not in (int, float):
                raise ValueError("invalid numeric job field")
            width, height, steps, guidance = int(width), int(height), int(steps), float(guidance)
            chosen_seed = _seed(fields.get("seed", base_seed if base_seed is not None else "random"))
            requested = fields.get("output")
            if requested is not None and (not isinstance(requested, str) or not requested):
                raise ValueError("invalid output path")
            # Validate before reserving names; a bad job must not consume output paths.
            validate_job(Job(next_index, prompt, Path("output.png"), width, height, steps, 0, guidance))
            used_seeds: set[int] = set()
            for variation in range(args.count):
                output = _unique_output(args.output_dir, reserved, requested)
                actual_seed = _actual_seed(chosen_seed, variation)
                if chosen_seed is None:
                    while actual_seed in used_seeds:
                        actual_seed = _actual_seed(None)
                used_seeds.add(actual_seed)
                job = Job(next_index, prompt, output, width, height, steps, actual_seed, guidance)
                jobs.append(job)
                next_index += 1
        except (TypeError, ValueError, OverflowError):
            failures.append(Failure(next_index, f"input line {line_no}: invalid job"))
            next_index += 1
    return jobs, failures


def batch_main(argv: list[str]) -> int:
    parser = _batch_parser()
    args = parser.parse_args(argv)
    began = time.monotonic()
    try:
        jobs, parse_failures = parse_batch_jobs(args)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Batch input error: {type(exc).__name__}")
        return 2
    if not jobs and not parse_failures:
        print("No jobs in input")
        return 2
    history = History()

    def record(result: Result) -> None:
        try:
            history.append(result)
        except (OSError, ValueError):
            print(f"Job {result.job.index}: local history could not be saved")

    summary = _run_jobs(jobs, model_path=args.model_path, on_complete=record) if jobs else Summary(total=0)
    try:
        history.ensure(summary.completed)
    except (OSError, ValueError):
        print("Local history could not be saved")
    all_failures = parse_failures + summary.failed
    for failure in all_failures:
        print(f"Job {failure.index}: {failure.message}")
    print(f"Total: {len(jobs) + len(parse_failures)} | Completed: {len(summary.completed)} | Failed: {len(all_failures)} | Elapsed: {time.monotonic() - began:.2f} s")
    if summary.interrupted:
        print("Batch interrupted; completed PNGs and history are retained")
        return 130
    return 1 if all_failures else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":
        return batch_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="mlx-image", description="Interactive local MLX image generation", epilog="Batch mode: mlx-image batch FILE --help"
    )
    parser.add_argument("--model-path", type=Path, help="Local model snapshot directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    return InteractiveSession(model_path=args.model_path, output_dir=args.output_dir).run()
