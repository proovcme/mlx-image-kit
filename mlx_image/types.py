"""Lightweight job and result types shared by CLI and MLX engine."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Job:
    index: int
    prompt: str
    output: Path
    width: int
    height: int
    steps: int
    seed: int
    guidance: float


@dataclass(frozen=True)
class Result:
    job: Job
    timestamp: str
    elapsed_seconds: float
    peak_metal_gb: float


@dataclass(frozen=True)
class Failure:
    index: int
    message: str


@dataclass
class Summary:
    total: int
    completed: list[Result] = field(default_factory=list)
    failed: list[Failure] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    interrupted: bool = False


def validate_job(job: Job) -> None:
    if not job.prompt.strip():
        raise ValueError("empty prompt")
    if job.width <= 0 or job.height <= 0 or job.width % 16 or job.height % 16:
        raise ValueError("width and height must be positive multiples of 16")
    if job.steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= job.seed <= 0xFFFFFFFF:
        raise ValueError("seed must be between 0 and 4294967295")
    if not 0 <= job.guidance < float("inf"):
        raise ValueError("guidance must be a finite nonnegative number")
    if job.output.suffix.lower() != ".png":
        raise ValueError("output must be a PNG path")
