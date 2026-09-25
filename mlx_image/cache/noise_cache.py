"""Conservative full-noise reuse gate for the Qwen-Image 2.1 denoising loop.

This is an experimental output-change predictor, not a port of TeaCache's
dual-stream residual cache or its old Qwen-Image coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class CacheConfig:
    threshold: float
    max_consecutive_skips: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("cache threshold must be finite and positive")
        if self.max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be positive")


def relative_l1(current, previous) -> float:
    """Dimensionless change; invalid values fail closed in the caller."""
    a = current.astype(mx.float32)
    b = previous.astype(mx.float32)
    difference = mx.mean(mx.abs(a - b))
    baseline = mx.mean(mx.abs(b))
    mx.eval(difference, baseline)
    return float((difference / mx.maximum(baseline, 1e-8)).item())


class NoiseCache:
    """One generation's last computed noise and observed change only."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.input = None
        self.noise = None
        self.last_input_change: float | None = None
        self.last_output_change: float | None = None
        self.consecutive_skips = 0
        self.forward_count = 0

    def decide(self, *, step: int, total: int, current_input) -> tuple[bool, float | None]:
        """Return (reuse, predicted output change); invalid evidence executes forward."""
        if step == 1 or step == total or self.forward_count < 2:
            return False, None
        if self.consecutive_skips >= self.config.max_consecutive_skips:
            return False, None
        if self.input is None or self.noise is None:
            return False, None
        denominator = self.last_input_change
        output_change = self.last_output_change
        if denominator is None or output_change is None or not all(
            math.isfinite(v) for v in (denominator, output_change)
        ) or denominator <= 0:
            return False, None
        input_change = relative_l1(current_input, self.input)
        metric = output_change * input_change / denominator
        if not math.isfinite(metric) or metric < 0:
            return False, metric
        if metric <= self.config.threshold:
            self.consecutive_skips += 1
            return True, metric
        return False, metric

    def observe_forward(self, current_input, noise) -> float | None:
        output_change = relative_l1(noise, self.noise) if self.noise is not None else None
        input_change = relative_l1(current_input, self.input) if self.input is not None else None
        self.input = current_input
        self.noise = noise
        self.last_input_change = input_change
        self.last_output_change = output_change
        self.consecutive_skips = 0
        self.forward_count += 1
        return output_change
