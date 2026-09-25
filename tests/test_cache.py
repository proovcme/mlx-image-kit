"""Synthetic checks for isolated, fail-safe balanced noise reuse."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from mlx_image.cache import CacheConfig, NoiseCache
from mlx_image import engine
from mlx_image.types import Job


class CacheStateTests(unittest.TestCase):
    def setUp(self):
        self.cache = NoiseCache(CacheConfig(threshold=0.2, max_consecutive_skips=2))
        self.first = mx.array([1.0, 2.0])
        self.second = mx.array([1.1, 2.1])
        self.noise1 = mx.array([1.0, 1.0])
        self.noise2 = mx.array([1.05, 1.05])

    def prime(self):
        self.assertEqual(self.cache.decide(step=1, total=8, current_input=self.first), (False, None))
        self.cache.observe_forward(self.first, self.noise1)
        self.assertEqual(self.cache.decide(step=2, total=8, current_input=self.second), (False, None))
        self.cache.observe_forward(self.second, self.noise2)

    def test_first_and_last_never_skip(self):
        self.prime()
        self.assertFalse(self.cache.decide(step=8, total=8, current_input=self.second)[0])

    def test_reuse_limit_and_reset(self):
        self.prime()
        self.assertTrue(self.cache.decide(step=3, total=8, current_input=self.second)[0])
        self.assertTrue(self.cache.decide(step=4, total=8, current_input=self.second)[0])
        self.assertFalse(self.cache.decide(step=5, total=8, current_input=self.second)[0])
        self.cache.reset()
        self.assertIsNone(self.cache.noise)
        self.assertEqual(self.cache.forward_count, 0)
        self.assertFalse(self.cache.decide(step=3, total=8, current_input=self.second)[0])

    def test_invalid_metric_forces_forward(self):
        self.prime()
        with patch("mlx_image.cache.noise_cache.relative_l1", return_value=math.nan):
            reuse, metric = self.cache.decide(step=3, total=8, current_input=self.second)
        self.assertFalse(reuse)
        self.assertTrue(math.isnan(metric))
        self.cache.last_output_change = math.inf
        self.assertFalse(self.cache.decide(step=3, total=8, current_input=self.second)[0])

    def test_distinct_jobs_have_distinct_state(self):
        self.prime()
        other = NoiseCache(self.cache.config)
        self.assertIsNone(other.noise)
        self.assertEqual(other.forward_count, 0)
        other.observe_forward(self.first, self.noise1)
        self.assertEqual(other.forward_count, 1)
        self.assertEqual(self.cache.forward_count, 2)


class EngineLoopTests(unittest.TestCase):
    def setUp(self):
        self.job = Job(1, "Synthetic neutral object", Path("synthetic.png"), 256, 256, 8, 1977, 1.0)

    def run_loop(self, cache_config=None):
        class Scheduler:
            sigmas = []

            @staticmethod
            def scale_model_input(latents, timestep):
                return latents

            @staticmethod
            def step(*, noise, timestep, latents):
                return latents + noise * 0.01

        class FakeConfig:
            width = height = 256
            image_path = None
            init_time_step = None
            time_steps = list(range(8, 0, -1))
            scheduler = Scheduler()

        calls = []
        records = []

        def transformer(*, t, **kwargs):
            calls.append(t)
            return mx.array([0.1 + t * 0.01])

        with (
            patch.object(engine, "Config", return_value=FakeConfig()),
            patch.object(engine, "Img2Img", return_value=None),
            patch.object(engine.LatentCreator, "create_for_txt2img_or_img2img", return_value=mx.array([1.0])),
            patch.object(engine, "relative_l1", side_effect=AssertionError("vanilla metric computed")) if cache_config is None else patch.object(engine, "relative_l1", wraps=engine.relative_l1),
        ):
            engine._denoise(self.job, transformer, mx.array([1.0]), mx.array([1.0]), object(),
                            on_diagnostic=records.append if cache_config else None,
                            cache_config=cache_config)
        return calls, records

    def test_off_uses_vanilla_path_and_calls_every_step(self):
        calls, records = self.run_loop()
        self.assertEqual(calls, list(range(8, 0, -1)))
        self.assertEqual(records, [])

    def test_new_denoise_job_starts_with_empty_cache(self):
        config = CacheConfig(threshold=100.0)
        first_calls, first_records = self.run_loop(config)
        second_calls, second_records = self.run_loop(config)
        self.assertEqual(first_calls, second_calls)
        self.assertTrue(first_records[0]["forward"])
        self.assertTrue(second_records[0]["forward"])
        self.assertTrue(second_records[-1]["forward"])
        self.assertTrue(any(record["reuse"] for record in second_records))

    def test_presentation_forward_events_match_cache_decisions_without_diagnostics(self):
        class Scheduler:
            sigmas = []

            @staticmethod
            def scale_model_input(latents, timestep):
                return latents

            @staticmethod
            def step(*, noise, timestep, latents):
                return latents + noise * 0.01

        class FakeConfig:
            width = height = 256
            image_path = None
            init_time_step = 0
            num_inference_steps = 8
            scheduler = Scheduler()

            def __init__(self):
                self._time_steps = None

            @property
            def time_steps(self):
                if self._time_steps is None:
                    raise AssertionError("library progress would be created")
                return self._time_steps

        forwards = []
        steps = []
        transformer_calls = []

        def transformer(*, t, **kwargs):
            transformer_calls.append(t)
            return mx.array([0.1 + t * 0.01])

        with (
            patch.object(engine, "Config", return_value=FakeConfig()),
            patch.object(engine, "Img2Img", return_value=None),
            patch.object(engine.LatentCreator, "create_for_txt2img_or_img2img", return_value=mx.array([1.0])),
        ):
            engine._denoise(
                self.job, transformer, mx.array([1.0]), mx.array([1.0]), object(),
                on_step=lambda step, total: steps.append(step),
                cache_config=CacheConfig(threshold=100.0),
                show_library_progress=False,
                on_forward=forwards.append,
            )
        self.assertEqual(steps, list(range(1, 9)))
        self.assertEqual(len(forwards), 8)
        self.assertEqual(sum(forwards), len(transformer_calls))
        self.assertTrue(forwards[0])
        self.assertTrue(forwards[-1])
        self.assertIn(False, forwards)


if __name__ == "__main__":
    unittest.main()
