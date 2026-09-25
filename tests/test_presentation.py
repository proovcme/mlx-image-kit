"""Terminal presentation tests with synthetic events; no model or private data."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate
from mlx_image.presentation import Presenter, run_presented
from mlx_image.types import Failure, Job, Result, Summary

PROMPT = "A red ceramic teapot beside a window"


class TtyBuffer(io.StringIO):
    def isatty(self):
        return True


def job(cache="off", index=1):
    return Job(index, PROMPT, Path(f"/synthetic/image-{index}.png"), 256, 256, 20, 42, 1.0, cache)


def result(item):
    return Result(item, "2026-09-25T12:00:00+03:00", 4.0, 13.3)


def fake_run(jobs, **kwargs):
    completed = []
    kwargs["on_stage_event"]("model_ready", None)
    kwargs["on_stage_event"]("text_encoder_ready", None)
    for item in jobs:
        kwargs["on_stage_timing"](item.index, "prompt_encoding", 1.8)
        kwargs["on_stage_event"]("prompt_encoded", item.index)
    kwargs["on_stage_event"]("transformer_ready", None)
    for item in jobs:
        kwargs["on_stage_event"]("denoising_start", item.index)
        for step in range(1, item.steps + 1):
            if kwargs["on_forward"]:
                kwargs["on_forward"](item.index, step in (1, 20) or step % 3 != 0)
            kwargs["on_step"](item.index, step, item.steps)
        kwargs["on_stage_timing"](item.index, "denoising", 220.0)
    kwargs["on_stage_event"]("vae_ready", None)
    for item in jobs:
        kwargs["on_stage_event"]("decoding_start", item.index)
        kwargs["on_stage_timing"](item.index, "vae_decode", 5.1)
        output = result(item)
        completed.append(output)
        kwargs["on_complete"](output)
    return Summary(len(jobs), completed=completed, elapsed_seconds=230.0)


class PresentationTests(unittest.TestCase):
    def test_normal_off_and_balanced_counts_and_summary(self):
        for mode in ("off", "balanced"):
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()) as output:
                run_presented(fake_run, [job(mode)])
            screen = output.getvalue()
            self.assertIn("Preparing", screen)
            self.assertIn("Generating", screen)
            self.assertIn("Decoding", screen)
            self.assertIn("✓ Saved", screen)
            self.assertIn("Denoising 20/20", screen)
            self.assertIn("  total       3:50", screen)
            self.assertIn("  prompt      1.8 s", screen)
            self.assertIn("  denoising   3:40", screen)
            self.assertIn("  VAE         5.1 s", screen)
            self.assertIn("peak Metal  13.30 GB", screen)
            if mode == "balanced":
                self.assertIn("forwards    14/20", screen)
                self.assertIn("skipped     6/20", screen)
                self.assertIn("computed 14 · skipped 6", screen)
            else:
                self.assertNotIn("computed", screen)
                self.assertNotIn("skipped", screen)
            self.assertNotIn(PROMPT, screen)

    def test_tty_has_one_dynamic_line_and_no_color(self):
        stream = TtyBuffer()
        presenter = Presenter([job("balanced")], stream=stream)
        presenter.begin()
        presenter.event("denoising_start", 1)
        for step in range(1, 21):
            presenter.forward(1, step % 3 != 0)
            presenter.step(1, step, 20)
        screen = stream.getvalue()
        self.assertEqual(screen.count("\r"), 20)
        self.assertIn("20/20", screen)
        self.assertNotIn("\x1b[", screen)
        self.assertNotIn(PROMPT, screen)

    def test_non_tty_emits_few_lines(self):
        stream = io.StringIO()
        presenter = Presenter([job()], stream=stream)
        presenter.begin()
        presenter.event("denoising_start", 1)
        for step in range(1, 21):
            presenter.step(1, step, 20)
        self.assertLessEqual(stream.getvalue().count("Denoising"), 5)
        self.assertNotIn("\r", stream.getvalue())

    def test_no_color_respected(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            stream = TtyBuffer()
            presenter = Presenter([job()], stream=stream)
            presenter.begin()
            presenter.step(1, 1, 20)
        self.assertNotIn("\x1b[", stream.getvalue())

    def test_quiet_only_paths_and_failure_visible_on_stderr(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_presented(fake_run, [job()], mode="quiet")
        self.assertEqual(output.getvalue(), "/synthetic/image-1.png\n")
        with (
            patch("sys.argv", ["generate.py", "--prompt", PROMPT, "--quiet"]),
            patch("mlx_image.engine.run_jobs", return_value=Summary(1, failed=[Failure(1, "model snapshot not found")])),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(generate.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("model snapshot not found", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_quiet_argument_error_is_short_and_has_no_stdout(self):
        with (
            patch("sys.argv", ["generate.py", "--prompt", PROMPT, "--quiet", "--cache", "unknown"]),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as caught:
                generate.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("invalid choice", stderr.getvalue())
        self.assertNotIn("usage:", stderr.getvalue())

    def test_invalid_size_never_starts_model(self):
        with (
            patch("sys.argv", ["generate.py", "--prompt", PROMPT, "--width", "257", "--quiet"]),
            patch("mlx_image.engine.run_jobs") as runner,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(generate.main(), 2)
        runner.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("positive multiples of 16", stderr.getvalue())

    def test_verbose_shows_model_source_but_not_prompt(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_presented(fake_run, [job("balanced")], mode="verbose", model_path=Path("/synthetic/model"))
        self.assertIn("Model source: /synthetic/model", output.getvalue())
        self.assertIn("loader", output.getvalue())
        self.assertNotIn(PROMPT, output.getvalue())

    def test_hub_progress_suppression_does_not_swallow_exception(self):
        def broken_runner(jobs, **kwargs):
            raise RuntimeError("synthetic failure")

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run_presented(broken_runner, [job()])

    def test_direct_uses_shared_presenter_and_quiet(self):
        with (
            patch("sys.argv", ["generate.py", "--prompt", PROMPT, "--quiet"]),
            patch("mlx_image.engine.run_jobs", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(generate.main(), 0)
        self.assertEqual(output.getvalue(), str(Path("output.png").resolve()) + "\n")

    def test_batch_isolation_and_quiet_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from mlx_image.cli import History, batch_main

            source = root / "jobs.txt"
            source.write_text(PROMPT + "\nA blue cup on a shelf\n")
            with (
                patch("mlx_image.cli._run_jobs", side_effect=fake_run),
                patch("mlx_image.cli.History", return_value=History(root / "history.jsonl")),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(batch_main([str(source), "--quiet", "--output-dir", str(root / "out")]), 0)
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(line.endswith(".png") for line in lines))
            self.assertNotIn(PROMPT, output.getvalue())

    def test_batch_quiet_failure_keeps_stdout_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            from mlx_image.cli import History, batch_main

            source = Path(directory) / "jobs.txt"
            source.write_text(PROMPT + "\n")
            with (
                patch("mlx_image.cli._run_jobs", return_value=Summary(1, failed=[Failure(1, "model snapshot not found")])),
                patch("mlx_image.cli.History", return_value=History(Path(directory) / "history.jsonl")),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(batch_main([str(source), "--quiet"]), 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("job 1: model snapshot not found", stderr.getvalue())
            self.assertNotIn(PROMPT, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
