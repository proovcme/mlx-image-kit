"""CLI behavior tests with neutral, synthetic prompts and no model downloads."""

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from mlx_image.cli import History, InteractiveSession, _batch_parser, parse_batch_jobs
from mlx_image.engine import Job, Result, Summary, run_jobs

PROMPT = "A red ceramic teapot on a wooden table"


class BatchParsingTests(unittest.TestCase):
    def test_txt_count_fixed_seed_and_unique_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.txt"
            source.write_text(PROMPT + "\nA small wooden cabin beside a mountain lake\n")
            args = _batch_parser().parse_args([str(source), "--count", "4", "--seed", "1977", "--output-dir", str(root / "outputs")])
            jobs, failures = parse_batch_jobs(args)
            self.assertFalse(failures)
            self.assertEqual(len(jobs), 8)
            self.assertEqual([j.seed for j in jobs[:4]], [1977, 1978, 1979, 1980])
            self.assertEqual(len({j.output for j in jobs}), 8)
            self.assertTrue(all(j.prompt in (PROMPT, "A small wooden cabin beside a mountain lake") for j in jobs))

    def test_jsonl_overrides_and_bad_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "jobs.jsonl"
            source.write_text(
                json.dumps({"prompt": PROMPT, "output": "image-a.png", "width": 768, "height": 1152, "steps": 3, "seed": 42, "guidance": 1.5})
                + "\n{invalid json}\n"
                + json.dumps({"prompt": "A lighthouse during a storm"}) + "\n"
            )
            args = _batch_parser().parse_args([str(source), "--count", "2", "--seed", "1977", "--output-dir", str(root / "outputs")])
            jobs, failures = parse_batch_jobs(args)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len(failures), 1)
            self.assertEqual([j.seed for j in jobs[:2]], [42, 43])
            self.assertEqual((jobs[0].width, jobs[0].height, jobs[0].steps, jobs[0].guidance), (768, 1152, 3, 1.5))
            self.assertEqual([j.output.name for j in jobs[:2]], ["image-a.png", "image-a_2.png"])
            self.assertEqual([j.seed for j in jobs[2:]], [1977, 1978])

    def test_random_variations_use_separate_actual_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.txt"
            source.write_text(PROMPT + "\n")
            args = _batch_parser().parse_args([str(source), "--count", "3", "--output-dir", str(root / "outputs")])
            with patch("mlx_image.cli.secrets.randbits", side_effect=[10, 10, 11, 12]):
                jobs, failures = parse_batch_jobs(args)
            self.assertFalse(failures)
            self.assertEqual([j.seed for j in jobs], [10, 11, 12])

    def test_fixed_seed_wraps_at_32_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.txt"
            source.write_text(PROMPT + "\n")
            args = _batch_parser().parse_args([str(source), "--count", "2", "--seed", "4294967295", "--output-dir", str(root / "outputs")])
            jobs, failures = parse_batch_jobs(args)
            self.assertFalse(failures)
            self.assertEqual([j.seed for j in jobs], [4294967295, 0])

    def test_invalid_job_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Job(1, PROMPT, Path(directory) / "image.png", 257, 256, 20, 42, 1.0)
            summary = run_jobs([job], model_path=Path(directory), progress=False)
            self.assertEqual(len(summary.failed), 1)
            self.assertEqual(len(summary.completed), 0)


class HistoryTests(unittest.TestCase):
    def _result(self, directory: Path) -> Result:
        job = Job(1, PROMPT, directory / "image.png", 256, 256, 20, 1977, 1.0)
        return Result(job, datetime.now().astimezone().isoformat(timespec="seconds"), 3.5, 4.2)

    def test_history_records_actual_seed_and_prompt_privately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".history" / "history.jsonl"
            history = History(path)
            history.append(self._result(Path(directory)))
            records = history.read()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["seed"], 1977)
            self.assertEqual(records[0]["prompt"], PROMPT)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_failed_replacement_preserves_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".history" / "history.jsonl"
            history = History(path)
            history.append(self._result(Path(directory)))
            original = path.read_bytes()
            with patch("mlx_image.cli.os.replace", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    history.append(self._result(Path(directory)))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(path.parent.glob("*.tmp"))), 0)

    def test_ensure_recovers_completed_result_without_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".history" / "history.jsonl"
            history = History(path)
            result = self._result(Path(directory))
            history.ensure([result])
            history.ensure([result])
            self.assertEqual(len(history.read()), 1)


class InteractiveTests(unittest.TestCase):
    def test_paste_preserves_full_multiline_prompt_until_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = []
            lines = [
                "A red ceramic teapot on a wooden table.",
                "",
                "  A small wooden cabin beside a mountain lake.",
                "",
                "A lighthouse during a storm.",
                "",
            ]
            expected = "\n".join(lines)

            def runner(jobs, *, model_path=None, on_complete=None):
                generated.extend(jobs)
                result = Result(jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)
                if on_complete:
                    on_complete(result)
                return Summary(1, completed=[result], elapsed_seconds=1.0)

            entries = iter(["/paste", *lines, "/end", "/quit"])

            def read_input(prompt):
                entry = next(entries)
                if entry == "/end":
                    self.assertEqual(generated, [])
                return entry

            session = InteractiveSession(
                output_dir=root / "outputs",
                history=History(root / ".history" / "history.jsonl"),
                runner=runner,
            )
            with patch("builtins.input", side_effect=read_input), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(session.run(), 0)

            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].prompt, expected)
            self.assertEqual(session.history.read()[0]["prompt"], expected)
            self.assertIn("Type a prompt, or /paste for multiline. /help for commands.", output.getvalue())
            self.assertIn("Paste multiline prompt. Finish with /end. Cancel with /cancel.", output.getvalue())

    def test_cancel_discards_paste_and_short_prompt_still_generates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = []

            def runner(jobs, *, model_path=None, on_complete=None):
                generated.extend(jobs)
                result = Result(jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)
                if on_complete:
                    on_complete(result)
                return Summary(1, completed=[result], elapsed_seconds=1.0)

            session = InteractiveSession(
                output_dir=root / "outputs",
                history=History(root / ".history" / "history.jsonl"),
                runner=runner,
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for line in ("/paste", "A lighthouse during a storm.", "", "/cancel"):
                    self.assertTrue(session.handle(line))
                self.assertEqual(generated, [])
                self.assertTrue(session.handle(PROMPT))
                self.assertTrue(session.handle("/help"))

            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].prompt, PROMPT)
            self.assertEqual(len(session.history.read()), 1)
            self.assertIn("/paste /end /cancel", output.getvalue())

    def test_commands_repeat_last_history_and_open_without_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = []
            opened = []

            def runner(jobs, *, model_path=None, on_complete=None):
                job = jobs[0]
                generated.append(job)
                job.output.parent.mkdir(parents=True, exist_ok=True)
                job.output.touch()
                result = Result(job, "2026-09-24T12:00:00+03:00", 1.0, 4.2)
                if on_complete:
                    on_complete(result)
                return Summary(1, completed=[result], elapsed_seconds=1.0)

            session = InteractiveSession(
                output_dir=root / "outputs",
                history=History(root / ".history" / "history.jsonl"),
                runner=runner,
                opener=lambda args, check: opened.append(args),
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for command in ("/status", "/portrait", "/landscape", "/square", "/size 256x256", "/steps 3", "/seed 1977", "/guidance 1.5"):
                    self.assertTrue(session.handle(command))
                self.assertTrue(session.handle(PROMPT))
                self.assertTrue(session.handle("/last"))
                self.assertTrue(session.handle("/history"))
                self.assertTrue(session.handle("/repeat"))
                self.assertTrue(session.handle("/open"))
                self.assertTrue(session.handle("/seed random"))
                self.assertFalse(session.handle("/quit"))
            self.assertEqual((generated[0].width, generated[0].height, generated[0].steps, generated[0].seed, generated[0].guidance), (256, 256, 3, 1977, 1.5))
            self.assertEqual((generated[1].prompt, generated[1].seed, generated[1].width, generated[1].steps, generated[1].guidance), (PROMPT, 1977, 256, 3, 1.5))
            self.assertNotEqual(generated[0].output, generated[1].output)
            self.assertEqual(opened[0][0], "open")
            self.assertNotIn(PROMPT, output.getvalue())
            self.assertEqual(len(session.history.read()), 2)


if __name__ == "__main__":
    unittest.main()
