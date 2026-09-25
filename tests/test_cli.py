"""CLI behavior tests with neutral, synthetic prompts and no model downloads."""

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import generate

from mlx_image import engine
from mlx_image.cli import History, InteractiveSession, _batch_parser, batch_main, main, parse_batch_jobs
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

    def test_old_history_without_cache_field_remains_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / ".history" / "history.jsonl")
            history.append(self._result(root))
            record = history.read()[0]
            record.pop("cache")
            history.path.write_text(json.dumps(record) + "\n")
            observed = []

            def runner(jobs, **kwargs):
                observed.extend(jobs)
                return Summary(1)

            session = InteractiveSession(history=history, output_dir=root, runner=runner)
            with contextlib.redirect_stdout(io.StringIO()):
                session.handle("/cache experimental")
                session.handle("/repeat")
            self.assertEqual(observed[0].cache_mode, "off")


class InteractiveTests(unittest.TestCase):
    def test_cache_command_status_history_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []

            def runner(jobs, *, model_path=None, on_complete=None, cache_config=None):
                observed.append((jobs[0], cache_config))
                result = Result(jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)
                if on_complete:
                    on_complete(result)
                return Summary(1, completed=[result], elapsed_seconds=1.0)

            session = InteractiveSession(history=History(root / ".history" / "history.jsonl"),
                                         output_dir=root / "outputs", runner=runner)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                session.handle("/cache experimental")
                session.handle("/status")
                session.handle(PROMPT)
                session.handle("/cache off")
                session.handle("/repeat")
            self.assertEqual([job.cache_mode for job, _ in observed], ["experimental", "experimental"])
            self.assertEqual([config.threshold for _, config in observed], [0.08, 0.08])
            self.assertEqual([record["cache"] for record in session.history.read()], ["experimental", "experimental"])
            self.assertIn("cache      experimental", output.getvalue())
            self.assertNotIn(PROMPT, output.getvalue())

    def test_direct_and_batch_cache_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.txt"
            source.write_text(PROMPT + "\n")
            args = _batch_parser().parse_args([str(source), "--cache", "experimental"])
            jobs, failures = parse_batch_jobs(args)
            self.assertFalse(failures)
            self.assertEqual(jobs[0].cache_mode, "experimental")
            started = []
            with patch("mlx_image.cli.InteractiveSession.run", autospec=True,
                       side_effect=lambda session: started.append(session.settings.cache) or 0) as run:
                self.assertEqual(main(["--cache", "experimental"]), 0)
                self.assertEqual(len(run.call_args.args), 1)
            self.assertEqual(started, ["experimental"])

            observed = []

            def runner(batch_jobs, *, model_path=None, on_complete=None, cache_config=None):
                observed.append((batch_jobs, cache_config))
                return Summary(len(batch_jobs))

            with patch("mlx_image.cli._run_jobs", side_effect=runner), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch_main([str(source), "--cache", "experimental"]), 0)
            self.assertEqual(observed[0][0][0].cache_mode, "experimental")
            self.assertEqual(observed[0][1].threshold, 0.08)

            direct = []

            def direct_runner(direct_jobs, *, model_path=None, cache_config=None):
                direct.append((direct_jobs[0], cache_config))
                return Summary(1, completed=[Result(direct_jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)])

            with (
                patch("sys.argv", ["generate.py", "--prompt", PROMPT, "--cache", "experimental"]),
                patch("mlx_image.engine.run_jobs", side_effect=direct_runner),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(generate.main(), 0)
            self.assertEqual(direct[0][0].cache_mode, "experimental")
            self.assertEqual(direct[0][1].threshold, 0.08)

    def test_paste_preserves_full_multiline_prompt_until_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = []
            lines = [
                "A red ceramic teapot on a wooden table. ‘Soft light’ & $shapes.",
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
            with patch("builtins.input", side_effect=read_input), patch("mlx_image.cli._version", return_value="0.3.0"), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(session.run(), 0)

            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].prompt, expected)
            self.assertEqual(session.history.read()[0]["prompt"], expected)
            self.assertIn("MLX Image 0.3.0", output.getvalue())
            self.assertIn("  /paste multiline · /help commands · /quit exit", output.getvalue())
            self.assertIn("MULTILINE PROMPT\nPaste your prompt below.\nFinish with /end · cancel with /cancel", output.getvalue())
            self.assertNotIn(expected, output.getvalue())

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
            self.assertIn("/paste              multiline prompt (/end to generate, /cancel to discard)", output.getvalue())

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


class InteractiveUxTests(unittest.TestCase):
    def make_session(self, root, generated, *, model_path=None):
        def runner(jobs, *, model_path=None, on_complete=None):
            generated.extend(jobs)
            result = Result(jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)
            if on_complete:
                on_complete(result)
            return Summary(1, completed=[result], elapsed_seconds=1.0)

        return InteractiveSession(
            model_path=model_path,
            output_dir=root / "outputs",
            history=History(root / ".history" / "history.jsonl"),
            runner=runner,
        )

    def test_startup_help_and_status_are_readable_without_color(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            session = self.make_session(Path(directory), generated, model_path=Path("/synthetic/model"))
            with (
                patch("mlx_image.cli._version", return_value="0.3.0"),
                patch("builtins.input", side_effect=["/help", "/status", "/quit"]) as read_input,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(session.run(), 0)
            screen = output.getvalue()
            self.assertIn("MLX Image 0.3.0\nQwen-Image 2.1 · MLX 4-bit", screen)
            self.assertIn("1152×768 · 20 steps · seed random · guidance 1.0", screen)
            for heading in ("Prompt", "Image", "Generation", "History", "Other"):
                self.assertIn(heading, screen)
            self.assertIn("  source     local snapshot", screen)
            self.assertIn("  directory  ", screen)
            self.assertEqual(read_input.call_args_list[0].args[0], "image › ")
            self.assertNotIn("\x1b[", screen)
            self.assertEqual(generated, [])

    def test_preset_settings_and_human_validation_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            session = self.make_session(Path(directory), generated)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for command, size in (("/portrait", (768, 1152)), ("/landscape", (1152, 768)), ("/square", (1024, 1024)), ("/size 256x512", (256, 512))):
                    self.assertTrue(session.handle(command))
                    self.assertEqual((session.settings.width, session.settings.height), size)
                for command in ("/size 1000x777", "/steps banana", "/steps 30", "/seed 1977", "/seed random", "/guidance -2", "/guidance 1.5"):
                    self.assertTrue(session.handle(command))
                self.assertTrue(session.handle("/status"))
            screen = output.getvalue()
            for feedback in (
                "✓ size 768×1152", "✓ size 1152×768", "✓ size 1024×1024", "✓ size 256×512",
                "✗ width and height must be divisible by 16", "✗ steps must be an integer",
                "✓ steps 30", "✓ seed 1977", "✓ seed random",
                "✗ guidance must be greater than 0", "✓ guidance 1.5", "  source     HF cache",
            ):
                self.assertIn(feedback, screen)
            self.assertEqual((session.settings.width, session.settings.height, session.settings.steps, session.settings.seed, session.settings.guidance), (256, 512, 30, None, 1.5))
            self.assertNotIn("Traceback", screen)
            self.assertEqual(generated, [])

    def test_empty_paste_and_cancel_do_not_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            session = self.make_session(Path(directory), generated)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for command in ("/paste", "", "", "/end", "/paste", PROMPT, "/cancel"):
                    self.assertTrue(session.handle(command))
            self.assertEqual(generated, [])
            self.assertEqual(session.history.read(), [])
            self.assertIn("✗ prompt is empty; nothing generated", output.getvalue())

    def test_ctrl_c_in_normal_and_multiline_input_and_ctrl_d_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            session = self.make_session(Path(directory), generated)
            with patch("builtins.input", side_effect=[KeyboardInterrupt(), "/paste", PROMPT, KeyboardInterrupt(), "/paste", PROMPT, EOFError()]), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(session.run(), 0)
            self.assertEqual(generated, [])
            self.assertIsNone(session._paste_lines)
            self.assertIn("Input cancelled", output.getvalue())
            self.assertEqual(output.getvalue().count("Multiline prompt cancelled"), 2)

    def test_last_history_repeat_and_open_failure_hide_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            session = self.make_session(Path(directory), generated)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                session.handle("/open")
                session.handle("/size 256x256")
                session.handle("/steps 3")
                session.handle("/seed 1977")
                session.handle("/guidance 1.5")
                session.handle(PROMPT)
                session.handle("/last")
                session.handle("/history")
                session.handle("/open")
                session.handle("/repeat")
            self.assertEqual(len(generated), 2)
            self.assertEqual((generated[1].prompt, generated[1].seed, generated[1].width, generated[1].height, generated[1].steps, generated[1].guidance), (PROMPT, 1977, 256, 256, 3, 1.5))
            self.assertEqual(len(session.history.read()), 2)
            screen = output.getvalue()
            self.assertIn("✗ no generated image yet", screen)
            self.assertIn("✗ last image no longer exists", screen)
            self.assertIn("Last generation", screen)
            self.assertIn("#  time", screen)
            self.assertIn("REPEAT\n256×256 · 3 steps · seed 1977 · guidance 1.5", screen)
            self.assertNotIn(PROMPT, screen)

    def test_interrupt_during_generation_exits_interactive_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            session = InteractiveSession(
                output_dir=Path(directory) / "outputs",
                history=History(Path(directory) / ".history" / "history.jsonl"),
                runner=lambda jobs, **kwargs: Summary(1, interrupted=True),
            )
            with patch("builtins.input", return_value=PROMPT), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(session.run(), 130)
            self.assertEqual(session.history.read(), [])
            self.assertIn("Generation interrupted; leaving interactive mode", output.getvalue())


class BatchUxTests(unittest.TestCase):
    def test_batch_summary_uses_indices_and_never_prints_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.txt"
            source.write_text(PROMPT + "\n", encoding="utf-8")
            history = History(root / ".history" / "history.jsonl")
            observed = []

            def runner(jobs, *, model_path=None, on_complete=None):
                observed.extend(jobs)
                result = Result(jobs[0], "2026-09-24T12:00:00+03:00", 1.0, 4.2)
                if on_complete:
                    on_complete(result)
                return Summary(1, completed=[result], elapsed_seconds=1.0)

            with patch("mlx_image.cli._run_jobs", side_effect=runner), patch("mlx_image.cli.History", return_value=history), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(batch_main([str(source), "--output-dir", str(root / "outputs")]), 0)
            screen = output.getvalue()
            self.assertIn("BATCH\n1 jobs · 1152×768 default · 20 steps", screen)
            self.assertIn("Completed  1", screen)
            self.assertIn("Failed     0", screen)
            self.assertIn("Peak Metal 4.20 GB", screen)
            self.assertNotIn(PROMPT, screen)
            self.assertNotIn("teapot", observed[0].output.name)
            self.assertEqual(history.read()[0]["prompt"], PROMPT)

    def test_engine_stage_progress_never_repeats_prompt(self):
        class FakeQwen:
            pass

        class FakeCache:
            def clear(self):
                pass

        def init_config(qwen, config):
            qwen.prompt_cache = FakeCache()

        def init_tokenizers(qwen, snapshot):
            qwen.tokenizers = {"qwen21": object()}

        def denoise(job, transformer, embeds, mask, config, on_step):
            for step in range(1, job.steps + 1):
                on_step(step, job.steps)
            return "latents"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = {}
            job = Job(1, PROMPT, root / "image.png", 256, 256, 3, 1977, 1.0)
            with (
                patch.object(engine, "_snapshot", return_value=Path("synthetic")),
                patch.object(engine.ModelConfig, "qwen_image_21", return_value=object()),
                patch.object(engine, "_load_text_encoder", return_value=object()),
                patch.object(engine, "QwenImage21", FakeQwen),
                patch.object(engine.Qwen21Initializer, "_init_config", side_effect=init_config),
                patch.object(engine.Qwen21Initializer, "_init_tokenizers", side_effect=init_tokenizers),
                patch.object(engine.Qwen21PromptEncoder, "encode_prompt", return_value=("embeds", "mask")),
                patch.object(engine.mx, "savez", side_effect=lambda path, **values: storage.__setitem__(path, values)),
                patch.object(engine.mx, "load", side_effect=lambda path: storage[path]),
                patch.object(engine.mx, "eval"),
                patch.object(engine.mx, "reset_peak_memory"),
                patch.object(engine.mx, "clear_cache"),
                patch.object(engine.mx, "get_peak_memory", return_value=4 * 1024**3),
                patch.object(engine, "_load_transformer", return_value=object()),
                patch.object(engine, "_denoise", side_effect=denoise),
                patch.object(engine, "_load_vae", return_value=object()),
                patch.object(engine.Qwen21LatentCreator, "unpack_latents", return_value=object()),
                patch.object(engine.VAEUtil, "decode", return_value=object()),
                patch.object(engine, "_save_png"),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                summary = run_jobs([job])
            self.assertEqual(len(summary.completed), 1)
            screen = output.getvalue()
            for label in ("Loading text encoder...", "Encoding prompt...", "Text encoder released", "Loading transformer...", "Denoising", "[01/01]", "Transformer released", "Loading VAE...", "Decoding", "VAE released"):
                self.assertIn(label, screen)
            self.assertNotIn(PROMPT, screen)


if __name__ == "__main__":
    unittest.main()
