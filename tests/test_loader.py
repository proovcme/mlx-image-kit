"""Native Q4 loader contract tests; no model download or private prompts."""

import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_image import engine


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def squeeze(self):
        return FakeTensor(tuple(dimension for dimension in self.shape if dimension != 1))


class FakeModel:
    def __init__(self, events):
        self.events = events
        self.loaded = None

    def update(self, weights, *, strict):
        self.events.append(("load", strict))
        self.loaded = weights

    def parameters(self):
        return {}


class LoaderRegressionTests(unittest.TestCase):
    def test_text_encoder_native_q4_mapping_and_strict_order(self):
        mapping = engine.build_text_encoder_mapping()
        pairs = [(item.to_pattern, tuple(item.from_pattern)) for item in mapping]
        self.assertEqual(len(pairs), 904)
        self.assertIn(("embed_tokens.weight", ("language_model.model.embed_tokens.weight",)), pairs)
        self.assertIn(("layers.35.mlp.down_proj.biases", ("language_model.model.layers.35.mlp.down_proj.biases",)), pairs)

        events = []
        model = FakeModel(events)
        weights = {"language_model.model.embed_tokens.weight": object()}
        mapped = {"embed_tokens.weight": object()}

        def quantize(target, *, group_size, bits, mode):
            self.assertIs(target, model)
            events.append(("quantize", group_size, bits, mode))

        def remap(source, rules):
            self.assertIs(source, weights)
            self.assertEqual(len(rules), 904)
            events.append(("remap",))
            return mapped

        with (
            patch.object(engine.mx, "load", return_value=weights),
            patch.object(engine.Qwen21TextEncoder, "__new__", return_value=model),
            patch.object(engine.nn, "quantize", side_effect=quantize),
            patch.object(engine.WeightMapper, "apply_mapping", side_effect=remap),
            patch.object(engine.mx, "eval"),
        ):
            self.assertIs(engine._load_text_encoder(Path("synthetic")), model)

        self.assertEqual(events, [("quantize", 64, 4, "affine"), ("remap",), ("load", True)])
        self.assertIs(model.loaded, mapped)

    def test_transformer_native_q4_remaps_before_strict_load(self):
        events = []
        model = FakeModel(events)
        weights = {
            "modulation.0.weight": object(),
            "time_text_embed.linear_1.weight": object(),
            "other.weight": object(),
        }

        def quantize(target, *, group_size, bits):
            self.assertIs(target, model)
            events.append(("quantize", group_size, bits))

        def unflatten(items):
            events.append(("remap",))
            return dict(items)

        with (
            patch.object(engine.mx, "load", return_value=weights),
            patch.object(engine.Qwen21Transformer, "__new__", return_value=model),
            patch.object(engine.nn, "quantize", side_effect=quantize),
            patch.object(engine, "tree_unflatten", side_effect=unflatten),
            patch.object(engine.mx, "eval"),
        ):
            self.assertIs(engine._load_transformer(Path("synthetic")), model)

        self.assertEqual(events, [("quantize", 64, 4), ("remap",), ("load", True)])
        self.assertEqual(set(model.loaded), {
            "modulation.layers.1.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "other.weight",
        })

    def test_vae_renames_and_skips_incompatible_shapes_like_original(self):
        events = []
        model = FakeModel(events)
        checkpoint = {
            "block.gamma": FakeTensor((1, 1, 1, 4)),
            "block.downsampler.resample.1.weight": FakeTensor((4, 4)),
            "block.upsampler.resample.1.weight": FakeTensor((4, 4)),
            "shape_mismatch.weight": FakeTensor((3, 3)),
            "unknown.weight": FakeTensor((4, 4)),
        }
        parameters = {
            "block.weight": FakeTensor((4,)),
            "block.downsampler.conv.weight": FakeTensor((4, 4)),
            "block.upsampler.conv.weight": FakeTensor((4, 4)),
            "shape_mismatch.weight": FakeTensor((4, 4)),
        }

        with (
            patch.object(engine.mx, "load", return_value=checkpoint),
            patch.object(engine.Qwen21VAE, "__new__", return_value=model),
            patch.object(engine, "tree_flatten", return_value=list(parameters.items())),
            patch.object(engine, "tree_unflatten", side_effect=lambda items: dict(items)),
            patch.object(engine.mx, "eval"),
        ):
            self.assertIs(engine._load_vae(Path("synthetic")), model)

        self.assertEqual(events, [("load", False)])
        self.assertEqual(set(model.loaded), {
            "block.weight",
            "block.downsampler.conv.weight",
            "block.upsampler.conv.weight",
        })
        self.assertEqual(model.loaded["block.weight"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
