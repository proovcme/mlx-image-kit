# MLX image generation with Qwen-Image 2.1 4-bit

## What this is

A small command-line script for local Qwen-Image 2.1 generation with MLX and native 4-bit weights. It uses the `mlx-community/Qwen-Image-2.1-MLX-4bit` model and the Qwen-Image 2.1 components supplied by `mflux`.

## Tested hardware

Mac mini M4 with 24 GB unified memory.

## Verified benchmark

One local run of the source pipeline at 1152 × 768, 20 steps, guidance 1.0: about 458 seconds total, with about 12.7 GB peak Metal memory and no swap use. Prompt encoding took about 8.6 seconds, denoising about 431 seconds, and VAE decoding about 8.8 seconds. The packaged script also completed an offline 256 × 256, 20-step smoke test using local weights. Performance depends on the Apple SoC, unified memory, resolution, step count, and dependency versions.

## How it works

The script loads and quantizes the text encoder, encodes the prompt, and releases the encoder. It then loads the native 4-bit transformer, runs denoising, and releases the transformer. Finally it loads the VAE, decodes the latents, and saves a PNG.

## Native Q4 loader workaround

The script includes a loader workaround required by the tested setup. The text encoder loader maps checkpoint keys from the `language_model.model` hierarchy into the quantized encoder module before strict loading. For the transformer, it creates a 4-bit module with group size 64, renames checkpoint keys beginning with `modulation.0.` and `time_text_embed.linear_` to their module paths, and loads the remapped weights strictly. The VAE loader separately renames some keys and selects tensors with matching shapes. This describes what the tested code does; the underlying cause of the checkpoint/module naming differences has not been established.

## Installation

Use Python 3.12 on Apple Silicon:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The model weights are not part of this repository. By default, `generate.py` resolves the model from the Hugging Face cache and downloads missing files from `mlx-community/Qwen-Image-2.1-MLX-4bit` on first use; the model is about 10.5 GB. Pass `--model-path` with a local model snapshot directory to use existing files. The snapshot must contain `text_encoder/model.safetensors`, `transformer/model.safetensors`, and `vae/model.safetensors`, along with the tokenizer files required by `mflux`. Keep snapshots outside this repository.

## Licenses

The MIT license in this repository applies to this script, not to model weights. The [MLX model card](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit) identifies the weights as governed by the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). That license limits model use to noncommercial research or evaluation unless a separate commercial license is obtained. Read its full terms before using the weights.

## Usage

```sh
python generate.py \
  --prompt "A red ceramic teapot on a wooden table, soft window light" \
  --output output.png \
  --width 1152 \
  --height 768 \
  --steps 20 \
  --seed 1977 \
  --guidance 1.0
```

Use `python generate.py --help` for all options. The default PNG output and `.safetensors` weights are ignored by Git; other model files may not be. Keep model snapshots outside this repository.
