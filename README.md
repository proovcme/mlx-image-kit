# MLX Image Kit

[English](README.md) · [Русский](README.ru.md)

**Local image generation on Apple Silicon with Qwen-Image 2.1, a native 4-bit MLX loader, staged execution, and an optional denoising cache.** Use the interactive CLI, a script-friendly direct command, or a sequential batch. Vanilla generation is the default.

![Sunlit glass conservatory after rain, with a cat in a wicker chair](assets/showcase.png)

Qwen-Image 2.1 · native 4-bit MLX · Mac mini M4, 24 GB unified memory<br>
1152×768 · 20 steps · `balanced` cache

## One model, many visual languages

Every image below was generated locally with the same 4-bit model and 20 denoising steps. These are showcase outputs, not paired benchmarks. Open an image to inspect it at full resolution.

| Woodblock print | 1970s comic |
| --- | --- |
| <a href="assets/gallery/woodblock.png"><img src="assets/gallery/woodblock.png" alt="Ukiyo-e inspired storm sea and lighthouse" width="360"></a> | <a href="assets/gallery/comic.png"><img src="assets/gallery/comic.png" alt="Retro futuristic city in a Franco-Belgian comic style" width="360"></a> |
| **Isometric game scene** | **Exploded-view illustration** |
| <a href="assets/gallery/isometric.png"><img src="assets/gallery/isometric.png" alt="Cutaway isometric Mars base" width="360"></a> | <a href="assets/gallery/exploded-view.png"><img src="assets/gallery/exploded-view.png" alt="Exploded-view cassette recorder with part labels" width="360"></a> |
| **1960s print poster** |  |
| <a href="assets/gallery/poster.png"><img src="assets/gallery/poster.png" alt="Italian coffee poster with large CAFFÈ LUNA typography" width="360"></a> |  |

The poster renders its two requested text lines clearly. The exploded view demonstrates short labels, but its mechanism is an illustration, **not an engineering assembly drawing**. Generated text and fine details should always be reviewed before publication.

## Measured `off` vs `balanced`

`off` runs the vanilla denoising loop. `balanced` may reuse a previous transformer result when consecutive steps are sufficiently similar. It is an explicit choice; the default is **`off`**.

| 1152×768 · 20 steps | Total, `off` | Total, `balanced` | Denoising speedup | SSIM |
| --- | ---: | ---: | ---: | ---: |
| Paired test 1 | 377.08 s | 227.49 s | 1.67× | 0.9645 |
| Paired test 2 | 448.13 s | 299.06 s | 1.53× | 0.9732 |

These are two measured workloads on a Mac mini M4 with 24 GB unified memory, separate from the gallery. Results vary by prompt and machine; visual differences remain possible. `off` is the reference mode. The cache is scoped to one image, including each job in a batch. The mechanism is project-specific, inspired by approaches such as [TeaCache](https://arxiv.org/abs/2411.19108) and [Cache-DiT](https://github.com/vipshop/cache-dit); it is not a direct port.

## Quick start

Requires an Apple Silicon Mac and Python 3.12 or newer. The tested configuration is listed above; other memory sizes have not been benchmarked here.

```sh
git clone https://github.com/proovcme/mlx-image-kit.git
cd mlx-image-kit
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
mlx-image
```

The first run uses the Hugging Face cache and downloads missing model files. To use an existing snapshot instead, pass `--model-path PATH`. Model weights are not included in this repository. Review the [model license](#model-and-license) before using them.

The interactive prompt accepts one line immediately, or multiple lines with `/paste`:

```text
image › /paste
Paste your prompt below.
Finish with /end · cancel with /cancel
│ A red ceramic teapot on a worn wooden table.
│
│ Soft daylight through an old window.
│ /end
```

Blank lines and Unicode are preserved. Generation starts only after `/end`; `/cancel` discards the multiline prompt. The prompt is not echoed in the generation summary.

### Interactive commands

| Command | What it does |
| --- | --- |
| `/portrait`, `/landscape`, `/square` | Select 768×1152, 1152×768, or 1024×1024 |
| `/size WIDTHxHEIGHT` | Set dimensions divisible by 16 |
| `/steps N` | Set denoising steps |
| `/seed N` or `/seed random` | Fix or randomize the seed |
| `/guidance X` | Set guidance |
| `/cache off` or `/cache balanced` | Select vanilla or optional cached denoising |
| `/status` | Show current settings, including cache mode |
| `/paste`, `/end`, `/cancel` | Enter, finish, or discard a multiline prompt |
| `/last`, `/history`, `/repeat` | Inspect or repeat completed local jobs |
| `/open` | Open the last PNG on macOS |
| `/help`, `/quit` | Show help or exit |

Defaults: 1152×768, 20 steps, guidance 1.0, random seed, cache `off`. `/repeat` restores the complete prompt, seed, dimensions, guidance, steps, and cache mode. Old history entries without a cache field are treated as `off`.

### Direct command

```sh
python generate.py \
  --prompt "A red ceramic teapot on a wooden table" \
  --output output.png \
  --width 1152 --height 768 \
  --steps 20 --seed 1977 --guidance 1.0 \
  --cache balanced
```

Omit `--cache` for vanilla generation. `--cache off` is also accepted. Use `--model-path PATH` for an existing local snapshot.

### Sequential batch

A `.txt` file contains one prompt per nonempty line. A `.jsonl` file can also set `output`, `width`, `height`, `steps`, `seed`, and `guidance` for each job:

```jsonl
{"prompt":"A red ceramic teapot on a wooden table","output":"image-a.png","width":1152,"height":768,"steps":20,"seed":1977,"guidance":1.0}
{"prompt":"A small wooden cabin by a lake","output":"image-b.png","width":768,"height":1152,"steps":20,"seed":42,"guidance":1.0}
```

```sh
mlx-image batch local/prompts.txt --cache balanced
mlx-image batch local/jobs.jsonl --cache balanced --output-dir outputs/
```

The batch-wide cache mode defaults to `off`. `--count N` makes N variations per prompt; a fixed seed increments for each variation. Batch jobs run sequentially, reuse the loaded transformer, and each gets fresh cache state. A failed job is reported by number without printing its prompt. Run `mlx-image batch --help` for all flags.

## Why the native Q4 loader matters

The working path in [`mlx_image/engine.py`](mlx_image/engine.py) loads the local Qwen-Image 2.1 MLX 4-bit snapshot through a custom loader. It does not substitute the default mflux or Hugging Face loader.

| Component | Loading behavior |
| --- | --- |
| Text encoder | Create `Qwen21TextEncoder`; quantize to affine 4-bit with group size 64; remap 904 keys from `language_model.model.*`; load with `strict=True`. Order: **quantize → remap → strict load**. |
| Transformer | Create native Q4 `Qwen21Transformer` with group size 64; remap `modulation.0.*` to `modulation.layers.1.*` and `time_text_embed.linear_*` to `time_text_embed.timestep_embedder.linear_*`; load with `strict=True`. The tested local snapshot needs 3 + 6 such key remaps. |
| VAE | Map `.gamma`/`.beta` and convolution paths, then select only keys with matching target shapes. The VAE retains the working prototype's `strict=False` update behavior. |

Execution is staged: encode prompts and release the text encoder; load the transformer, denoise and release it; load the VAE, decode and save PNGs. Batch mode spills intermediate arrays to a temporary directory and loads each heavy component once. Loader behavior is covered by [`tests/test_loader.py`](tests/test_loader.py) without downloading weights.

## Local data and output

Generated PNGs, model weights, diagnostic files, `local/`, and `.history/` are ignored by Git. The CLI stores completed prompt history locally for `/repeat`; history is never included in this repository. The six curated images under `assets/` are the only generated images intended for the public README. Their PNG metadata is empty; no source prompt or private output path is included.

Run the model-free CLI, cache, and loader regression tests with:

```sh
python -m unittest discover -s tests -q
```

## Model and license

This repository's [MIT license](LICENSE) covers its code, **not the model weights**. The default weights come from [`mlx-community/Qwen-Image-2.1-MLX-4bit`](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit). The underlying [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) limits use of the model materials to noncommercial research or evaluation unless a separate commercial license is obtained. Read the full terms before use.
