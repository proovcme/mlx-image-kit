# MLX Image

Local Qwen-Image 2.1 4-bit generation on Apple Silicon using MLX. The `mlx-image` command offers an interactive prompt and staged batch generation; `generate.py` remains available for scripts.

## Quick start

Use Python 3.12 on Apple Silicon:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
mlx-image
```

The startup screen shows the installed package version and current generation settings:

```text
MLX Image 0.3.0
Qwen-Image 2.1 · MLX 4-bit

  1152×768 · 20 steps · seed random · guidance 1.0

  Type a prompt
  /paste multiline · /help commands · /quit exit

image ›
```

A one-line prompt starts generation immediately:

```text
image › A red ceramic teapot on a wooden table, soft window light
```

For multiple paragraphs, start `/paste` and finish with `/end`. The CLI preserves blank lines, Unicode, quotes, and shell characters literally; it generates only after `/end`. `/cancel` discards everything entered in this mode.

```text
image › /paste

MULTILINE PROMPT
Paste your prompt below.
Finish with /end · cancel with /cancel
│ A red ceramic teapot on a wooden table.
│
│ A small wooden cabin beside a mountain lake.
│
│ A lighthouse during a storm.
│ /end
```

Before generation, the CLI shows the size, steps, resolved numeric seed, and guidance without repeating the prompt. On success it shows the PNG path, elapsed time, seed, and measured peak Metal memory when available. Ctrl+C at the main prompt clears the current input; during `/paste` it discards that prompt. Ctrl+D exits and discards any unfinished multiline prompt. If generation is interrupted, the CLI exits because the model's state may be unsafe to reuse; only completed jobs enter history.

The seed and timestamp shown above illustrate the output format; an actual run chooses a random seed by default. Images go to `./outputs/` with timestamp names and a numeric suffix on collisions. Filenames never derive from prompts. Completed interactive and batch jobs are recorded privately in `./.history/history.jsonl` for `/repeat`; this folder is ignored by Git in this repository.

## Interactive commands

| Command | Action |
| --- | --- |
| `/portrait`, `/landscape`, `/square` | Set 768x1152, 1152x768, or 1024x1024 |
| `/size WIDTHxHEIGHT` | Set a size divisible by 16 |
| `/steps N` | Set inference steps |
| `/seed N`, `/seed random` | Set a fixed or random seed |
| `/guidance X` | Set guidance |
| `/cache off`, `/cache experimental` | Choose vanilla denoising or the optional experimental cache |
| `/status` | Show current settings, including cache mode |
| `/last` | Show the last generation's time, size, steps, seed, guidance, and output |
| `/history` | Show up to 10 recent generations, newest first, without prompts |
| `/repeat` | Repeat the last full prompt with exactly the same seed and settings |
| `/open` | Open the last PNG with macOS `open`; report if it is missing |
| `/paste` | Start entering a multiline prompt, preserving blank lines |
| `/end` | Finish a multiline prompt and generate one image |
| `/cancel` | Discard a multiline prompt without generating |
| `/help`, `/quit` | Show commands or exit |

`/help` groups commands by prompt, image, generation, history, and other actions. `/status` shows settings, cache mode, model, precision, runtime, source (HF cache or local snapshot), and output directory. Setting commands give short confirmation; invalid values give a short error and keep the session running. Defaults are 1152×768, 20 steps, guidance 1.0, a random seed, and cache `off`. The actual seed is shown before generation and saved in local history. Use `mlx-image --model-path PATH` to select an existing model snapshot, `--output-dir PATH` to choose where images go, or `--cache experimental` to start in experimental mode. The CLI remains readable without ANSI color.

## Batch generation

Place private input files outside a Git repository or in its ignored `local/` directory. A TXT file has one prompt per nonempty line:

```text
A red ceramic teapot on a wooden table
A small wooden cabin beside a mountain lake
A lighthouse during a storm
```

```sh
mlx-image batch local/prompts.txt --landscape --steps 20 --seed random --output-dir outputs/
mlx-image batch local/prompts.txt --count 4
mlx-image batch local/prompts.txt --cache experimental
```

A JSONL file can override settings for each job:

```jsonl
{"prompt":"A red ceramic teapot on a wooden table","output":"image-a.png","width":1152,"height":768,"steps":20,"seed":1977,"guidance":1.0}
{"prompt":"A small wooden cabin beside a mountain lake","output":"image-b.png","width":768,"height":1152,"steps":20,"seed":42,"guidance":1.0}
```

```sh
mlx-image batch local/jobs.jsonl --output-dir outputs/
```

`--count N` makes N images for each prompt. A fixed seed uses that seed, then seed + 1, seed + 2, and so on (wrapping at 2³²); `random` chooses a new seed for each variation. Each actual seed is recorded in local history. Job-specific JSONL fields override batch defaults. Auto-generated filenames contain only timestamps and numeric suffixes. Explicit JSONL `output` names are used as provided, with suffixes added if needed to avoid overwrites.

Batch accepts `--portrait`, `--landscape`, `--square`, or `--size WIDTHxHEIGHT`, plus `--steps`, `--seed`, `--guidance`, `--cache`, `--count`, `--output-dir`, and `--model-path`. It shows a job count and default settings, then encoding, denoising, decoding, and model-release progress. Its summary reports completed and failed counts, elapsed time, and measured peak Metal memory when available. Failures identify the job number without printing its prompt. Run `mlx-image batch --help` for the full option list.

The batch engine validates jobs before loading weights. It loads the text encoder once and writes one prompt embedding at a time to a temporary directory. It then loads the transformer once, sequentially denoises jobs, and stores latents temporarily. Finally it loads the VAE once to decode and save each PNG. Intermediate arrays are removed after use; peak unified memory does not grow linearly with job count. A failed job is reported by index without printing its prompt, and completed PNGs are retained. Ctrl+C stops the batch and preserves history for completed jobs.

## Direct CLI

For scripts and automation, the original direct interface remains available:

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

It also accepts `--model-path PATH` and `--cache off|experimental`. Run `python generate.py --help` or `mlx-image batch --help` for options.

## Experimental denoising cache

The default `off` mode runs the original v0.3.0 denoising loop: one native Q4 transformer forward for every step. `experimental` keeps the last computed noise tensor within one image. After two full forwards, it estimates the next output change from the last measured output change and the change in the transformer's input. It reuses the last noise only when that estimate is at or below the locally calibrated threshold `0.08`. The first and last steps always execute, and at most two consecutive steps can reuse a result. Invalid metrics force a normal forward. This is a project-specific output-reuse experiment; it is not a direct port of TeaCache's old Qwen-Image residual cache or coefficients.

```text
image › /cache experimental
image › /status
  cache      experimental
image › /cache off
```

The mode is stored in private local history and `/repeat` restores it; old history entries without a cache field use `off`. Batch reuses the loaded transformer but creates a fresh cache for every image. The mechanism runs only after the custom native Q4 transformer loader has completed. It does not change the scheduler, weights, quantization, text encoder, VAE, or model loading. Cache mode is an explicit image-quality tradeoff; compare it with `off` for your own scenes.

The [TeaCache paper](https://arxiv.org/abs/2411.19108) and [mlx-teacache's Qwen-Image variant](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/qwen-image.md) informed this experiment, but that Qwen variant targets the older dual-stream transformer. Qwen-Image 2.1 uses a [single-stream block-causal transformer](https://github.com/QwenLM/Qwen-Image-2.1#architecture). The older fitted coefficients did not predict output change reliably in our three 20-step neutral scenes, so this project measures the last actual noise-output change instead. No third-party acceleration framework is required at runtime.

## Tested hardware and results

Mac mini M4 with 24 GB unified memory.

| Check | Resolution | Steps | Guidance | Total | Peak Metal | Swap |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original working pipeline benchmark | 1152x768 | 20 | 1.0 | about 458 s | about 12.7 GB | 0 |
| Earlier packaged script smoke test | 256x256 | 20 | 1.0 | 35.85 s | 4.23 GB | No increase during test |
| Current shared-engine direct smoke test | 256x256 | 20 | 1.0 | 38.27 s | 4.23 GB | Not measured |
| Previous v0.3.0 regression | 1152x768 | 20 | 1.0 | 415.10 s | 12.70 GB | Not reported |
| This branch, cache off A | 1152x768 | 20 | 1.0 | 377.08 s | 13.33 GB | −32 MB system-wide delta |
| This branch, cache experimental B | 1152x768 | 20 | 1.0 | 227.49 s | 13.33 GB | 0 MB system-wide delta |

For the original benchmark, prompt encoding took about 8.6 s, denoising about 431 s, and VAE decoding about 8.8 s. The previous v0.3.0 regression measured 1.75 s, 395.50 s, and 10.10 s respectively. The current engine also completed two 256x256 jobs with one load of each model component. A 10-job, 2-step check stayed near 4.24 GB peak Metal while continuing after one deliberate save failure. The new A/B used the same neutral synthetic prompt and seed 19780415 for both runs. Cache off took 367.35 s denoising with 20 transformer calls; experimental took 220.35 s with 12 calls, a 1.67× denoising speedup. Prompt encoding was about 0.3 s and VAE decoding about 4.9 s in both runs. The A/B PNGs had SSIM 0.964 and PSNR 32.78 dB; visual inspection found small detail changes but no new obvious structural failure. System-wide swap was already around 17 GB before A and did not rise during either run, so the swap observation cannot be compared directly with the original zero-swap benchmark. Performance and image quality depend on the Apple SoC, unified memory, prompt, resolution, step count, and dependency versions.

## How it works

Each run follows the same order: load and quantize the text encoder, encode prompts, release the encoder, load the native 4-bit transformer, denoise, release the transformer, load the VAE, decode, and save PNGs. The batch mode uses temporary disk storage between stages to keep only one job's embeddings or latents active at a time.

## Native Q4 loader workaround

The working pipeline depends on a custom native 4-bit loader in [`mlx_image/engine.py`](mlx_image/engine.py). It does not replace the loader with the default mflux or Hugging Face model-loading path. The loader first gets a snapshot from the local Hugging Face cache, downloading missing files only when needed, or uses an existing `--model-path` snapshot. Each component is loaded and released in sequence to limit unified-memory use.

| Component | Native Q4 loading behavior |
| --- | --- |
| Text encoder | Construct `Qwen21TextEncoder`, quantize to 4-bit affine with group size 64, map 904 checkpoint keys from `language_model.model.*` to module paths, then load with `strict=True`. The order is **quantize → remap → strict load**. |
| Transformer | Construct `Qwen21Transformer`, quantize to native 4-bit with group size 64, rename `modulation.0.*` to `modulation.layers.1.*` and `time_text_embed.linear_*` to `time_text_embed.timestep_embedder.linear_*`, then load with `strict=True`. |
| VAE | Rename `.gamma`/`.beta` to `.weight`/`.bias`, adapt downsampler and upsampler convolution paths, and apply the same fallback `.conv` mapping as the working prototype. Only checkpoint tensors whose mapped key exists and whose shape matches the VAE parameter are loaded; VAE update uses `strict=False`, as in the prototype. |

The loader was compared against the original working benchmark implementation, not just this README. In the existing local snapshot, all 904 text-encoder mapping sources are present; the transformer has 3 `modulation.0.*` keys and 6 `time_text_embed.linear_*` keys requiring remapping, with no target-key collisions. For the VAE, 226 checkpoint tensors match the target keys and shapes, 12 have no matching target key, and none fail the shape check. Each of the three current loaders also completed a read-only load of that local snapshot without a prompt or image render. These checks do not claim that every future snapshot has the same keys. [`tests/test_loader.py`](tests/test_loader.py) locks down the quantization order, strict loading, remaps, and VAE shape-selection behavior without downloading weights. The cause of the upstream naming differences has not been established.

## Model and licenses

The weights are not part of this repository. By default the script uses the Hugging Face cache and downloads missing files from [`mlx-community/Qwen-Image-2.1-MLX-4bit`](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit) on first use; the model is about 10.5 GB. `--model-path` accepts a local snapshot containing the text encoder, transformer, VAE, and tokenizer files. Keep snapshots outside this repository.

The repository's MIT license applies to this code, not to the weights. The model card identifies the weights as governed by the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which limits use to noncommercial research or evaluation unless a separate commercial license is obtained. Read its full terms before using the weights.
