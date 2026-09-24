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

Type a prompt to generate an image. For example:

```text
MLX IMAGE
Qwen-Image 2.1 · 4-bit
1152x768 | 20 steps | seed random | guidance 1.0
> A red ceramic teapot on a wooden table, soft window light
Generating 1152x768 | 20 steps | seed 1977 | guidance 1.0
...
Saved: outputs/YYYYMMDD_HHMMSS.png
```

The seed and timestamp shown above illustrate the output format; an actual run chooses a random seed by default. Images go to `./outputs/` with timestamp names and a numeric suffix on collisions. Filenames never derive from prompts. Completed interactive and batch jobs are recorded privately in `./.history/history.jsonl` for `/repeat`; this folder is ignored by Git in this repository.

## Interactive commands

| Command | Action |
| --- | --- |
| `/portrait`, `/landscape`, `/square` | Set 768x1152, 1152x768, or 1024x1024 |
| `/size WIDTHxHEIGHT` | Set a size divisible by 16 |
| `/steps N` | Set inference steps |
| `/seed N`, `/seed random` | Set a fixed or random seed |
| `/guidance X` | Set guidance |
| `/status` | Show current settings |
| `/last`, `/history` | Show completed job details without printing prompts |
| `/repeat` | Repeat the last prompt with exactly the same seed and settings |
| `/open` | Open the last PNG with macOS `open` |
| `/help`, `/quit` | Show commands or exit |

Defaults are 1152x768, 20 steps, guidance 1.0, and a random seed. The actual seed is shown before generation and saved in local history. Use `mlx-image --model-path PATH` to select an existing model snapshot.

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

It also accepts `--model-path PATH`. Run `python generate.py --help` or `mlx-image batch --help` for options.

## Tested hardware and results

Mac mini M4 with 24 GB unified memory.

| Check | Resolution | Steps | Guidance | Total | Peak Metal | Swap |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original working pipeline benchmark | 1152x768 | 20 | 1.0 | about 458 s | about 12.7 GB | 0 |
| Earlier packaged script smoke test | 256x256 | 20 | 1.0 | 35.85 s | 4.23 GB | No increase during test |
| Current shared-engine direct smoke test | 256x256 | 20 | 1.0 | 38.27 s | 4.23 GB | Not measured |

For the original benchmark, prompt encoding took about 8.6 s, denoising about 431 s, and VAE decoding about 8.8 s. The current engine also completed two 256x256 jobs with one load of each model component. A 10-job, 2-step check stayed near 4.24 GB peak Metal while continuing after one deliberate save failure. Performance depends on the Apple SoC, unified memory, resolution, step count, and dependency versions. The current package has not been rebenchmarked at 1152x768.

## How it works

Each run follows the same order: load and quantize the text encoder, encode prompts, release the encoder, load the native 4-bit transformer, denoise, release the transformer, load the VAE, decode, and save PNGs. The batch mode uses temporary disk storage between stages to keep only one job's embeddings or latents active at a time.

## Native Q4 loader workaround

The script includes a loader workaround required by the tested setup. The text encoder loader maps checkpoint keys from the `language_model.model` hierarchy into the quantized encoder module before strict loading. For the transformer, it creates a 4-bit module with group size 64, renames checkpoint keys beginning with `modulation.0.` and `time_text_embed.linear_` to their module paths, and loads the remapped weights strictly. The VAE loader separately renames some keys and selects tensors with matching shapes. This describes the code's behavior; the underlying cause of the naming differences has not been established.

## Model and licenses

The weights are not part of this repository. By default the script uses the Hugging Face cache and downloads missing files from [`mlx-community/Qwen-Image-2.1-MLX-4bit`](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit) on first use; the model is about 10.5 GB. `--model-path` accepts a local snapshot containing the text encoder, transformer, VAE, and tokenizer files. Keep snapshots outside this repository.

The repository's MIT license applies to this code, not to the weights. The model card identifies the weights as governed by the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which limits use to noncommercial research or evaluation unless a separate commercial license is obtained. Read its full terms before using the weights.
