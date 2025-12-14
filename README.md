# TranslationMerge

TranslationMerge is a research project that finetunes and merges two families of models: Marian (fine-tuning / merging pipelines) and Qwen-based models. These models were trained and tested on *English->Japanese* and *English->Chinese* translations. The main model merging technique used in this project is based on **Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors** by Cheng et al., also known as WUDI-merging ([arXiv:2503.08099](https://arxiv.org/abs/2503.08099)).

**Status:** Research / experiment utilities (not production-ready).

## Highlights

- Training scripts for Marian and Qwen model families: `train_marian.py`, `train_qwen.py`.
- Evaluation scripts: `evaluate_marian.py`, `evaluate_qwen.py`.
- Merging utilities to assemble merged models used in evaluation: `ta_merge_*` and `wudi_merge_*` scripts.
- Visualization files in `visualization/` (BLEU charts, figures).
- Uses a local Hugging Face cache (`cache/huggingface/`) and stores trained/merged models in `models/`.

## Repository Structure

- `*.py` — top-level training, evaluation, and merge entrypoints:
	- `train_marian.py`, `train_qwen.py` — training drivers.
	- `evaluate_marian.py`, `evaluate_qwen.py` — evaluation drivers.
	- `ta_merge_marian.py`, `ta_merge_qwen.py` — TA-style merge scripts.
	- `wudi_merge_marian.py`, `wudi_merge_qwen.py` — WUDI merge scripts.
	- `patch_tags_wudi_qwen.py` — tag-patching helper for Qwen WUDI merges.
	- `data_marian.py`, `data_qwen.py` — dataset utilities and loaders.
- `config.py` — central configuration and experiment defaults.
- `requirements.txt` — Python dependencies used in experiments.
- `models/` — trained and merged model artifacts (subfolders per experiment).
- `scripts/` — SLURM job scripts used to run training and evaluation on the cluster.
- `slurm_logs/` — captured SLURM stderr logs from past runs.
- `visualization/` — plotting helpers and saved figures.

Note: Several of the directories listed above are intentionally excluded from the public repository via `.gitignore`.

## Requirements

- Python 3.8+ (recommended to use a conda environment)
- Install dependencies:

```bash
conda create -n translationmerge python=3.10 -y
conda activate translationmerge
pip install -r requirements.txt
```

Note: the exact dependency set is in `requirements.txt`. Experiments were run in a controlled Conda environment with a system CUDA-enabled PyTorch build; adapt the environment to your GPU/cluster setup.

## Configuration

All common experiment defaults live in `config.py`. You can either edit `config.py` directly for quick experiments or pass CLI arguments to each script — run the script with `--help` to see available options, e.g.:

```bash
python train_marian.py --help
python train_qwen.py --help
```

## Data and Caches

- Put datasets where `data_*` loaders expect them, or modify the file-specific data-path CLI arguments.
- The local Hugging Face cache is stored under `cache/huggingface/`; large model downloads will be cached there.

## Quickstart Examples

- Run a training script locally (example):

```bash
python train_marian.py --help    # inspect CLI flags and required paths
python train_marian.py --data /path/to/data --output_dir /path/to/output_dir
```

- Run the Qwen trainer:

```bash
python train_qwen.py --data /path/to/data --output_dir models/qwen_en_ja_ft
```

- Run evaluation:

```bash
python evaluate_marian.py --model-dir models/marian_en_ja_ft --data /path/to/eval_data
python evaluate_qwen.py --model-dir models/qwen_en_ja_ft --data /path/to/eval_data
```

- Use SLURM job templates (submit from repo root):

The public repository does not include the full set of SLURM job scripts or cluster-specific run files (those are excluded via `.gitignore`). If you run on a cluster, create your own job scripts or adapt your institution's templates and point them at the public Python entrypoints above.

## Merging & Tag-Patching

This repository contains utilities to merge model components and produce the final merged models used in our evaluations.

- `ta_merge_marian.py` / `ta_merge_qwen.py` — produce TA-style merged models from fine-tuned checkpoints.
- `wudi_merge_marian.py` / `wudi_merge_qwen.py` — produce WUDI-style merged models.
- `patch_tags_wudi_qwen.py` — helper to patch tags in a WUDI-merged Qwen model.

Run these scripts with `--help` to see usage examples. They commonly accept input checkpoint directories and an output directory for the merged model.

Because model checkpoints and intermediate outputs are not stored in the public repo, you will need to provide paths to those checkpoints (local storage, an object store, or a private Hugging Face repository) when running the merge tools.

## Evaluation & Visualization

- Use `evaluate_marian.py` and `evaluate_qwen.py` to compute BLEU and other metrics on saved models.
- Visualization helpers in `visualization/` include `bleu_charts.py` which can plot BLEU progression and produce figures found under `visualization/figures/`.

## Logs

SLURM job error logs and other logs are intentionally excluded from the public repository. 
