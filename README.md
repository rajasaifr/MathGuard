# MathGuard ? Lightweight Verification of LLM Mathematical Reasoning

MathGuard explores whether feedback from a small mathematical claim checker can improve a larger language model's judgments. This repository contains a LoRA fine-tuning and evaluation notebook, synthetic datasets, and preserved experimental CSVs. It evaluates claim validity, rather than formally verifying complete proofs or reasoning traces.

## Architecture

1. **Baseline:** `Qwen/Qwen3-4B-Instruct-2507` judges a claim and produces a brief explanation. The notebook loads it in 4-bit NF4 with float16 computation; standard evaluation uses double quantization.
2. **Specialist:** `Qwen/Qwen2.5-0.5B-Instruct`, fine-tuned with LoRA, independently returns a validity label and explanation.
3. **Combined:** the main model receives the original claim, its baseline response, and specialist feedback. The prompt asks it to verify independently because the specialist may be wrong.

Generation uses `do_sample=False`, with up to 110 new tokens for the main model and 80 for the specialist. The label parser takes the first whole-word `VALID` or `INVALID`, case-insensitively; absent labels become `UNKNOWN`. This is model-assisted checking, not a symbolic correctness guarantee. SymPy was included in the original installation command but is not used as a verifier.


The local workspace also retains ignored `local_archive/` (original notebook, datasets, CSVs, and SHA-256 manifest) and `math_slm_lora/` (the existing adapter and tokenizer). These are not intended for GitHub. The workspace directory was kept in place to preserve its existing Git repository.

## Datasets

The original notebook generates six balanced synthetic claim families: principal square roots, cancellation at a zero denominator, stationary points versus extrema, ordering from squared values, inequality division, and squared absolute values. Training uses integer parameters 2?21; the standard test uses 30?33. Python and Torch seeds are set to 42 in the notebook.

| File | Rows | VALID / INVALID | Distinct claim texts |
| --- | ---: | ---: | ---: |
| train.json | 240 | 120 / 120 | 202 |
| test.json | 48 | 24 / 24 | 42 |
| hard_test.json | 24 | 12 / 12 | 24 |

Standard examples contain `problem`, `label`, `reason`, and `type`; hard examples contain `problem` and `label`. The hard set is a manually specified collection covering analysis, linear algebra, probability, sequences/series, and graph theory. It is not an established external benchmark. All JSONs are preserved byte-for-byte.

**Dataset caveats:** two distinct claim texts occur in both training and the standard test, because the stationary-point template does not use its integer parameter. Both splits also contain repeated texts. Numerical parameter separation therefore does not imply complete text separation. Template explanations are shared across labels and are not always tailored to an individual example. Some hard claims leave domain assumptions implicit (for example, interior points for the derivative-at-a-minimum statement). Labels are preserved, not independently certified.

## LoRA fine-tuning

The notebook and saved adapter configuration support the following setup:

| Setting | Value |
| --- | --- |
| Base specialist | Qwen/Qwen2.5-0.5B-Instruct |
| Target modules | q_proj, v_proj |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Bias / task | none / CAUSAL_LM |
| Epochs | 3 |
| Per-device batch / gradient accumulation | 2 / 4 |
| Learning rate | 0.0002 |
| Maximum sequence length | 256 |
| Precision / objective | FP16 / completion-only loss |
| Checkpoint strategy | no intermediate saves; final adapter saved |

Training uses TRL's `SFTTrainer` with prompt/completion conversations. The assistant completion contains the dataset label and reason. The original notebook metadata specifies a Colab T4 GPU. No training was executed during repository preparation.

## Evaluation and recorded results

Accuracy is the fraction of rows whose parsed prediction matches the preserved reference label. The table below is calculated directly from existing CSV correctness columns, not a new inference run. Rows and predictions were checked against their datasets and recorded correctness flags.

| System | Standard test | Hard test |
| --- | ---: | ---: |
| Main model baseline | 46/48 (95.83%) | 20/24 (83.33%) |
| LoRA specialist | 40/48 (83.33%) | 16/24 (66.67%) |
| Main model + specialist | 48/48 (100.00%) | 21/24 (87.50%) |

On the standard set, the combined system corrects two baseline mistakes and regresses on zero cases. On the hard set, it corrects four mistakes and regresses on three cases, for a net gain of one correct answer. The CSVs retain raw responses, predictions, labels, and correctness flags; combined files also retain baseline and specialist correctness.

These are small, single recorded evaluations with no confidence intervals or repeated-seed study. There is no measured latency, memory benchmark, comparison to an untrained specialist, or main-model self-revision control. Thus the data do not isolate the effect of LoRA or specialist feedback from the effect of another main-model generation. ?Lightweight? describes the specialist's relative model size and adapter training, not a measured end-to-end efficiency improvement. Hard combined evaluation omits the double-quantization option used in the standard baseline; prompts also differ across phases. No generalization or formal verification claim follows from the perfect standard score.

## Usage and reproducibility

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab notebooks/mathguard.ipynb
```

The notebook defaults to `RUN_EXPERIMENTS = False`: running all cells only reads and summarizes saved results with the Python standard library. It does not download models or train. A CUDA-capable environment is required for the optional original model workflow; bitsandbytes installation depends on platform support.

To run a new experiment, explicitly set `RUN_EXPERIMENTS = True` and execute cells in order. This includes model downloads, LoRA training, and all evaluations. New adapters, training files, and result CSVs go into an ignored, unique `runs/<timestamp>/` directory; `data/` and `results/` remain read-only inputs. The historical adapter remains locally in `math_slm_lora/`; the full sequential notebook creates its own new adapter for subsequent hard evaluation.

The cleaned notebook removes package-uninstall troubleshooting, duplicate helper definitions, stale Colab Drive paths, download/widget output, and an empty cell. It fixes the deleted-model dependency by loading the main model before combined evaluation. Recorded outputs remain in the archived original notebook and CSVs. No model code was executed during cleanup.

`requirements.txt` follows the notebook's actual installation requirements and imports, with JupyterLab added for local use. It is not an exact environment lock: the original run did not capture all package versions or model revision hashes. Exact numerical reproduction is therefore not guaranteed. No dataset or project license was supplied; no license has been invented.

## Local preservation and Git hygiene

`local_archive/sha256.json` records original notebook, dataset, and result hashes. `.gitignore` excludes the original archive, trained adapters/models, training/checkpoint directories, common weight extensions, environments, caches, credentials, token files, and temporary files. Keep tokens in environment variables rather than notebook cells. Git ignore rules do not remove files already tracked; repository preparation checks the index separately.

Nothing has been committed, uploaded, or pushed by this preparation.
