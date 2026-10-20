# MathGuard

### Lightweight Verification of LLM Mathematical Reasoning

MathGuard investigates whether a LoRA-trained small language model can improve a larger language model's mathematical judgments, especially when mathematical assumptions are overlooked.

## 1. Research Motivation and Blind Spot

My interest in mathematics and research on language-model reliability motivated me to investigate whether LLMs can distinguish valid mathematical conclusions from statements that omit necessary assumptions.

For example, a stationary point does not necessarily indicate a local extremum, and algebraic cancellation is invalid when the denominator is zero.

MathGuard studies assumption-sensitive mathematical claim verification, rather than complete proof verification.

## 2. Research Questions

1. How accurately does Qwen3-4B identify valid and invalid mathematical claims?
2. Can a LoRA-adapted 0.5B model learn to recognize mathematical reasoning errors?
3. Does feedback from the smaller model improve the larger model's judgments?

## 3. Architecture

- **Main LLM:** `Qwen/Qwen3-4B-Instruct-2507`, loaded using 4-bit NF4 quantization.
- **Specialist SLM:** `Qwen/Qwen2.5-0.5B-Instruct`, fine-tuned with LoRA.
- **Combined system:** The main model receives its original answer and feedback from the specialist, then independently revisits its judgment.

The SLM offers advisory feedback, not formal mathematical verification.

## 4. Dataset

| Dataset | Examples | Description |
|---|---:|---|
| Training | 240 | Six families of synthetic mathematical claims |
| Standard evaluation | 48 | Familiar mathematical structures with different numerical parameters |
| Hard evaluation | 24 | Different mathematical structures from analysis, algebra, probability, and graph theory |

The six synthetic families cover square roots, algebraic cancellation, stationary points, squared inequalities, division of inequalities, and absolute values.

The standard split shares templates and some exact examples with training. The hard test is a small, separately specified set, not an established benchmark.

## 5. LoRA Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Rank / Alpha | 8 / 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 3 |
| Learning rate | `2e-4` |
| Effective batch size | 8 |
| Max sequence length | 256 |
| Precision | FP16 |

Training uses supervised fine-tuning with validity labels and explanations.

## 6. Experimental Results

| System | Standard (48) | Hard (24) |
|---|---:|---:|
| Qwen3-4B alone | 95.83% | 83.33% |
| LoRA-trained SLM | 83.33% | 66.67% |
| **Qwen3-4B + SLM** | **100.00%** | **87.50%** |

On the standard evaluation, feedback corrected two baseline errors and introduced no regressions.

On the hard evaluation, feedback corrected four errors but introduced three regressions, producing a net improvement of one correct answer out of 24.

These results suggest that specialist feedback can sometimes improve mathematical judgments, but that even correct feedback is not necessarily used correctly by the main model.

## 7. Proposed Path Forward

**Assumption-focused data:** Curate counterfactual mathematical claims that differ by one essential assumption, such as positivity, domain restrictions, or invertibility.

**Specialist adaptation:** Train the SLM to identify the exact invalid inference or missing assumption, rather than predicting only `VALID` or `INVALID`.

**Verification-gated revision:** Introduce symbolic tools where applicable, and accept revised judgments only when supported by a valid mathematical check.

These are proposed improvements, not experimentally demonstrated solutions.

## 8. Repository Structure

```text
MathGuard/
├── notebooks/
│   └── mathguard.ipynb
├── data/
│   ├── train.json
│   ├── test.json
│   └── hard_test.json
├── results/
│   ├── baseline.csv
│   ├── slm_results.csv
│   ├── combined_results.csv
│   ├── hard_baseline.csv
│   ├── hard_slm.csv
│   └── hard_combined.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Reproducibility

```bash
pip install -r requirements.txt
jupyter lab notebooks/mathguard.ipynb
```

The notebook defaults to reviewing saved results. Set `RUN_EXPERIMENTS = True` to execute the model and training workflows in a compatible GPU environment. Experiments were conducted using Google Colab with a T4 GPU.

## 10. Limitations

- Standard evaluation shares training templates and contains some exact overlap.
- The hard set contains only 24 claims, with some implicit mathematical assumptions.
- No repeated-seed evaluation or statistical confidence intervals were conducted.
- There is no untrained-SLM or LLM-only self-revision control.
- Latency and compute efficiency were not measured.
- Claims were not formally verified, and some labels require independent checking.
- Prompt and quantization differences limit strict comparisons across phases.

The results are preliminary and do not establish broad mathematical generalization.

## 11. Conclusion

MathGuard demonstrates a functioning LLM–SLM mathematical claim-checking pipeline. The LoRA-trained specialist provides feedback that can improve some decisions, but the larger model can also make incorrect revisions.

Future work could investigate assumption-focused training and symbolic verification to make model collaboration more reliable.

**Models:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
