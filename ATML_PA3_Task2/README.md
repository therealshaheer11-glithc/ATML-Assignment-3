# Task 2: Vision-language models and visual grounding

This package contains the completed **196-visual-token SigLIP–Qwen experiment**, its frozen selections, final implementation, raw recorded predictions and training logs, and reproducible CPU analysis. The optional 49-token experiment was not run. The assignment report is separate.

## Reproduce the saved results

From this directory, using Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python reproduce.py
```

On Windows, activate with `.venv\Scripts\activate`. The command needs no GPU, model weights, account credentials or dataset download. It verifies the SHA-256 inventory and original artifact receipts, checks the frozen splits, replays all 4,500 training updates, validates 160 evaluation batches, rescores 1,280 raw answers, recomputes the 4,000-pair bootstrap intervals, and regenerates ten figures in PNG and SVG. Outputs go to `reproduced/`. Success prints `"status": "passed"`; any failed check exits with an error. Do not use `python -O`, which disables assertions in the original validation code.

For verification only, run `python reproduce.py --verify-only`. The equivalent notebook is [notebooks/Reproduce_results.ipynb](notebooks/Reproduce_results.ipynb). The CPU command reproduces analysis of the recorded run; it does not retrain the model or regenerate answers.

## Final held-out results

Scores are local subset results, not VQA server scores. Grounding uses normalized exact match; natural VQA uses all ten reference answers and the leave-one-out consensus formula.

| Image input | Grounding, 128 questions | Natural VQA, 192 questions |
|---|---:|---:|
| Correct | 71.875% | 54.167% |
| Complementary partner | 11.719% | 25.208% |
| Unrelated | 35.938% | 30.208% |
| No image | 40.625% | 35.208% |

The correct-over-partner accuracy gains are **60.2 percentage points** for grounding (95% paired interval 46.9–72.7) and **29.0 points** for natural VQA (19.3–38.4). The paired comparison holds the question and visual prefix length fixed. Counting is a weakness: 34/64 correct, compared with 58/64 presence answers. The natural yes/no subgroup's gain interval includes zero. See [the audit](audit/MANUAL_AUDIT.md) for qualifications and [the complete tables](results/tables/) for all metrics and intervals.

## What is included

| Location | Contents |
|---|---|
| `task2/` | Final model interface, curriculum, sampler, loss, training, evaluation and analysis code |
| `data/frozen/` | Exact selected IDs, targets, references, caption order, annotation pools and human-review provenance |
| `results/` | Original numerical results, ten figures, tables and resource measurements |
| `results/source/` | Recorded raw generations, 4,500 update logs, 12 development records, recipes and receipts |
| `workflow/` | Original final-revision Colab loaders and packaged workers; prerequisites and stage order |
| `environment/` | Original observed environments and the environment used for package verification |
| `provenance/` | Original archive and source-package hashes |
| `audit/` | Experiment audit and publication review |
| `third_party/` | Attributed VQA normalization code and retained upstream source |

Open [results/all_final_pairs.html](results/all_final_pairs.html) in a browser for all 160 pairs and four prediction conditions. It loads images from the original COCO URLs and therefore requires internet access; no JPEGs are embedded. For a verified offline gallery, obtain the 320 original val2014 images listed in `results/source/gallery_image_receipts.json` and run:

```sh
python reproduce.py --images /path/to/val2014
```

The command checks every image hash before copying it into the ignored `reproduced/` folder. Image permissions remain with the original sources. Example-selection rule: the gallery contains every final pair, with no selection based on model performance.

## Training and answer generation

See [workflow/README.md](workflow/README.md) before using the GPU stages. Those scripts are the actual, hash-bound Colab workflow used for this run. They depend on the original Drive directory and earlier verified artifacts. They are **not a one-click fresh training pipeline**. Re-running them without those prerequisites will stop at the verification checks. The final checkpoint, cached features, source archives and model weights are deliberately external; their identities and observed verification receipts are included. Do not remove integrity checks to bypass missing state.

Model revisions, seed and schedules are in [task2/protocol.py](task2/protocol.py). This run used seed 6304, microbatch 8 with accumulation 2, 2,250/1,500/750 updates, frozen SigLIP and original Qwen weights, a two-layer GELU connector, and rank-16 LoRA in stages 2–3. The approved initial FP16 gradient scale was 128. Full measured environment and decoding settings are in `results/source/evaluation_recipe.json`; per-stage resource costs are in `results/resources.json`.

## Qualifications and attribution

- The original prepared dataset manifest records `dataset_frozen: false`; the later passed `results/source/freeze_receipt.json` records the actual freeze. These historical records are preserved rather than rewritten.
- One grounding development image/question occurs in two pairs. It never enters training or final evaluation; all 320 original final-evaluation images are distinct.
- One prediction-independent manual TV-image replacement was approved during training-data review. The earlier proposal receipt still says pending; `data/frozen/provenance/approval_A012.json` and `human_chat_acceptance.json` record the later acceptance. Raw evidence is preserved unchanged.
- The original evaluation log counted trainable parameters before freezing. Actual evaluation trainable parameters were zero; no evaluation updates occurred.
- Historical checkpoint binaries were not all available locally for this audit. CPU verification checks the saved evidence; it does not claim a second GPU replication.
- The original execution notebook contains retired and repeated cells. Its cell code is preserved, but output and private notebook metadata have been removed. Use the documented final loaders, not Run all.

Materially reused VQA normalization code is attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). COCO, VQAv2, SigLIP and Qwen retain their respective terms. LLM coding assistance was used; the submitting author remains responsible for understanding the implementation. No assignment PDF, full raw datasets, model weights or checkpoint binaries are included.

To upload, put the **contents of this folder** at the intended GitHub task location. Keep `SHA256_MANIFEST.json` with them. If combining with Tasks 1 and 3, add links to their READMEs in the repository's top-level README. This package has not been pushed to GitHub and does not include the deferred report.
