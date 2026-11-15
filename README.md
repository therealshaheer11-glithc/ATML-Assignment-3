# ATML Assignment 3

**Shaheer Ahmad Masood · Advanced Topics in Machine Learning · Fall 2026**

Fine-tuning and retention, visual grounding, and test-time compute. This repository contains the final report and the available code, recorded results, figures, execution receipts, and CPU verification tools for all three tasks.

**[Read the final report](report/PA3_Report_Shaheer_Ahmad_Masood.pdf)**

Repository address recorded in the report: [therealshaheer11-glithc/ATML-Assignment-3](https://github.com/therealshaheer11-glithc/ATML-Assignment-3). Preparing this package does not confirm publication to that address.

## Explore the work

| Task | Contents | Evidence limits |
|---|---|---|
| [1: Fine-tuning and retention](ATML_PA3_Task1/README.md) | Qwen3-0.6B; Full FT, L2-SP, LoRA-4, LoRA-16 and OPLoRA; seven benchmark summaries, training receipts, source and historical workflows | Aggregate results are recorded completion receipts. Final raw-result collection, trained-model audit, diagnostics and qualitative annotation remain incomplete. |
| [2: Visual grounding](ATML_PA3_Task2/README.md) | SigLIP-Qwen, 196 visual tokens; 4,500 training updates, 1,280 recorded answers, controlled image comparisons, analysis and figures | Optional 49-token run not performed; weights and raw images external. |
| [3: Test-time compute](ATML_PA3_Task3/README.md) | 1,600 held-out offline questions and 250 live calls on 50 MATH questions; selection, refinement and adaptive stopping | Original private inputs external; offline scorer-timeout uncertainty retained. |

## Verify the package

From the repository root, this command checks file hashes, original task manifests, nested archives and Task 1 receipt arithmetic using only Python's standard library:

```sh
python3 -B verify_package.py
```

For all saved-result numerical checks, use Python 3.12 and NumPy:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-verify.txt
python -B verify_all.py
```

The second command set also replays Task 2's recorded sampling history, rescores its 1,280 answers, checks its ten tables, and recomputes Task 3's saved-array statistics and live-output extraction. A successful run ends with `"status": "PASS"`. It requires no GPU, model download, credentials or original Drive workspace. Do not run Python with `-O`: original task verifiers use assertions.

These checks reproduce saved-result analysis. They do not repeat training, generate new answers, recover missing Task 1 evidence, or freshly symbolically grade Task 3.

## Reproduce figures or inspect experiments

Each task README explains its analysis commands, data, provenance and limitations. Use separate environments for full Task 2 and Task 3 figure regeneration because their original Matplotlib pins differ. Historical Colab scripts require the original assets and Drive state; they are retained as execution evidence and are not a fresh-install training pipeline.

Useful starting points:

- [Task 1 benchmark table](ATML_PA3_Task1/results/BENCHMARK_TABLE.md), [resource table](ATML_PA3_Task1/results/RESOURCE_TABLE.md), and [status and gaps](ATML_PA3_Task1/STATUS_AND_GAPS.md)
- [Task 2 numerical tables](ATML_PA3_Task2/results/tables/), [figures and gallery](ATML_PA3_Task2/results/), and [audit](ATML_PA3_Task2/audit/MANUAL_AUDIT.md)
- [Task 3 offline results](ATML_PA3_Task3/results/offline/), [live results](ATML_PA3_Task3/results/live/), and [figures](ATML_PA3_Task3/figures/)
- [Release notes](docs/RELEASE_NOTES.md) and [verification evidence](audit/FINAL_RELEASE_CHECKS.json)

## Package and attribution

Extract the upload ZIP and place the **contents of `ATML-Assignment-3/`** at the repository root. Keep the three task-directory names and manifests intact. No nested Git repository is included. Upload and publication are a separate step.

The supplied final PDF is preserved byte-for-byte. Its editable LaTeX source was not supplied. The earlier report handoff is retained under `docs/history/`; its statements that the report was pending describe that earlier snapshot. All three task directories are preserved byte-for-byte from the supplied handoff, including historical README status wording. The current root README and release notes describe this release.

Raw source datasets, private complete-result envelopes and model checkpoints remain external as documented by each task. Upstream attribution and third-party notices are retained. No blanket license is imposed on third-party materials. LLM assistance was used; the submitting author remains responsible for the work.
