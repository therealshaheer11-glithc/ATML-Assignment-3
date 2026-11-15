# ATML PA3 — Task 3: Test-time compute

Completed sampling/selection, signal analysis, adaptive inference and the fixed 50-question live comparison. This folder is intended to be uploaded as `task3/` inside the assignment repository, or used as a standalone Task 3 repository. Report writing and actual GitHub publication are separate steps.

## Results

| Live strategy | Correct | Mean input tokens | Mean output tokens | Mean inference seconds |
|---|---:|---:|---:|---:|
| Single longer | 33/50 (66%) | 108.38 | 640.04 | 27.40 |
| Parallel self-consistency | 30/50 (60%) | 325.14 | 1849.96 | 78.89 |
| Sequential refinement | 31/50 (62%) | 858.06 | 1157.92 | 49.67 |

All paired 95% accuracy-difference intervals include zero. Parallel candidates were generated serially at batch size 1; runtime sums measured inference and is not concurrent serving latency. Shared initial input/output tokens and runtime are charged to both strategies. The sole beneficial revision recovered a capped response with no extracted answer; no harmful revision was observed. All 24 capped outputs and 23 missing extractions remain included. None of the 50 long or 50 revision calls was capped. Live scoring produced no warnings/timeouts.

Offline: 1,000 supplied questions per dataset, split 200 development/800 held-out; 32 candidates/question and ten fixed nested orders. Fixed-32 MATH SC accuracy is 73.8875%, K-RWC 75.55%; MathVista SC 61.5375%, VisConf-RWC 63.125%. Full methods, costs, paired differences and adaptive settings are in results/offline and configs/. Thirteen reviewed figures are included. Open index.html locally for the gallery; GitHub renders figures directly but does not execute HTML notebooks.

## Verify and reproduce published numerical results (CPU)

```sh
python -m pip install -r requirements-results.txt
python verify_results.py
```

The verifier checks the release manifest, recomputes every offline method accuracy/token mean and paired accuracy interval from the included question arrays and saved bootstrap indices, and recomputes live strategy means/intervals from the 50 derived question records. It requires neither data downloads nor GPU. Task3_Results_CPU.ipynb runs the same verifier without hidden state; its outputs are intentionally cleared. In Colab set accelerator None, upload/extract this repository folder, set the working directory to it and run the notebook. Dependencies must already be installed using the command above.

For plot regeneration from included saved summaries:

```sh
python src/offline_plots.py results/offline
```

The original live presentation script expects the private raw snapshot, because it produces a complete prompt/response browser. Public numerical verification does not need that snapshot.

## Experimental implementation and full-evidence reproduction

src/scoring.py and src/selection.py implement the frozen answer rules, grouping, quality scores, ties and prefix-only stopping. src/development.py and src/evaluation.py implement the dev/evaluation separation; run them through the historical workers with src/ on PYTHONPATH. src/live_execution/ preserves the exact GPU implementation, including watchdogs, immutable per-call saving, original independent seeds and A002 integrity checking. src/colab_history/ preserves the original CPU stage/worker/check source. These historical modules depend on the original stage packages, reference manifests and Drive workspace; they are not advertised as a clean-install one-cell launcher. No GPU code runs from the results notebook or verifier.

For full CPU rescoring, obtain the private final live snapshot and offline export listed in DATA.md, extract them as private/live and private/offline, install the exact scorer requirements, then run:

```sh
python -m pip install --require-hashes -r configs/task3_scorer_requirements.lock
python -m pip install tokenizers==0.23.3
python src/live_analysis.py private/live private/offline reproduced_live
python src/live_present.py private/live reproduced_live
python src/offline_analysis.py private/offline reproduced_offline
python src/offline_allocation.py private/offline reproduced_allocation
```

For a fresh experiment, reacquire course data and model assets separately, follow historical stages 01, 02R1, 03–12, diagnostic 13D if relevant, approved 14, then guarded 15. Exact historical packages are preserved in the private project backup; this upload contains their source implementation. The old 13 starter predates A002 and is not the approved continuation. A changed Colab software image must pass the frozen checks or undergo explicit compatibility review; never bypass them. All existing generation is complete; reproducing the results requires no new GPU runs.

## Qualifications

The offline MATH scorer recorded 78 bounded comparison timeouts on test:4928; exact comparison pairs were not logged, so the precise effect is unknown. Original grades, groups, policies and denominators are preserved. No held-out retuning or relabeling occurred. Bootstrap uses 1,000 paired question-level resamples after order averaging, with pointwise percentile intervals. Adaptive held-out intervals do not establish a 1pp noninferiority guarantee. Correlations and model/dataset comparisons do not establish causal visual grounding.

All 250 live envelopes and matching attempts were audited. Five successful generation sessions total 7,976.323 seconds; unique measured inference totals 6,509.086 seconds. Earlier diagnostics/failures are excluded from those totals; complete billed GPU time and physical deallocation are not independently proven. The downloaded original execution notebook is included in notebooks/ with code preserved and outputs/private metadata cleared. Its Block 15 matches the approved starter after normalizing three local variable names. It contains historical retired cells, requires original upload packages/Drive state, and must not be run with Run all. The separate CPU results notebook is newly reconstructed. Inspecting this downloaded copy does not independently resolve the earlier cloud-save conflict.

See audit/REVIEW.md for this packaging review, DATA.md for omitted inputs, and third_party/NOTICE.md for attribution and license scope. No global license has been chosen for unrelated original coursework. Do not upload the private complete-results ZIP in place of this folder.
