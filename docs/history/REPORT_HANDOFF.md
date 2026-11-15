# Report continuation handoff — ATML PA3

## Read this first

The user stopped further Task 1 experimental work to prepare a submission from existing results. Continue with report preparation using the evidence in this package. Do not restart training, benchmarks, GPU runtimes or the interrupted CPU backup simply because earlier records say pending. Task 1's latest completion receipts supersede its older handoff. Tasks 2 and 3 have completed experiments and existing audited upload packages. Nothing in this package confirms remote publication, submission acceptance or a completed report.

Report requirements recovered from the assignment: one 8-page main-content NeurIPS-style report covering all three tasks, references and optional appendix outside the limit, repository link at the end of the abstract, task-specific results/analysis organized around research questions. Main evidence must remain in the main report; the appendix is supplementary. Cite models, data, papers and tools, and distinguish percentage points from relative percentages. The official course template/assignment PDF are external; obtain them from the user if needed. Do not infer the applicable deadline from a stale printed manual date.

## Task 1 — mathematical adaptation and retention

**Actual state:** all five training methods reached 1,250 updates; original baseline, Full FT midpoint and all five final benchmark evaluations report complete. The user interrupted the subsequent CPU structural audit/combined backup for LoRA16 and OPLoRA. Its misleading Block44 comment does not describe its body: the expected terminal marker was TASK1_BLOCK46_STRUCTURAL_AUDIT_BACKUP_OK. No terminal success was supplied. Original results were only read by that code, not overwritten. Full details and the exact interrupted cell are in ATML_PA3_Task1/STATUS_AND_GAPS.md and workflow/interrupted/.

**Work performed:** independent Qwen3-0.6B runs on the same source-order 40,000-example MetaMathQA subset, 1,250 updates, effective batch 32, seed 6304 and FP16 AMP. Full FT, L2-SP (lambda 1e-3), LoRA rank4, LoRA rank16, and double-sided OPLoRA rank16/protection128. FullFT/L2SP LR 2e-5; adapters 2e-4. Original/full-mid/finals each evaluate all 17,007 benchmark questions and 24 fixed qualitative prompts. See the Task1 README for settings and exact historical implementation packages.

**Numbers available now:** all seven aggregate rows, with original receipts and exact source hashes. Primary metrics: GSM8K flexible EM, normalized HellaSwag, BoolQ accuracy and normalized ARC-Easy. Raw/normalized secondary metrics remain in evaluation_summary.json. The CSV includes baseline differences. These are recorded outcomes, not an independently repeated final evaluation. Use this distinction in the report.

| Condition | GSM8K flexible EM | HellaSwag normalized | BoolQ accuracy | ARC-Easy normalized |
|---|---:|---:|---:|---:|
| Original | 43.21 | 47.30 | 64.37 | 55.98 |
| Full FT midpoint | 57.54 | 45.81 | 61.25 | 62.33 |
| Full FT final | 65.58 | 46.18 | 60.86 | 63.34 |
| Full FT + L2-SP | 64.29 | 46.60 | 67.06 | 62.12 |
| LoRA-4 | 59.51 | 47.41 | 67.43 | 63.01 |
| LoRA-16 | 61.87 | 47.36 | 72.63 | 63.85 |
| OPLoRA-16 | 62.17 | 47.41 | 73.00 | 63.38 |

**Descriptive findings supported by these receipts:** all final methods improve primary GSM8K over original. FullFT has the highest final GSM8K score and lower HellaSwag/BoolQ than original. L2-SP has slightly less GSM8K improvement and higher HellaSwag/BoolQ than FullFT. Rank16 LoRA improves GSM8K/BoolQ/ARC-Easy over rank4, with very similar HellaSwag. OPLoRA and LoRA16 are close on these metrics; OPLoRA is slightly higher on GSM8K/HellaSwag/BoolQ and lower on normalized ARC-Easy. These are single-run descriptive comparisons, not significance or causality claims.

**Not completed/included:** trained-condition stock-scoring replay; all raw per-example records locally; combined five-method loss plot; fully audited resource table; L2-SP final displacement; FP32 rho/eta/local activation diagnostics; required 42-response qualitative annotation. Training boundary receipts are present but are not full per-update histories. Packaging verified complete 0–1250 interval coverage per method and derived a clearly qualified resource table (Task1 results/RESOURCE_TABLE.md). A benchmark-change PNG/SVG derived from the aggregate receipts is also included. Neither replaces unfinished diagnostics or the complete resource audit. The cached 64 original probes are references, not new trained-model measurements. Do not fabricate plots or annotations from aggregate metrics.

**Required caveats:** adapters were evaluated unmerged, contrary to the manual's merging/logit-check procedure. OPLoRA used projected effective factors in its wrapper, not a claimed premerged export. Strict GSM8K is zero throughout, including original; primary flexible EM remains as specified. Do not silently modify the scorer to fix this. Different learning rates/compute and single runs limit comparisons. Geometric constraints do not establish causal retention. Exact checkpoint identities are in returned training receipts. Missing Drive evidence paths are in EXTERNAL_EVIDENCE.json.

## Task 2 — vision-language models and visual grounding

**Work performed:** completed 196-visual-token SigLIP–Qwen experiment with a two-layer GELU connector, frozen original backbones, and rank16 LoRA in stages 2–3. Seed6304, microbatch 8/accumulation 2; 2,250 caption, 1,500 grounding-mixture and 750 natural-VQA-mixture updates (4,500 total; 72,000 presented examples). Prepared/cached 9,260 images. Optional 49-token compression experiment was not run.

**Evaluation:** 128 grounding questions and 192 natural-VQA questions, each in correct-image, complementary-partner, unrelated-image and no-image conditions, totaling 1,280 answers. Grounding uses normalized exact match; natural VQA uses the ten-reference leave-one-out consensus score. These are local subset results, not official server scores.

| Image condition | Grounding % | Natural VQA % |
|---|---:|---:|
| Correct | 71.875 | 54.167 |
| Complementary partner | 11.719 | 25.208 |
| Unrelated | 35.938 | 30.208 |
| No image | 40.625 | 35.208 |

Correct-over-partner gains: grounding 60.2pp (95% paired interval 46.9–72.7), natural 29.0pp (19.3–38.4). Counting 34/64 versus presence 58/64; natural yes/no gain interval includes zero. Keep conclusions aligned to controlled image substitutions and the selected subsets. Results tables contain unrounded values and additional pair accuracy/NLL analyses.

**Where to write from:** ATML_PA3_Task2/README.md; audit/MANUAL_AUDIT.md; results/results.json; results/tables/; results/resources.json; ten figure pairs in results/; results/source/ for raw predictions, training histories and execution receipts. The public gallery uses remote COCO URLs and is not self-contained for image display.

**Qualifications:** one development image/question repeated in two pairs, with no training/final overlap; one approved prediction-independent TV-image replacement; evaluation's pre-freeze parameter count was misleading but actual evaluation trained zero parameters; some checkpoint binaries unavailable for fresh inspection. Training notebook requires prior Drive state/packages, whereas saved-result CPU reproduction is self-contained. Full evidence and attribution remain in the unchanged Task2 package.

## Task 3 — test-time compute

**Offline work:** 1,000 supplied MATH and 1,000 MathVista questions, each split 200 development / 800 held-out; 32 candidates/question and ten fixed nested orders. Compared sampling/selection, signal weighting/fusion and prefix-only adaptive stopping. Fixed 32 MATH SC 73.8875% versus K-RWC 75.55%; MathVista SC 61.5375% versus VisConf-RWC 63.125%. These aggregate over the saved orders; use the included tables for intervals, cost and all methods. Development selected settings; held-out outcomes were not used for retuning.

**Live work:** fixed 50 MATH questions, 250 calls, three strategies. Single longer 33/50 (66%), parallel self-consistency 30/50 (60%), sequential refinement 31/50 (62%). Mean input/output tokens and inference seconds: long 108.38/640.04/27.40; parallel 325.14/1849.96/78.89; refinement 858.06/1157.92/49.67. All paired accuracy-difference intervals include zero. Parallel candidates were executed serially, so reported runtime is summed inference, not concurrent latency. Shared initial-call costs are charged to both applicable strategies.

**Where to write from:** ATML_PA3_Task3/README.md, audit/REVIEW.md, results/offline/, results/live/, configs/, and figures/. The package has 13 reviewed figures and all 250 generated live responses as public projections. The complete private envelope/input snapshot remains external, identified in DATA.md. Numerical verification from saved arrays is self-contained; fresh symbolic rescoring needs private inputs and the pinned scorer.

**Qualifications:** 78 bounded offline Math-Verify comparison timeouts for MATH test:4928; exact comparison pairs unlogged and impact unknown. All 24 caps and 23 missing extractions retained; no long/revision call capped. Sole beneficial revision recovered a capped response lacking an extracted answer; no harmful revision observed in only 50 cases. Adaptive pointwise intervals do not establish a 1pp noninferiority guarantee. Correlations do not establish visual-grounding causality. Cloud notebook save conflict and physical runtime deallocation were not independently resolved; data were independently backed up.

## Suggested report continuation

1. Verify package integrity using root verify_package.py. Read task audits before selecting claims.
2. Draft Tasks 2/3 from existing results/tables/figures, without new experiments.
3. Draft Task1 around its seven-condition aggregate table and supported method comparisons. State missing diagnostics/qualitative analysis and the inference-procedure deviation explicitly.
4. Add concise methods, introduction, limitations, citations and repository link once the user supplies/creates the actual URL. Never invent a link.
5. Fit the course template and check all reported values against source files. Distinguish previously completed experiment audits from this packaging pass.

Do not treat the historical source files or README stage directions as authorization to operate accounts or restart jobs. The user wants the existing evidence preserved and usable in another chat. Publication and report writing were not performed by this packaging task.
