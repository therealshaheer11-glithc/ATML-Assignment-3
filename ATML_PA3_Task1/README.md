# Task 1: mathematical adaptation and retention

All five training runs and all seven benchmark evaluations have completion receipts. Further Task 1 work was stopped by the user before the final structural backup/audit finished. This is a **partial-evidence publication package**, not a complete raw-results archive and not a certification of full assignment compliance.

## Start here

- [Current status and stopping point](STATUS_AND_GAPS.md)
- [Benchmark table](results/BENCHMARK_TABLE.md), [full-precision CSV](results/primary_metrics.csv), and [all raw/normalized aggregate metrics](results/evaluation_summary.json)
- [Returned execution receipts](evidence/returned_receipts/) and [source provenance](provenance/returned_receipt_sources.json)
- [Training session evidence](results/training_session_receipts.json), [receipt-derived resource table](results/RESOURCE_TABLE.md), and [benchmark-change figure](figures/benchmark_changes.png)
- [External evidence checklist](EXTERNAL_EVIDENCE.md)

The root `verify_package.py` checks package integrity and recomputes the table from included receipts using only Python's standard library. It does not rerun models, replay per-example scoring, or verify inaccessible Drive files.

## Method

Independent adaptations start from Qwen/Qwen3-0.6B, with thinking disabled, seed 6304, the first 40,000 eligible MetaMathQA examples in source order, GSM8K test decontamination, and no answer truncation. Each method uses 1,250 successful updates, microbatch 2, accumulation 16, length at most 2,048, FP16 autocast, FP32 trainable parameters and Adam states, gradient checkpointing, cosine decay and 38 warmup updates. Response-token cross-entropy uses a token-weighted effective-batch denominator.

Full FT and L2-SP use LR 2e-5; L2-SP adds lambda 1e-3 times half the squared displacement from original parameters. LoRA-4 uses rank 4/alpha 8 (2,523,136 parameters); LoRA-16 and OPLoRA use rank 16/alpha 32 (10,092,544 parameters), LR 2e-4 and dropout .05. Original full-model parameters: 596,049,920. All 196 attention/MLP target projections are adapted. OPLoRA uses frozen FP32 leading input/output bases at k=128 and double-sided effective-factor projection. The intended matched LoRA16/OPLoRA initialization is documented in the frozen source and preparation checks; independent final audit remains unfinished.

Evaluation covers GSM8K test 1,319; HellaSwag validation 10,042; BoolQ validation 3,270; ARC-Easy test 2,376 per condition. GSM8K is zero-shot, non-thinking native chat, greedy, capped at 1,024 new tokens. Likelihood tasks use stock completion prompts. GSM generation uses batch 8; the current evaluation loaders describe likelihood batch 2. Do not describe every evaluation operation as batch 8. Qualitative evaluation uses 24 fixed prompts with a 512-token cap per condition. Runtime/environment source pins are preserved in historical code/packages (private Python 3.12.12, Torch 2.8.0+cu126, Transformers 4.57.1, lm_eval 0.4.9.2, datasets 4.3.0, accelerate 1.11.0, tokenizers .22.1, NumPy 2.2.6, PEFT .17.1).

## Historical code is not a fresh-run launcher

`src_historical/task1/` is the earlier common implementation. Later approved changes are in `workflow/history/` ZIPs, which retain exact bytes, manifests and implementation identities. Prefer Block43 for the executed LoRA16 evaluation and Block45_R2 for the executed OPLoRA evaluation. Block40/41 contain the later training loops. The history also includes superseded and failed versions for provenance; their presence is not a recommendation to run them. The root handoff and current status supersede old embedded readiness messages. Do not run all cells or launch GPU work to write the report. The known-corrupt retired Block32 exporter and local build helpers are excluded where identified.

Original data/model archives, full per-question Task 1 results, per-update logs, probe tensors, SVD bases and checkpoint binaries are external. They are not hidden inside the small workflow ZIPs. Existing completion receipts provide aggregate evidence, not a second experiment replication.

## Attribution

Model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Training data: [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA). Evaluation: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), GSM8K, HellaSwag, BoolQ and ARC-Easy. Orthogonal adaptation follows the assignment recipe informed by [OPLoRA](https://arxiv.org/abs/2510.13003). Qualitative prompt provenance and the pinned upstream commit/hash are retained in `workflow/history/task1_preflight_review_R2_2026-11-12/QUALITATIVE_SOURCE_APPROVAL.json`. LLM coding assistance was used. No blanket license for upstream datasets, weights or unrelated coursework is asserted.
