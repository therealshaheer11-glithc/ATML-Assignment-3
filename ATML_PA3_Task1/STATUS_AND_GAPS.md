# Current Task 1 status — 15 November 2026

## Confirmed recorded completion

Original baseline, Full FT midpoint (625), and all five final evaluations (1250) each report 17,007 benchmark answers and 24 qualitative responses. This is 119,049 example-condition evaluations and 168 qualitative generations in the receipts; the raw records themselves are not all local. Training receipts reach 1,250 updates for Full FT, L2-SP, LoRA4, LoRA16 and OPLoRA. OPLoRA's final receipt covers only the resumed 1072–1250 segment; it is not its total training time. The original baseline also has a user-returned successful complete stock-scoring replay/export receipt. Trained-condition final replay is not established.

The old handoff said LoRA16 evaluation and OPLoRA training/evaluation were pending. That is superseded by the later completion receipts included here. Do not restart those runs based on the old handoff.

## Exact stopping point

After completion of both final adapter evaluations, the user stopped a CPU-only collection/checking/backup cell. Its first comment incorrectly says Block44 adapter export; its body expects completed LoRA16/OPLoRA summaries, reads both result trees, checks IDs/contracts/hash chains, reaggregates stored scores and builds a combined evaluation ZIP. Its success marker would be TASK1_BLOCK46_STRUCTURAL_AUDIT_BACKUP_OK. No terminal success output was supplied. The completed combined backup cannot be claimed. The interrupted script and structured stop record are preserved in workflow/interrupted/.

This cell contains no training or inference. It does not modify the source evaluation files or model checkpoints. An interrupted temporary ZIP or pending Drive copy is not a verified export. No actual interruption offset, current runtime allocation or Drive contents were inspected in this packaging pass. Do not promise that a ZIP was published or that physical GPU release occurred.

A receipt-derived resource table is now included in results/RESOURCE_TABLE.md. Packaging verified continuous, nonoverlapping 0–1250 update coverage for each method before summing recorded training/checkpoint segments. This does not fill missing full logs, method-specific overhead or billing evidence.

## Still incomplete or unavailable here

1. Complete local collection of Task 1 per-example outputs, GSM8K extractions/cap flags, qualitative responses, full training histories and numerical diagnostic inputs.
2. Final independent trained-condition scoring/metric/manual-compliance audit. Included aggregate values trace to returned receipts, not local per-example recomputation.
3. Combined SFT curves, L2-SP penalty analysis and full-model displacement. Session boundary receipts do not replace per-update losses.
4. Fully reconciled resource comparison across all training sessions, failures, checkpointing and preprocessing. Do not turn one resume segment into full-run time or equate summed inference with billed runtime.
5. FP32 down_proj diagnostics at layers 0,13,27: rho, eta, and correction response on the same 64 cached original activations. Cached original probes are references, not completed trained-model diagnostics. Recoverable in principle from preserved weights/bases/probes, but not computed or audited here.
6. Required review of 42 fixed qualitative responses (first/last question, three formats, seven conditions). Do not invent annotations or use examples selected for dramatic outcomes.
7. Report and final submission. This package is not a report, a remote publication, or proof of late acceptance.

## Methodological qualifications for the report

- Manual requires merged adapters and a merged/unmerged logit check. Actual final adapter evaluations used the training AdapterLinear wrapper with FP32 factors and frozen FP16 base; OPLoRA projects inside the wrapper. This is a known departure from the prescribed inference/export procedure. Do not describe inference as merged, or claim an audited equivalence check that is absent.
- Strict-match GSM8K is zero for all seven conditions, including baseline. The prescribed primary metric is stock flexible-extraction exact match, which is preserved unchanged. The strict-score pattern still needs an extraction/format explanation; it is not grounds to invent corrected numbers or label all answers incorrect under the primary metric.
- Single fixed training runs do not quantify run-to-run variability. Differences are descriptive; no Task 1 statistical-significance claim is established.
- Full FT and adapters have different prescribed learning rates; equal update/example budgets are not equal compute.
- Geometric constraints alone do not prove retention or causal explanations of behavior.

## Current scope

User stopped further Task 1 experimental work and requested packaging existing evidence for report writing. Preserve checkpoints and original records. Do not resume training, benchmarks or the interrupted backup without a new request. Writing may proceed from the recorded table and explicit limitations; missing diagnostic sections must be disclosed.
