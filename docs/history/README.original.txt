# ATML PA3 — all three tasks

Combined GitHub upload and report-continuation package, assembled 15 November 2026. **Read [REPORT_HANDOFF.md](REPORT_HANDOFF.md) first when continuing in another chat.**

| Task | Included evidence | Completion limits |
|---|---|---|
| [Task 1: adaptation and retention](ATML_PA3_Task1/README.md) | Seven complete benchmark summaries, training/evaluation receipts, source and historical Colab packages | Final raw-result collection/backup, trained-model audit, diagnostics and qualitative annotation unfinished |
| [Task 2: visual grounding](ATML_PA3_Task2/README.md) | Existing upload package, unchanged: recorded predictions, training logs, splits, analysis, figures and CPU reproduction | Optional 49-token experiment not run; weights/raw images external |
| [Task 3: test-time compute](ATML_PA3_Task3/README.md) | Existing upload package, unchanged: offline/live numerical results, 250 generated responses, figures, code and CPU verification | Original private inputs/raw envelopes external; documented scorer-timeout uncertainty |

## Use on GitHub

Extract this ZIP and upload the **contents of ATML_PA3_All_Tasks/** as the repository root. Keep the three task directories and their manifests intact. This task prepared files only; it did not create/push a remote repository. No final report or LMS submission is included.

## Verify locally without models or a GPU

```sh
python3 verify_package.py
```

This standard-library check verifies the combined SHA-256 inventory, unchanged Task 2/3 inner inventories, ZIP integrity and Task 1 table-to-receipt arithmetic. It does not certify unseen Drive evidence or rerun trained Task 1 scoring.

Use each task's README for deeper CPU reproduction. Task 2: `python reproduce.py --verify-only` from its directory; Task 3: `python verify_results.py` from its directory. Install each task's own requirements in a separate environment when reproducing its full analysis: their original plotting-version pins differ. Historical GPU scripts depend on original assets/Drive state and must not be treated as a clean-install launcher.

## What this package is and is not

The provided Task 2 and Task 3 archive contents are preserved byte-for-byte. Task 1 aggregates recovered from earlier returned logs are now included with source provenance. Missing Task 1 payloads are explicitly listed in ATML_PA3_Task1/EXTERNAL_EVIDENCE.json. No checkpoint, missing diagnostic, confidence interval, annotation or successful interrupted backup has been fabricated. See audit/ for the checks actually performed.

Upstream attribution and limitations are retained in task READMEs/notices. No assignment PDF, email correspondence, account credentials, full raw source dataset or large model/checkpoint binary has been added. No global license is imposed on third-party materials. LLM assistance was used; the student remains responsible for the submitted work.
