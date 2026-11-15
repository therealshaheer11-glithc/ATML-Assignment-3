# Task 2 publication review — 14 November 2026

## Verified scope

The assignment manual was reread (pages 1, 6–10 and 15). The original November 10 experiment audit remains the detailed requirement-to-evidence mapping. Its statement that submission packaging was outstanding is superseded by this package; the report and public upload remain separate.

The CPU reproduction command passed against the actual recorded run:

- All 4,500 updates and 72,000 example presentations replayed exactly from seed 6304.
- All 12 development records and all 160 evaluation batches were verified against the frozen plan and original artifact receipts.
- All 1,280 predictions were rescored, including an independent explicit ten-reference leave-one-out calculation.
- All saved result statistics and 4,000-complete-pair bootstrap intervals matched (floating tolerance 1e-12).
- All ten CSV tables reproduced byte-for-byte, including per-question, per-pair, training, development and resource tables.
- Recorded resource measurements matched their saved sessions; complete successful session coverage was 0–4,500 with no gap or overlap.
- Nine original implementation checks passed with the locked official normalizer; no development/final-test image entered training.
- Ten figures regenerated as PNG and SVG and were visually checked for readable axes, legends, intervals and stage boundaries.
- The optional offline gallery path verified all 320 supplied local image hashes and produced the complete 160-pair gallery. These test images are outside the upload package. The distributed gallery instead references original source URLs.
- Original Block 7 was reconstructed byte-for-byte from retained non-image members and the separately supplied original photo, then matched its original SHA-256.

## Package inspection

The final package contains the final code, original configurations, exact selections, raw run records, machine-readable results, observed environments, attributed normalization source, and reproduction instructions. Only final Colab stage revisions are supplied for execution. Original source/model/data identities are retained unchanged. The historical notebook preserves cell source and removes outputs and private metadata.

All included executable Python sources were parsed. The preserved upstream VQA Python 2 reference is explicitly identified as historical source, not an entry point. Included ZIPs were checked for CRC errors and their member manifests. A recursive scan, including nested ZIP members, found no original JPEGs, assignment PDFs, model/checkpoint binaries, personal macOS home paths, common access-token patterns, private keys or personal email patterns. This is a targeted scan, not a claim of universal secret detection. Colab Drive paths remain as necessary provenance; they are not access credentials.

The package SHA-256 inventory covers all distributed files except the inventory itself. The original results export inventory also remains under provenance; it describes the original export, including images intentionally excluded from this publication package.

## Limits that remain

No second GPU training/evaluation run was performed. Training stages retain hash-bound dependencies on the original Drive artifacts, so full fresh-Drive reconstruction is not claimed as tested. Checkpoint and cache binaries remain external. Historical intermediate checkpoints could not all be independently reread locally. The repeated development item, one manually approved training TV replacement, evaluation parameter-count log timing, subgroup uncertainty and weaker counting performance remain disclosed in the original audit and README. None is silently corrected by modifying recorded evidence.

This is the mandatory 196-token Task 2 experiment only. The optional 49-token experiment, combined report, GitHub publication and LMS submission are not included. No new claim of meeting the assignment deadline is made.
