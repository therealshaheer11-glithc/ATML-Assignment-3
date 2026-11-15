# Original training workflow

The CPU reproduction entry point in the top-level README is self-contained. The stages here preserve the **actual original Colab execution**, including source hashes and artifact dependencies. Do not run the historical notebook with Run all.

## Prerequisites

Original durable root: `/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a`.

The frozen dataset manifest lists earlier source locks under `original_source_locks`. Block 7 expects those original annotation/VQAv2 selections and receipts, `code/vqa_normalization.py`, and the original COCO annotation archive (SHA-256 `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009`). Earlier preparation scripts are included in `early_preparation/` for provenance; independently rerunning earlier selection/review is not a substitute for the frozen accepted selection.

Subsequent stages validate their predecessors' manifests and receipts. Some reference files bind to artifacts of this specific run, including timestamped paths and hashes. A fresh Drive will not satisfy them. Recreating the entire run on a different Drive requires explicit setup of the sources and regeneration of dependent receipts; it has **not** been tested as a fresh installation by this package. No checks have been weakened for publication.

Raw archives, 9,260 image files, feature shards, full pretrained weights, and training checkpoints must be obtained or generated externally. Their recorded identities are retained. The final checkpoint was recorded as 77,397,849 bytes with SHA-256 `0aaafc7b02763e239590356320367fbe6ca7f5f4fae981c1d9d0feaedb5f2f19`. It was verified on the original Colab run; this package does not contain its bytes or claim to have reloaded it locally.

## Final stage order

For each row, select the indicated runtime, paste the matching `loaders/Task_2_Block_*.py` into one Colab cell, and upload the correspondingly named ZIP when requested. Wait for the full verified result before moving on. Original loaders verify archive and code hashes before execution. GPU controllers can intentionally release the runtime after a bounded segment; a disconnect alone is not evidence of success.

| Stage | Runtime | Final package revision | Purpose |
|---|---|---|---|
| 7 | CPU | `07_CPU_Preflight` | Verify original sources and freeze accepted dataset |
| 8 | CPU | `08_Image_Preparation` | Fetch and verify original images |
| 9 | CPU | `09_Cache_Input_Packing` | Pack verified image inputs |
| 10 | CPU | `10_Model_Preparation` | Fetch and verify pinned original model archives |
| 11 | GPU | `11R1_GPU_Feature_Cache` | Cache deterministic 196×768 FP16 features |
| 12 | CPU | `12_CPU_Text_Preparation` | Verify tokenizer, caps, masks and text inputs |
| 13 | GPU | `13R2_GPU_Initialization` | Calibrate and verify initialization, initial scale 128 |
| 14 | GPU | `14R1_GPU_Training` | Run/resume bounded segments to 4,500 updates |
| 15 | GPU | `15_GPU_Final_Evaluation` | Restore final weights and generate 1,280 answers |
| 16 | CPU | `16_CPU_Results_and_Plots` | Score, plot and export final evidence |

Only final revisions are supplied as executable packages. Failed/superseded revisions remain documented in the historical notebook and original audit. GPU environment details are in `environment/observed_evaluation.json`; the original workers install/verify pinned dependencies and manage the recorded TorchAO import isolation. Quantization is not used.

## Reconstruct Block 7 without distributing its photograph

The original Block 7 ZIP includes one original COCO photo. Its exact non-image members are referenced by `block07_archive.json`, which preserves archive metadata and hashes; unique auxiliary bytes are in `block07_sources/`. Obtain the original `COCO_train2014_000000153620.jpg` from the source URL in `data/frozen/provenance/manual_tv_image_receipt.json`, then run from the package root:

```sh
python workflow/prepare_block07.py --image /path/to/COCO_train2014_000000153620.jpg
```

The helper verifies the image and every member, reconstructs the original ZIP under the ignored `reconstructed/` folder, and verifies the original archive SHA-256. Different ZIP/zlib implementations can produce different compressed bytes; the helper fails rather than claiming equivalence. The local reconstruction was tested against the original ZIP byte-for-byte. Never commit that reconstructed archive because it contains the source photo.

## Status interpretation

A verified success/proceed message identifies the completed boundary. A verified continuation message means rerun the same training stage to reach its remaining update budget. A STOP/refer message, traceback, missing final status or unexpected disconnect requires inspection of the saved logs before proceeding. The complete production logs and final evaluation receipts in `results/source/` are the evidence that this run finished.
