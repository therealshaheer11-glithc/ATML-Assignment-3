# Task 2 experiment audit · 10 November 2026

The mandatory 196-token experiment meets the computational requirements I could verify. The final results reproduce from the raw answers. I found no issue that warrants another training or evaluation run. The saved notebook is an execution history with external dependencies. Submission packaging and the assignment report are separate outstanding work.

## Requirements and evidence

| Manual | Check against the actual run | Finding |
|---|---|---|
| p. 7, visual interface | The recorded SigLIP probe produced 196 × 768 FP16 features in row-major order, with one final normalization, a frozen encoder and exact repeatability on its probe images. The completed cache covers 9,260 images. The training worker loads cached features and the original frozen Qwen3-0.6B. | Matches the specified interface. |
| pp. 7–9, connector and learning | The implementation inserts the two-layer GELU connector output before user text, disables thinking, masks the context and padding, and averages causal target-token loss including end-of-turn within each example. The training-only calibration uses 256 images and 256 caption IDs. The recorded RMS scale is 0.05459058. | Connector gradients were nonzero on every production update. LoRA B gradients appear in Stages 2 and 3. |
| pp. 7–8, data | Revalidated 6,000 training images with three captions each, 96 development captions, 24/64 grounding development/test pairs, and 1,536/32/96 natural train/dev/test pairs. Checked all three natural answer-type quotas, ten references per example, eight-vote minimum, distinct normalized pair answers, original-string targets and split isolation. | The training pool contains 8,733 image IDs. The full training log uses 8,709 distinct IDs. All 527 distinct development and final-test image IDs are outside both groups. |
| pp. 8–9, curriculum | Replayed seed 6304 from the frozen dataset and compared each complete draw and example record with the actual logs. All 72,000 examples match in order, including both complete caption passes and all resumed segments. Stage totals are 2,250, 1,500 and 750 updates. | The implemented probabilities produce the recorded mixtures. Microbatch 8, accumulation 2, stage learning rates and warmups, AdamW, dynamic FP16 scaling and unscale-before-norm-1 clipping match the recipe. |
| p. 8, checkpoints and development | The worker persists full resume state every 50 updates and at stage endpoints, derives the scheduler position from the saved stage counter, and writes endpoint weights. The actual training logs have boundaries at all 18 required multiples of 250. All 12 scheduled development records are present, including the untrained baseline. Evaluation restored the final checkpoint at 4,500. | Resume validation and the final checkpoint have direct run evidence. The local results ZIP contains receipts and measurements. Individual historical Drive weight files, including the update-3,000 binary and Stage 1–2 weights, were unavailable for a fresh byte-level inspection. |
| pp. 8–9, final protocol | Rebuilt the frozen mismatch map from the dataset. Verified all 160 batches of eight against the plan, condition-specific image IDs, original references, and token termination. Generation omits targets, uses greedy decoding with thinking off, and drops the entire visual prefix in the no-image condition. The original-answer NLL excludes end-of-turn. | 1,280 complete answers for 320 questions under four conditions. Every answer terminated and contained text. |
| pp. 9–10, statistics and figures | Independently scored all raw answers with grounding exact match or the explicit ten-reference leave-one-out formula. Regenerated all 224 condition/type metric rows, 84 advantage rows and 35 partner-following rows, including the 4,000-pair bootstrap intervals. Checked the actual figure montage and the first saved pair of every question type against the photos and answers. | Scores and intervals reproduce. The export includes ten figures, machine-readable tables, measured resource records and a gallery of all 160 pairs. The 49-token repeat is optional. |

## What the measurements support

| Held-out measure | Grounding | Natural VQA |
|---|---:|---:|
| Correct image | 71.9% | 54.2% |
| Partner image, scored against original references | 11.7% | 25.2% |
| Unrelated image | 35.9% | 30.2% |
| No image | 40.6% | 35.2% |
| Correct-image gain over partner swap, percentage points (95% paired interval) | 60.2 [46.9, 72.7] | 29.0 [19.3, 38.4] |
| Both canonical answers correct with original images | 35/64 pairs | 23/96 pairs |
| Increase in original-answer NLL on swapping | 2.19 | 1.44 |

Caption development loss fell from 5.62 to 2.44 during alignment. Stage 2 brought grounding development NLL from 8.26 to 0.84. During Stage 3, natural development NLL improved from 2.08 to 1.31 while grounding NLL rose slightly to 0.87. These stage-specific measurements avoid conflating a change in the training mixture with an improvement on the same task.

The same-length partner swap gives the clearest evidence that images helped. The no-image baseline also removes 196 input positions. Presence was strong (58/64). Counting was weaker (34/64), particularly for a target of three (4/16). Natural yes/no had a 10.3-point correct-over-swap advantage, with an interval from −3.6 to +24.6 points. The aggregate natural result is more conclusive than that subgroup. In a saved pair asking how many fence wires are visible, the model answered two on both images despite references of three and four. The manual specifies no minimum accuracy.

## Qualifications for handoff

- One development image (190236) appears in two keyboard-counting pairs with the same question and target of three. This gives the example extra weight in development monitoring. It stays entirely within grounding development. The final evaluation uses 320 distinct original image IDs.
- One prediction-independent manual replacement used COCO train2014 image 153620 in the training annotation audit. Its count was checked against COCO. Disclose this exception to automated candidate selection when describing curation.
- The raw evaluation log reports 6,424,576 “trainable” parameters from a count taken immediately before freezing the Stage 3 model. The evaluation worker then freezes both the model and connector. The resource summary gives the correct evaluation count of zero. No weights were updated in evaluation.
- Earlier download, processor-check and optional-library failures, the FP16 initialization overflow, and an interrupted training-cell display remain visible in the notebook. The repaired initialization passed with an initial dynamic loss scale of 128. The saved production examples replay exactly across recovery, and the final checkpoint was verified before evaluation.
- Runtime figures have different scopes. Update time was 126.6 minutes. Successful training sessions took 250.2 minutes including staging, development and checkpoint work. Failed attempts and setup time fall outside that total. The 39.9 milliseconds per generated answer measures batch-eight throughput.
- The attached notebook validates as a 57-cell notebook. It uses manual uploads, prior Drive state and superseded cells. A fresh-user reproduction still needs a top-level README, the final code/configs and frozen IDs, environment details, and upstream attribution. Publishable machine-readable results should omit the 320 original JPEGs. This audit did not establish a public repository, LMS submission, an extension to the manual's November 6 deadline, or completion of the deferred three-task report.

## Audited inputs

| Input | SHA-256 |
|---|---|
| Original assignment PDF, pages 1, 6–10 and 15 | `ce74ac092dbdb83b44afe75c3f5a22e1ef3f2f2bb6048f8175e21d63e76659c9` |
| Attached notebook | `270cb18b22b2246c9210531269b94454ac9fd3c9a140841bfd508fe88c894263` |
| Frozen dataset manifest | `82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2` |
| Results ZIP recorded by the notebook and rehashed locally | `6c225c36b540740c2cbee3035f6756f477741d8409be0bcdc0cb6ddc21262e02` |

The ZIP contains 678 manifest-listed payloads plus the manifest itself (679 members in total). CRC, member inventory, payload sizes, SHA-256 digests and extracted payload bytes checked out. The audit used the saved run and inspected its code. It did not regenerate model weights or predictions on a second GPU.
