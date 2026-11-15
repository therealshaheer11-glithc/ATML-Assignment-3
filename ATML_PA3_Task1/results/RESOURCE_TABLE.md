# Task 1 resource measurements from returned receipts

Derived from returned terminal receipts with continuous nonoverlapping 0–1250 update coverage. Training/checkpoint time are sums of reported segments, not full billed compute; initialization, preprocessing, failed attempts and some profiling/transfer costs are excluded. Hardware differs across some runs; peak is the largest reported allocation across segments. Full per-update and cost audit remains unfinished.

| Method | Trainable parameters | Training seconds | Checkpoint seconds | Peak allocated GiB | Final checkpoint bytes |
|---|---:|---:|---:|---:|---:|
| full | 596,049,920 | 9384.30 | 4321.13 | 10.685 | 7,154,073,600 |
| l2sp | 596,049,920 | 8663.61 | 2891.36 | 15.614 | 7,154,094,080 |
| lora4 | 2,523,136 | 14169.46 | 22.06 | 1.812 | 31,815,680 |
| lora16 | 10,092,544 | 13834.63 | 52.30 | 1.926 | 122,654,720 |
| oplora | 10,092,544 | 20815.33 | 52.93 | 2.252 | 122,654,720 |

Final checkpoint size refers to the reported packed checkpoint, including training state; it is not inference adapter weight size. FullFT/L2SP archives and adapter archives have different contents. Source receipts retain exact identities. Two observed total-memory capacities occur for L2SP and LoRA4; do not present these as a controlled same-hardware speed benchmark.
