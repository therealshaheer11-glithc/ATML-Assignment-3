# Task 1 recorded benchmark results

Percentages; full precision and changes in percentage points are in primary_metrics.csv. All rows derive from saved completion receipts. Per-question replay of the six trained evaluations remains unfinished.

| Condition | GSM8K flexible EM | HellaSwag normalized | BoolQ accuracy | ARC-Easy normalized |
|---|---:|---:|---:|---:|
| Original | 43.21 | 47.30 | 64.37 | 55.98 |
| Full FT midpoint | 57.54 | 45.81 | 61.25 | 62.33 |
| Full FT final | 65.58 | 46.18 | 60.86 | 63.34 |
| Full FT + L2-SP | 64.29 | 46.60 | 67.06 | 62.12 |
| LoRA-4 | 59.51 | 47.41 | 67.43 | 63.01 |
| LoRA-16 | 61.87 | 47.36 | 72.63 | 63.85 |
| OPLoRA-16 | 62.17 | 47.41 | 73.00 | 63.38 |

Raw and normalized accuracies, and strict GSM8K scores, are preserved in evaluation_summary.json. Strict GSM8K is zero in all seven conditions; the required primary metric is flexible-extraction exact match. Do not silently replace filters or claim the strict-score explanation has been audited.
