# Datasheet — fracatlas-v7-split-v2-dedup

- fingerprint: `e7b7fac5fd286799`
- images: 4083  |  positives: 719  |  prevalence: 17.6%
- groups (patients): 3950  |  sources: fracatlas

## Counts per split and body part

| split | body_part | images | positives | prevalence |
|---|---|---:|---:|---:|
| test | hand | 177 | 53 | 29.9% |
| test | hip | 27 | 2 | 7.4% |
| test | leg | 329 | 38 | 11.6% |
| test | mixed | 54 | 16 | 29.6% |
| test | shoulder | 14 | 1 | 7.1% |
| train | hand | 990 | 289 | 29.2% |
| train | hip | 138 | 8 | 5.8% |
| train | leg | 1598 | 160 | 10.0% |
| train | mixed | 303 | 82 | 27.1% |
| train | shoulder | 76 | 7 | 9.2% |
| val | hand | 114 | 39 | 34.2% |
| val | hip | 14 | 0 | 0.0% |
| val | leg | 197 | 14 | 7.1% |
| val | mixed | 41 | 8 | 19.5% |
| val | shoulder | 11 | 2 | 18.2% |

## Known issues

- 0 near-duplicate pairs across splits (pHash Hamming <= 2) after `regroup-duplicates` (148 before, 45 identical)
- no patient identifier: groups = near-duplicate clusters, multi-view studies may still straddle splits
- hardware (metal) is a near-perfect proxy for the label: 97/99 hardware images are fractured (shortcut risk)
- JPEG source: windowing already applied, original bit depth lost
