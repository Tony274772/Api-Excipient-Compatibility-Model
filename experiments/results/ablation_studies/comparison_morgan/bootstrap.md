# Paired bootstrap (stratified, mean over seeds)

diff = model X − model Y. A 95% CI that excludes 0 indicates a difference beyond test-set sampling noise.

| comparison | split | metric | X | Y | X mean | Y mean | diff | 95% CI | p | seeds X better |
|---|---|---|---|---|---|---|---|---|---|---|
| A | test | pr_auc | baseline_4level_5d | ablation_3level_5d | 0.6987 | 0.6993 | -0.0006 | [-0.0064, +0.0065] | 0.801 | 2/5 |
| A | test | roc_auc | baseline_4level_5d | ablation_3level_5d | 0.9112 | 0.9107 | +0.0006 | [-0.0019, +0.0046] | 0.809 | 4/5 |
| B | test | pr_auc | ablation_3level_5d | ablation_3level_1d | 0.6993 | 0.7050 | -0.0057 | [-0.0210, +0.0098] | 0.474 | 3/5 |
| B | test | roc_auc | ablation_3level_5d | ablation_3level_1d | 0.9107 | 0.9067 | +0.0039 | [-0.0034, +0.0126] | 0.345 | 4/5 |
| A+B | test | pr_auc | baseline_4level_5d | ablation_3level_1d | 0.6987 | 0.7050 | -0.0063 | [-0.0223, +0.0095] | 0.438 | 3/5 |
| A+B | test | roc_auc | baseline_4level_5d | ablation_3level_1d | 0.9112 | 0.9067 | +0.0045 | [-0.0029, +0.0129] | 0.267 | 4/5 |
| A | val | pr_auc | baseline_4level_5d | ablation_3level_5d | 0.6905 | 0.6898 | +0.0007 | [-0.0128, +0.0155] | 0.960 | 3/5 |
| A | val | roc_auc | baseline_4level_5d | ablation_3level_5d | 0.9244 | 0.9241 | +0.0004 | [-0.0011, +0.0021] | 0.665 | 3/5 |
| B | val | pr_auc | ablation_3level_5d | ablation_3level_1d | 0.6898 | 0.6744 | +0.0154 | [-0.0027, +0.0339] | 0.099 | 4/5 |
| B | val | roc_auc | ablation_3level_5d | ablation_3level_1d | 0.9241 | 0.9190 | +0.0051 | [-0.0003, +0.0111] | 0.071 | 4/5 |
| A+B | val | pr_auc | baseline_4level_5d | ablation_3level_1d | 0.6905 | 0.6744 | +0.0161 | [-0.0054, +0.0376] | 0.140 | 5/5 |
| A+B | val | roc_auc | baseline_4level_5d | ablation_3level_1d | 0.9244 | 0.9190 | +0.0054 | [+0.0000, +0.0113] | 0.050 | 4/5 |
