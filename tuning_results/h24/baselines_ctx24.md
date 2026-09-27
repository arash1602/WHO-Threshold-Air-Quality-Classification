# Naive baselines on the nested folds (ctx 24, horizon 24, 48 h embargo)

Both baselines use only the **input window**, the same 24 h the models see.
Labels were re-derived and checked against the stored fold files.

- **WHO rule** (zero parameters): `mean(OPCN3PM25) > 15 or mean(OPCN3PM10) > 45`.
  Its ranking view scores `max(pm25/15, pm10/45)`, which fits nothing either.
- **Logistic regression** (one hyperparameter): window means of `OPCN3PM25`, `OPCN3PM10`, `SHT31TE`, `SHT31HE`, `Absolute_humidity`,
  standardized on train, `class_weight="balanced"`; `C` chosen by mean inner-fold
  validation PR-AUC, threshold = median of that `C`'s inner F2-max thresholds,
  refit on `refit_idx` and scored once on the outer test.

## single-blocked

| fold | n test | pos rate | WHO PR-AUC | WHO F2 | WHO precision | WHO recall | logreg C | logreg PR-AUC | logreg F2 @0.5 | logreg F2 @thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Antwerp-b1 | 1007 | 0.207 | 0.365 | 0.447 | 0.338 | 0.486 | 0.01 | 0.523 | 0.379 | 0.700 |
| Antwerp-b2 | 1596 | 0.402 | 0.671 | 0.659 | 0.610 | 0.673 | 0.01 | 0.771 | 0.247 | 0.492 |
| Antwerp-b3 | 2652 | 0.436 | 0.458 | 0.238 | 0.424 | 0.215 | 0.01 | 0.800 | 0.846 | 0.799 |
| Antwerp-b4 | 2513 | 0.301 | 0.424 | 0.271 | 0.428 | 0.248 | 10 | 0.552 | 0.445 | 0.585 |
| Antwerp-b5 | 2115 | 0.321 | 0.294 | 0.122 | 0.166 | 0.115 | 0.1 | 0.525 | 0.740 | 0.761 |
| Oslo-b1 | 1193 | 0.085 | 0.080 | 0.106 | 0.046 | 0.157 | 0.01 | 0.722 | 0.626 | 0.802 |
| Oslo-b2 | 1184 | 0.097 | 0.200 | 0.257 | 0.217 | 0.270 | 0.1 | 0.465 | 0.653 | 0.631 |
| Oslo-b3 | 534 | 0.094 | 0.133 | 0.166 | 0.098 | 0.200 | 0.01 | 0.638 | 0.559 | 0.773 |
| Oslo-b4 | 457 | 0.053 | 0.079 | 0.207 | 0.082 | 0.333 | 1 | 0.063 | 0.123 | 0.000 |
| Oslo-b5 | 358 | 0.073 | 0.075 | 0.065 | 0.041 | 0.077 | 0.1 | 0.163 | 0.116 | 0.000 |
| Zagreb-b1 | 139 | 0.439 | 0.876 | 0.656 | 0.974 | 0.607 | 10 | 0.869 | 0.736 | 0.726 |
| Zagreb-b2 | 621 | 0.419 | 0.807 | 0.392 | 0.927 | 0.342 | 10 | 0.946 | 0.925 | 0.870 |
| Zagreb-b3 | 967 | 0.274 | 0.777 | 0.398 | 0.968 | 0.347 | 10 | 0.691 | 0.652 | 0.674 |
| Zagreb-b4 | 569 | 0.195 | 0.674 | 0.385 | 0.760 | 0.342 | 0.01 | 0.675 | 0.616 | 0.674 |
| Zagreb-b5 | 681 | 0.157 | 0.613 | 0.471 | 0.662 | 0.439 | 0.1 | 0.665 | 0.857 | 0.883 |

**Means over 15 folds.** WHO rule PR-AUC 0.435 +/- 0.277, F2 0.323 +/- 0.179. Logistic regression PR-AUC 0.605 +/- 0.232, F2 @0.5 0.568 +/- 0.248, F2 @threshold 0.625 +/- 0.265.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.653 | WHO rule 0.435 | +0.218 | 15/15 |
| nested5 | GRU | 0.653 | logreg 0.605 | +0.049 | 10/15 |
| nested5 | TCN | 0.655 | WHO rule 0.435 | +0.220 | 14/15 |
| nested5 | TCN | 0.655 | logreg 0.605 | +0.051 | 10/15 |
| nested5 | FUSION | 0.633 | WHO rule 0.435 | +0.197 | 13/15 |
| nested5 | FUSION | 0.633 | logreg 0.605 | +0.028 | 9/15 |

## multi

| fold | n test | pos rate | WHO PR-AUC | WHO F2 | WHO precision | WHO recall | logreg C | logreg PR-AUC | logreg F2 @0.5 | logreg F2 @thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout-Antwerp | 13883 | 0.321 | 0.411 | 0.325 | 0.421 | 0.308 | 10 | 0.535 | 0.423 | 0.711 |
| heldout-Oslo | 5671 | 0.068 | 0.097 | 0.149 | 0.083 | 0.186 | 0.01 | 0.244 | 0.405 | 0.324 |
| heldout-Zagreb | 4162 | 0.276 | 0.687 | 0.393 | 0.813 | 0.348 | 0.01 | 0.623 | 0.638 | 0.724 |

**Means over 3 folds.** WHO rule PR-AUC 0.398 +/- 0.241, F2 0.289 +/- 0.103. Logistic regression PR-AUC 0.467 +/- 0.162, F2 @0.5 0.489 +/- 0.106, F2 @threshold 0.586 +/- 0.185.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.560 | WHO rule 0.398 | +0.161 | 3/3 |
| nested5 | GRU | 0.560 | logreg 0.467 | +0.092 | 3/3 |
| nested5 | TCN | 0.517 | WHO rule 0.398 | +0.118 | 3/3 |
| nested5 | TCN | 0.517 | logreg 0.467 | +0.049 | 2/3 |
| nested5 | FUSION | 0.471 | WHO rule 0.398 | +0.073 | 2/3 |
| nested5 | FUSION | 0.471 | logreg 0.467 | +0.004 | 2/3 |

