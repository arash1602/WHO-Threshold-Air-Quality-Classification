# Naive baselines on the nested folds (ctx 24, horizon 1, 25 h embargo)

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
| Antwerp-b1 | 2674 | 0.273 | 0.681 | 0.630 | 0.588 | 0.642 | 0.1 | 0.822 | 0.730 | 0.823 |
| Antwerp-b2 | 1965 | 0.364 | 0.599 | 0.459 | 0.588 | 0.436 | 10 | 0.828 | 0.739 | 0.870 |
| Antwerp-b3 | 2432 | 0.341 | 0.474 | 0.304 | 0.488 | 0.277 | 10 | 0.701 | 0.703 | 0.808 |
| Antwerp-b4 | 2934 | 0.295 | 0.726 | 0.549 | 0.662 | 0.527 | 10 | 0.741 | 0.616 | 0.668 |
| Antwerp-b5 | 1949 | 0.417 | 0.478 | 0.250 | 0.407 | 0.228 | 1 | 0.645 | 0.803 | 0.862 |
| Oslo-b1 | 473 | 0.235 | 0.261 | 0.226 | 0.197 | 0.234 | 0.01 | 0.795 | 0.808 | 0.850 |
| Oslo-b2 | 682 | 0.145 | 0.885 | 0.484 | 0.896 | 0.434 | 0.01 | 0.757 | 0.557 | 0.795 |
| Oslo-b3 | 1055 | 0.062 | 0.114 | 0.258 | 0.091 | 0.477 | 0.01 | 0.437 | 0.575 | 0.702 |
| Oslo-b4 | 1031 | 0.028 | 0.037 | 0.145 | 0.044 | 0.345 | 0.01 | 0.087 | 0.072 | 0.000 |
| Oslo-b5 | 935 | 0.031 | 0.104 | 0.167 | 0.094 | 0.207 | 0.01 | 0.314 | 0.084 | 0.000 |
| Zagreb-b1 | 526 | 0.576 | 0.791 | 0.240 | 1.000 | 0.201 | 1 | 0.953 | 0.755 | 0.899 |
| Zagreb-b2 | 380 | 0.339 | 0.945 | 0.760 | 0.969 | 0.721 | 0.1 | 0.879 | 0.804 | 0.804 |
| Zagreb-b3 | 1131 | 0.233 | 0.719 | 0.291 | 0.835 | 0.250 | 10 | 0.644 | 0.686 | 0.707 |
| Zagreb-b4 | 878 | 0.180 | 0.867 | 0.581 | 0.923 | 0.532 | 0.1 | 0.900 | 0.828 | 0.842 |
| Zagreb-b5 | 634 | 0.158 | 0.875 | 0.704 | 0.819 | 0.680 | 1 | 0.892 | 0.838 | 0.850 |

**Means over 15 folds.** WHO rule PR-AUC 0.570 +/- 0.300, F2 0.403 +/- 0.197. Logistic regression PR-AUC 0.693 +/- 0.234, F2 @0.5 0.640 +/- 0.236, F2 @threshold 0.699 +/- 0.281.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.742 | WHO rule 0.570 | +0.172 | 13/15 |
| nested5 | GRU | 0.742 | logreg 0.693 | +0.049 | 11/15 |
| nested5 | TCN | 0.747 | WHO rule 0.570 | +0.177 | 12/15 |
| nested5 | TCN | 0.747 | logreg 0.693 | +0.054 | 15/15 |
| nested5 | FUSION | 0.747 | WHO rule 0.570 | +0.177 | 13/15 |
| nested5 | FUSION | 0.747 | logreg 0.693 | +0.054 | 13/15 |

## multi

| fold | n test | pos rate | WHO PR-AUC | WHO F2 | WHO precision | WHO recall | logreg C | logreg PR-AUC | logreg F2 @0.5 | logreg F2 @thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout-Antwerp | 14014 | 0.315 | 0.578 | 0.434 | 0.557 | 0.411 | 10 | 0.660 | 0.623 | 0.705 |
| heldout-Oslo | 5777 | 0.062 | 0.140 | 0.271 | 0.141 | 0.352 | 10 | 0.096 | 0.183 | 0.220 |
| heldout-Zagreb | 4199 | 0.271 | 0.781 | 0.439 | 0.912 | 0.389 | 10 | 0.738 | 0.719 | 0.709 |

**Means over 3 folds.** WHO rule PR-AUC 0.500 +/- 0.267, F2 0.381 +/- 0.078. Logistic regression PR-AUC 0.498 +/- 0.286, F2 @0.5 0.508 +/- 0.233, F2 @threshold 0.545 +/- 0.229.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.630 | WHO rule 0.500 | +0.130 | 3/3 |
| nested5 | GRU | 0.630 | logreg 0.498 | +0.132 | 3/3 |
| nested5 | TCN | 0.605 | WHO rule 0.500 | +0.105 | 2/3 |
| nested5 | TCN | 0.605 | logreg 0.498 | +0.107 | 2/3 |
| nested5 | FUSION | 0.609 | WHO rule 0.500 | +0.110 | 2/3 |
| nested5 | FUSION | 0.609 | logreg 0.498 | +0.112 | 3/3 |

