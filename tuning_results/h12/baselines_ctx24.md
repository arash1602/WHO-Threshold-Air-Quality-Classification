# Naive baselines on the nested folds (ctx 24, horizon 12, 36 h embargo)

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
| Antwerp-b1 | 1480 | 0.218 | 0.596 | 0.641 | 0.452 | 0.715 | 10 | 0.732 | 0.584 | 0.806 |
| Antwerp-b2 | 2733 | 0.317 | 0.460 | 0.329 | 0.421 | 0.312 | 10 | 0.703 | 0.759 | 0.791 |
| Antwerp-b3 | 1834 | 0.423 | 0.523 | 0.363 | 0.570 | 0.332 | 0.1 | 0.720 | 0.555 | 0.738 |
| Antwerp-b4 | 2515 | 0.357 | 0.690 | 0.509 | 0.632 | 0.485 | 0.01 | 0.648 | 0.606 | 0.742 |
| Antwerp-b5 | 2382 | 0.348 | 0.373 | 0.151 | 0.271 | 0.136 | 0.01 | 0.597 | 0.804 | 0.746 |
| Oslo-b1 | 892 | 0.120 | 0.145 | 0.171 | 0.103 | 0.206 | 0.01 | 0.868 | 0.808 | 0.856 |
| Oslo-b2 | 587 | 0.170 | 0.677 | 0.406 | 0.661 | 0.370 | 0.01 | 0.709 | 0.612 | 0.611 |
| Oslo-b3 | 1018 | 0.058 | 0.089 | 0.187 | 0.064 | 0.356 | 0.01 | 0.652 | 0.786 | 0.663 |
| Oslo-b4 | 911 | 0.027 | 0.047 | 0.176 | 0.063 | 0.320 | 1 | 0.278 | 0.174 | 0.050 |
| Oslo-b5 | 469 | 0.058 | 0.092 | 0.123 | 0.074 | 0.148 | 1 | 0.295 | 0.000 | 0.000 |
| Zagreb-b1 | 497 | 0.571 | 0.763 | 0.251 | 0.984 | 0.211 | 0.1 | 0.917 | 0.730 | 0.932 |
| Zagreb-b2 | 621 | 0.217 | 0.876 | 0.712 | 0.868 | 0.681 | 0.1 | 0.831 | 0.744 | 0.687 |
| Zagreb-b3 | 812 | 0.299 | 0.793 | 0.294 | 0.924 | 0.251 | 10 | 0.689 | 0.696 | 0.823 |
| Zagreb-b4 | 618 | 0.180 | 0.837 | 0.641 | 0.848 | 0.604 | 10 | 0.845 | 0.759 | 0.749 |
| Zagreb-b5 | 771 | 0.145 | 0.723 | 0.570 | 0.769 | 0.536 | 10 | 0.772 | 0.852 | 0.870 |

**Means over 15 folds.** WHO rule PR-AUC 0.512 +/- 0.285, F2 0.368 +/- 0.194. Logistic regression PR-AUC 0.684 +/- 0.178, F2 @0.5 0.631 +/- 0.232, F2 @threshold 0.671 +/- 0.266.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.732 | WHO rule 0.512 | +0.219 | 14/15 |
| nested5 | GRU | 0.732 | logreg 0.684 | +0.048 | 11/15 |
| nested5 | TCN | 0.765 | WHO rule 0.512 | +0.253 | 14/15 |
| nested5 | TCN | 0.765 | logreg 0.684 | +0.081 | 15/15 |
| nested5 | FUSION | 0.750 | WHO rule 0.512 | +0.238 | 13/15 |
| nested5 | FUSION | 0.750 | logreg 0.684 | +0.067 | 14/15 |

## multi

| fold | n test | pos rate | WHO PR-AUC | WHO F2 | WHO precision | WHO recall | logreg C | logreg PR-AUC | logreg F2 @0.5 | logreg F2 @thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout-Antwerp | 13953 | 0.315 | 0.489 | 0.379 | 0.485 | 0.359 | 0.1 | 0.619 | 0.665 | 0.755 |
| heldout-Oslo | 5730 | 0.063 | 0.116 | 0.210 | 0.110 | 0.271 | 0.1 | 0.131 | 0.334 | 0.346 |
| heldout-Zagreb | 4182 | 0.272 | 0.728 | 0.411 | 0.850 | 0.364 | 10 | 0.629 | 0.680 | 0.680 |

**Means over 3 folds.** WHO rule PR-AUC 0.444 +/- 0.252, F2 0.333 +/- 0.088. Logistic regression PR-AUC 0.460 +/- 0.232, F2 @0.5 0.560 +/- 0.160, F2 @threshold 0.594 +/- 0.178.

### Paired per-fold test PR-AUC vs the models

| sweep | arch | model | baseline | mean diff | model wins |
| --- | --- | --- | --- | --- | --- |
| nested5 | GRU | 0.606 | WHO rule 0.444 | +0.161 | 3/3 |
| nested5 | GRU | 0.606 | logreg 0.460 | +0.146 | 3/3 |
| nested5 | TCN | 0.603 | WHO rule 0.444 | +0.159 | 3/3 |
| nested5 | TCN | 0.603 | logreg 0.460 | +0.143 | 3/3 |
| nested5 | FUSION | 0.584 | WHO rule 0.444 | +0.140 | 3/3 |
| nested5 | FUSION | 0.584 | logreg 0.460 | +0.124 | 3/3 |

