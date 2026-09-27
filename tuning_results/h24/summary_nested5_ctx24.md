# Nested CV results (schema 5, context_size=24)

Outer-fold test pr_auc, mean +/- std across outer folds. Each outer fold selected its own hyperparameters by inner CV and was scored once; these numbers estimate the tuning+training procedure and are not conditioned on the single hyperparameter set listed. `hp_agreement` is the fraction of outer folds whose own winner matched that set. Built from every schema-5 `best_*.json` in this directory.

| experiment | arch | nested_test_pr_auc_mean | nested_test_pr_auc_std | nested_test_f1_mean | nested_test_f2_mean | nested_test_f2_at_threshold_mean | nested_test_f1_at_threshold_mean | recall_at_threshold | precision_at_threshold | threshold_prob_median | inner_val_pr_auc_mean | hp_agreement | n_outer | hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | FUSION | 0.4712 | 0.1376 | 0.4957 | 0.6167 | 0.5932 | 0.4461 | 0.8686 | 0.3270 | 0.258 | 0.6919 | 0.33 | 3 | {"fusion_dim": 32, "cnn_channels": 32, "kernel_size": 5, "gru_hidden": 64, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | GRU | 0.5598 | 0.2067 | 0.5284 | 0.5958 | 0.6089 | 0.4803 | 0.8340 | 0.3697 | 0.196 | 0.6842 | 0.00 | 3 | {"hidden_size": 32, "num_layers": 2, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | TCN | 0.5167 | 0.2059 | 0.5157 | 0.5650 | 0.5851 | 0.4481 | 0.7811 | 0.3321 | 0.202 | 0.6879 | 0.33 | 3 | {"tcn_channels": [64, 64, 64], "kernel_size": 2, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | FUSION | 0.6326 | 0.2159 | 0.5621 | 0.6172 | 0.6451 | 0.5381 | 0.7611 | 0.4358 | 0.263 | 0.6746 | 0.00 | 15 | {"fusion_dim": 32, "cnn_channels": 64, "kernel_size": 3, "gru_hidden": 64, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | GRU | 0.6534 | 0.2413 | 0.5434 | 0.5875 | 0.6722 | 0.5630 | 0.8066 | 0.4604 | 0.202 | 0.6890 | 0.00 | 15 | {"hidden_size": 32, "num_layers": 4, "dropout": 0.1, "learning_rate": 0.0003, "weight_decay": 0.0} |
| single-blocked | TCN | 0.6552 | 0.2290 | 0.5635 | 0.6132 | 0.6727 | 0.5717 | 0.7772 | 0.4703 | 0.338 | 0.6775 | 0.20 | 15 | {"tcn_channels": [64, 64, 64], "kernel_size": 2, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |

`nested_test_f1_mean` and `nested_test_f2_mean` are at probability 0.5. The `*_at_threshold` columns are at each outer fold's tuned threshold: the median of the winning combo's inner-fold F2-maximising val thresholds (test data never involved). `threshold_prob_median` is the median of those thresholds over outer folds, as a probability. PR-AUC does not depend on the threshold.
