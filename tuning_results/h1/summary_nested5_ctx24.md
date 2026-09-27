# Nested CV results (schema 5, context_size=24)

Outer-fold test pr_auc, mean +/- std across outer folds. Each outer fold selected its own hyperparameters by inner CV and was scored once; these numbers estimate the tuning+training procedure and are not conditioned on the single hyperparameter set listed. `hp_agreement` is the fraction of outer folds whose own winner matched that set. Built from every schema-5 `best_*.json` in this directory.

| experiment | arch | nested_test_pr_auc_mean | nested_test_pr_auc_std | nested_test_f1_mean | nested_test_f2_mean | nested_test_f2_at_threshold_mean | nested_test_f1_at_threshold_mean | recall_at_threshold | precision_at_threshold | threshold_prob_median | inner_val_pr_auc_mean | hp_agreement | n_outer | hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | FUSION | 0.6095 | 0.1960 | 0.5951 | 0.6526 | 0.6476 | 0.5671 | 0.7329 | 0.4870 | 0.319 | 0.8037 | 0.33 | 3 | {"fusion_dim": 32, "cnn_channels": 64, "kernel_size": 5, "gru_hidden": 64, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | GRU | 0.6302 | 0.2149 | 0.5931 | 0.6677 | 0.6722 | 0.5868 | 0.7533 | 0.4931 | 0.407 | 0.8064 | 0.00 | 3 | {"hidden_size": 128, "num_layers": 2, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | TCN | 0.6051 | 0.2215 | 0.5774 | 0.6487 | 0.6706 | 0.5451 | 0.8124 | 0.4254 | 0.176 | 0.8118 | 0.00 | 3 | {"tcn_channels": [32, 32, 32, 32], "kernel_size": 2, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | FUSION | 0.7471 | 0.2139 | 0.6348 | 0.6924 | 0.6888 | 0.5829 | 0.8089 | 0.5271 | 0.335 | 0.7847 | 0.13 | 15 | {"fusion_dim": 64, "cnn_channels": 64, "kernel_size": 5, "gru_hidden": 64, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | GRU | 0.7419 | 0.2289 | 0.6210 | 0.6919 | 0.7081 | 0.6233 | 0.8008 | 0.5662 | 0.323 | 0.7924 | 0.20 | 15 | {"hidden_size": 128, "num_layers": 4, "dropout": 0.1, "learning_rate": 0.0003, "weight_decay": 0.0} |
| single-blocked | TCN | 0.7472 | 0.2255 | 0.6319 | 0.6897 | 0.6984 | 0.6079 | 0.7996 | 0.6369 | 0.327 | 0.8073 | 0.33 | 15 | {"tcn_channels": [32, 32, 32], "kernel_size": 3, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |

`nested_test_f1_mean` and `nested_test_f2_mean` are at probability 0.5. The `*_at_threshold` columns are at each outer fold's tuned threshold: the median of the winning combo's inner-fold F2-maximising val thresholds (test data never involved). `threshold_prob_median` is the median of those thresholds over outer folds, as a probability. PR-AUC does not depend on the threshold.
