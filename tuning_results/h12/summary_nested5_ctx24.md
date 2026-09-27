# Nested CV results (schema 5, context_size=24)

Outer-fold test pr_auc, mean +/- std across outer folds. Each outer fold selected its own hyperparameters by inner CV and was scored once; these numbers estimate the tuning+training procedure and are not conditioned on the single hyperparameter set listed. `hp_agreement` is the fraction of outer folds whose own winner matched that set. Built from every schema-5 `best_*.json` in this directory.

| experiment | arch | nested_test_pr_auc_mean | nested_test_pr_auc_std | nested_test_f1_mean | nested_test_f2_mean | nested_test_f2_at_threshold_mean | nested_test_f1_at_threshold_mean | recall_at_threshold | precision_at_threshold | threshold_prob_median | inner_val_pr_auc_mean | hp_agreement | n_outer | hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | FUSION | 0.5838 | 0.2383 | 0.5449 | 0.6301 | 0.6340 | 0.5027 | 0.7891 | 0.3803 | 0.207 | 0.7845 | 0.33 | 3 | {"fusion_dim": 64, "cnn_channels": 64, "kernel_size": 5, "gru_hidden": 32, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | GRU | 0.6055 | 0.2182 | 0.5659 | 0.6538 | 0.6539 | 0.5278 | 0.8015 | 0.4072 | 0.378 | 0.7853 | 0.67 | 3 | {"hidden_size": 16, "num_layers": 2, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| multi | TCN | 0.6030 | 0.2391 | 0.5688 | 0.6206 | 0.6209 | 0.5180 | 0.7414 | 0.4191 | 0.350 | 0.7814 | 0.33 | 3 | {"tcn_channels": [64, 64, 64], "kernel_size": 2, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | FUSION | 0.7502 | 0.1809 | 0.6254 | 0.6994 | 0.6869 | 0.5972 | 0.7799 | 0.6062 | 0.298 | 0.7751 | 0.07 | 15 | {"fusion_dim": 64, "cnn_channels": 32, "kernel_size": 3, "gru_hidden": 32, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | GRU | 0.7317 | 0.2229 | 0.6120 | 0.6749 | 0.6751 | 0.5816 | 0.7792 | 0.5201 | 0.415 | 0.7759 | 0.13 | 15 | {"hidden_size": 32, "num_layers": 4, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0} |
| single-blocked | TCN | 0.7649 | 0.1785 | 0.6438 | 0.7114 | 0.6911 | 0.5938 | 0.7878 | 0.5135 | 0.286 | 0.7830 | 0.13 | 15 | {"tcn_channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0} |

`nested_test_f1_mean` and `nested_test_f2_mean` are at probability 0.5. The `*_at_threshold` columns are at each outer fold's tuned threshold: the median of the winning combo's inner-fold F2-maximising val thresholds (test data never involved). `threshold_prob_median` is the median of those thresholds over outer folds, as a probability. PR-AUC does not depend on the threshold.
