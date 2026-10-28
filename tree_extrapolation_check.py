"""Why the gradient-boosted trees lose most on single-city fold Antwerp-b1 (h = 1).

    python tree_extrapolation_check.py      # -> tables/tree_extrapolation_check.txt

Reports, for the refit split and the outer test block of Antwerp-b1, the 5th, 50th
and 95th percentiles of the input window means, and the training and test average
precision of a gradient-boosted tree fitted with the settings selected for that fold.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import naive_baselines as nb
import extra_baselines as eb
from compare_models import nested_plan

FOLD, H = "Antwerp-b1", 1


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("AQM_DATA_DIR", os.path.join(here, "senseurcity_data_v03", "dataset"))
    nb.HORIZON = H
    common = dict(context_size=nb.CONTEXT_SIZE, overlap=nb.OVERLAP, horizon=H)
    spec = next(s for s in nested_plan(data_dir, experiment="single", protocol="blocked",
                                       cities=["Antwerp"], k_outer=nb.K_OUTER, n_inner=nb.N_INNER,
                                       block_hours=nb.BLOCK_HOURS, **common) if s.outer_id == FOLD)
    te = eb.window_arrays(nb._split(data_dir, spec.test_cities, bands=spec.test_bands, **common))
    isp = next(i for i in spec.inner if i.idx == spec.refit_idx)
    tr = eb.window_arrays(nb._split(data_dir, spec.train_cities, bands=isp.train_bands, **common))
    with open(os.path.join(here, "tuning_results", f"h{H}", "extra_baselines_ctx24.json"),
              encoding="utf-8") as f:
        cfg = next(r for r in json.load(f) if r["outer"] == FOLD)["gbm"]["config"]
    g = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                       random_state=nb.SEED, **cfg).fit(tr[0], tr[2])
    out = [f"{FOLD}, h={H}: selected tree settings {cfg}",
           f"  tree AP: training split {average_precision_score(tr[2], g.predict_proba(tr[0])[:, 1]):.3f}, "
           f"test block {average_precision_score(te[2], g.predict_proba(te[0])[:, 1]):.3f}",
           "  window means, 5th / 50th / 95th percentile: training split | test block"]
    for j, name in enumerate(nb.INPUT_FEATURES):
        out.append(f"  {name:18s} {np.percentile(tr[1][:, j], [5, 50, 95]).round(1)} | "
                   f"{np.percentile(te[1][:, j], [5, 50, 95]).round(1)}")
    text = "\n".join(out)
    with open(os.path.join(here, "tables", "tree_extrapolation_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
