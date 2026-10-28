"""Independent verification of the seed-refit results behind Table 15.

    python seed_verify.py      # -> tables/seed_verify.txt

Shares no code with seed_analysis.py: it reads the raw JSON lines and the shipped
sweep records directly, with plain Python and NumPy.

1. Integrity of seed_results/seed_refit.jsonl:
   - exactly one record for every (horizon, design, outer fold, architecture, seed)
     with seeds 1-4, i.e. 3 x (15 + 3) x 3 x 4 = 648 records, no duplicates;
   - every record trained with the paper's settings (100 epochs, deterministic);
   - every record refitted the configuration the sweep selected for that fold
     (its combo equals the shipped winner_combo);
   - every record's stored seed-42 reference equals the shipped outer-test AP;
   - every score is a finite number in [0, 1].
2. Recomputes Table 15 from scratch and compares it, cell by cell, with the
   table in tables/tab_seed.tex.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

ARCHS = ("GRU", "TCN", "FUSION")
HORIZONS = (1, 12, 24)
DESIGNS = {"single-blocked": "single", "multi": "multi"}
SEEDS = (1, 2, 3, 4)


def shipped(rd, h, arch, exp):
    out = {}
    with open(os.path.join(rd, f"h{h}", f"nested5_{arch}_{exp}_ctx24_blocked.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "outer":
                out[r["outer"]] = r
    return out


def baseline_lr(rd, h, design):
    with open(os.path.join(rd, f"h{h}", "baselines_ctx24.json"), encoding="utf-8") as f:
        return {r["outer"]: r["logreg"]["pr_auc"] for r in json.load(f)
                if r["experiment"] == design}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "tuning_results")
    out, problems = [], []
    say = out.append
    with open(os.path.join(here, "seed_results", "seed_refit.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # ---------------------------------------------------------------- 1. integrity
    seen = {}
    for r in rows:
        k = (r["horizon"], r["design"], r["outer"], r["arch"], r["seed"])
        if k in seen:
            problems.append(f"duplicate {k}")
        seen[k] = r
    expected = set()
    ref = {}
    for h in HORIZONS:
        for design, exp in DESIGNS.items():
            for a in ARCHS:
                rec = shipped(rd, h, a, exp)
                for fold, orec in rec.items():
                    ref[(h, design, fold, a)] = orec
                    for s in SEEDS:
                        expected.add((h, design, fold, a, s))
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing:
        problems.append(f"{len(missing)} expected records missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected records, e.g. {sorted(extra)[:3]}")
    n_cfg = n_ref = n_set = n_val = 0
    for k, r in seen.items():
        orec = ref.get(k[:4])
        if orec is None:
            continue
        if r["combo"] != orec["winner_combo"]:
            n_cfg += 1
        if abs(r["paper_pr_auc"] - orec["test"]["pr_auc"]) > 1e-9:
            n_ref += 1
        if r.get("epochs") != 100 or r.get("deterministic") is not True:
            n_set += 1
        for m in ("pr_auc", "roc_auc", "f2_at_threshold"):
            v = r.get(m)
            if v is None or not math.isfinite(v) or not 0 <= v <= 1:
                n_val += 1
    say(f"records: {len(rows)} (expected {len(expected)}); duplicates "
        f"{len(rows) - len(seen)}; missing {len(missing)}; unexpected {len(extra)}")
    say(f"configuration differs from the selected one: {n_cfg}")
    say(f"seed-42 reference differs from the shipped score: {n_ref}")
    say(f"not trained with 100 epochs, deterministic: {n_set}")
    say(f"non-finite or out-of-range scores: {n_val}")
    for name, n in (("config", n_cfg), ("reference", n_ref), ("settings", n_set), ("values", n_val)):
        if n:
            problems.append(f"{n} records fail the {name} check")

    # ------------------------------------------------- 2. recompute Table 15
    def fold_matrix(h, design, a):
        folds = sorted(f for (hh, d, f, aa) in ref if (hh, d, aa) == (h, design, a))
        M = np.array([[ref[(h, design, f, a)]["test"]["pr_auc"]]
                      + [seen[(h, design, f, a, s)]["pr_auc"] for s in SEEDS] for f in folds])
        return folds, M

    cells = []
    for h in HORIZONS:
        for a in ARCHS:
            row = []
            for design in DESIGNS:
                folds, M = fold_matrix(h, design, a)
                means = M.mean(axis=0)
                row.append(f"{means.min():.3f}--{means.max():.3f}")
                row.append(f"{np.mean(M.std(axis=1, ddof=1)):.3f}")
                if design == "single-blocked":
                    row.append(f"{M[:, 0].std():.3f}")
                else:
                    lr = baseline_lr(rd, h, "multi")
                    wins = [(M[:, j] > np.array([lr[f] for f in folds])).sum()
                            for j in range(M.shape[1])]
                    lo, hi = min(wins), max(wins)
                    row.append(f"{lo}/3" if lo == hi else f"{lo}--{hi}/3")
            cells.append(row)

    with open(os.path.join(here, "tables", "tab_seed.tex"), encoding="utf-8") as f:
        tab = [re.findall(r"\d\.\d{3}--\d\.\d{3}|\d\.\d{3}|\d(?:--\d)?/3", l)
               for l in f if "/3" in l]
    diffs = [(i, j, a, b) for i, (ra, rb) in enumerate(zip(cells, tab))
             for j, (a, b) in enumerate(zip(ra, rb)) if a != b]
    say(f"Table 15 recomputed independently: {len(cells)} rows x {len(cells[0])} cells, "
        f"{len(diffs)} differences from tables/tab_seed.tex")
    if diffs or len(cells) != len(tab):
        problems.append(f"Table 15 mismatch: {diffs[:5]}")

    say("RESULT: " + ("all checks passed" if not problems else "PROBLEMS: " + "; ".join(problems)))
    with open(os.path.join(here, "tables", "seed_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
