"""Seed sensitivity of the reported results, from seed_refit.py's output.

    python seed_analysis.py        # -> tables/seed_analysis.txt, tables/tab_seed.tex

Each (horizon, design, outer fold, architecture) has the paper's seed-42 score plus
one score per extra seed. Reported:

* seed SD     standard deviation across seeds of an outer fold's AP, averaged over folds
* fold SD     between-fold standard deviation of AP (the spread the paper reports)
* mean range  minimum and maximum, over seeds, of the AP averaged over folds
* within city: for each seed, the Nadeau--Bengio and Wilcoxon tests against the
  logistic regression with Holm adjustment, as in the paper; how many of the nine
  differences are significant under each seed
* across cities: for each seed, on how many held-out cities each architecture beats
  the logistic regression and the gradient-boosted trees
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import make_tables as mt
from stats_extra import corrected_t, holm, test_train_ratio

HORIZONS = (1, 12, 24)


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "tuning_results")
    sd = os.path.join(here, "seed_results")
    data = mt.load(rd, HORIZONS, 5)
    out = []
    say = out.append

    chk = os.path.join(sd, "seed_check.jsonl")
    if os.path.exists(chk):
        rows = load_rows(chk)
        worst = max(abs(r["pr_auc"] - r["paper_pr_auc"]) for r in rows)
        say(f"Reproduction check (seed 42, {len(rows)} refits): largest |AP - paper AP| = {worst:.4f}")
    else:
        say("Reproduction check not run (seed_results/seed_check.jsonl missing)")

    rows = load_rows(os.path.join(sd, "seed_refit.jsonl"))
    ap = defaultdict(dict)        # (h, exp, arch) -> {(outer, seed): AP}
    f2 = defaultdict(dict)
    for r in rows:
        exp = "single" if r["design"] == "single-blocked" else "multi"
        ap[(r["horizon"], exp, r["arch"])][(r["outer"], r["seed"])] = r["pr_auc"]
        f2[(r["horizon"], exp, r["arch"])][(r["outer"], r["seed"])] = r["f2_at_threshold"]
    new_seeds = sorted({r["seed"] for r in rows})
    seeds = [42] + new_seeds
    say(f"seeds: 42 (paper) + {new_seeds}; {len(rows)} refits")

    def matrix(h, exp, a):
        """folds x seeds AP matrix; seed 42 from the shipped sweep."""
        e = data[h][exp]
        M = np.full((len(e["folds"]), len(seeds)), np.nan)
        M[:, 0] = mt.ap(e, a)
        for j, s in enumerate(new_seeds, start=1):
            for i, f in enumerate(e["folds"]):
                M[i, j] = ap[(h, exp, a)].get((f, s), np.nan)
        return M

    summary = {}
    for exp in ("single", "multi"):
        say(f"\n{exp.upper()}")
        for h in HORIZONS:
            for a in mt.ARCHS:
                M = matrix(h, exp, a)
                if np.isnan(M).any():
                    say(f"  h={h:2d} {a:6s} incomplete ({int(np.isnan(M).sum())} missing)")
                    continue
                seed_sd = float(np.mean(M.std(axis=1, ddof=1)))
                fold_sd = float(M[:, 0].std())
                means = M.mean(axis=0)
                summary[(exp, h, a)] = dict(seed_sd=seed_sd, fold_sd=fold_sd,
                                            lo=float(means.min()), hi=float(means.max()),
                                            paper=float(means[0]), M=M)
                say(f"  h={h:2d} {a:6s} mean AP {means[0]:.3f} (seed 42), range over seeds "
                    f"{means.min():.3f}-{means.max():.3f}; seed SD per fold {seed_sd:.3f}, "
                    f"between-fold SD {fold_sd:.3f}")

    # Within-city tests per seed
    say("\nWITHIN CITY vs logistic regression, per seed (Holm over 9; significant at 0.05)")
    sig_counts = {}
    for j, s in enumerate(seeds):
        fam = []
        for h in HORIZONS:
            e = data[h]["single"]
            lr = mt.lr(e, "pr_auc")
            r = test_train_ratio(e)
            for a in mt.ARCHS:
                if ("single", h, a) not in summary:
                    continue
                d = summary[("single", h, a)]["M"][:, j] - lr
                fam.append((d.mean(), int((d > 0).sum()), corrected_t(d, r),
                            stats.wilcoxon(d, method="exact").pvalue))
        if len(fam) < 9:
            continue
        nb = holm([x[2] for x in fam])
        w = holm([x[3] for x in fam])
        sig_counts[s] = (int((nb < .05).sum()), int((w < .05).sum()))
        say(f"  seed {s:2d}: diff {min(x[0] for x in fam):+.3f} to {max(x[0] for x in fam):+.3f}; "
            f"Nadeau-Bengio {sig_counts[s][0]}/9, Wilcoxon {sig_counts[s][1]}/9")

    # Across cities per seed
    say("\nACROSS CITIES: held-out cities on which the architecture beats LR / GBT, per seed")
    extra = {h: {(r["design"], r["outer"]): r for r in
                 json.load(open(os.path.join(rd, f"h{h}", "extra_baselines_ctx24.json"),
                                encoding="utf-8"))} for h in HORIZONS}
    city_wins = {}
    for h in HORIZONS:
        e = data[h]["multi"]
        lr = mt.lr(e, "pr_auc")
        gbt = np.array([extra[h][("multi", f)]["gbm"]["pr_auc"] for f in e["folds"]])
        for a in mt.ARCHS:
            if ("multi", h, a) not in summary:
                continue
            M = summary[("multi", h, a)]["M"]
            w_lr = [(M[:, j] > lr).sum() for j in range(len(seeds))]
            w_gb = [(M[:, j] > gbt).sum() for j in range(len(seeds))]
            city_wins[(h, a)] = (min(w_lr), max(w_lr), min(w_gb), max(w_gb))
            say(f"  h={h:2d} {a:6s} >LR " + " ".join(map(str, w_lr))
                + "   >GBT " + " ".join(map(str, w_gb)) + "   (seeds " + " ".join(map(str, seeds)) + ")")

    # LaTeX table
    tex = [r"\begin{tabular}{clcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{Within city} & \multicolumn{3}{c}{Leave one city out} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"$h$ & Arch. & Mean AP & Seed SD & Fold SD & Mean AP & Seed SD & $>$LR \\",
           r"\midrule"]
    for h in HORIZONS:
        for i, a in enumerate(mt.ARCHS):
            lead = rf"\multirow{{3}}{{*}}{{{h}}}" if i == 0 else ""
            cells = []
            for exp in ("single", "multi"):
                v = summary.get((exp, h, a))
                if v is None:
                    cells += ["--", "--"] + (["--"] if exp == "single" else [])
                    continue
                cells += [f"{v['lo']:.3f}--{v['hi']:.3f}", f"{v['seed_sd']:.3f}"]
                if exp == "single":
                    cells.append(f"{v['fold_sd']:.3f}")
            cw = city_wins.get((h, a))
            cells.append("--" if cw is None else
                         (f"{cw[0]}/3" if cw[0] == cw[1] else f"{cw[0]}--{cw[1]}/3"))
            tex.append(f"{lead} & {a} & " + " & ".join(cells) + r" \\")
        if h != HORIZONS[-1]:
            tex.append(r"\cmidrule(lr){1-8}")
    tex += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.join(here, "tables"), exist_ok=True)
    with open(os.path.join(here, "tables", "seed_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(here, "tables", "tab_seed.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(out))
    print("\n" + "\n".join(tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
