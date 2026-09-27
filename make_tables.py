"""Manuscript tables and in-text statistics from a finished sweep. No retraining.

Reads only what ``find_best_model.py`` and ``naive_baselines.py`` write:

    tuning_results/h<H>/nested<schema>_<arch>_<exp>_ctx24_blocked.jsonl
    tuning_results/h<H>/best_<arch>_<exp>_ctx24_blocked_logz.json
    tuning_results/h<H>/baselines_ctx24.json

and writes, to ``--out`` (default ``tables/``):

    tab_cv_splits.tex          single-city window counts and prevalence per outer fold
    tab_cv_splits_multi.tex    the same per held-out city
    tab_arch_within_city.tex   architecture comparison, single-city-blocked
    tab_baselines.tex          deep models against the WHO rule and logistic regression
    tab_identifiability.tex    hyperparameter agreement / distinct winners / margin
    tab_multi_city.tex         leave-one-city-out results per held-out city
    paper_numbers.txt          every statistic the text quotes that is not in a table

Conventions (the same as ``find_best_model._finalize``): means and standard
deviations are over outer folds, the standard deviation with ddof = 0. "AP" is
sklearn's average precision, stored as ``pr_auc``. Paired tests are two-sided
paired t-tests over the folds common to both predictors.

Train + val counts are the sum of the inner validation splits: the inner groups
partition the development blocks, so their validation splits together hold every
development window exactly once. The script checks that no inner fold was
dropped before relying on that.

    python make_tables.py
    python make_tables.py --horizons 1 --out tables_h1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import os

import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_rel

ARCHS = ("GRU", "TCN", "FUSION")
EXPERIMENTS = {"single": "single-blocked", "multi": "multi"}   # cache name -> baselines name
CITIES = ("Antwerp", "Oslo", "Zagreb")
CONTEXT = 24
LOGREG_PARAMS = 6       # five window-mean coefficients + intercept
WHO_PARAMS = 0          # the WHO rule fits nothing


# --------------------------------------------------------------------------- IO

def read_cache(d, schema, arch, exp):
    """``(inner, outer)``: ``{(outer, combo): rec}`` and ``{outer: rec}``."""
    path = os.path.join(d, f"nested{schema}_{arch}_{exp}_ctx{CONTEXT}_blocked.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run the sweep for this horizon first")
    inner, outer = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "inner":
                inner[(r["outer"], r["combo"])] = r
            elif r.get("kind") == "outer":
                outer[r["outer"]] = r
    return inner, outer


def read_best(d, arch, exp):
    path = os.path.join(d, f"best_{arch}_{exp}_ctx{CONTEXT}_blocked_logz.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_baselines(d, exp):
    path = os.path.join(d, f"baselines_ctx{CONTEXT}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run naive_baselines.py --horizon for it first")
    with open(path, encoding="utf-8") as f:
        return {r["outer"]: r for r in json.load(f) if r["experiment"] == EXPERIMENTS[exp]}


def load(results_dir, horizons, schema):
    """``data[h][exp]`` with per-arch caches, pooled winners and baselines."""
    data = {}
    for h in horizons:
        d = os.path.join(results_dir, f"h{h}")
        data[h] = {}
        for exp in EXPERIMENTS:
            e = {"inner": {}, "outer": {}, "best": {}}
            for a in ARCHS:
                e["inner"][a], e["outer"][a] = read_cache(d, schema, a, exp)
                e["best"][a] = read_best(d, a, exp)
            e["base"] = read_baselines(d, exp)
            folds = list(e["outer"][ARCHS[0]])
            for a in ARCHS:
                if set(e["outer"][a]) != set(folds):
                    raise SystemExit(f"h{h} {exp}: {a} scored different folds")
            if set(e["base"]) != set(folds):
                raise SystemExit(f"h{h} {exp}: baselines scored different folds")
            e["folds"] = sorted(folds, key=fold_order)
            data[h][exp] = e
    return data


def fold_order(outer_id):
    if outer_id.startswith("heldout-"):
        return (CITIES.index(outer_id.split("-", 1)[1]), 0)
    city, b = outer_id.rsplit("-", 1)
    return (CITIES.index(city), int(b[1:]))


def city_of(outer_id):
    return outer_id.split("-", 1)[1] if outer_id.startswith("heldout-") \
        else outer_id.rsplit("-", 1)[0]


# ----------------------------------------------------------------- quantities

_PARAMS: dict = {}


def n_params(arch, hp):
    """Trainable parameters of one grid point, built exactly as the tuner builds it."""
    key = (arch, json.dumps(hp, sort_keys=True))
    if key not in _PARAMS:
        from nested_worker import build_model
        hp = dict(hp)
        if "tcn_channels" in hp:
            hp["tcn_channels"] = tuple(hp["tcn_channels"])
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(arch, hp, 5)
        _PARAMS[key] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _PARAMS[key]


def ap(e, a):
    return np.array([e["outer"][a][f]["test"]["pr_auc"] for f in e["folds"]])


def test_metric(e, a, m, tuned=False):
    key = "test_at_threshold" if tuned else "test"
    return np.array([e["outer"][a][f][key][m] for f in e["folds"]])


def who(e, m):
    return np.array([e["base"][f]["who"][m] for f in e["folds"]])


def lr(e, m):
    return np.array([e["base"][f]["logreg"][m] for f in e["folds"]])


def prevalence(e):
    return np.array([e["outer"][ARCHS[0]][f]["test"]["pos_rate"] for f in e["folds"]])


def pooled_params(e, a):
    return n_params(a, e["best"][a]["hyperparameters"])


def split_counts(e, f):
    """``(dev_n, dev_pos, test_n, test_pos)`` for one outer fold."""
    rec = e["outer"][ARCHS[0]][f]
    if rec["dropped_inner"] or rec["n_inner_used"] != 3:
        raise SystemExit(f"{f}: an inner fold was dropped; train + val cannot be summed")
    inner = e["inner"][ARCHS[0]][(f, 0)]
    dev_n = sum(r["val"]["tp"] + r["val"]["fp"] + r["val"]["tn"] + r["val"]["fn"]
                for r in inner["inner"])
    dev_pos = sum(r["val"]["tp"] + r["val"]["fn"] for r in inner["inner"])
    t = rec["test"]
    return dev_n, dev_pos, t["tp"] + t["fp"] + t["tn"] + t["fn"], t["tp"] + t["fn"]


def selection_stats(e, a):
    """Identifiability measures of one architecture's grid search.

    agreement  fraction of folds whose winner is the pooled best combo
    distinct   number of different winning combos
    margin     winner's lead over the runner-up in inner-val AP, in standard errors
               of a combo mean (mean inner-fold std / sqrt(n_inner)), averaged
    within / across   mean pairwise Spearman correlation of the combo ranking
               between folds of the same city / of different cities
    """
    inner, outer, folds = e["inner"][a], e["outer"][a], e["folds"]
    combos = sorted({c for _, c in inner})
    gaps = []
    for f in folds:
        means = np.array([inner[(f, c)]["val_pr_auc_mean"] for c in combos])
        stds = np.array([inner[(f, c)]["val_pr_auc_std"] for c in combos])
        se = np.nanmean(stds) / math.sqrt(max(inner[(f, combos[0])]["n_inner_used"], 1))
        top = np.sort(means[~np.isnan(means)])[::-1]
        gaps.append((top[0] - top[1]) / se if se > 0 else np.nan)
    winners = [outer[f]["winner_combo"] for f in folds]
    R = np.array([[inner[(f, c)]["val_pr_auc_mean"] for c in combos] for f in folds])
    within, across = [], []
    for i, j in itertools.combinations(range(len(folds)), 2):
        rho = spearmanr(R[i], R[j]).statistic
        (within if city_of(folds[i]) == city_of(folds[j]) else across).append(rho)
    ranks = np.array([rankdata(-row) for row in R])
    m_, n_ = ranks.shape
    S = ((ranks.sum(0) - ranks.sum(0).mean()) ** 2).sum()
    return {"agreement": e["best"][a]["hp_agreement"],
            "distinct": len(set(winners)),
            "margin": float(np.nanmean(gaps)),
            "within": float(np.mean(within)) if within else float("nan"),
            "across": float(np.mean(across)) if across else float("nan"),
            "kendall_w": float(12 * S / (m_ ** 2 * (n_ ** 3 - n_)))}


def paired(x, y):
    """``(mean diff, p, wins of x)`` for a two-sided paired t-test of x against y."""
    d = np.asarray(x) - np.asarray(y)
    return float(d.mean()), float(ttest_rel(x, y).pvalue), int((d > 0).sum())


# ------------------------------------------------------------------ formatting

def n(v):
    """Integer with a LaTeX thin-space thousands separator."""
    return f"{int(v):,}".replace(",", "\\,")


def f3(v):
    return f"{v:.3f}"


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def pval(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def write(out_dir, name, text):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"  wrote {path}")


HEADER = "% {label}  --  requires \\usepackage{{booktabs,multirow}}\n" \
         "% Generated by make_tables.py from the sweep results; do not edit numbers by hand.\n"


# ---------------------------------------------------------------------- tables

def tab_cv_splits(data, horizons, exp):
    single = exp == "single"
    label = "tab:cv_splits" if single else "tab:cv_splits_multi"
    first = "Outer fold" if single else "Held-out city"
    rows = []
    folds = data[horizons[0]][exp]["folds"]
    for f in folds:
        name = f.replace("-b", "--b") if single else city_of(f)
        cells = [name]
        for h in horizons:
            dn, dp, tn, tp = split_counts(data[h][exp], f)
            cells += [n(dn), f3(dp / dn), n(tn), f3(tp / tn)]
        rows.append(" & ".join(cells) + " \\\\")
        if single and f.endswith("-b5") and f != folds[-1]:
            rows.append("\\addlinespace[2pt]")
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{$h = {h}$\\,h}}" for h in horizons)
    rules = "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(horizons)))
    sub = " & Train\\,+\\,val & $\\pi$ & Test & $\\pi$" * len(horizons)
    if single:
        caption = "Single-City Nested Cross-Validation: Window Counts and Prevalence per Outer Fold"
        note = ("Counts are $24 \\times 5$ input windows at 50\\,\\% overlap, and $\\pi$ is "
                "prevalence, the fraction of those windows whose label is positive. Outer fold "
                "$i$ tests on block group $i$; the four remaining groups form the train\\,+\\,val "
                "pool, which the inner loop subdivides further, so no single inner fold trains on "
                "the whole pool. The five outer folds of a city share data by construction, so "
                "these counts must not be summed.")
    else:
        caption = ("Leave-One-City-Out Cross-Validation: Window Counts and Prevalence per "
                   "Held-Out City")
        note = ("Counts are $24 \\times 5$ input windows at 50\\,\\% overlap, and $\\pi$ is "
                "prevalence, the fraction of those windows whose label is positive. Each fold "
                "trains on two cities and tests on the third, so the test split is the entire "
                "held-out city and the train\\,+\\,val pool is the other two cities, which the "
                "inner loop subdivides further. Each city appears once as a test split and twice "
                "inside a pool, so these counts must not be summed.")
    return (HEADER.format(label=label)
            + "\\begin{table*}[!t]\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n\\footnotesize\n"
            "\\renewcommand{\\arraystretch}{1.1}\n"
            f"\\begin{{tabular}}{{l{'rcrc' * len(horizons)}}}\n\\toprule\n"
            f"\\multirow{{2}}{{*}}{{{first}}} & {head} \\\\\n{rules}\n{sub} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\vspace{2pt}\n\\par\n"
            "\\begin{minipage}{\\linewidth}\n\\scriptsize\n" + note
            + "\n\\end{minipage}\n\\end{table*}\n")


def tab_arch_within_city(data, horizons):
    rows = []
    for k, h in enumerate(horizons):
        e = data[h]["single"]
        aps = {a: ap(e, a) for a in ARCHS}
        stack = np.vstack([aps[a] for a in ARCHS])
        won = {a: int((stack.argmax(axis=0) == i).sum()) for i, a in enumerate(ARCHS)}
        f2t = {a: test_metric(e, a, "f2", tuned=True).mean() for a in ARCHS}
        order = sorted(ARCHS, key=lambda a: -aps[a].mean())
        best_ap = max(ARCHS, key=lambda a: aps[a].mean())
        best_f2 = max(ARCHS, key=lambda a: f2t[a])
        for i, a in enumerate(order):
            lead = f"\\multirow{{3}}{{*}}{{{h}}}" if i == 0 else ""
            rows.append(
                f"{lead}\n & {a:<6} & "
                + bold(f3(aps[a].mean()), a == best_ap) + f"\\,$\\pm$\\,{f3(aps[a].std())}"
                + f" & {f3(test_metric(e, a, 'roc_auc').mean())}"
                + f" & {f3(test_metric(e, a, 'f1').mean())}"
                + f" & {f3(test_metric(e, a, 'f2').mean())}"
                + " & " + bold(f3(f2t[a]), a == best_f2)
                + f" & {won[a]} & {n(pooled_params(e, a))} \\\\")
        if k < len(horizons) - 1:
            rows.append("\\cmidrule(lr){1-9}")
    prev = ", ".join(f3(prevalence(data[h]["single"]).mean()) for h in horizons)
    return (HEADER.format(label="tab:arch_within_city")
            + "\\begin{table*}[!t]\n\\caption{Architecture Comparison Within City "
            "(Single-City-Blocked, 15 Outer Folds per Horizon)}\n"
            "\\label{tab:arch_within_city}\n\\centering\n\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{tabular}{clccccccr}\n\\toprule\n"
            "$h$ & Architecture & AP & ROC-AUC & $F_1$@0.5 & $F_2$@0.5 & $F_2$@tuned & Won & "
            "Params \\\\\n\\midrule\n" + "\n".join(rows).replace("\n\n", "\n")
            + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n\\parbox{\\textwidth}{\\footnotesize\n"
            "Means over the 15 outer folds; AP carries its between-fold standard deviation. "
            "\\emph{AP} is average precision, $\\sum_n (R_n - R_{n-1})P_n$, the standard estimator "
            "of the area under the precision--recall curve. Its chance level is the prevalence, "
            f"which is {prev} at $h$ = {', '.join(str(h) for h in horizons)}. $F_\\beta$ uses "
            "$\\beta=2$, weighting recall above precision. ``@0.5'' is at probability 0.5; "
            "``@tuned'' is at the median of the winning configuration's inner-fold "
            "$F_2$-maximising thresholds. \\emph{Won} counts folds on which an architecture "
            "attains the highest AP. \\emph{Params} is the trainable-parameter count of the "
            "pooled configuration. Bold marks the best AP and the best $F_2$@tuned within each "
            "horizon.}\n\\end{table*}\n")


def tab_baselines(data, horizons):
    def cell(x):
        return f"{f3(x.mean())}\\,$\\pm$\\,{f3(x.std())}"
    lines = []
    rows = {"WHO rule (0 param.)": [], "Logistic regression (6)": [],
            **{a: [] for a in ARCHS}}
    for h in horizons:
        e = data[h]["single"]
        preds = {"WHO rule (0 param.)": (who(e, "pr_auc"), who(e, "f2")),
                 "Logistic regression (6)": (lr(e, "pr_auc"), lr(e, "f2_at_threshold")),
                 **{a: (ap(e, a), test_metric(e, a, "f2", tuned=True)) for a in ARCHS}}
        best_ap = max(ARCHS, key=lambda a: preds[a][0].mean())
        best_f2 = max(ARCHS, key=lambda a: preds[a][1].mean())
        for name, (x, y) in preds.items():
            rows[name] += [bold(f3(x.mean()), name == best_ap) + f"\\,$\\pm$\\,{f3(x.std())}"
                           if name in ARCHS else cell(x),
                           bold(f3(y.mean()), name == best_f2)]
    for i, (name, cells) in enumerate(rows.items()):
        lines.append(f"{name:<23} & " + " & ".join(cells) + " \\\\")
        if i == 1:
            lines.append("\\midrule")
    prev = ", ".join(f3(prevalence(data[h]["single"]).mean()) for h in horizons)
    lr_who, arch_lr = [], {a: [] for a in ARCHS}
    for h in horizons:
        e = data[h]["single"]
        d, p, w = paired(lr(e, "pr_auc"), who(e, "pr_auc"))
        lr_who.append(f"${d:+.3f}$ (${pval(p)}$, {w} of {len(e['folds'])} folds)")
        for a in ARCHS:
            d, p, _ = paired(ap(e, a), lr(e, "pr_auc"))
            arch_lr[a].append(f"${d:+.3f}$ (${pval(p)}$)")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$h={h}$}}" for h in horizons)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(horizons)))
    return (HEADER.format(label="tab:baselines")
            + "\\begin{table*}[!t]\n\\caption{Deep Models Against Zero- and Low-Parameter "
            "Baselines, by Forecast Horizon}\n\\label{tab:baselines}\n\\centering\n"
            "\\renewcommand{\\arraystretch}{1.15}\n"
            f"\\begin{{tabular}}{{l{'cc' * len(horizons)}}}\n\\toprule\n & {head} \\\\\n{rules}\n"
            "Predictor" + " & AP & $F_2$" * len(horizons) + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n"
            "\\parbox{\\textwidth}{\\footnotesize\n"
            "Both baselines observe only the 24\\,h input window the networks receive. AP is "
            "average precision; $F_2$ is taken at each predictor's tuned operating point (the "
            f"WHO rule's is its fixed limits). Prevalence is {prev}. Against the WHO rule the "
            "logistic regression gains " + ", ".join(lr_who) + ". Against the logistic "
            "regression the architectures gain, at $h$ = " + ", ".join(str(h) for h in horizons)
            + " respectively: " + "; ".join(f"{a} " + ", ".join(v) for a, v in arch_lr.items())
            + ". Significance is a two-sided paired $t$-test over the "
            f"{len(data[horizons[0]]['single']['folds'])} common folds within a horizon.\n"
            "}\n\\end{table*}\n")


def tab_identifiability(data, horizons):
    stats = {h: {a: selection_stats(data[h]["single"], a) for a in ARCHS} for h in horizons}
    lines = []
    for a in ARCHS:
        cells = ([f"{stats[h][a]['agreement']:.2f}" for h in horizons]
                 + [str(stats[h][a]["distinct"]) for h in horizons]
                 + [f"{stats[h][a]['margin']:.2f}" for h in horizons])
        lines.append(f"{a:<6} & " + " & ".join(cells) + " \\\\")
    k = len(horizons)
    hs = " & ".join(f"$h{{=}}{h}$" for h in horizons)
    nf = len(data[horizons[0]]["single"]["folds"])
    return (HEADER.format(label="tab:identifiability")
            + "\\begin{table*}[!t]\n\\caption{Hyperparameter Identifiability Across the "
            "32-Point Grids}\n\\label{tab:identifiability}\n\\centering\n"
            "\\renewcommand{\\arraystretch}{1.15}\n"
            f"\\begin{{tabular}}{{l{'c' * 3 * k}}}\n\\toprule\n"
            f" & \\multicolumn{{{k}}}{{c}}{{Winner agreement}} & \\multicolumn{{{k}}}{{c}}"
            f"{{Distinct winners (of {nf})}} & \\multicolumn{{{k}}}{{c}}{{Margin (SE)}} \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(3))
            + f"\nArchitecture & {hs} & {hs} & {hs} \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n\\parbox{\\textwidth}{\\footnotesize\n"
            "\\emph{Winner agreement} is the fraction of outer folds whose selected configuration "
            "equals the pooled best configuration. \\emph{Distinct winners} counts how many "
            "different configurations win at least one fold. \\emph{Margin} is the winning "
            "configuration's lead over the runner-up in inner-validation AP, expressed in "
            "standard errors of a configuration mean and averaged over folds.}\n\\end{table*}\n")


def tab_multi_city(data, horizons):
    lines = []
    for k, h in enumerate(horizons):
        e = data[h]["multi"]
        lr_ap = lr(e, "pr_auc")
        preds = {a: (ap(e, a), test_metric(e, a, "f2", tuned=True), pooled_params(e, a))
                 for a in ARCHS}
        preds["Logistic regression"] = (lr_ap, lr(e, "f2_at_threshold"), LOGREG_PARAMS)
        preds["WHO rule"] = (who(e, "pr_auc"), who(e, "f2"), WHO_PARAMS)
        best = max(ARCHS, key=lambda a: preds[a][0].mean())
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Horizon }}$h = {h}$\\,\\textit{{h}}}} \\\\")
        for name, (x, f2, params) in preds.items():
            beats = "--" if name == "Logistic regression" else f"{int((x > lr_ap).sum())}/{len(x)}"
            lines.append(f"{name} & " + " & ".join(f3(v) for v in x)
                         + " & " + bold(f3(x.mean()), name == best)
                         + f" & {beats} & {f3(f2.mean())} & {n(params)} \\\\")
        if k < len(horizons) - 1:
            lines.append("\\midrule")
    e1 = data[horizons[0]]["multi"]
    sizes = ", ".join(
        f"{city_of(f)} {n(split_counts(e1, f)[2])} at {f3(e1['outer'][ARCHS[0]][f]['test']['pos_rate'])}"
        for f in e1["folds"])
    cities = " & ".join(city_of(f) for f in e1["folds"])
    return (HEADER.format(label="tab:multi_city")
            + "\\begin{table*}[!t]\n\\caption{Leave-One-City-Out Generalization: Average "
            "Precision and $F_2$ by Held-Out City}\n\\label{tab:multi_city}\n\\centering\n"
            "\\footnotesize\n\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{tabular}{lccccccr}\n\\toprule\n"
            "\\multirow{2}{*}{Predictor} & \\multicolumn{3}{c}{Average precision by held-out "
            "city} & \\multirow{2}{*}{Mean AP} & \\multirow{2}{*}{Cities $>$ LR} & "
            "\\multirow{2}{*}{Mean $F_2$} & \\multirow{2}{*}{Params} \\\\\n\\cmidrule(lr){2-4}\n"
            f" & {cities} & & & & \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n\\vspace{2pt}\n\\par\n"
            "\\begin{minipage}{\\textwidth}\n\\scriptsize\n"
            f"Test windows and prevalence at $h = {horizons[0]}$\\,h are {sizes}. Mean AP and "
            "mean $F_2$ are unweighted means over the three folds. Cities $>$ LR counts the "
            "held-out cities on which the predictor's AP exceeds that of the logistic "
            "regression. $F_2$ is evaluated at the WHO rule's own fixed limits and at the "
            "validation-tuned threshold for every other predictor. Params is the "
            "trainable-parameter count of the pooled configuration. Bold marks the highest "
            "mean AP among the three architectures.\n\\end{minipage}\n\\end{table*}\n")


# --------------------------------------------------------------- in-text report

def paper_numbers(data, horizons):
    out = []

    def say(s=""):
        out.append(s)

    def rng(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return f"{min(vals):.3f} to {max(vals):.3f}"

    say("# In-text statistics (generated by make_tables.py; not a table in the paper)")
    say("# means and std over outer folds, std ddof=0; paired t-tests two-sided")

    say("\n== Single-city-blocked ==")
    who_gap, lr_f2, f2_best = [], [], []
    for h in horizons:
        e = data[h]["single"]
        prev = prevalence(e)
        say(f"\n-- h = {h} --")
        say(f"mean prevalence over folds {prev.mean():.3f}")
        aps = {a: ap(e, a) for a in ARCHS}
        means = {a: aps[a].mean() for a in ARCHS}
        say("AP mean: " + ", ".join(f"{a} {means[a]:.3f}" for a in ARCHS)
            + f"; WHO {who(e, 'pr_auc').mean():.3f}; LR {lr(e, 'pr_auc').mean():.3f}")
        say(f"best-minus-worst architecture AP {max(means.values()) - min(means.values()):.3f}; "
            f"between-fold AP std {rng([aps[a].std() for a in ARCHS])}")
        for a, b in itertools.combinations(ARCHS, 2):
            d, p, w = paired(aps[a], aps[b])
            say(f"  paired AP {a} - {b}: {d:+.4f} ({pval(p)}), {a} wins {w}/{len(aps[a])}")
        for a in ARCHS:
            d, p, w = paired(aps[a], who(e, "pr_auc"))
            who_gap.append(d)
            say(f"  {a} vs WHO rule AP {d:+.3f} ({pval(p)}), wins {w}")
        for a in ARCHS:
            d, p, w = paired(aps[a], lr(e, "pr_auc"))
            say(f"  {a} vs LR AP {d:+.3f} ({pval(p)}), wins {w}")
        for a in ARCHS:
            d, p, w = paired(test_metric(e, a, "f2", tuned=True), lr(e, "f2_at_threshold"))
            lr_f2.append(p)
            say(f"  {a} vs LR F2@tuned {d:+.3f} ({pval(p)}), wins {w}")
        best = max(ARCHS, key=lambda a: test_metric(e, a, "f2", tuned=True).mean())
        d = test_metric(e, best, "f2", tuned=True).mean() - lr(e, "f2_at_threshold").mean()
        f2_best.append(d)
        say(f"  best F2@tuned architecture {best} leads LR by {d:+.3f}")
        # Per-fold detail the text quotes.
        span = {f: [e["outer"][a][f]["test"]["pr_auc"] for a in ARCHS] for f in e["folds"]}
        say("  per fold: prevalence | AP " + "/".join(ARCHS) + " | spread | AP/prevalence")
        for i, f in enumerate(e["folds"]):
            v = span[f]
            say(f"    {f:<11} {prev[i]:.3f} | " + " ".join(f"{x:.3f}" for x in v)
                + f" | {max(v) - min(v):.3f} | " + " ".join(f"{x / prev[i]:.1f}" for x in v))
        below = sum(int(x <= prev[i]) for i, f in enumerate(e["folds"]) for x in span[f])
        say(f"  points on or below AP = prevalence: {below}")
        for a in ARCHS:
            lowest = [e["folds"][i] for i in np.argsort(aps[a])[:3]]
            say(f"  three lowest-AP folds, {a}: {', '.join(lowest)}")
        say("  pooled-configuration params: "
            + ", ".join(f"{a} {pooled_params(e, a):,}" for a in ARCHS)
            + " (orders of magnitude over the 6-parameter LR: "
            + ", ".join(f"{a} {math.log10(pooled_params(e, a) / LOGREG_PARAMS):.2f}"
                        for a in ARCHS) + ")")
        for a in ARCHS:
            s = selection_stats(e, a)
            say(f"  selection {a}: agreement {s['agreement']:.2f}, distinct {s['distinct']}, "
                f"margin {s['margin']:.2f} SE, rank Spearman within city {s['within']:.2f} / "
                f"across cities {s['across']:.2f}, Kendall's W {s['kendall_w']:.2f}")
    say(f"\nall horizons: architectures over the WHO rule by {rng(who_gap)}; best architecture's "
        f"F2@tuned lead over LR {', '.join(f'{d:+.3f}' for d in f2_best)}; "
        f"smallest architecture-vs-LR F2 p-value {min(lr_f2):.3f}")

    say("\n== Multi-city (leave-one-city-out) ==")
    drops, city_drop, f2_who = [], {c: [] for c in CITIES}, []
    for h in horizons:
        e, s = data[h]["multi"], data[h]["single"]
        prev = prevalence(e)
        say(f"\n-- h = {h} --")
        say("prevalence: " + ", ".join(f"{city_of(f)} {p:.3f}" for f, p in zip(e["folds"], prev)))
        lr_ap, lr_roc = lr(e, "pr_auc"), lr(e, "roc_auc")
        say("LR per city AP " + " ".join(f"{x:.3f}" for x in lr_ap)
            + " | ROC-AUC " + " ".join(f"{x:.3f}" for x in lr_roc))
        for a in ARCHS:
            x = ap(e, a)
            roc = test_metric(e, a, "roc_auc")
            drops.append(ap(s, a).mean() - x.mean())
            say(f"{a}: AP " + " ".join(f"{v:.3f}" for v in x) + " | ROC-AUC "
                + " ".join(f"{v:.3f}" for v in roc)
                + " | margin over LR " + " ".join(f"{v:+.3f}" for v in x - lr_ap)
                + f" (mean {np.mean(x - lr_ap):+.3f})"
                + " | AP/prevalence " + " ".join(f"{v / p:.1f}" for v, p in zip(x, prev))
                + f" | single-city mean minus this {ap(s, a).mean() - x.mean():.3f}")
            for i, f in enumerate(e["folds"]):
                c = city_of(f)
                within = np.mean([s["outer"][a][g]["test"]["pr_auc"] for g in s["folds"]
                                  if city_of(g) == c])
                city_drop[c].append(within - x[i])
                say(f"    held-out {c}: {x[i]:.3f} vs within-city mean {within:.3f} "
                    f"(loss {within - x[i]:.3f})")
            f2 = test_metric(e, a, "f2", tuned=True)
            f2_who.append(f2.mean() - who(e, "f2").mean())
            say(f"    F2@tuned beats LR on {int((f2 > lr(e, 'f2_at_threshold')).sum())}/3 "
                f"cities, beats WHO rule on {int((f2 > who(e, 'f2')).sum())}/3; mean F2 "
                f"minus WHO rule {f2.mean() - who(e, 'f2').mean():+.3f}")
            st = selection_stats(e, a)
            say(f"    selection: distinct winners {st['distinct']}, margin {st['margin']:.2f} SE")
        for a, b in itertools.combinations(ARCHS, 2):
            say(f"  mean paired AP {a} - {b}: {np.mean(ap(e, a) - ap(e, b)):+.3f}")
        stack = np.vstack([ap(e, a) for a in ARCHS])
        say("  best architecture per held-out city: "
            + ", ".join(f"{city_of(f)} {ARCHS[int(i)]}" for f, i in
                        zip(e["folds"], stack.argmax(axis=0))))
        below = int((stack <= prev[None, :]).sum())
        say(f"  architecture points on or below AP = prevalence: {below}")
    say(f"\nall horizons: single-city mean AP minus multi-city mean AP {rng(drops)}")
    for c in CITIES:
        say(f"  held-out {c} loss against its own within-city folds {rng(city_drop[c])}")
    say(f"  architectures' mean F2 over the WHO rule {rng(f2_who)}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------ main

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--results-dir", default=os.path.join(here, "tuning_results"))
    ap_.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap_.add_argument("--schema", type=int, default=5, help="5 = logz (the paper), 4 = minmax")
    ap_.add_argument("--out", default=os.path.join(here, "tables"))
    args = ap_.parse_args()

    data = load(args.results_dir, args.horizons, args.schema)
    hs = args.horizons
    write(args.out, "tab_cv_splits.tex", tab_cv_splits(data, hs, "single"))
    write(args.out, "tab_cv_splits_multi.tex", tab_cv_splits(data, hs, "multi"))
    write(args.out, "tab_arch_within_city.tex", tab_arch_within_city(data, hs))
    write(args.out, "tab_baselines.tex", tab_baselines(data, hs))
    write(args.out, "tab_identifiability.tex", tab_identifiability(data, hs))
    write(args.out, "tab_multi_city.tex", tab_multi_city(data, hs))
    write(args.out, "paper_numbers.txt", paper_numbers(data, hs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
