"""Reproduce every result in the paper, stage by stage.

    python reproduce.py                       # tables + figures from the shipped results
    python reproduce.py --stages all          # retrain everything, then tables + figures
    python reproduce.py --stages sweep --horizons 12

Stages, in order (each runs only when named; ``all`` is every stage):

    geometry   leak and embargo assertions on the fold plans     (needs the data)
    sweep      nested-CV grid search, 3 architectures x 2 designs (needs the data + GPU;
               about 2.5 h per horizon on an RTX 4060 laptop GPU with 3 workers)
    baselines  WHO rule + logistic regression on the same folds   (needs the data)
    tables     LaTeX tables and in-text statistics  -> tables/
    figures    figures 1-4 (single and multi) and 5 -> figures/   (fig 5 needs the data)

The default, ``tables figures``, rebuilds the manuscript assets from the results in
tuning_results/ without touching the data, except figure 5 (the raw-data skewness
plot), which is skipped with a note when the data is absent.

The sweep resumes from tuning_results/h<H>/ caches. To retrain from scratch, move the
shipped nested5_* files out of the way first (or pass --out-root to write elsewhere),
because a resume onto a complete cache retrains nothing.

The data location is --data-dir, else $AQM_DATA_DIR, else ./senseurcity_data_v03/dataset.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("geometry", "sweep", "baselines", "tables", "figures")


def run(args: list[str], log: str | None = None) -> None:
    cmd = [sys.executable, "-u", *args]
    print(f"\n$ {' '.join(cmd)}" + (f"  >> {log}" if log else ""), flush=True)
    if log:
        with open(log, "a", encoding="utf-8") as f:
            code = subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    else:
        code = subprocess.call(cmd, cwd=HERE)
    if code != 0:
        raise SystemExit(f"failed ({code}): {' '.join(args)}" + (f"; see {log}" if log else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["tables", "figures"],
                    choices=[*STAGES, "all"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR")
                    or os.path.join(HERE, "senseurcity_data_v03", "dataset"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "tuning_results"),
                    help="results root; each horizon uses <out-root>/h<H>/")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent trainings sharing the GPU (~1.4 GB host RAM each)")
    args = ap.parse_args()
    stages = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    have_data = os.path.isdir(args.data_dir)
    data = ["--data-dir", args.data_dir]
    needs_data = {"geometry", "sweep", "baselines"} & set(stages)
    if needs_data and not have_data:
        raise SystemExit(f"stages {sorted(needs_data)} need the dataset, not found at "
                         f"{args.data_dir}. Run download_data.py or set AQM_DATA_DIR.")

    if "geometry" in stages:
        for h in args.horizons:
            run(["test_nested_geometry.py", "--materialize", "--horizon", str(h), *data])

    if "sweep" in stages:
        for h in args.horizons:
            out = os.path.join(args.out_root, f"h{h}")
            os.makedirs(out, exist_ok=True)
            for exp in ("single-blocked", "multi"):
                run(["find_best_model.py", "--experiments", exp, "--workers",
                     str(args.workers), "--deterministic", "--scaling", "logz",
                     "--horizon", str(h), "--out-dir", out, *data],
                    log=os.path.join(out, "nested5_run.log"))

    if "baselines" in stages:
        for h in args.horizons:
            run(["naive_baselines.py", "--horizon", str(h),
                 "--out-dir", os.path.join(args.out_root, f"h{h}"), *data])

    if "tables" in stages:
        run(["make_tables.py", "--results-dir", args.out_root,
             "--horizons", *map(str, args.horizons)])

    if "figures" in stages:
        for exp in ("single", "multi"):
            run(["make_figures.py", "--results-dir", args.out_root, "--experiment", exp,
                 "--horizons", *map(str, args.horizons)])
        if have_data:
            run(["make_skew_figure.py", *data])
        else:
            print(f"\n[skip] figure 5 (make_skew_figure.py) needs the dataset at {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
