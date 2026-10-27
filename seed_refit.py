"""Seed sensitivity: refit every outer fold's selected configuration with new seeds.

    python seed_refit.py --smoke                 # ~2 minutes: one fold, one seed, few epochs
    python seed_refit.py --check                 # seed 42 on three folds; must match the paper
    python seed_refit.py                         # the full run (seeds 1-4, every fold)
    python seed_analysis.py                      # -> tables/seed_*.txt and .tex

The hyperparameter search is NOT repeated. For each (horizon, design, outer fold,
architecture) the configuration and decision threshold selected in the shipped sweep
are read from ``tuning_results/h<H>/nested5_*.jsonl`` and refitted exactly as the
sweep refitted them -- same folds, same refit split, same trainer, epochs and early
stopping -- with only the random seed changed. The spread of the outer-test scores
across seeds is the seed variance of the reported pipeline, conditional on the
selected configuration.

Results are appended to ``seed_results/seed_refit.jsonl`` one training at a time,
so an interrupted run resumes where it stopped. Run with ``--deterministic``
(the default here, as in the paper's sweep) so the run itself is reproducible.

Cost: 3 horizons x 18 outer folds x 3 architectures x 4 seeds = 648 trainings plus
54 fold materializations. The models are small; a GPU is much faster but a CPU works.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
import time

import find_best_model as fbm
from compare_models import materialize_outer, nested_plan
from nested_worker import Job

ARCHS = ("GRU", "TCN", "FUSION")
DESIGNS = ("single-blocked", "multi")
HORIZONS = (1, 12, 24)
CHECK_FOLDS = {"single-blocked": ("Antwerp-b1", "Oslo-b2"), "multi": ("heldout-Zagreb",)}


def outer_records(results_dir, h, arch, experiment):
    path = os.path.join(results_dir, f"h{h}",
                        f"nested5_{arch}_{experiment}_ctx24_blocked.jsonl")
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "outer":
                recs[r["outer"]] = r
    return recs


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                with contextlib.suppress(json.JSONDecodeError, KeyError):
                    r = json.loads(line)
                    keys.add((r["horizon"], r["design"], r["outer"], r["arch"], r["seed"]))
    return keys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR")
                    or os.path.join(here, "senseurcity_data_v03", "dataset"))
    ap.add_argument("--results-dir", default=os.path.join(here, "tuning_results"))
    ap.add_argument("--out-dir", default=os.path.join(here, "seed_results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=DESIGNS)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel trainings; 2-3 suits a laptop GPU")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="refit seed 42 on three folds per horizon; scores must match the "
                         "shipped results, which confirms the setup reproduces the paper")
    ap.add_argument("--smoke", action="store_true",
                    help="one fold, one architecture, one seed, 3 epochs: tests the setup only")
    args = ap.parse_args()

    deterministic = not args.no_deterministic
    epochs = fbm.EPOCHS
    seeds = args.seeds
    horizons, designs, archs = args.horizons, args.designs, ARCHS
    out_name = "seed_refit.jsonl"
    if args.check:
        seeds, out_name = [fbm.SEED], "seed_check.jsonl"
    if args.smoke:
        seeds, horizons, designs, archs = [1], [1], ["single-blocked"], ("TCN",)
        epochs, out_name = 3, "seed_smoke.jsonl"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, out_name)
    if args.smoke and os.path.exists(out_path):
        os.remove(out_path)
    done = done_keys(out_path)
    runner = fbm._Runner(args.workers, deterministic)
    tmp_dir = tempfile.mkdtemp(prefix="aqm_seed_")
    t_start = time.time()
    n_new = 0
    try:
        for h in horizons:
            fbm.HORIZON = h
            common = {"context_size": fbm.CONTEXT_SIZE, "overlap": fbm.OVERLAP, "horizon": h}
            for design in designs:
                experiment, protocol = fbm.EXPERIMENTS[design]
                recs = {a: outer_records(args.results_dir, h, a, experiment) for a in archs}
                plan = nested_plan(args.data_dir, experiment=experiment, protocol=protocol,
                                   cities=list(fbm.CITIES),
                                   k_outer=fbm.K_OUTER, n_inner=fbm.N_INNER,
                                   block_hours=fbm.BLOCK_HOURS, **common)
                if args.check:
                    plan = [s for s in plan if s.outer_id in CHECK_FOLDS[design]]
                if args.smoke:
                    plan = plan[:1]
                for spec in plan:
                    todo = [(a, s) for a in archs for s in seeds
                            if (h, design, spec.outer_id, a, s) not in done
                            and spec.outer_id in recs[a]]
                    if not todo:
                        continue
                    print(f"\nh={h} {design} {spec.outer_id}: materializing "
                          f"({len(todo)} trainings)", flush=True)
                    bundle = materialize_outer(args.data_dir, spec, **common,
                                               scaling=fbm.DEFAULT_SCALING)
                    if bundle is None:
                        continue
                    ff = fbm._export_outer(bundle, tmp_dir, f"h{h}_{design}_{spec.outer_id}")
                    del bundle
                    jobs = []
                    for a, s in todo:
                        r = recs[a][spec.outer_id]
                        combo = r["winner_combo"]
                        hp = fbm._grid_for(a)[combo]
                        if json.dumps(fbm._jsonify(hp), sort_keys=True) != \
                                json.dumps(r["hp"], sort_keys=True):
                            raise SystemExit(f"{a} {spec.outer_id}: grid point {combo} does not "
                                             "match the recorded configuration; the grid changed")
                        jobs.append(Job(arch=a, combo=combo, hp=hp, fold=ff.refit_fold,
                                        dev_path=ff.dev_path, test_path=ff.test_path,
                                        input_channels=fbm.N_INPUT_CHANNELS, seed=s,
                                        epochs=epochs, patience=fbm.PATIENCE,
                                        batch_size=fbm.DEFAULT_BATCH, eval_batch=fbm.EVAL_BATCH,
                                        threshold=r["threshold"]["logit"],
                                        threshold_beta=fbm.THRESHOLD_BETA))
                    futures = [runner.submit(j) for j in jobs]
                    for fut in futures:
                        jb, res = fut.result()
                        ref = recs[jb.arch][spec.outer_id]
                        t, tt = res["test_metrics"] or {}, res.get("test_metrics_at_threshold") or {}
                        row = {"horizon": h, "design": design, "outer": spec.outer_id,
                               "arch": jb.arch, "seed": jb.seed, "combo": jb.combo,
                               "pr_auc": t.get("pr_auc"), "roc_auc": t.get("roc_auc"),
                               "f2_at_threshold": tt.get("f2"),
                               "best_epoch": res["best_epoch"], "secs": round(res["secs"], 1),
                               "paper_pr_auc": ref["test"].get("pr_auc"),
                               "epochs": epochs, "deterministic": deterministic}
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                        n_new += 1
                        print(f"  {jb.arch:6s} seed {jb.seed}: AP {row['pr_auc']:.3f} "
                              f"(paper, seed 42: {row['paper_pr_auc']:.3f})  "
                              f"F2 {row['f2_at_threshold']:.3f}  {row['secs']:.0f}s", flush=True)
                    for p in (ff.dev_path, ff.test_path):
                        with contextlib.suppress(OSError):
                            os.remove(p)
    finally:
        runner.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mins = (time.time() - t_start) / 60
    print(f"\n{n_new} trainings in {mins:.1f} min -> {out_path}")
    if args.smoke:
        print("Smoke test passed: the setup trains and scores. Now run --check.")
    if args.check:
        rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        worst = max(abs(r["pr_auc"] - r["paper_pr_auc"]) for r in rows)
        print(f"largest difference from the paper's seed-42 AP: {worst:.4f}")
        print("Reproduces the paper." if worst < 0.005 else
              "Differs from the paper by more than rounding: different hardware or library "
              "versions can do this (see README, Determinism). Report this number with the "
              "results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
