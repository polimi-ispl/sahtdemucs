#!/usr/bin/env python3
"""
compare_ablation.py — summarise a decoder-depth ablation sweep.

Scans ``<out-root>/*/`` for the run directories produced by
``htdemucsspatial/train.py`` (each holds ``config.json`` + ``history.csv``),
prints a comparison table sorted by best total validation loss, and optionally
plots the validation curves of every run on shared axes.

Works while the sweep is still running — ``history.csv`` is appended per epoch.

Usage::

    python htdemucsspatial/compare_ablation.py /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial
    python htdemucsspatial/compare_ablation.py <out-root> --plot ablation.png --sort valid_ild
    python htdemucsspatial/compare_ablation.py <out-root> --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ["valid_total", "valid_td", "valid_ild", "valid_itd"]


def load_run(run_dir: Path):
    """Return a summary dict for one run directory, or None if it has no history."""
    hist_path, cfg_path = run_dir / "history.csv", run_dir / "config.json"
    if not hist_path.exists():
        return None
    with hist_path.open(encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    # Skip a history.csv written by another trainer (e.g. sahtdemucs.train, whose
    # columns are train_si/train_ild rather than the td/ild/itd triplet here).
    if not rows or not all(m in rows[0] for m in METRICS):
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    best = min(rows, key=lambda r: r["valid_total"])
    return {
        "run":         run_dir.name,
        "strategy":    cfg.get("freeze_strategy", run_dir.name),
        "trainable":   int(cfg.get("n_trainable", 0)),
        "trainable_%": 100.0 * cfg.get("n_trainable", 0) / max(cfg.get("n_total", 1), 1),
        "epochs":      int(rows[-1]["epoch"]),
        "target":      int(cfg.get("epochs", 0)),
        "best_epoch":  int(best["epoch"]),
        "s/epoch":     sum(r["seconds"] for r in rows) / len(rows),
        **{m: best[m] for m in METRICS},
        "_rows":       rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root", type=Path, help="parent directory of the run directories")
    ap.add_argument("--sort", default="valid_total", choices=METRICS + ["trainable", "run"])
    ap.add_argument("--plot", type=Path, help="write a PNG with the validation curves")
    ap.add_argument("--csv", type=Path, help="write the summary table as CSV")
    args = ap.parse_args()

    runs = [r for r in (load_run(d) for d in sorted(args.out_root.iterdir()) if d.is_dir())
            if r is not None]
    if not runs:
        print(f"No run with a history.csv under {args.out_root}")
        return 1
    runs.sort(key=lambda r: r[args.sort] if args.sort != "run" else r["run"])

    hdr = (f"{'run':<22} {'trainable':>12} {'%':>5} {'epochs':>9} {'s/ep':>6} "
           f"{'best@':>6} {'valid_total':>12} {'valid_td':>10} {'valid_ild':>11} {'valid_itd':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        print(f"{r['run']:<22} {r['trainable']:>12,} {r['trainable_%']:>5.1f} "
              f"{str(r['epochs']) + '/' + str(r['target']):>9} {r['s/epoch']:>6.0f} "
              f"{r['best_epoch']:>6} {r['valid_total']:>12.4f} {r['valid_td']:>10.4f} "
              f"{r['valid_ild']:>11.5f} {r['valid_itd']:>11.5f}")
    print(f"\nsorted by {args.sort} (lower is better) — {len(runs)} runs under {args.out_root}")

    if args.csv:
        fields = [k for k in runs[0] if not k.startswith("_")]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in runs:
                w.writerow({k: r[k] for k in fields})
        print(f"summary written to {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(METRICS), figsize=(5.5 * len(METRICS), 4), dpi=150)
        for ax, metric in zip(axes, METRICS):
            for r in runs:
                ax.plot([row["epoch"] for row in r["_rows"]],
                        [row[metric] for row in r["_rows"]],
                        label=r["run"], lw=1.5)
            ax.set_xlabel("epoch"); ax.set_ylabel(metric); ax.set_title(metric)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Decoder-depth ablation — {args.out_root.name}")
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"plot written to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
