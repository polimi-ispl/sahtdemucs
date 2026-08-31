#!/usr/bin/env python3
"""
train.py — headless SA-HTDemucs training (one run = one process).

Trains the frozen HTDemucs backbone + per-source ``SpatialCueModule`` heads with
:class:`~sahtdemucs.losses.SpatialLoss`, an Adam + ``ReduceLROnPlateau`` recipe
and a "best on validation ILD" checkpoint rule — no notebook state, no live
plots, and every artefact of a run lives in its own directory, so several
configurations can be launched in parallel and compared afterwards.

Layout produced for a run::

    <out-root>/<run-name>/
        spatial_modules_<run-name>.pt   best checkpoint (lowest valid ILD),
                                        a bare ``spatial_modules`` state_dict
        last.pt                         latest epoch incl. optimiser/scheduler
                                        state (for --resume), unless --no-save-last
        config.json                     full argv + environment + spatial config
        history_<run-name>.json         per-epoch curves, one list per metric
        history.csv                     one row per epoch (appended live)
        train.log                       full textual log

``notebook/TestSAHTDemucs.ipynb`` discovers these run directories under a shared
``RUNS_ROOT``, rebuilds each head set from its ``config.json`` and compares them
on the test split against the frozen HTDemucs baseline; ``sahtdemucs.separate``
does the same headless, for a single run.

Examples
--------
Single run::

    python -m sahtdemucs.train \
        --dataset-root /nas/home/macerbi/Dataset/binauralMUSDB18HQ \
        --out-root     /nas/home/macerbi/sahtdemucs/runs \
        --epochs 250

Smoke test (a couple of minutes, exercises the whole path)::

    python -m sahtdemucs.train ... --epochs 1 \
        --limit-train-batches 4 --limit-valid-batches 2

Overfitting sanity check on a single fixed segment::

    python -m sahtdemucs.train ... --overfit --epochs 300

The script can also be run directly (``python sahtdemucs/train.py ...``); it
inserts the repository root into ``sys.path`` itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make the repo root importable when the script is run as a plain file.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demucs.pretrained import get_model                       # noqa: E402
from sahtdemucs.dataset import MusdbSpatialDataset            # noqa: E402
from sahtdemucs.losses import SpatialLoss                     # noqa: E402
from sahtdemucs.model import SAHTDemucs                       # noqa: E402

SOURCES = ["drums", "bass", "other", "vocals"]

# Columns of history.csv (history_<run>.json keeps the notebook's own layout).
HISTORY_FIELDS = [
    "epoch", "train_total", "train_si", "train_ild", "train_itd",
    "valid_total", "valid_si", "valid_ild", "valid_itd", "lr", "seconds",
]

# Keys copied into config.json["spatial_config"] — everything `separate.py`
# needs to rebuild the SpatialCueModule heads that match the checkpoint.
SPATIAL_KEYS = [
    "spatial_arch", "hidden", "n_fft", "hop_length", "n_bands",
    "ild_scale", "band_scale", "sample_rate", "max_lag", "use_gb",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless SA-HTDemucs training (frozen HTDemucs + SpatialCueModule heads).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="binauralMUSDB18HQ root (contains train/ and test/)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="parent directory that will hold one sub-directory per run")
    p.add_argument("--run-name", default="",
                   help="run directory name (default: current timestamp)")
    p.add_argument("--tag", default="",
                   help="optional suffix appended to the run directory name")

    # Training hyper-parameters (defaults mirror the notebook)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--valid-split", type=float, default=0.2)
    p.add_argument("--crops-per-track", type=int, default=4)
    p.add_argument("--min-rms", type=float, default=1e-4,
                   help="skip near-silent random crops (0 disables the check)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234,
                   help="shared by every run of a sweep so all configs see the same crops")
    p.add_argument("--valid-seed", type=int, default=42,
                   help="seed used once to materialise the fixed validation crops")

    # Scheduler (ReduceLROnPlateau on the validation ILD term)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=30)
    p.add_argument("--min-lr", type=float, default=1e-6)

    # Loss weights
    p.add_argument("--lambda-si", type=float, default=10.0)
    p.add_argument("--lambda-ild", type=float, default=1.0)
    p.add_argument("--lambda-itd", type=float, default=0.0,
                   help="structural no-op for the ILD-gain head; wired for forward compatibility")
    p.add_argument("--si-margin-db", type=float, default=0.2,
                   help="tolerated SI-SDR degradation (dB) w.r.t. the frozen backbone")
    p.add_argument("--itd-max-lag", type=int, default=64)
    p.add_argument("--itd-beta", type=float, default=20.0)

    # SpatialCueModule configuration (must match at inference time)
    p.add_argument("--spatial-arch", choices=["cnn2d", "cnn1d"], default="cnn2d")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=4096)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--n-bands", type=int, default=64)
    p.add_argument("--ild-scale", type=float, default=15.0,
                   help="max |delta ILD| the head can apply, in dB")
    p.add_argument("--band-scale", choices=["mel", "linear"], default="mel")
    p.add_argument("--no-global-branch", dest="use_gb", action="store_false", default=True,
                   help="disable the cnn2d global branch")

    # Runtime
    p.add_argument("--device", default="auto",
                   help='"auto" (GPU with most free memory), "cuda:N" or "cpu"')
    p.add_argument("--overfit", action="store_true",
                   help="sanity check: train and validate on one fixed segment")
    p.add_argument("--no-cache-valid", dest="cache_valid", action="store_false", default=True,
                   help="re-draw validation crops every epoch instead of caching them once")
    p.add_argument("--no-save-last", dest="save_last", action="store_false", default=True,
                   help="do not keep last.pt (no --resume support, half the disk per run)")
    p.add_argument("--resume", action="store_true",
                   help="continue from last.pt in the run directory if present")
    p.add_argument("--limit-train-batches", type=int, default=0, help="0 = all")
    p.add_argument("--limit-valid-batches", type=int, default=0, help="0 = all")
    p.add_argument("--log-every", type=int, default=25,
                   help="progress line every N training batches; without it the "
                        "first epoch runs silently and a slow run is "
                        "indistinguishable from a hung one (0 = epoch end only)")
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path | None, name: str = "sahtdemucs") -> logging.Logger:
    """Log to stdout and, when ``log_path`` is given, to that file as well."""
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def pick_device(spec: str) -> torch.device:
    """Resolve --device; "auto" returns the visible CUDA device with most free VRAM."""
    if spec != "auto":
        return torch.device(spec)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0                      # GPU fully occupied or unavailable
        if free > best_free:
            best_free, best_idx = free, i
    return torch.device(f"cuda:{best_idx}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def append_history(csv_path: Path, row: dict) -> None:
    """Append one epoch to history.csv, writing the header on first use."""
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in HISTORY_FIELDS})


def cache_batches(loader: DataLoader, seed: int) -> list:
    """Materialise a loader once, so every epoch sees the exact same crops.

    Crop positions are drawn from the ``random`` module inside the dataset, so
    the RNG is seeded here and restored afterwards to leave the training stream
    untouched.  Batches stay on the CPU and are moved device-side per epoch.
    """
    py_rng, th_rng = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    batches = [(mix, targets) for mix, targets in loader]
    random.setstate(py_rng)
    torch.set_rng_state(th_rng)
    return batches


@torch.no_grad()
def run_valid(model, batches, loss_fn, device, limit: int = 0):
    """Evaluate the loss and its three components over the validation batches."""
    model.eval()
    tot = si = ild = itd = 0.0
    n = 0
    for i, (mix, targets) in enumerate(batches):
        if limit and i >= limit:
            break
        mix, targets = mix.to(device), targets.to(device)
        estimates, raw_estimates, _ = model(mix)
        total, l_si, l_ild, l_itd = loss_fn(estimates, targets, raw_estimates)
        tot += total.item(); si += l_si.item(); ild += l_ild.item(); itd += l_itd.item()
        n += 1
    d = max(n, 1)
    return tot / d, si / d, ild / d, itd / d


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = args.out_root / (run_name + (f"__{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"spatial_modules_{run_name}.pt"
    last_path = run_dir / "last.pt"
    hist_json = run_dir / f"history_{run_name}.json"
    hist_csv  = run_dir / "history.csv"

    log = setup_logger(run_dir / "train.log")
    seed_everything(args.seed)
    device = pick_device(args.device)

    log.info(f"=== run {run_dir.name} (pid {os.getpid()} on {socket.gethostname()}) ===")
    log.info(f"run dir    : {run_dir}")
    log.info(f"device     : {device}"
             + (f" ({torch.cuda.get_device_name(device)}, "
                f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB)"
                if device.type == "cuda" else ""))
    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if args.overfit:
        log.info("*** OVERFIT MODE - single fixed segment ***")

    # ── Model: frozen HTDemucs backbone + per-source spatial heads ────────────
    bag  = get_model("htdemucs")
    base = bag.models[0] if hasattr(bag, "models") else bag

    sample_rate = base.samplerate                            # 44100 Hz
    seg_len     = int(float(base.segment) * sample_rate)     # ~8 s

    spatial_config = {
        "spatial_arch": args.spatial_arch, "hidden": args.hidden,
        "n_fft": args.n_fft, "hop_length": args.hop_length, "n_bands": args.n_bands,
        "ild_scale": args.ild_scale, "band_scale": args.band_scale,
        "sample_rate": sample_rate, "max_lag": args.itd_max_lag, "use_gb": args.use_gb,
    }
    model = SAHTDemucs(base, sources=base.sources, **spatial_config).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"sources    : {list(base.sources)}")
    log.info(f"segment    : {float(base.segment):.2f} s ({seg_len} samples) @ {sample_rate} Hz")
    log.info(f"trainable  : {n_trainable:,}   frozen: {n_frozen:,}")

    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "run_name": run_name, "started": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "pid": os.getpid(), "python": platform.python_version(),
        "torch": torch.__version__, "git_commit": git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "sources": list(base.sources), "sample_rate": sample_rate,
        "segment_samples": seg_len,
        "n_trainable": n_trainable, "n_frozen": n_frozen,
        "spatial_config": spatial_config,
    }, indent=2), encoding="utf-8")

    # ── Loss — STFT settings must match the SpatialCueModule configuration ────
    loss_fn = SpatialLoss(
        lambda_si=args.lambda_si, lambda_ild=args.lambda_ild, lambda_itd=args.lambda_itd,
        si_margin_db=args.si_margin_db, n_fft=args.n_fft, hop_length=args.hop_length,
        n_bands=args.n_bands, band_scale=args.band_scale, sample_rate=sample_rate,
        itd_max_lag=args.itd_max_lag, itd_beta=args.itd_beta,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    full_ds = MusdbSpatialDataset(
        args.dataset_root, split="train", sources=SOURCES,
        segment_len=seg_len, sample_rate=sample_rate,
        augment=not args.overfit,
        crops_per_track=1 if args.overfit else args.crops_per_track,
        min_rms=args.min_rms,
    )

    if args.overfit:
        # One fixed (mix, targets) pair, drawn once and reused every epoch: the
        # loss must converge to ~0, otherwise the gradient path is broken.
        torch.manual_seed(0)
        mix_fixed, tgt_fixed = full_ds[0]                      # (2, T), (S, 2, T)
        fixed_ds = TensorDataset(mix_fixed.unsqueeze(0), tgt_fixed.unsqueeze(0))
        train_loader = DataLoader(fixed_ds, batch_size=1, shuffle=False, num_workers=0)
        valid_loader = DataLoader(fixed_ds, batch_size=1, shuffle=False, num_workers=0)
        n_valid_items = 1
        log.info(f"overfit    : track '{full_ds.tracks[0].name}', "
                 f"mix={tuple(mix_fixed.shape)} targets={tuple(tgt_fixed.shape)}")
    else:
        # Split at the *track* level so no crop of the same song leaks across
        # the train/valid boundary (which random_split cannot guarantee).
        n_valid = max(1, int(len(full_ds.tracks) * args.valid_split))
        n_train = len(full_ds.tracks) - n_valid
        train_ds = full_ds.subset(full_ds.tracks[:n_train], crops_per_track=args.crops_per_track)
        valid_ds = full_ds.subset(full_ds.tracks[n_train:], crops_per_track=1, augment=False)
        n_valid_items = len(valid_ds)

        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=pin)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=pin)
        log.info(f"train      : {n_train} tracks x {args.crops_per_track} crops = "
                 f"{len(train_ds)} items ({len(train_loader)} batches @batch={args.batch_size})")
        log.info(f"valid      : {n_valid} tracks ({len(valid_loader)} batches)")

    # Cache the validation crops once so only model quality moves the curve.
    valid_batches = (cache_batches(valid_loader, args.valid_seed)
                     if args.cache_valid else valid_loader)
    if args.cache_valid:
        log.info(f"valid cache: {len(valid_batches)} batches ({n_valid_items} segments)")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    history = {
        "train_total": [], "train_si": [], "train_ild": [], "train_itd": [],
        "valid_total": [], "valid_si": [], "valid_ild": [], "valid_itd": [],
        "lr": [],
    }
    start_epoch, best_valid = 1, math.inf

    if args.resume and last_path.exists():
        # weights_only=False: last.pt also carries the optimiser/scheduler state.
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.spatial_modules.load_state_dict(ck["spatial_state"])
        optimizer.load_state_dict(ck["optim_state"])
        if ck.get("sched_state"):
            scheduler.load_state_dict(ck["sched_state"])
        start_epoch = ck["epoch"] + 1
        best_valid  = ck.get("best_valid", math.inf)
        if hist_json.exists():
            history = json.loads(hist_json.read_text(encoding="utf-8"))
        log.info(f"resumed from {last_path} at epoch {start_epoch} (best valid ild={best_valid:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    t_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            tot = si = ild = itd = 0.0
            n_batches = 0
            n_steps   = args.limit_train_batches or len(train_loader)

            for step, (mix, targets) in enumerate(train_loader):
                if args.limit_train_batches and step >= args.limit_train_batches:
                    break
                mix, targets = mix.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)

                estimates, raw_estimates, _ = model(mix)
                total, l_si, l_ild, l_itd = loss_fn(estimates, targets, raw_estimates)
                total.backward()
                optimizer.step()

                tot += total.item(); si += l_si.item(); ild += l_ild.item(); itd += l_itd.item()
                n_batches += 1

                if args.log_every and n_batches % args.log_every == 0:
                    per_batch = (time.time() - t0) / n_batches
                    # Same breakdown as the epoch line, so a run can be read at a
                    # glance without waiting for the epoch to close.
                    log.info(f"  epoch {epoch} [{n_batches}/{n_steps}]  "
                             f"train={tot / n_batches:.4f} (si={si / n_batches:.3f} "
                             f"ild={ild / n_batches:.3f})  {per_batch:.2f}s/batch  "
                             f"epoch eta {(n_steps - n_batches) * per_batch / 60:.1f} min")

            d  = max(n_batches, 1)
            tr = (tot / d, si / d, ild / d, itd / d)
            va = run_valid(model, valid_batches, loss_fn, device,
                           limit=args.limit_valid_batches)

            # The ILD term is what the spatial heads actually optimise, so it
            # drives both the scheduler and the "best checkpoint" rule.
            scheduler.step(va[2])
            current_lr = optimizer.param_groups[0]["lr"]

            for key, value in zip(
                ("train_total", "train_si", "train_ild", "train_itd",
                 "valid_total", "valid_si", "valid_ild", "valid_itd", "lr"),
                (*tr, *va, current_lr),
            ):
                history[key].append(value)
            hist_json.write_text(json.dumps(history), encoding="utf-8")

            elapsed = time.time() - t0
            append_history(hist_csv, dict(zip(
                HISTORY_FIELDS, (epoch, *tr, *va, current_lr, round(elapsed, 1)))))

            eta = (args.epochs - epoch) * (time.time() - t_start) / max(epoch - start_epoch + 1, 1)
            msg = (f"epoch {epoch:4d}/{args.epochs}  "
                   f"train={tr[0]:.4f} (si={tr[1]:.3f} ild={tr[2]:.3f})  "
                   f"valid={va[0]:.4f} (si={va[1]:.3f} ild={va[2]:.3f})  "
                   f"lr={current_lr:.2e}  {elapsed:.0f}s  eta={eta / 3600:.1f}h")

            if va[2] < best_valid:
                best_valid = va[2]
                # Bare state_dict, exactly what the notebook's load cell expects.
                torch.save(model.spatial_modules.state_dict(), ckpt_path)
                msg += "  <- best"
            if args.save_last:
                torch.save({
                    "epoch": epoch,
                    "spatial_state": model.spatial_modules.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "valid_total": va[0], "valid_ild": va[2],
                    "best_valid": best_valid,
                    "spatial_config": spatial_config,
                    "config": str(run_dir / "config.json"),
                }, last_path)
            log.info(msg)

    except KeyboardInterrupt:
        log.info("interrupted - history and the best checkpoint are preserved")

    log.info(f"done. best valid ild = {best_valid:.4f}")
    log.info(f"checkpoint      : {ckpt_path}")
    log.info(f"history         : {hist_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
