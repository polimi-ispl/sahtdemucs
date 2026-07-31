#!/usr/bin/env python3
"""
train.py - headless HTDemucs spatial fine-tune (one run = one process).

Training half of the HTDemucs spatial fine-tune: no notebook state, no live
plots, and every artifact of a run lives in its own directory named after the
freeze strategy, so several runs can be launched in parallel on the same machine
and compared afterwards.  ``notebook/TestHTDemucsSpatial.ipynb`` then discovers
those run directories and compares them against the frozen baseline.

Layout produced for a run::

    <out-root>/<strategy>[__<tag>]/
        htdmcs_sp_<strategy>.pt     best checkpoint (lowest total validation loss)
        last.pt                     latest epoch (for --resume), unless --no-save-last
        config.json                 full argv + environment of the run
        history.csv                 one row per epoch (appended live)
        train.log                   full textual log

Examples
--------
Single run::

    python -m htdemucsspatial.train \
        --dataset-root /nas/home/macerbi/Dataset/binauralMUSMOISESDB \
        --out-root     /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial \
        --freeze-strategy dec_last2 --epochs 100

Smoke test (few minutes, verifies the whole path before launching the sweep)::

    python -m htdemucsspatial.train ... --epochs 1 \
        --limit-train-batches 4 --limit-valid-batches 2

Once a sweep is done, ``python htdemucsspatial/compare_ablation.py <out-root>``
summarises every run from its ``history.csv``.

The script can also be run directly (``python htdemucsspatial/train.py ...``); it
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
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Make the repo root importable when the script is run from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demucs.pretrained import get_model                        # noqa: E402
from sahtdemucs.dataset import MusdbSpatialDataset             # noqa: E402
from htdemucsspatial.freeze import apply_freeze_strategy       # noqa: E402
from htdemucsspatial.losses import HTDemucsSpatialLoss         # noqa: E402

SOURCES = ["drums", "bass", "other", "vocals"]
HISTORY_FIELDS = [
    "epoch", "train_total", "train_td", "train_ild", "train_itd",
    "valid_total", "valid_td", "valid_ild", "valid_itd", "lr", "seconds",
]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless HTDemucs spatial fine-tune (one freeze strategy per process).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="binauralMUSMOISESDB root (contains train/ and test/)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="parent directory that will hold one sub-directory per run")
    p.add_argument("--tag", default="",
                   help="optional suffix appended to the run directory name")

    # What to train
    p.add_argument("--freeze-strategy", required=True,
                   help="which blocks to train, e.g. 'dec_last2', 'enc_first1+dec_last1' "
                        "or 'all' (see htdemucsspatial/freeze.py for the grammar)")

    # Training hyper-parameters (defaults mirror the notebook)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--valid-split", type=float, default=0.2)
    p.add_argument("--crops-per-track", type=int, default=4)
    p.add_argument("--min-rms", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234,
                   help="shared by every run of a sweep so all strategies see the same crops")

    # Loss weights / sub-band configuration
    p.add_argument("--lambda-td",  type=float, default=0.9)
    p.add_argument("--lambda-ild", type=float, default=1e-4)
    p.add_argument("--lambda-itd", type=float, default=0.0)
    p.add_argument("--ild-n-fft",  type=int,   default=4096)
    p.add_argument("--ild-hop",    type=int,   default=512)
    p.add_argument("--ild-n-bands", type=int,  default=64)
    p.add_argument("--ild-band-scale", choices=["mel", "linear"], default="mel")
    p.add_argument("--itd-max-lag", type=int,   default=64)
    p.add_argument("--itd-beta",    type=float, default=20.0)

    # Runtime
    p.add_argument("--device", default="auto",
                   help='"auto" (GPU with most free memory), "cuda:N" or "cpu"')
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                   help="disable FP16 autocast")
    p.add_argument("--no-save-last", dest="save_last", action="store_false", default=True,
                   help="do not keep last.pt (halves the disk used per run)")
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

def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_path, encoding="utf-8")):
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


@torch.no_grad()
def run_valid(model, loader, loss_fn, device, amp, limit=0):
    """Validation pass with a frozen RNG, so the crops are identical every epoch."""
    model.eval()
    py_rng, th_rng = random.getstate(), torch.get_rng_state()
    random.seed(1234)
    torch.manual_seed(1234)
    tot = td = il = it = 0.0
    n = 0
    for i, (mix, targets) in enumerate(loader):
        if limit and i >= limit:
            break
        mix, targets = mix.to(device), targets.to(device)
        with autocast("cuda", enabled=amp):
            estimates = model(mix)                    # (B, S, 2, T)
        total, l_td, l_il, l_it = loss_fn(estimates, targets)
        tot += total.item(); td += l_td.item(); il += l_il.item(); it += l_it.item()
        n += 1
    random.setstate(py_rng)
    torch.set_rng_state(th_rng)
    d = max(n, 1)
    return tot / d, td / d, il / d, it / d


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)

    run_name = args.freeze_strategy + (f"__{args.tag}" if args.tag else "")
    run_dir  = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"htdmcs_sp_{args.freeze_strategy}.pt"
    last_path = run_dir / "last.pt"
    hist_path = run_dir / "history.csv"

    log = setup_logger(run_dir / "train.log")
    seed_everything(args.seed)
    device = pick_device(args.device)
    amp    = args.amp and device.type == "cuda"

    log.info(f"=== run {run_name} (pid {os.getpid()} on {socket.gethostname()}) ===")
    log.info(f"strategy   : {args.freeze_strategy}")
    log.info(f"run dir    : {run_dir}")
    log.info(f"device     : {device}"
             + (f" ({torch.cuda.get_device_name(device)}, "
                f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB)"
                if device.type == "cuda" else ""))
    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}  amp={amp}")

    # ── Model + freeze strategy ───────────────────────────────────────────────
    bag   = get_model("htdemucs")
    model = bag.models[0] if hasattr(bag, "models") else bag
    model = model.to(device)

    sample_rate = model.samplerate                          # 44100 Hz
    seg_len     = int(float(model.segment) * sample_rate)   # ~8 s

    groups      = apply_freeze_strategy(model, args.freeze_strategy)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    log.info(f"trainable  : {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")
    for name, params in groups.items():
        log.info(f"    [{name}] {sum(p.numel() for p in params):,}")

    # config.json — everything needed to reproduce or interpret the run
    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "run_name": run_name, "started": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "pid": os.getpid(), "python": platform.python_version(),
        "torch": torch.__version__, "git_commit": git_commit(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "n_trainable": n_trainable, "n_total": n_total,
        "trainable_groups": {k: sum(p.numel() for p in v) for k, v in groups.items()},
        "sample_rate": sample_rate, "segment_samples": seg_len,
    }, indent=2), encoding="utf-8")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = HTDemucsSpatialLoss(
        lambda_td=args.lambda_td, lambda_ild=args.lambda_ild, lambda_itd=args.lambda_itd,
        n_fft=args.ild_n_fft, hop_length=args.ild_hop, n_bands=args.ild_n_bands,
        band_scale=args.ild_band_scale, sample_rate=sample_rate,
        itd_max_lag=args.itd_max_lag, itd_beta=args.itd_beta,
    )

    # ── Data (track-level split, identical across runs thanks to --seed) ──────
    full_ds = MusdbSpatialDataset(
        args.dataset_root, split="train", sources=SOURCES,
        segment_len=seg_len, sample_rate=sample_rate, augment=True,
        crops_per_track=args.crops_per_track, min_rms=args.min_rms,
    )
    n_valid = max(1, int(len(full_ds.tracks) * args.valid_split))
    n_train = len(full_ds.tracks) - n_valid
    train_ds = full_ds.subset(full_ds.tracks[:n_train], crops_per_track=args.crops_per_track)
    valid_ds = full_ds.subset(full_ds.tracks[n_train:], crops_per_track=1, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)
    log.info(f"train      : {n_train} tracks x {args.crops_per_track} crops = "
             f"{len(train_ds)} items ({len(train_loader)} batches @batch={args.batch_size}, "
             f"accum={args.accum_steps} -> effective {args.batch_size * args.accum_steps})")
    log.info(f"valid      : {n_valid} tracks ({len(valid_loader)} batches)")

    # ── Optimiser / scheduler (Adam, no weight decay — HTDemucs recipe) ───────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0,
    )
    scaler    = GradScaler("cuda", enabled=amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20,
    )

    start_epoch, best_valid = 1, math.inf
    if args.resume and last_path.exists():
        # weights_only=False: last.pt also carries the optimiser/scheduler state.
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        if ck.get("sched_state"):
            scheduler.load_state_dict(ck["sched_state"])
        if ck.get("scaler_state"):
            scaler.load_state_dict(ck["scaler_state"])
        start_epoch = ck["epoch"] + 1
        best_valid  = ck.get("best_valid", math.inf)
        log.info(f"resumed from {last_path} at epoch {start_epoch} (best={best_valid:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    t_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            tot = td = il = it = 0.0
            n_batches = 0
            n_steps   = args.limit_train_batches or len(train_loader)
            optimizer.zero_grad(set_to_none=True)

            for step, (mix, targets) in enumerate(train_loader):
                if args.limit_train_batches and step >= args.limit_train_batches:
                    break
                mix, targets = mix.to(device), targets.to(device)
                with autocast("cuda", enabled=amp):
                    estimates = model(mix)                       # (B, S, 2, T)
                # Loss in FP32 (inputs are cast inside the loss)
                total, l_td, l_il, l_it = loss_fn(estimates, targets)
                scaler.scale(total / args.accum_steps).backward()

                tot += total.item(); td += l_td.item(); il += l_il.item(); it += l_it.item()
                n_batches += 1

                if args.log_every and n_batches % args.log_every == 0:
                    per_batch = (time.time() - t0) / n_batches
                    # Same breakdown as the epoch line, so a run can be read at a
                    # glance without waiting for the epoch to close.
                    log.info(f"  epoch {epoch} [{n_batches}/{n_steps}]  "
                             f"train={tot / n_batches:.4f} (td={td / n_batches:.3f} "
                             f"ild={il / n_batches:.3f} itd={it / n_batches:.3f})  "
                             f"{per_batch:.2f}s/batch  "
                             f"epoch eta {(n_steps - n_batches) * per_batch / 60:.1f} min")

                last_step = n_steps - 1
                if (step + 1) % args.accum_steps == 0 or step == last_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=args.clip_grad,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            d = max(n_batches, 1)
            tr = (tot / d, td / d, il / d, it / d)
            va = run_valid(model, valid_loader, loss_fn, device, amp,
                           limit=args.limit_valid_batches) if len(valid_ds) else (float("nan"),) * 4

            elapsed = time.time() - t0
            row = dict(zip(HISTORY_FIELDS,
                           (epoch, *tr, *va, scheduler.get_last_lr()[0], round(elapsed, 1))))
            append_history(hist_path, row)

            eta = (args.epochs - epoch) * (time.time() - t_start) / max(epoch - start_epoch + 1, 1)
            msg = (f"epoch {epoch:4d}/{args.epochs}  "
                   f"train={tr[0]:.4f} (td={tr[1]:.3f} ild={tr[2]:.3f} itd={tr[3]:.3f})  "
                   f"valid={va[0]:.4f} (td={va[1]:.3f} ild={va[2]:.3f} itd={va[3]:.3f})  "
                   f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s  eta={eta / 3600:.1f}h")

            payload = {
                "epoch": epoch, "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if amp else None,
                "train_loss": tr[0], "valid_loss": va[0],
                "valid_ild": va[2], "valid_itd": va[3],
                "freeze_strategy": args.freeze_strategy,
                "best_valid": min(best_valid, va[0]),
                "config": str(run_dir / "config.json"),
            }
            if va[0] < best_valid:
                best_valid = va[0]
                payload["best_valid"] = best_valid
                torch.save(payload, ckpt_path)
                msg += "  <- best"
            if args.save_last:
                torch.save(payload, last_path)
            log.info(msg)

    except KeyboardInterrupt:
        log.info("interrupted — history and the best checkpoint are preserved")

    log.info(f"done. best valid = {best_valid:.4f}")
    log.info(f"checkpoint      : {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
