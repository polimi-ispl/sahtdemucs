#!/usr/bin/env python3
"""
separate.py - headless SA-HTDemucs inference and evaluation.

Headless counterpart of ``notebook/TestSAHTDemucs.ipynb`` for a single run:
loads a trained ``spatial_modules`` checkpoint on top of the frozen HTDemucs
backbone, separates one or more tracks with overlap-add chunking, and — when
ground-truth stems are available — reports SI-SDR and per-sub-band ILD MAE
against them, optionally next to the raw HTDemucs baseline.

The sub-band configuration (FFT size, hop, number of bands, band scale) is read
from the run's ``config.json`` when present, so the metric always uses the same
band layout as the training loss.  Command-line flags override it.

Layout produced under ``--out-dir``::

    <track>/{drums,bass,other,vocals}.wav   estimates (unless --no-save-stems)
    baseline/<track>/*.wav                  backbone-only stems (with --baseline
                                            and --save-baseline-stems)
    metrics.json                            per-track, per-source raw metrics
    ild_mae_ranking.csv                     tracks ranked best -> worst
    ild_mae_per_band.zip                    notebook-format tab-separated curves
    ild_mae_per_band.png                    with --plot

Examples
--------
Separate the test split and evaluate against its stems, with the baseline::

    python -m sahtdemucs.separate \
        --ckpt  /nas/home/macerbi/sahtdemucs/runs/20260531_155238/spatial_modules_20260531_155238.pt \
        --input /nas/home/macerbi/Dataset/binauralMUSDB18HQ/test \
        --out-dir /nas/home/macerbi/sahtdemucs/runs/20260531_155238/estimates \
        --evaluate --baseline --plot

Separate a single file (no metrics — no ground truth)::

    python -m sahtdemucs.separate --ckpt run/spatial_modules_run.pt \
        --input song.wav --out-dir estimates/

Re-score stems separated in an earlier session, without re-running the model::

    python -m sahtdemucs.separate --from-estimates estimates/ \
        --input /path/to/binauralMUSDB18HQ/test --out-dir estimates/ --evaluate

The script can also be run directly (``python sahtdemucs/separate.py ...``); it
inserts the repository root into ``sys.path`` itself.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torchaudio

# Make the repo root importable when the script is run as a plain file.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sahtdemucs.dataset import load_audio                     # noqa: E402
from sahtdemucs.metrics import si_sdr, ild_bands_mae, itd_bands_mae   # noqa: E402
from sahtdemucs.spatial import mel_bin_assignment             # noqa: E402
from sahtdemucs.train import SPATIAL_KEYS, pick_device, setup_logger  # noqa: E402

SOURCES = ["drums", "bass", "other", "vocals"]
COLORS  = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

# Defaults of the SpatialCueModule configuration — used only when neither the
# checkpoint's config.json nor a command-line flag provides a value.
DEFAULT_SPATIAL = {
    "spatial_arch": "cnn2d", "hidden": 64, "n_fft": 4096, "hop_length": 512,
    "n_bands": 64, "ild_scale": 15.0, "band_scale": "mel", "sample_rate": 44100,
    "max_lag": 64, "use_gb": True,
}


class Track(NamedTuple):
    """One item to process: a mixture to separate plus where its stems live."""
    name: str
    mix_path: Optional[Path]     # None when scoring pre-computed estimates
    gt_dir: Optional[Path]       # directory holding the ground-truth stems


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SA-HTDemucs inference + SI-SDR / per-sub-band ILD MAE evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=Path,
                   help="spatial_modules checkpoint (bare state_dict or last.pt); "
                        "omit only with --from-estimates or --baseline-only")
    p.add_argument("--input", type=Path, required=True,
                   help="a .wav mixture, a track directory containing mixture.wav, "
                        "or a directory of such track directories (e.g. <dataset>/test)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="where estimates and metric files are written")
    p.add_argument("--from-estimates", type=Path, default=None,
                   help="score stems already written under this directory instead "
                        "of running the model; pass --ckpt as well so the band "
                        "config is read from the run's config.json")

    # What to compute
    p.add_argument("--evaluate", action="store_true",
                   help="compute SI-SDR and per-sub-band ILD MAE against the stems "
                        "found next to each mixture")
    p.add_argument("--baseline", action="store_true",
                   help="also evaluate the frozen HTDemucs backbone (no spatial correction)")
    p.add_argument("--baseline-only", action="store_true",
                   help="run and evaluate only the backbone; no checkpoint needed")
    p.add_argument("--itd", action="store_true",
                   help="also report per-sub-band ITD MAE (slower)")
    p.add_argument("--plot", action="store_true",
                   help="write ild_mae_per_band.png (matplotlib, Agg backend)")

    # Output control
    p.add_argument("--no-save-stems", dest="save_stems", action="store_false", default=True,
                   help="evaluate without writing the separated WAVs")
    p.add_argument("--save-baseline-stems", action="store_true",
                   help="also write the backbone-only stems under <out-dir>/baseline/")
    p.add_argument("--limit-tracks", type=int, default=0, help="0 = all")
    p.add_argument("--shifts", type=int, default=0,
                   help="random-shift passes averaged by apply_model; 0 keeps the "
                        "run reproducible (demucs defaults to 1, which randomises "
                        "the estimates and moves SI-SDR run to run)")

    # SpatialCueModule configuration — defaults come from the run's config.json
    p.add_argument("--spatial-arch", choices=["cnn2d", "cnn1d"], default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--n-bands", type=int, default=None)
    p.add_argument("--ild-scale", type=float, default=None)
    p.add_argument("--band-scale", choices=["mel", "linear"], default=None)
    p.add_argument("--max-lag", type=int, default=None)
    p.add_argument("--no-global-branch", dest="use_gb", action="store_false", default=None)

    p.add_argument("--device", default="auto",
                   help='"auto" (GPU with most free memory), "cuda:N" or "cpu"')
    return p.parse_args(argv)


# ── Track discovery ───────────────────────────────────────────────────────────

def discover_tracks(path: Path) -> List[Track]:
    """Resolve --input into a list of tracks.

    Accepts a single ``.wav`` file, a track directory containing ``mixture.wav``
    (ground-truth stems, if any, sit next to it), or a parent directory holding
    one sub-directory per track — the MUSDB18-HQ ``test/`` layout.
    """
    if path.is_file():
        return [Track(path.stem, path, path.parent)]
    if not path.is_dir():
        raise FileNotFoundError(f"--input not found: {path}")

    if (path / "mixture.wav").exists():
        return [Track(path.name, path / "mixture.wav", path)]

    tracks = [
        Track(d.name, d / "mixture.wav", d)
        for d in sorted(path.iterdir(), key=lambda p: p.name.lower())
        if d.is_dir() and (d / "mixture.wav").exists()
    ]
    if tracks:
        return tracks

    # Fall back to a flat directory of mixtures — no ground truth available.
    wavs = sorted(path.glob("*.wav"), key=lambda p: p.name.lower())
    if not wavs:
        raise FileNotFoundError(
            f"No mixture.wav track directories and no .wav files under {path}")
    return [Track(w.stem, w, None) for w in wavs]


# ── Model ─────────────────────────────────────────────────────────────────────

def resolve_spatial_config(args: argparse.Namespace) -> Dict:
    """Merge the SpatialCueModule config: run config.json < command-line flags."""
    cfg = dict(DEFAULT_SPATIAL)
    source = "built-in defaults"

    if args.ckpt is not None:
        payload_cfg = None
        cfg_json = args.ckpt.parent / "config.json"
        if cfg_json.exists():
            stored = json.loads(cfg_json.read_text(encoding="utf-8"))
            payload_cfg, source = stored.get("spatial_config"), str(cfg_json)
        if payload_cfg:
            cfg.update({k: v for k, v in payload_cfg.items() if k in SPATIAL_KEYS})

    overrides = {
        "spatial_arch": args.spatial_arch, "hidden": args.hidden, "n_fft": args.n_fft,
        "hop_length": args.hop_length, "n_bands": args.n_bands,
        "ild_scale": args.ild_scale, "band_scale": args.band_scale,
        "max_lag": args.max_lag, "use_gb": args.use_gb,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["_source"] = source
    return cfg


def build_model(args: argparse.Namespace, cfg: Dict, device: torch.device, log):
    """Instantiate SA-HTDemucs and load the spatial heads from ``--ckpt``.

    ``--ckpt`` accepts both checkpoint flavours written by ``train.py``: the bare
    ``spatial_modules`` state_dict (the notebook format) and the richer
    ``last.pt`` payload, whose stored config wins over ``config.json``.
    """
    from demucs.pretrained import get_model
    from sahtdemucs.model import SAHTDemucs

    bag  = get_model("htdemucs")
    base = bag.models[0] if hasattr(bag, "models") else bag
    cfg["sample_rate"] = base.samplerate

    state = None
    if args.ckpt is not None:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "spatial_state" in payload:
            state = payload["spatial_state"]
            stored = payload.get("spatial_config") or {}
            cfg.update({k: v for k, v in stored.items() if k in SPATIAL_KEYS})
            cfg["_source"] = f"{args.ckpt} (last.pt payload)"
            log.info(f"checkpoint : {args.ckpt} (epoch {payload.get('epoch', '?')})")
        else:
            state = payload
            log.info(f"checkpoint : {args.ckpt}")

    model_kwargs = {k: cfg[k] for k in SPATIAL_KEYS}
    model = SAHTDemucs(base, sources=base.sources, **model_kwargs).to(device)
    if state is not None:
        model.spatial_modules.load_state_dict(state)
    model.eval()
    log.info(f"spatial cfg: {json.dumps(model_kwargs)}  (from {cfg['_source']})")
    return model, base.samplerate


@torch.no_grad()
def separate_baseline(model, wav: torch.Tensor, shifts: int = 0) -> torch.Tensor:
    """Run the HTDemucs backbone only — no spatial correction. Returns (S, 2, T)."""
    from demucs.apply import apply_model
    return apply_model(model.base_model, wav.unsqueeze(0),
                       shifts=shifts, progress=False).squeeze(0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def score_track(stems: Dict[int, torch.Tensor], track: Track, cfg: Dict,
                sample_rate: int, with_itd: bool) -> Dict[str, Dict]:
    """SI-SDR and per-sub-band ILD (and optionally ITD) MAE for one track.

    ``stems`` maps the source index to a ``(2, T)`` estimate on the CPU; sources
    whose ground-truth stem is missing are skipped.
    """
    out: Dict[str, Dict] = {}
    for i, src in enumerate(SOURCES):
        tgt_path = track.gt_dir / f"{src}.wav" if track.gt_dir else None
        if tgt_path is None or not tgt_path.exists() or i not in stems:
            continue
        tgt = load_audio(tgt_path, sample_rate)
        est = stems[i]
        T   = min(tgt.shape[-1], est.shape[-1])
        tgt, est = tgt[:, :T], est[:, :T]

        band_kwargs = dict(
            n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"]),
            n_bands=int(cfg["n_bands"]), scale=cfg["band_scale"],
            sample_rate=int(sample_rate),
        )
        res = {
            "si_sdr": si_sdr(est, tgt),
            "ild_bands_mae": ild_bands_mae(est, tgt, **band_kwargs).tolist(),
        }
        if with_itd:
            res["itd_bands_mae"] = itd_bands_mae(
                est, tgt, max_lag=int(cfg["max_lag"]), **band_kwargs).tolist()
        out[src] = res
    return out


def band_centre_hz(cfg: Dict, sample_rate: int) -> np.ndarray:
    """Centre frequency of each sub-band — the x-axis of the per-band MAE curve."""
    n_fft, n_bands = int(cfg["n_fft"]), int(cfg["n_bands"])
    if cfg["band_scale"] == "mel":
        assign = mel_bin_assignment(n_fft, n_bands, int(sample_rate))
        bin_hz = np.arange(n_fft // 2 + 1) * (sample_rate / n_fft)
        return np.array([
            bin_hz[(assign == m).numpy()].mean() if (assign == m).any() else 0.0
            for m in range(n_bands)
        ])
    return np.linspace(0, sample_rate / 2, n_bands, endpoint=False)


def mean_bands(results: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Per-source per-band ILD MAE, averaged over all scored tracks."""
    out = {}
    for src in SOURCES:
        curves = [t[src]["ild_bands_mae"] for t in results.values() if src in t]
        if curves:
            out[src] = np.mean(np.asarray(curves, dtype=float), axis=0)
    return out


# ── Reporting ─────────────────────────────────────────────────────────────────

def write_ranking(results: Dict[str, Dict], path: Path, log) -> None:
    """Rank tracks by the source-mean scalar ILD MAE (best -> worst)."""
    rows = []
    for name, track_res in results.items():
        per_src = {src: float(np.mean(track_res[src]["ild_bands_mae"]))
                   for src in SOURCES if src in track_res}
        if not per_src:
            continue
        row = {"track": name, **per_src, "mean": float(np.mean(list(per_src.values())))}
        rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda r: r["mean"])

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track", *SOURCES, "mean"])
        w.writeheader()
        w.writerows(rows)

    log.info("scalar ILD MAE (dB) per track - best -> worst")
    header = f"{'track':<40}" + "".join(f"{s:>9}" for s in [*SOURCES, "mean"])
    log.info(header)
    log.info("-" * len(header))
    for r in rows:
        log.info(f"{r['track'][:39]:<40}"
                 + "".join(f"{r.get(s, float('nan')):>9.3f}" for s in [*SOURCES, "mean"]))
    log.info(f"best : {rows[0]['track']}  ({rows[0]['mean']:.3f} dB)")
    log.info(f"worst: {rows[-1]['track']}  ({rows[-1]['mean']:.3f} dB)")


def report_si_sdr(results: Dict[str, Dict], baseline: Optional[Dict[str, Dict]], log) -> None:
    """Per-source SI-SDR, next to the backbone baseline when it was computed."""
    def collect(res, src):
        return [t[src]["si_sdr"] for t in res.values() if src in t]

    log.info("SI-SDR (dB) - SA-HTDemucs" + (" vs HT-Demucs baseline" if baseline else ""))
    head = f"{'source':<10}{'mean':>10}{'std':>10}"
    if baseline:
        head += f"{'bl mean':>12}{'bl std':>10}"
    log.info(head)
    log.info("-" * len(head))
    all_sp, all_bl = [], []
    for src in SOURCES:
        sp = collect(results, src)
        if not sp:
            continue
        all_sp += sp
        line = f"{src:<10}{np.mean(sp):>+10.2f}{np.std(sp):>10.2f}"
        if baseline:
            bl = collect(baseline, src)
            all_bl += bl
            line += f"{np.mean(bl):>+12.2f}{np.std(bl):>10.2f}" if bl else f"{'-':>12}{'-':>10}"
        log.info(line)
    if all_sp:
        log.info("-" * len(head))
        line = f"{'all':<10}{np.mean(all_sp):>+10.2f}{np.std(all_sp):>10.2f}"
        if baseline and all_bl:
            line += f"{np.mean(all_bl):>+12.2f}{np.std(all_bl):>10.2f}"
        log.info(line)


def write_band_curves(curves: Dict[str, Dict[str, np.ndarray]], band_hz: np.ndarray,
                      zip_path: Path) -> None:
    """Write the per-band ILD MAE curves as a ZIP of tab-separated .txt files."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tag, mae_dict in curves.items():
            for src, values in mae_dict.items():
                buf = io.StringIO()
                buf.write("frequency_hz\tild_mae_dB\n")
                for freq, val in zip(band_hz, values):
                    buf.write(f"{freq:.4f}\t{float(val):.6f}\n")
                zf.writestr(f"ild_mae_per_band_{src}_{tag}.txt", buf.getvalue())


def plot_band_curves(curves: Dict[str, Dict[str, np.ndarray]], band_hz: np.ndarray,
                     n_tracks: int, png_path: Path) -> None:
    """Per-band ILD MAE, one panel per model — the notebook's summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"sahtdemucs": "SA-HTDemucs", "htdemucs": "HT-Demucs baseline"}
    items  = list(curves.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6.5 * len(items), 4),
                             dpi=150, sharey=True, squeeze=False)
    for ax, (tag, mae_dict) in zip(axes[0], items):
        for i, src in enumerate(SOURCES):
            if src in mae_dict:
                ax.plot(band_hz, mae_dict[src], label=src, color=COLORS[i], lw=1.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("(dB)" if ax is axes[0][0] else "")
        ax.set_xscale("log")
        ax.set_xlim([20, 2e4])
        ax.set_title(titles.get(tag, tag))
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    plt.suptitle(f"Per-Band ILD MAE  ({n_tracks} test tracks)", fontsize=13)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(args.out_dir / "separate.log", name="sahtdemucs.separate")

    if args.ckpt is None and args.from_estimates is None and not args.baseline_only:
        log.error("--ckpt is required unless --from-estimates or --baseline-only is used")
        return 2

    tracks = discover_tracks(args.input)
    if args.limit_tracks:
        tracks = tracks[:args.limit_tracks]
    log.info(f"tracks     : {len(tracks)} from {args.input}")

    device = pick_device(args.device)
    cfg    = resolve_spatial_config(args)

    run_model = args.from_estimates is None
    model = None
    if run_model:
        log.info(f"device     : {device}")
        model, sample_rate = build_model(args, cfg, device, log)
    else:
        sample_rate = int(cfg["sample_rate"])
        log.info(f"scoring pre-computed estimates from {args.from_estimates}")

    want_baseline = (args.baseline or args.baseline_only) and run_model
    results: Dict[str, Dict] = {}
    results_bl: Dict[str, Dict] = {}

    for idx, track in enumerate(tracks, 1):
        t0 = time.perf_counter()
        stems: Dict[int, torch.Tensor] = {}

        if run_model and not args.baseline_only:
            wav = load_audio(track.mix_path, sample_rate).to(device)
            out = model.separate(wav, progress=False, shifts=args.shifts)   # (S, 2, T)
            stems = {i: out[i].cpu() for i in range(out.shape[0])}
            if args.save_stems:
                track_dir = args.out_dir / track.name
                track_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(SOURCES):
                    torchaudio.save(str(track_dir / f"{src}.wav"), stems[i], sample_rate)
        elif not run_model:
            est_dir = args.from_estimates / track.name
            for i, src in enumerate(SOURCES):
                stem_path = est_dir / f"{src}.wav"
                if stem_path.exists():
                    stems[i] = load_audio(stem_path, sample_rate)
            if not stems:
                log.info(f"[{idx}/{len(tracks)}] {track.name} - SKIP (no stems in {est_dir})")
                continue

        if args.evaluate and stems:
            results[track.name] = score_track(stems, track, cfg, sample_rate, args.itd)

        if want_baseline:
            wav_bl = load_audio(track.mix_path, sample_rate).to(device)
            raw    = separate_baseline(model, wav_bl, args.shifts).cpu()    # (S, 2, T)
            raw_stems = {i: raw[i] for i in range(raw.shape[0])}
            if args.save_baseline_stems:
                bl_dir = args.out_dir / "baseline" / track.name
                bl_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(SOURCES):
                    torchaudio.save(str(bl_dir / f"{src}.wav"), raw_stems[i], sample_rate)
            if args.evaluate:
                results_bl[track.name] = score_track(
                    raw_stems, track, cfg, sample_rate, args.itd)

        log.info(f"[{idx}/{len(tracks)}] {track.name} - done ({time.perf_counter() - t0:.1f}s)")

    if not args.evaluate:
        log.info(f"estimates written to {args.out_dir}")
        return 0

    scored = results or results_bl
    if not scored:
        log.info("nothing scored - no ground-truth stems found next to the mixtures")
        return 0

    # ── Metric artefacts ──────────────────────────────────────────────────────
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "spatial_config": {k: cfg[k] for k in SPATIAL_KEYS},
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "n_tracks": len(scored),
        "tracks": results,
        "baseline": results_bl,
    }, indent=2), encoding="utf-8")

    if results:
        write_ranking(results, args.out_dir / "ild_mae_ranking.csv", log)
    report_si_sdr(results or results_bl, results_bl if results else None, log)

    band_hz = band_centre_hz(cfg, sample_rate)
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    if results:
        curves["sahtdemucs"] = mean_bands(results)
    if results_bl:
        curves["htdemucs"] = mean_bands(results_bl)
    write_band_curves(curves, band_hz, args.out_dir / "ild_mae_per_band.zip")
    if args.plot:
        plot_band_curves(curves, band_hz, len(scored),
                         args.out_dir / "ild_mae_per_band.png")

    log.info(f"metrics    : {metrics_path}")
    log.info(f"per-band   : {args.out_dir / 'ild_mae_per_band.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
