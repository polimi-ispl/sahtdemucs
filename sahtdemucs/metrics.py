"""
metrics.py — Evaluation metrics for stereo source separation.

These are inference-time metrics, kept separate from the training objective in
``losses.py``.  They operate on a single stereo pair ``(2, T)`` (estimate vs.
ground truth) and reuse the sub-band cue primitives from :mod:`spatial`, so the
evaluation is consistent with the ILD/ITD terms optimised during training.

    si_sdr           scalar scale-invariant SDR in dB over the whole signal.
    ild_bands_mae    per-sub-band ILD mean-absolute-error (n_bands,).
    itd_bands_mae    per-sub-band ITD mean-absolute-error in samples (n_bands,).

The ``*_bands_mae`` helpers accept the same band configuration as the training
loss (FFT size, hop, number of bands, ``scale`` = "linear" | "mel", …) so the
metric and the objective can be kept in sync from a single config.
"""

from __future__ import annotations

import numpy as np
import torch

from .spatial import (
    compute_ild_bands, compute_ild_bands_mel,
    compute_itd_bands, compute_itd_bands_mel,
)

__all__ = ["si_sdr", "ild_bands_mae", "itd_bands_mae"]


def si_sdr(est: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB between two stereo tensors ``(2, T)``.

    Both channels are flattened together and treated as a single signal, giving
    one scalar per source.  For the per-waveform, batched variant used inside
    the training loss see :func:`sahtdemucs.losses._si_sdr_db`.
    """
    e = est.float().reshape(-1)
    t = tgt.float().reshape(-1)
    e = e - e.mean()
    t = t - t.mean()
    alpha = (e * t).sum() / (t * t).sum().clamp(min=eps)
    proj  = alpha * t
    noise = e - proj
    return 10 * torch.log10(
        (proj ** 2).sum() / (noise ** 2).sum().clamp(min=eps)
    ).item()


def ild_bands_mae(
    est: torch.Tensor,
    tgt: torch.Tensor,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    scale: str = "linear",
    sample_rate: int = 44100,
) -> np.ndarray:
    """Per-sub-band ILD MAE ``(n_bands,)`` between estimate and target ``(2, T)``.

    Uses the same band ``scale`` ("linear" or "mel") as the training loss so the
    metric is consistent with the objective.
    """
    if scale == "mel":
        def _fn(l, r):
            return compute_ild_bands_mel(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands), sample_rate=int(sample_rate),
            )
    else:
        def _fn(l, r):
            return compute_ild_bands(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands),
            )
    ild_est = _fn(est[0].unsqueeze(0).cpu(), est[1].unsqueeze(0).cpu())
    ild_tgt = _fn(tgt[0].unsqueeze(0).cpu(), tgt[1].unsqueeze(0).cpu())
    return torch.abs(ild_est - ild_tgt).mean(dim=-1).squeeze(0).numpy()   # (n_bands,)


def itd_bands_mae(
    est: torch.Tensor,
    tgt: torch.Tensor,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    scale: str = "linear",
    sample_rate: int = 44100,
    max_lag: int = 64,
    beta: float = 20.0,
) -> np.ndarray:
    """Per-sub-band ITD MAE ``(n_bands,)`` in samples, between estimate and target.

    Same band layout and GCC-PHAT settings (``max_lag``, ``beta``) as the
    training loss.
    """
    if scale == "mel":
        def _fn(l, r):
            return compute_itd_bands_mel(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands), sample_rate=int(sample_rate),
                max_lag=int(max_lag), beta=float(beta),
            )
    else:
        def _fn(l, r):
            return compute_itd_bands(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands),
                max_lag=int(max_lag), beta=float(beta),
            )
    itd_est = _fn(est[0].unsqueeze(0).cpu(), est[1].unsqueeze(0).cpu())
    itd_tgt = _fn(tgt[0].unsqueeze(0).cpu(), tgt[1].unsqueeze(0).cpu())
    return torch.abs(itd_est - itd_tgt).mean(dim=-1).squeeze(0).numpy()   # (n_bands,)
