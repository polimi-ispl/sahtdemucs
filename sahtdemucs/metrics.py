"""
metrics.py - Evaluation metrics for stereo source separation.

These are inference-time metrics, kept separate from the training objective in
``losses.py``. They operate on a single stereo pair ``(2, T)`` (estimate vs.
ground truth) and reuse the sub-band cue primitives from :mod:`spatial`, so the
evaluation is consistent with the ILD/ITD terms optimized during training.

    si_sdr           scalar scale-invariant SDR in dB over the whole signal.
    ild_bands_mae    per-sub-band ILD mean-absolute-error (n_bands,).
    itd_bands_mae    per-sub-band ITD mean-absolute-error in samples (n_bands,).
    ipd_bands_mae    per-sub-band IPD mean-absolute-error in radians (n_bands,).
    ic_bands_mae     per-sub-band interaural-coherence MAE, unitless (n_bands,).

The ``*_bands_mae`` helpers accept the same band configuration as the training
loss (FFT size, hop, number of bands, ``scale`` = "linear" | "mel", …) so the
metric and the objective can be kept in sync from a single config.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .spatial import (
    compute_ild_bands, compute_ild_bands_mel,
    compute_itd_bands, compute_itd_bands_mel,
    audible_band_mask, mel_bin_assignment,
    interaural_coherence_bands
)

__all__ = ["si_sdr", "ild_bands_mae", "itd_bands_mae", "ipd_bands_mae", "ic_bands_mae"]

# ------------------------------------------------------------------------------
# Scale Invariant Signal To Distortion Ratio
# ------------------------------------------------------------------------------

def si_sdr(
        est: torch.Tensor,
        tgt: torch.Tensor,
        eps: float = 1e-8
) -> float:
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

# ------------------------------------------------------------------------------
# Interaural Level Difference MAE
# ------------------------------------------------------------------------------

def ild_bands_mae(
    est: torch.Tensor,
    tgt: torch.Tensor,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    scale: str = "linear",
    sample_rate: int = 44100,
    floor_db: Optional[float] = None,
) -> np.ndarray:
    """Per-sub-band ILD MAE ``(n_bands,)`` between estimate and target ``(2, T)``.

    Uses the same band ``scale`` ("linear" or "mel") as the training loss so the
    metric is consistent with the objective.

    With ``floor_db`` (e.g. -40.0) the error is averaged only over the audible
    frames of each band of the *target* (see :func:`spatial.audible_band_mask`),
    so silent passages - whose ILD is noise - do not dominate the metric.  The
    mask depends on the target only, so every model is scored on the same cells;
    a band with no audible frame reports 0.  ``None`` averages over all frames.
    """
    if scale == "mel":
        def _fn(l, r, **kw):
            return compute_ild_bands_mel(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands), sample_rate=int(sample_rate), **kw,
            )
    else:
        def _fn(l, r, **kw):
            return compute_ild_bands(
                l, r, n_fft=int(n_fft), hop_length=int(hop_length),
                n_bands=int(n_bands), **kw,
            )
    ild_est = _fn(est[0].unsqueeze(0).cpu(), est[1].unsqueeze(0).cpu())
    if floor_db is None:
        ild_tgt = _fn(tgt[0].unsqueeze(0).cpu(), tgt[1].unsqueeze(0).cpu())
        return torch.abs(ild_est - ild_tgt).mean(dim=-1).squeeze(0).numpy()   # (n_bands,)

    ild_tgt, p_l, p_r = _fn(tgt[0].unsqueeze(0).cpu(), tgt[1].unsqueeze(0).cpu(),
                            return_power=True)
    mask = audible_band_mask(p_l + p_r, floor_db).float()
    err  = torch.abs(ild_est - ild_tgt) * mask
    return (err.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).squeeze(0).numpy()   # (n_bands,)

# ------------------------------------------------------------------------------
# Interaural Time Difference MAE
# ------------------------------------------------------------------------------

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
    floor_db: Optional[float] = None,
) -> np.ndarray:
    """Per-sub-band ITD MAE ``(n_bands,)`` in samples, between estimate and target.

    Same band layout and GCC-PHAT settings (``max_lag``, ``beta``) as the
    training loss.  Runs on the device of ``est`` / ``tgt`` (GPU recommended:
    the band-limited GCC-PHAT is the costliest metric).  ``floor_db`` masks the
    silent cells of the target exactly as in :func:`ild_bands_mae`.
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
    est, tgt = est.float(), tgt.float()
    itd_est = _fn(est[0:1], est[1:2]).squeeze(0)                     # (n_bands, T_frames)
    itd_tgt = _fn(tgt[0:1], tgt[1:2]).squeeze(0)
    err = torch.abs(itd_est - itd_tgt)
    if floor_db is None:
        return err.mean(dim=-1).cpu().numpy()                          # (n_bands,)

    band_idx = _band_assignment(n_fft, n_bands, scale, sample_rate, tgt.device)
    S_l, S_r = _stft_pair(tgt, n_fft, hop_length)
    mask = _target_mask(S_l, S_r, band_idx, n_bands, floor_db)
    return _masked_time_mean(err, mask)

# ------------------------------------------------------------------------------
# Interaural Phase Difference MAE
# ------------------------------------------------------------------------------

def ipd_bands_mae(
    est: torch.Tensor,
    tgt: torch.Tensor,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    scale: str = "linear",
    sample_rate: int = 44100,
    floor_db: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-sub-band IPD MAE ``(n_bands,)`` in radians, between estimate and target.

    The interaural phase difference of every STFT cell is the phase of the
    cross-spectrum ``X_L · conj(X_R)``; its error is the wrapped phase distance
    ``|angle(C_est · conj(C_tgt))|`` in ``[0, π]``, so no peak picking is
    involved (unlike the GCC-PHAT ITD).  Within a band the per-bin errors are
    weighted by the target cross-spectrum magnitude ``|S_L||S_R|``, so bins
    where the target has no energy do not count; the band error is then averaged
    over frames (only the audible ones of the target with ``floor_db``).

    Above ~1.5 kHz the IPD wraps several times across the ITD range, so the
    phase error there no longer maps to a perceptual time cue.
    """
    band_idx = _band_assignment(n_fft, n_bands, scale, sample_rate, tgt.device)
    E_l, E_r = _stft_pair(est, n_fft, hop_length)
    S_l, S_r = _stft_pair(tgt, n_fft, hop_length)

    c_est = E_l * E_r.conj()
    c_tgt = S_l * S_r.conj()
    d = torch.angle(c_est * c_tgt.conj()).abs()                        # (F, T) in [0, π]
    w = c_tgt.abs()

    num = _band_sum(w * d, band_idx, n_bands)                          # (n_bands, T)
    den = _band_sum(w, band_idx, n_bands)
    err = num / den.clamp(min=eps)

    valid = den > eps
    if floor_db is not None:
        valid = valid & _target_mask(S_l, S_r, band_idx, n_bands, floor_db)
    return _masked_time_mean(err, valid)

# ------------------------------------------------------------------------------
# Interaural coherence MAE
# ------------------------------------------------------------------------------

def ic_bands_mae(
    est: torch.Tensor,
    tgt: torch.Tensor,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    scale: str = "linear",
    sample_rate: int = 44100,
    n_avg: int = 9,
    floor_db: Optional[float] = None,
) -> np.ndarray:
    """Per-sub-band interaural-coherence MAE ``(n_bands,)`` (ΔIC, unitless).

    ``|IC_est(k, t) - IC_tgt(k, t)|`` averaged over frames (only the audible
    ones of the target with ``floor_db``); see
    :func:`interaural_coherence_bands` for the IC definition.
    """
    band_idx = _band_assignment(n_fft, n_bands, scale, sample_rate, tgt.device)
    E_l, E_r = _stft_pair(est, n_fft, hop_length)
    S_l, S_r = _stft_pair(tgt, n_fft, hop_length)

    ic_est = interaural_coherence_bands(E_l, E_r, band_idx, n_bands, n_avg)
    ic_tgt = interaural_coherence_bands(S_l, S_r, band_idx, n_bands, n_avg)
    err = torch.abs(ic_est - ic_tgt)
    if floor_db is None:
        return err.mean(dim=-1).cpu().numpy()
    return _masked_time_mean(err, _target_mask(S_l, S_r, band_idx, n_bands, floor_db))

# ------------------------------------------------------------------------------
# Shared STFT / band helpers
# ------------------------------------------------------------------------------

def _band_assignment(n_fft, n_bands, scale, sample_rate, device) -> torch.Tensor:
    """``(n_fft // 2 + 1,)`` bin → band map; same partitions as :mod:`spatial`.

    Linear bands drop the top remainder bins (``-1``), like
    :func:`spatial.compute_ild_bands`.
    """
    n_fft, n_bands = int(n_fft), int(n_bands)
    if scale == "mel":
        return mel_bin_assignment(n_fft, n_bands, int(sample_rate)).to(device)
    F_bins = n_fft // 2 + 1
    bpb    = F_bins // n_bands
    idx = torch.full((F_bins,), -1, dtype=torch.long)
    idx[:bpb * n_bands] = torch.arange(bpb * n_bands) // bpb
    return idx.to(device)

def _stft_pair(x: torch.Tensor, n_fft: int, hop_length: int):
    """Complex STFTs ``(F_bins, T_frames)`` of the two channels of ``x`` ``(2, T)``."""
    x = x.float()
    window = torch.hann_window(int(n_fft), device=x.device)
    X = torch.stft(x, int(n_fft), int(hop_length), window=window, return_complex=True)
    return X[0], X[1]

def _masked_time_mean(err: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean of ``err`` ``(n_bands, T)`` over the masked frames; 0 for empty bands."""
    mask = mask.float()
    return ((err * mask).sum(-1) / mask.sum(-1).clamp(min=1)).cpu().numpy()

def _band_sum(x: torch.Tensor, band_idx: torch.Tensor, n_bands: int) -> torch.Tensor:
    """Sum the bins of ``x`` ``(F_bins, T)`` into bands → ``(n_bands, T)``."""
    keep = band_idx >= 0
    out = x.new_zeros((n_bands, x.shape[-1]))
    return out.index_add_(0, band_idx[keep], x[keep])

def _target_mask(S_l, S_r, band_idx, n_bands, floor_db) -> torch.Tensor:
    """Audible (band, frame) cells of the target, on mean band power."""
    power  = _band_sum(S_l.abs().pow(2) + S_r.abs().pow(2), band_idx, n_bands)
    counts = torch.bincount(band_idx[band_idx >= 0], minlength=n_bands).clamp(min=1)
    return audible_band_mask(power / counts.unsqueeze(-1), floor_db)