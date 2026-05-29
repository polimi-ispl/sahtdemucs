"""
losses.py — Loss functions for spatial-aware source separation.

Classes
-------
SpatialLoss
    Combines two terms, each averaged over the S sources:

        Loss = (1/S) · Σ_s [ λ_si  · SIDegradationLoss(est_s, raw_s, tgt_s)
                             + λ_ild · MSE(ILD_bands_est_s, ILD_bands_gt_s) ]

Term details
------------
SI-SDR degradation penalty (dB, always ≥ 0)
    Penalises the spatial correction only when it reduces SI-SDR below the
    frozen HTDemucs baseline by more than ``si_margin_db``.

    Given the raw HTDemucs output ``raw_s`` and the spatially corrected
    output ``est_s``, define:

        L_si = mean( ReLU( SI-SDR(raw_s, tgt_s) − SI-SDR(est_s, tgt_s) − margin ) )

    Properties
    ~~~~~~~~~~
    * Always non-negative (ReLU).
    * Zero (no gradient) when the correction does not hurt SI-SDR beyond
      the margin — so the spatial head trains freely when it improves ILD
      without degrading separation.
    * In dB units: typical degradations are 0–3 dB, same order of magnitude
      as ILD corrections, so LAMBDA_SI and LAMBDA_ILD are directly comparable.
    * Directly interpretable: ``si_margin_db=0.5`` means "allow at most 0.5 dB
      SI-SDR degradation due to spatial correction".

    Why not plain -SI-SNR?
    ~~~~~~~~~~~~~~~~~~~~~~
    The original formulation returned -SI-SNR directly (≈ -8 dB for good
    separation). This is negative throughout training and has units of dB,
    while ILD MSE is in dB² at scale 10–100.  The resulting scale mismatch
    means LAMBDA_SI must be ~50x larger than LAMBDA_ILD just to balance
    gradient magnitudes — and even then the sign is unconventional.

Sub-band ILD MSE (dB²)
    Penalises errors in the per-sub-band Interaural Level Difference.
    The STFT spectrum is divided into *n_bands* frequency bands (linear or
    Mel scale); the ILD of each band is computed from the mean power in that
    band.  MSE is taken over all (batch, band, frame) entries.
    Typical values at the start of training: 10–100 dB².
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import compute_ild_bands, compute_ild_bands_mel

__all__ = ["SpatialLoss"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _si_sdr_db(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-waveform SI-SDR in dB.

    Args:
        estimate: ``(B, 2, T)``
        target:   ``(B, 2, T)``

    Returns:
        ``(B * 2,)`` tensor of SI-SDR values in dB.
    """
    B, C, T = estimate.shape
    est = estimate.reshape(B * C, T)
    tgt = target.reshape(B * C, T)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    dot    = (est * tgt).sum(dim=-1)
    tgt_sq = (tgt ** 2).sum(dim=-1) + eps
    proj   = (dot / tgt_sq).unsqueeze(-1) * tgt
    noise  = est - proj

    return 10.0 * torch.log10(
        (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
    )


# ──────────────────────────────────────────────────────────────────────────────
# Combined spatial loss
# ──────────────────────────────────────────────────────────────────────────────

class SpatialLoss(nn.Module):
    """Combine SI-SDR degradation penalty with sub-band ILD supervision.

    Args:
        lambda_si:     weight for the SI-SDR degradation penalty (default 1.0)
        lambda_ild:    weight for the sub-band ILD MSE term (default 1.0)
        si_margin_db:  tolerated SI-SDR degradation in dB relative to the raw
                       HTDemucs baseline; degradation below this margin is not
                       penalised (default 0.5 dB)
        n_fft:         STFT FFT size for sub-band ILD computation (default 2048)
        hop_length:    STFT hop size (default 512)
        n_bands:       number of frequency sub-bands (default 32)
        band_scale:    frequency band spacing — ``"linear"`` (default) or
                       ``"mel"``.  Must match the SpatialCueModule config.
        sample_rate:   audio sample rate in Hz, used only when
                       ``band_scale="mel"`` (default 44100)
    """

    def __init__(
        self,
        lambda_si: float   = 1.0,
        lambda_ild: float  = 1.0,
        si_margin_db: float = 0.5,
        n_fft: int         = 2048,
        hop_length: int    = 512,
        n_bands: int       = 32,
        band_scale: str    = "linear",
        sample_rate: int   = 44100,
    ) -> None:
        super().__init__()
        self.lambda_si    = lambda_si
        self.lambda_ild   = lambda_ild
        self.si_margin_db = si_margin_db
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_bands      = n_bands
        self.band_scale   = band_scale
        self.sample_rate  = sample_rate

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
        raw_estimates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss over all sources.

        Args:
            estimates:     ``(B, S, 2, T)`` spatially corrected model outputs
            targets:       ``(B, S, 2, T)`` ground-truth sources
            raw_estimates: ``(B, S, 2, T)`` raw HTDemucs outputs before spatial
                           correction.  Required when ``lambda_si > 0``; if
                           ``None``, the SI-SDR degradation term is skipped.

        Returns:
            total:    combined weighted loss (call ``.backward()`` on this)
            si_part:  weighted SI-SDR degradation contribution (for logging)
            ild_part: weighted ILD MSE contribution (for logging)
        """
        B, S, C, T = estimates.shape

        loss_si  = torch.tensor(0.0, device=estimates.device)
        loss_ild = torch.tensor(0.0, device=estimates.device)

        for s in range(S):
            est_s = estimates[:, s]     # (B, 2, T)
            tgt_s = targets[:, s]       # (B, 2, T)

            # ── SI-SDR degradation penalty (dB, ≥ 0) ─────────────────────────
            if self.lambda_si > 0 and raw_estimates is not None:
                raw_s   = raw_estimates[:, s]                    # (B, 2, T)
                si_raw  = _si_sdr_db(raw_s,  tgt_s)             # (B*2,) dB
                si_corr = _si_sdr_db(est_s,  tgt_s)             # (B*2,) dB
                loss_si = loss_si + F.relu(
                    si_raw - si_corr - self.si_margin_db
                ).mean()

            # ── Sub-band ILD MSE (dB²) ────────────────────────────────────────
            if self.lambda_ild > 0:
                if self.band_scale == "mel":
                    def _ild(l, r):
                        return compute_ild_bands_mel(
                            l, r, self.n_fft, self.hop_length,
                            self.n_bands, self.sample_rate,
                        )
                else:
                    def _ild(l, r):
                        return compute_ild_bands(
                            l, r, self.n_fft, self.hop_length, self.n_bands,
                        )
                ild_est  = _ild(est_s[:, 0], est_s[:, 1])   # (B, n_bands, T_frames)
                ild_gt   = _ild(tgt_s[:, 0], tgt_s[:, 1])
                loss_ild = loss_ild + F.mse_loss(ild_est, ild_gt)

        total    = (self.lambda_si * loss_si + self.lambda_ild * loss_ild) / S
        si_part  = self.lambda_si  * loss_si  / S
        ild_part = self.lambda_ild * loss_ild / S
        return total, si_part, ild_part