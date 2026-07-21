"""
losses.py — Loss functions for spatial-aware source separation.

Classes
-------
SpatialLoss
    Combines up to three terms, each averaged over the S sources:

        Loss = (1/S) · Σ_s [ λ_si  · SIDegradationLoss(est_s, raw_s, tgt_s)
                             + λ_ild · MSE(ILD_bands_est_s, ILD_bands_gt_s)
                             + λ_itd · MSE(ITD_bands_est_s, ITD_bands_gt_s) ]

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

Sub-band ITD MSE (samples²)
    Penalises errors in the per-sub-band Interaural Time Difference, estimated
    with a band-limited, differentiable GCC-PHAT (soft-argmax) that shares the
    band layout with the ILD term, so both metrics have shape
    ``(B, n_bands, T_frames)``.  MSE is taken over all (batch, band, frame)
    entries.  Because the ITD is expressed in **samples** (range ±itd_max_lag),
    this term is on a much larger numerical scale than the ILD (dB²): with
    itd_max_lag=64 the per-entry error can reach ~10³ samples², so ``lambda_itd``
    typically needs to be one to two orders of magnitude smaller than
    ``lambda_ild`` to balance the gradient magnitudes.

    Disabled by default (``lambda_itd=0.0``).  Note that a magnitude-only spatial
    correction (e.g. the ILD-gain SpatialCueModule of SA-HTDemucs) cannot affect
    the ITD, because PHAT whitening removes all magnitude information — so this
    term only produces useful gradients for models that can alter interaural
    phase/time (e.g. the full HTDemucs backbone fine-tune).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import (
    compute_ild_bands,
    compute_ild_bands_mel,
    compute_itd_bands,
    compute_itd_bands_mel,
)

__all__ = ["SpatialLoss", "HTDemucsSpatialLoss"]


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
        lambda_itd:    weight for the sub-band ITD MSE term (default 0.0, i.e.
                       disabled).  The ITD is in samples², a much larger scale
                       than the ILD (dB²), so this is usually ≪ ``lambda_ild``.
        si_margin_db:  tolerated SI-SDR degradation in dB relative to the raw
                       HTDemucs baseline; degradation below this margin is not
                       penalised (default 0.5 dB)
        n_fft:         STFT FFT size for sub-band ILD/ITD computation (default 2048)
        hop_length:    STFT hop size (default 512)
        n_bands:       number of frequency sub-bands (default 32)
        band_scale:    frequency band spacing — ``"linear"`` (default) or
                       ``"mel"``.  Must match the SpatialCueModule config.
        sample_rate:   audio sample rate in Hz, used only when
                       ``band_scale="mel"`` (default 44100)
        itd_max_lag:   maximum lag searched by the GCC-PHAT, in samples
                       (default 64 → ±1.45 ms @ 44100 Hz)
        itd_beta:      soft-argmax temperature for the ITD estimator (default 20.0)
    """

    def __init__(
        self,
        lambda_si: float   = 1.0,
        lambda_ild: float  = 1.0,
        lambda_itd: float  = 0.0,
        si_margin_db: float = 0.5,
        n_fft: int         = 2048,
        hop_length: int    = 512,
        n_bands: int       = 32,
        band_scale: str    = "linear",
        sample_rate: int   = 44100,
        itd_max_lag: int   = 64,
        itd_beta: float    = 20.0,
    ) -> None:
        super().__init__()
        self.lambda_si    = lambda_si
        self.lambda_ild   = lambda_ild
        self.lambda_itd   = lambda_itd
        self.si_margin_db = si_margin_db
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_bands      = n_bands
        self.band_scale   = band_scale
        self.sample_rate  = sample_rate
        self.itd_max_lag  = itd_max_lag
        self.itd_beta     = itd_beta

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
        raw_estimates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            itd_part: weighted ITD MSE contribution (for logging)
        """
        B, S, C, T = estimates.shape

        loss_si  = torch.tensor(0.0, device=estimates.device)
        loss_ild = torch.tensor(0.0, device=estimates.device)
        loss_itd = torch.tensor(0.0, device=estimates.device)

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

            # ── Sub-band ITD MSE (samples²) ───────────────────────────────────
            if self.lambda_itd > 0:
                if self.band_scale == "mel":
                    def _itd(l, r):
                        return compute_itd_bands_mel(
                            l, r, self.n_fft, self.hop_length,
                            self.n_bands, self.sample_rate,
                            self.itd_max_lag, self.itd_beta,
                        )
                else:
                    def _itd(l, r):
                        return compute_itd_bands(
                            l, r, self.n_fft, self.hop_length, self.n_bands,
                            self.itd_max_lag, self.itd_beta,
                        )
                itd_est  = _itd(est_s[:, 0], est_s[:, 1])   # (B, n_bands, T_frames)
                itd_gt   = _itd(tgt_s[:, 0], tgt_s[:, 1])
                loss_itd = loss_itd + F.mse_loss(itd_est, itd_gt)

        total = (
            self.lambda_si  * loss_si
            + self.lambda_ild * loss_ild
            + self.lambda_itd * loss_itd
        ) / S
        si_part  = self.lambda_si  * loss_si  / S
        ild_part = self.lambda_ild * loss_ild / S
        itd_part = self.lambda_itd * loss_itd / S
        return total, si_part, ild_part, itd_part

class HTDemucsSpatialLoss(nn.Module):
    """Original HTDemucs time-domain L1 + sub-band ILD & ITD MSE spatial terms.

        L = lambda_td  * time_L1
          + lambda_ild * subband_ILD_MSE
          + lambda_itd * subband_ITD_MSE

    The first term is exactly the HTDemucs training objective (Rouard et al.,
    2022): an L1 loss on the waveforms, with no frequency-domain component.
    The ILD (level) and ITD (time/phase) terms share the same band layout.

    Args:
        lambda_td:    weight for the time-domain L1 term (separation)
        lambda_ild:   weight for the sub-band ILD MSE term (level cue)
        lambda_itd:   weight for the sub-band ITD MSE term (time/phase cue).
                      ITD is in samples², a larger scale than ILD (dB²), so
                      this is usually << lambda_ild. 0.0 disables it.
        n_fft:        FFT size for the sub-band ILD/ITD computation
        hop_length:   STFT hop for the sub-band ILD/ITD computation
        n_bands:      number of frequency sub-bands
        band_scale:   "linear" or "mel"
        sample_rate:  audio sample rate (used only when band_scale="mel")
        itd_max_lag:  max GCC-PHAT lag in samples (±)
        itd_beta:     soft-argmax temperature for the ITD estimator
    """

    def __init__(
        self,
        lambda_td: float    = 1.0,
        lambda_ild: float   = 1.0,
        lambda_itd: float   = 0.0,
        n_fft: int          = 4096,
        hop_length: int     = 512,
        n_bands: int        = 64,
        band_scale: str     = "mel",
        sample_rate: int    = 44100,
        itd_max_lag: int    = 64,
        itd_beta: float     = 20.0,
    ) -> None:
        super().__init__()
        self.lambda_td    = lambda_td
        self.lambda_ild   = lambda_ild
        self.lambda_itd   = lambda_itd
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_bands      = n_bands
        self.band_scale   = band_scale
        self.sample_rate  = sample_rate
        self.itd_max_lag  = itd_max_lag
        self.itd_beta     = itd_beta

    def _ild(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Per-sub-band ILD (B, n_bands, T_frames)."""
        if self.band_scale == "mel":
            return compute_ild_bands_mel(
                left, right, self.n_fft, self.hop_length,
                self.n_bands, self.sample_rate,
            )
        return compute_ild_bands(
            left, right, self.n_fft, self.hop_length, self.n_bands,
        )

    def _itd(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Per-sub-band ITD via band-limited GCC-PHAT (B, n_bands, T_frames)."""
        if self.band_scale == "mel":
            return compute_itd_bands_mel(
                left, right, self.n_fft, self.hop_length,
                self.n_bands, self.sample_rate, self.itd_max_lag, self.itd_beta,
            )
        return compute_itd_bands(
            left, right, self.n_fft, self.hop_length, self.n_bands,
            self.itd_max_lag, self.itd_beta,
        )

    def forward(
        self,
        estimates: torch.Tensor,   # (B, S, 2, T)
        targets: torch.Tensor,     # (B, S, 2, T)
    ):
        # Compute everything in FP32 — log/division/soft-argmax fragile in FP16.
        estimates = estimates.float()
        targets   = targets.float()

        B, S, C, T = estimates.shape
        loss_td  = torch.zeros((), device=estimates.device)
        loss_ild = torch.zeros((), device=estimates.device)
        loss_itd = torch.zeros((), device=estimates.device)

        for s in range(S):
            est_s = estimates[:, s]   # (B, 2, T)
            tgt_s = targets[:, s]     # (B, 2, T)

            # ── Time-domain L1 (original HTDemucs separation loss) ───────────
            if self.lambda_td > 0:
                loss_td = loss_td + F.l1_loss(est_s, tgt_s)

            # ── Sub-band ILD MSE (level cue) ─────────────────────────────────
            if self.lambda_ild > 0:
                ild_est = self._ild(est_s[:, 0], est_s[:, 1])   # (B, n_bands, Tf)
                ild_gt  = self._ild(tgt_s[:, 0], tgt_s[:, 1])
                loss_ild = loss_ild + F.mse_loss(ild_est, ild_gt)

            # ── Sub-band ITD MSE (time/phase cue) ────────────────────────────
            if self.lambda_itd > 0:
                itd_est = self._itd(est_s[:, 0], est_s[:, 1])   # (B, n_bands, Tf)
                itd_gt  = self._itd(tgt_s[:, 0], tgt_s[:, 1])
                loss_itd = loss_itd + F.mse_loss(itd_est, itd_gt)

        td_part  = self.lambda_td  * loss_td  / S
        ild_part = self.lambda_ild * loss_ild / S
        itd_part = self.lambda_itd * loss_itd / S
        total    = td_part + ild_part + itd_part
        return total, td_part, ild_part, itd_part