"""
losses.py - Loss functions for spatial-aware source separation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from sahtdemucs.spatial import (
    compute_ild_bands,
    compute_ild_bands_mel,
    compute_itd_bands,
    compute_itd_bands_mel,
)

__all__ = ["HTDemucsSpatialLoss"]

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