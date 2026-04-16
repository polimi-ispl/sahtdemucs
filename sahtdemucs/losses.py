"""
losses.py — Loss functions for spatial-aware source separation.

Classes
-------
SISNRLoss
    Scale-Invariant Signal-to-Noise Ratio loss (negated dB).

SpatialLoss
    Combines two terms, each averaged over the S sources:

        Loss = (1/S) · Σ_s [ λ_si  · SI-SNR_loss(est_s, tgt_s)
                             + λ_ild · MSE(ILD_bands_est_s, ILD_bands_gt_s) ]

Term details
------------
SI-SNR loss (negated dB)
    Measures waveform reconstruction quality independently of absolute scale.
    A perfect estimate gives SI-SNR → ∞, so the negated loss → −∞.
    A random estimate gives roughly −0 dB, i.e. loss ≈ +30–40.

Sub-band ILD MSE (dB²)
    Penalises errors in the per-sub-band Interaural Level Difference.  The
    STFT spectrum is divided into *n_bands* equal-width frequency bands; the
    ILD of each band is computed from the mean power in that band.  Using
    frequency-resolved ILD (vs. a single broadband scalar) captures spectral
    spatial imbalances that a global gain cannot correct — e.g. a source panned
    right at high frequencies but centred at low ones.

    MSE is taken over all (batch, band) pairs so the loss magnitude is
    comparable to the broadband version.  Typical values at the start of
    training are in the range 10–100 dB².

ITD MSE (disabled — see commented code)
    Penalises errors in the Interaural Time Difference.  Disabled because a
    single scalar ITD over a full segment is not meaningful for polyphonic
    music.  Re-enable by restoring the commented-out blocks below.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import compute_ild_bands, compute_ild_bands_mel

__all__ = ["SISNRLoss", "SpatialLoss"]


# ──────────────────────────────────────────────────────────────────────────────
# SI-SNR loss
# ──────────────────────────────────────────────────────────────────────────────

class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss (negated, so lower = better).

    Operates on ``(B, 2, T)`` tensors and averages over batch and channels.

    The SI-SNR is defined as::

        SI-SNR = 10 · log10( ‖proj‖² / ‖noise‖² )

    where ``proj`` is the projection of the zero-mean estimate onto the
    zero-mean target, and ``noise = estimate − proj``.  Negating makes it a
    minimisation objective.
    """

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B, C, T = estimate.shape
        # Flatten batch and channel dims so every waveform is treated independently
        est = estimate.reshape(B * C, T)
        tgt = target.reshape(B * C, T)

        # Zero-mean both signals (SI-SNR is scale-invariant by construction)
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)
        est = est - est.mean(dim=-1, keepdim=True)

        # Project estimate onto target direction
        dot    = (est * tgt).sum(dim=-1, keepdim=True)
        tgt_sq = (tgt ** 2).sum(dim=-1, keepdim=True) + eps
        proj   = (dot / tgt_sq) * tgt       # signal component  (B*C, T)
        noise  = est - proj                 # distortion component (B*C, T)

        # SI-SNR in dB, then negate so minimising = maximising separation quality
        si_snr = 10.0 * torch.log10(
            (proj  ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
        )
        return -si_snr.mean()

# ──────────────────────────────────────────────────────────────────────────────
# Combined spatial loss
# ──────────────────────────────────────────────────────────────────────────────

class SpatialLoss(nn.Module):
    """Combine SI-SNR reconstruction loss with sub-band ILD supervision.

    Args:
        lambda_si:  weight for the SI-SNR penalty term  (default 1.0)
        lambda_ild: weight for the sub-band ILD MSE term (default 1.0)
        n_fft:      STFT FFT size for sub-band ILD computation (default 2048)
        hop_length: STFT hop size (default 512)
        n_bands:    number of frequency sub-bands (default 32)
        band_scale: frequency band spacing used for ILD — ``"linear"``
                    (default, equal-width linear bands) or ``"mel"``
                    (Mel-scale bands).  Must match the setting used in the
                    :class:`~htdemucswspatial.cue_module.SpatialCueModule`
                    so that the loss supervises the same frequency resolution
                    as the model output.
        sample_rate: audio sample rate in Hz, used only when
                    ``band_scale="mel"`` (default 44100)
    """

    def __init__(
        self,
        lambda_si: float  = 1.0,
        lambda_ild: float = 1.0,
        n_fft: int        = 2048,
        hop_length: int   = 512,
        n_bands: int      = 32,
        band_scale: str   = "linear",
        sample_rate: int  = 44100,
    ) -> None:
        super().__init__()
        self.lambda_si   = lambda_si
        self.lambda_ild  = lambda_ild
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_bands     = n_bands
        self.band_scale  = band_scale
        self.sample_rate = sample_rate
        self._si_snr     = SISNRLoss()

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss over all sources.

        Args:
            estimates: ``(B, S, 2, T)`` model outputs
            targets:   ``(B, S, 2, T)`` ground-truth sources

        Returns:
            Scalar loss tensor (mean over S sources).
        """
        B, S, C, T = estimates.shape

        loss_si  = torch.tensor(0.0, device=estimates.device)
        loss_ild = torch.tensor(0.0, device=estimates.device)
        # loss_itd = torch.tensor(0.0, device=estimates.device)  # ITD disabled

        for s in range(S):
            est_s = estimates[:, s]   # (B, 2, T)
            tgt_s = targets[:, s]     # (B, 2, T)

            # ── Waveform reconstruction ───────────────────────────────────────
            if self.lambda_si > 0:
                loss_si = loss_si + self._si_snr(est_s, tgt_s)

            # ── Sub-band ILD ──────────────────────────────────────────────────
            if self.lambda_ild > 0:
                _ild_fn = (
                    lambda l, r: compute_ild_bands_mel(
                        l, r, self.n_fft, self.hop_length,
                        self.n_bands, self.sample_rate,
                    )
                    if self.band_scale == "mel"
                    else lambda l, r: compute_ild_bands(
                        l, r, self.n_fft, self.hop_length, self.n_bands,
                    )
                )
                ild_est  = _ild_fn(est_s[:, 0], est_s[:, 1])  # (B, n_bands, T_frames)
                ild_gt   = _ild_fn(tgt_s[:, 0], tgt_s[:, 1])
                loss_ild = loss_ild + F.mse_loss(ild_est, ild_gt)

        # Average over sources so the total loss is independent of S
        total = (
            self.lambda_si  * loss_si
            + self.lambda_ild * loss_ild
        ) / S

        return total