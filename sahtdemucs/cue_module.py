"""
cue_module.py — Per-source spatial cue correction modules.

Two architectures are provided, both sharing the same STFT-domain gain
application logic:

SpatialCueModule (``arch="cnn1d"``)
    Temporal CNN: Conv1d layers operate along the time axis only.
    Each frequency band is processed independently in time — the model
    learns *when* to correct but not *which bands to couple together*.

SpatialCueModule2D (``arch="cnn2d"``)
    Spectro-temporal CNN: Conv2d layers jointly consider frequency and
    time.  By treating the ILD map ``(n_bands, T_frames)`` as a 2-D
    image the network can learn cross-band patterns — e.g. "apply a
    larger correction at low frequencies when high frequencies show a
    consistent ILD offset" — which the 1-D architecture cannot express.

Both modules
------------
    1. Compute the STFT of the separated source and measure its ILD in
       each of *n_bands* equal-width frequency sub-bands, producing a
       per-band, per-frame ILD map ``(B, n_bands, T_frames)``.

    2. Feed that map through a small CNN that predicts a correction
       offset Δ_ILD ∈ [−ild_scale, +ild_scale] dB with the same shape.

    3. Apply the correction symmetrically in the STFT domain so that
       total loudness is preserved.  The full pipeline is differentiable.

Selecting an architecture
-------------------------
Pass ``arch="cnn1d"`` or ``arch="cnn2d"`` to
:class:`~model.SAHTDemucs` (or use the factory
:func:`build_spatial_module` directly)::

    model = SAHTDemucs(base, sources=..., spatial_arch="cnn2d")
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import mel_bin_assignment, compute_ild_bands, compute_ild_bands_mel

__all__ = ["SpatialCueModule", "SpatialCueModule2D", "build_spatial_module"]


# ══════════════════════════════════════════════════════════════════════════════
# Shared base class
# ══════════════════════════════════════════════════════════════════════════════

class _BaseSpatialCueModule(nn.Module):
    """Common STFT-domain gain application shared by all architectures.

    Subclasses must set ``self.n_fft``, ``self.hop_length``,
    ``self.n_bands``, and ``self.ild_scale`` in their ``__init__``, and
    implement :meth:`_predict_delta`.
    """

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_subband_gain(
        self,
        signal: torch.Tensor,
        delta_ild: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Apply per-band, per-frame gain to *signal* in the STFT domain.

        The amplitude gain for each band k and frame t is::

            gain[k, t] = 10 ^ (−delta_ild[k, t] / 20)

        Pass ``delta_ild / 2`` for the right channel and
        ``−delta_ild / 2`` for the left channel so that both channels
        are modified symmetrically and total loudness is preserved.

        For ``band_scale="mel"`` the per-band gain is broadcast to STFT bins
        via the same Mel assignment used during analysis (each bin gets the
        gain of its Mel band), ensuring a consistent analysis–synthesis pair.

        Args:
            signal:    waveform ``(B, T)``
            delta_ild: per-band, per-frame gain exponent in dB
                       ``(B, n_bands, T_frames)``
            length:    original signal length T (for ISTFT trimming)

        Returns:
            corrected waveform ``(B, T)``
        """
        window = torch.hann_window(self.n_fft, device=signal.device)

        S = torch.stft(signal, self.n_fft, self.hop_length, window=window,
                       return_complex=True)          # (B, F_bins, T_frames)
        B, F_bins, T_frames = S.shape

        gain_per_band = 10.0 ** (-delta_ild / 20.0)  # (B, n_bands, T_cnn)

        if gain_per_band.shape[-1] != T_frames:
            gain_per_band = F.interpolate(
                gain_per_band, size=T_frames, mode="linear", align_corners=False
            )

        if self.band_scale == "mel":
            # Each STFT bin gets the gain of its Mel band — consistent with
            # the Mel analysis performed in forward().
            band_idx     = mel_bin_assignment(
                self.n_fft, self.n_bands, self.sample_rate
            ).to(signal.device)                          # (F_bins,)
            gain_per_bin = gain_per_band[:, band_idx, :]  # (B, F_bins, T_frames)
        else:
            # Original equal-width linear-band logic (preserved for compatibility)
            bpb    = F_bins // self.n_bands
            F_trim = self.n_bands * bpb
            gain_per_bin = gain_per_band.repeat_interleave(bpb, dim=1)  # (B, F_trim, T_frames)
            if F_trim < F_bins:
                edge = torch.ones(B, F_bins - F_trim, T_frames,
                                  device=signal.device, dtype=gain_per_bin.dtype)
                gain_per_bin = torch.cat([gain_per_bin, edge], dim=1)

        return torch.istft(S * gain_per_bin, self.n_fft, self.hop_length,
                           window=window, length=length)

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    def _predict_delta(self, ild_tf: torch.Tensor) -> torch.Tensor:
        """Map ILD map ``(B, n_bands, T_frames)`` → Δ ∈ [−1, +1] same shape."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        source_est: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a learned per-sub-band, per-frame ILD correction.

        The ILD analysis uses either equal-width linear bands
        (``band_scale="linear"``) or Mel-scale bands
        (``band_scale="mel"``).  The synthesis gain is applied with the same
        band assignment so that analysis and synthesis are consistent.

        Args:
            source_est: raw separated source ``(B, 2, T)``

        Returns:
            corrected: ILD-corrected source tensor ``(B, 2, T)``
            delta:     predicted correction ``(B, n_bands, T_frames)``
                       in [−1, +1] (multiply by ``ild_scale`` for dB)
        """
        B, _, T = source_est.shape
        l, r    = source_est[:, 0], source_est[:, 1]

        if self.band_scale == "mel":
            ild_tf = compute_ild_bands_mel(
                l, r, self.n_fft, self.hop_length, self.n_bands, self.sample_rate
            )
        else:
            ild_tf = compute_ild_bands(
                l, r, self.n_fft, self.hop_length, self.n_bands
            )                                          # (B, n_bands, T_frames)

        delta     = self._predict_delta(ild_tf)        # (B, n_bands, T_frames) ∈ [−1, +1]
        delta_ild = delta * self.ild_scale             # dB

        # Symmetric correction — preserves total loudness
        half = delta_ild / 2.0
        l_corrected = self._apply_subband_gain(l, half.neg(), T)
        r_corrected = self._apply_subband_gain(r, half,       T)
        corrected   = torch.stack([l_corrected, r_corrected], dim=1)

        return corrected, delta


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 1: temporal CNN (Conv1d)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialCueModule(_BaseSpatialCueModule):
    """Per-sub-band, per-frame ILD correction using a **temporal CNN** (Conv1d).

    Architecture::

        ILD map (B, n_bands, T_frames)
            → Conv1d(n_bands → hidden, k=kernel_size) → ReLU
            → Conv1d(hidden  → hidden, k=kernel_size) → ReLU
            → Conv1d(hidden  → n_bands, k=1)          → Tanh   ∈ [−1, +1]
            → × ild_scale                                        dB

    Conv1d layers operate along the time axis; each frequency band is
    processed independently.  The temporal receptive field provides
    natural smoothing of the gain trajectory.

    Args:
        hidden:      hidden channel width (default 64)
        n_fft:       STFT FFT size in samples (default 2048)
        hop_length:  STFT hop size in samples (default 512)
        n_bands:     number of frequency sub-bands (default 32)
        ild_scale:   maximum ILD correction magnitude in dB (default 6.0)
        kernel_size: Conv1d temporal kernel size (default 7, odd recommended)
        band_scale:  frequency band spacing — ``"linear"`` (default, equal-width
                     linear bands) or ``"mel"`` (Mel-scale bands, finer
                     resolution at low frequencies)
        sample_rate: audio sample rate in Hz, used only when
                     ``band_scale="mel"`` (default 44100)
        max_lag:     reserved for future ITD support (default 64)
    """

    def __init__(
        self,
        hidden: int       = 64,
        n_fft: int        = 2048,
        hop_length: int   = 512,
        n_bands: int      = 32,
        ild_scale: float  = 6.0,
        kernel_size: int  = 7,
        band_scale: str   = "linear",
        sample_rate: int  = 44100,
        max_lag: int      = 64,
    ) -> None:
        super().__init__()
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_bands     = n_bands
        self.ild_scale   = ild_scale
        self.band_scale  = band_scale
        self.sample_rate = sample_rate
        self.max_lag     = max_lag

        pad = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_bands, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden,  hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden,  n_bands, kernel_size=1),
            nn.Tanh(),
        )

    def _predict_delta(self, ild_tf: torch.Tensor) -> torch.Tensor:
        delta = self.cnn(ild_tf)
        return delta[..., :ild_tf.shape[-1]]


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2: spectro-temporal CNN (Conv2d)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialCueModule2D(_BaseSpatialCueModule):
    """Per-sub-band, per-frame ILD correction using a **spectro-temporal CNN** (Conv2d).

    Architecture::

        ILD map (B, n_bands, T_frames)
            → unsqueeze(1) → (B, 1, n_bands, T_frames)

        Local branch — 3 Conv2d layers with GroupNorm and internal residual:
            layer1: Conv2d(1→hidden, freq_k×time_k) → GroupNorm → ReLU   → h1
            layer2: Conv2d(hidden→hidden, freq_k×time_k) → GroupNorm → ReLU
            layer3: Conv2d(hidden→hidden, freq_k×time_k) → GroupNorm → ReLU + h1  → h3
            [receptive field: (2·freq_k−1) bands × (2·time_k−1) frames]

        Global context branch — temporal mean per frequency band:
            mean over T → (B, 1, n_bands, 1)
            Conv2d(1→hidden, freq_k×1) → ReLU → broadcast over T

        Fusion + projection:
            (h3 + global) → Conv2d(hidden→1, 1×1) → Tanh ∈ [−1, +1]
            → squeeze(1) → (B, n_bands, T_frames)
            → × ild_scale                             dB

    The global context branch captures the *DC component* of ILD (e.g. a
    source consistently panned right throughout the segment), freeing the
    local branch to focus on fine spectro-temporal variations.  The
    internal residual (layer1 → layer3) improves gradient flow without
    adding parameters.  GroupNorm stabilises training with small batches.
    The output projection is zero-initialised so corrections start near
    zero at the beginning of training.

    Args:
        hidden:         hidden channel width (default 32)
        n_fft:          STFT FFT size in samples (default 2048)
        hop_length:     STFT hop size in samples (default 512)
        n_bands:        number of frequency sub-bands (default 32)
        ild_scale:      maximum ILD correction magnitude in dB (default 6.0)
        freq_kernel:    Conv2d kernel size along the frequency axis (default 3)
        time_kernel:    Conv2d kernel size along the time axis (default 7)
        band_scale:     frequency band spacing — ``"linear"`` (default,
                        equal-width linear bands) or ``"mel"`` (Mel-scale
                        bands, finer resolution at low frequencies)
        sample_rate:    audio sample rate in Hz, used only when
                        ``band_scale="mel"`` (default 44100)
        max_lag:        reserved for future ITD support (default 64)
    """

    def __init__(
        self,
        hidden: int       = 32,
        n_fft: int        = 2048,
        hop_length: int   = 512,
        n_bands: int      = 32,
        ild_scale: float  = 6.0,
        freq_kernel: int  = 3,
        time_kernel: int  = 7,
        band_scale: str   = "linear",
        sample_rate: int  = 44100,
        max_lag: int      = 64,
    ) -> None:
        super().__init__()
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_bands     = n_bands
        self.ild_scale   = ild_scale
        self.band_scale  = band_scale
        self.sample_rate = sample_rate
        self.max_lag     = max_lag

        fpad   = freq_kernel // 2
        tpad   = time_kernel // 2
        k      = (freq_kernel, time_kernel)
        p      = (fpad, tpad)
        groups = min(8, hidden)   # GroupNorm groups — must divide hidden

        # Local branch: 3 layers, internal residual skip from layer1 to layer3
        self.local1 = nn.Sequential(
            nn.Conv2d(1,      hidden, kernel_size=k, padding=p),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(),
        )
        self.local2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=k, padding=p),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(),
        )
        self.local3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=k, padding=p),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(),
        )

        # Global context branch: collapse time, convolve over frequency bands
        self.global_conv = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(freq_kernel, 1), padding=(fpad, 0)),
            nn.ReLU(),
        )

        # Output projection — zero-initialised for near-zero corrections at init
        self.proj = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _predict_delta(self, ild_tf: torch.Tensor) -> torch.Tensor:
        x  = ild_tf.unsqueeze(1)          # (B, 1, n_bands, T)

        # Local branch with residual
        h1 = self.local1(x)               # (B, hidden, n_bands, T)
        h2 = self.local2(h1)              # (B, hidden, n_bands, T)
        h3 = self.local3(h2) + h1        # (B, hidden, n_bands, T)

        # Global context: mean over T → (B, 1, n_bands, 1) → (B, hidden, n_bands, 1)
        g  = x.mean(dim=-1, keepdim=True)
        g  = self.global_conv(g)          # broadcasts over T in the sum below

        out = torch.tanh(self.proj(h3 + g))   # (B, 1, n_bands, T)
        return out.squeeze(1)[..., :ild_tf.shape[-1]]


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

_ARCH_REGISTRY = {
    "cnn1d": SpatialCueModule,
    "cnn2d": SpatialCueModule2D,
}


def build_spatial_module(arch: str = "cnn1d", **kwargs) -> _BaseSpatialCueModule:
    """Instantiate a spatial cue module by architecture name.

    Args:
        arch:    ``"cnn1d"`` (temporal Conv1d, default) or
                 ``"cnn2d"`` (spectro-temporal Conv2d).
        **kwargs: forwarded to the chosen class constructor.

    Returns:
        An instance of the requested :class:`_BaseSpatialCueModule` subclass.

    Raises:
        ValueError: if *arch* is not a recognised key.

    Example::

        module = build_spatial_module("cnn2d", n_bands=32, ild_scale=6.0)
    """
    if arch not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unknown spatial_arch {arch!r}. "
            f"Choose one of: {list(_ARCH_REGISTRY)}"
        )
    return _ARCH_REGISTRY[arch](**kwargs)