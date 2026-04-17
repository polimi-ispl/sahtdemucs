"""
model.py — Plug-and-play wrapper that adds spatial cue correction to a
pre-trained HT-Demucs model from the official ``demucs`` package.

Motivation
----------
HTDemucs is a strong music source separator but does not explicitly model
spatial cues: its L and R channel estimates may not preserve the ILD and ITD
of the original mix.  SA-HTDemucs freezes the HT-Demucs weights and
attaches a lightweight :class:`~sahtdemucs.cue_module.SpatialCueModule` after
each source output, so only the small spatial heads need to be trained.

Each SpatialCueModule learns a per-source correction function f(ild_src) → Δ_ild
trained directly with ILD MSE loss.

Usage
-----
::

    from demucs.pretrained import get_model
    from sahtdemucs.model import SAHTDemucs
    from sahtdemucs.losses import SpatialLoss
    import torch

    base  = get_model("htdemucs")
    model = SAHTDemucs(base, sources=base.sources)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-4)

    # Training loop
    for mix, targets in train_loader:
        estimates, deltas = model(mix)
        loss = SpatialLoss()(estimates, targets)
        loss.backward()
        optimizer.step()

    # Inference
    estimates, _ = model(mix)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple

from .cue_module import build_spatial_module

__all__ = ["SAHTDemucs"]


class SAHTDemucs(nn.Module):
    """Attach :class:`~sahtdemucs.cue_module.SpatialCueModule` instances
    to a frozen, pre-trained HTDemucs model.

    The base model weights are frozen by default so that only the lightweight
    spatial correction heads need to be trained.  This makes fine-tuning on a
    relatively small spatially-annotated dataset practical.

    Args:
        base_model:    pre-trained ``HTDemucs`` (or any compatible) model that
                       accepts ``(B, 2, T)`` and returns ``(B, S, 2, T)``
        sources:       list of source names matching the base model's outputs
        spatial_arch:  which spatial cue architecture to use: ``"cnn1d"``
                       (default, temporal Conv1d) or ``"cnn2d"``
                       (spectro-temporal Conv2d — jointly models frequency
                       and time).  The string is forwarded to
                       :func:`~sahtdemucs.cue_module.build_spatial_module`.
        hidden:        hidden channel width of the correction CNN (default 64
                       for ``"cnn1d"``, 32 for ``"cnn2d"``)
        n_fft:         STFT FFT size for sub-band ILD (default 2048)
        hop_length:    STFT hop size in samples (default 512)
        n_bands:       number of frequency sub-bands (default 32)
        band_scale:    frequency band spacing — ``"linear"`` (default,
                       equal-width linear bands) or ``"mel"`` (Mel-scale
                       bands, improves ILD preservation at low frequencies)
        sample_rate:   audio sample rate in Hz, used only when
                       ``band_scale="mel"`` (default 44100, matches HT-Demucs)
        freeze_base:   if ``True`` (default), freeze all parameters of *base_model*

    Inputs:
        mix: ``(B, 2, T)``

    Outputs:
        estimates: ``(B, S, 2, T)`` with spatial cues preserved
    """

    def __init__(
        self,
        base_model: nn.Module,
        sources: List[str],
        spatial_arch: str = "cnn1d",
        max_lag: int      = 64,
        hidden: int       = 64,
        n_fft: int        = 2048,
        hop_length: int   = 512,
        n_bands: int      = 32,
        ild_scale: float  = 6.0,
        band_scale: str   = "linear",
        sample_rate: int  = 44100,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_model   = base_model
        self.sources      = list(sources)
        self.n_sources    = len(self.sources)
        self.spatial_arch = spatial_arch

        # Freeze the pre-trained backbone so only the spatial heads are trained
        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        # One spatial module per source — architecture selected by spatial_arch
        self.spatial_modules: nn.ModuleList = nn.ModuleList(
            [
                build_spatial_module(
                    arch        = spatial_arch,
                    hidden      = hidden,
                    n_fft       = n_fft,
                    hop_length  = hop_length,
                    n_bands     = n_bands,
                    ild_scale   = ild_scale,
                    band_scale  = band_scale,
                    sample_rate = sample_rate,
                    max_lag     = max_lag,
                )
                for _ in range(self.n_sources)
            ]
        )

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str):
        """Forward any attribute lookup not found on this wrapper to base_model.

        This allows demucs utilities such as ``apply_model`` to access
        model metadata (``samplerate``, ``audio_channels``, ``segment``, …)
        transparently, as if they were querying the original HT-Demucs model.
        """
        # nn.Module stores its own attributes in __dict__ and _modules;
        # only fall through to base_model if the attribute truly doesn't exist
        # on this object (prevents infinite recursion during __init__).
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def trainable_parameters(self):
        """Return only the SpatialCueModule parameters (base model excluded).

        Pass this iterator to the optimiser so the frozen backbone weights
        are never updated::

        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-4)
        """
        return self.spatial_modules.parameters()

    def count_trainable(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run HTDemucs separation, then apply per-source ILD correction.

        The base model always runs under ``torch.no_grad()`` (frozen).
        Each :class:`~sahtdemucs.cue_module.SpatialCueModule` receives the
        corresponding raw estimate and predicts a Δ_ILD correction from its
        own estimated ILD alone.

        Args:
            mix: `(B, 2, T)` stereo mixture

        Returns:
            estimates: `(B, S, 2, T)` ILD-corrected separated sources
            deltas:    list of S tensors `(B, n_bands, T_frames)` — raw CNN outputs in [−1, +1]
        """
        # ── Base model (always frozen, never needs grad) ──
        with torch.no_grad():
            raw_estimates = self.base_model(mix)  # (B, S, 2, T)

        # ── Per-source spatial correction ──
        estimates: List[torch.Tensor] = []
        deltas:    List[torch.Tensor] = []

        for s in range(self.n_sources):
            corrected, delta = self.spatial_modules[s](raw_estimates[:, s])
            estimates.append(corrected)
            deltas.append(delta)

        return torch.stack(estimates, dim=1), deltas  # (B, S, 2, T), [S × (B, 1)]

    # ------------------------------------------------------------------ #
    # Full-track inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def separate(
        self,
        wav: torch.Tensor,
        progress: bool = False,
    ) -> torch.Tensor:
        """Full-track inference with overlap-add chunking + spatial correction.

        Uses ``demucs.apply.apply_model`` for the backbone so that chunk
        boundaries are smoothed with overlap-add (unlike the naive manual
        chunking in ``separate_full_track``).  The spatial correction heads
        are then applied once on the **full-length** separated signal, giving
        more stable ILD estimates than per-chunk correction would.

        Args:
            wav:      ``(2, T)`` stereo waveform, already on the correct device.
            progress: if ``True``, show a tqdm progress bar over backbone chunks.

        Returns:
            ``(S, 2, T)`` separated sources with spatial cues corrected.
        """
        from demucs.apply import apply_model

        self.eval()
        device = wav.device

        # ── Step 1: backbone separation via overlap-add ───────────────────
        # apply_model expects (batch, channels, time) and returns the same
        # shape with an extra source dimension: (batch, sources, channels, time).
        # We call it on self.base_model (HT-Demucs), not on self, because our
        # forward() returns a tuple that apply_model cannot handle.
        raw = apply_model(
            self.base_model,
            wav.unsqueeze(0).to(device),   # (1, 2, T)
            progress=progress,
        ).squeeze(0)                        # (S, 2, T)

        # ── Step 2: spatial correction on the full signal ─────────────────
        estimates = []
        for s in range(self.n_sources):
            corrected, _ = self.spatial_modules[s](raw[s].unsqueeze(0))  # (1, 2, T)
            estimates.append(corrected.squeeze(0))   # (2, T)

        return torch.stack(estimates, dim=0)  # (S, 2, T)