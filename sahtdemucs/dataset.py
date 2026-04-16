"""
dataset.py — PyTorch Dataset wrappers for spatially-annotated audio datasets.

MusdbSpatialDataset
    Loads mixture + per-source stereo stems from a MUSDB18-HQ style directory
    layout and returns random fixed-length segments.

Expected directory structure::

    data_dir/
    ├── train/
    │   ├── track_name/
    │   │   ├── mixture.wav
    │   │   ├── drums.wav
    │   │   ├── bass.wav
    │   │   ├── other.wav
    │   │   └── vocals.wav
    │   └── ...
    └── test/
        └── ...

Notes
-----
* ``__len__`` returns the number of tracks (not segments).  One random
  segment is drawn per track per epoch, so the effective dataset size
  scales with the number of training epochs.
* Tracks shorter than ``segment_len`` are zero-padded on the right.
* Mono files are duplicated to stereo; files with > 2 channels are
  truncated to the first two channels.
* Resampling is performed on-the-fly if the file sample rate differs from
  ``sample_rate``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

__all__ = ["MusdbSpatialDataset"]

# Default source order matches the Demucs convention
DEFAULT_SOURCES: List[str] = ["drums", "bass", "other", "vocals"]


class MusdbSpatialDataset(Dataset):
    """Random-segment dataset over a MUSDB18-HQ style directory.

    Each item is a ``(mix, targets)`` tuple:
        * ``mix``    — ``(2, segment_len)`` stereo mixture
        * ``targets``— ``(S, 2, segment_len)`` per-source stereo stems

    Args:
        root:         path to the dataset root (contains ``train/`` and/or ``test/``)
        split:        ``"train"`` or ``"test"``
        sources:      ordered list of source stem names
        segment_len:  number of samples per training segment
        sample_rate:  target sample rate; files are resampled on the fly if needed
        augment:      if ``True``, apply random gain and channel-flip augmentation
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sources: List[str] = DEFAULT_SOURCES,
        segment_len: int = 44100 * 6,
        sample_rate: int = 44100,
        augment: bool = True,
    ) -> None:
        self.root        = Path(root) / split
        self.sources     = sources
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.augment     = augment

        # Collect all track subdirectories, sorted for reproducibility
        self.tracks: List[Path] = sorted(
            p for p in self.root.iterdir() if p.is_dir()
        )
        if not self.tracks:
            raise RuntimeError(
                f"No track directories found under {self.root}. "
                "Check that 'root' points to the dataset and 'split' is correct."
            )

    def __len__(self) -> int:
        # One item per track; a random segment is drawn at each __getitem__ call
        return len(self.tracks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track_dir = self.tracks[idx]

        # Load mixture and all source stems at the target sample rate
        mix   = self._load(track_dir / "mixture.wav")   # (2, T)
        stems = torch.stack(
            [self._load(track_dir / f"{src}.wav") for src in self.sources]
        )  # (S, 2, T)

        # ── Random crop ───────────────────────────────────────────────────────
        T = mix.shape[-1]
        if T > self.segment_len:
            # Pick a random start so the model sees diverse temporal positions
            start = random.randint(0, T - self.segment_len)
            mix   = mix[:, start: start + self.segment_len]
            stems = stems[:, :, start: start + self.segment_len]
        else:
            # Zero-pad on the right if the track is shorter than the segment
            pad   = self.segment_len - T
            mix   = torch.nn.functional.pad(mix,   (0, pad))
            stems = torch.nn.functional.pad(stems, (0, pad))

        # ── Data augmentation (train split only) ──────────────────────────────
        if self.augment:
            mix, stems = self._augment(mix, stems)

        return mix, stems

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> torch.Tensor:
        """Load a WAV file and return a ``(2, T)`` tensor at ``self.sample_rate``."""
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception:
            # Fall back to soundfile when torchaudio's backend (torchcodec)
            # cannot load the file (e.g. missing FFmpeg shared libraries).
            data, sr = sf.read(str(path), always_2d=True)   # (T, C)
            wav = torch.from_numpy(data.T).float()           # (C, T)

        # Ensure exactly 2 channels
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)       # mono → duplicate to stereo
        elif wav.shape[0] > 2:
            wav = wav[:2]                # keep first two channels only

        # Resample on the fly if the file rate differs from the target rate
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav

    @staticmethod
    def _augment(
        mix: torch.Tensor,
        stems: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random gain (±6 dB) and random left-right channel flip.

        Both transformations preserve the mixture = sum-of-stems identity and
        are applied consistently to mix and stems so the loss is still valid.

        Args:
            mix:   ``(2, T)``
            stems: ``(S, 2, T)``

        Returns:
            Augmented (mix, stems) with the same shapes.
        """
        # Random gain: uniform in log scale → multiplicative in linear scale
        gain  = 10.0 ** (random.uniform(-0.3, 0.3))   # ±6 dB range
        mix   = mix   * gain
        stems = stems * gain

        # Random channel swap: flip L↔R in both mix and all stems
        if random.random() < 0.5:
            mix   = mix.flip(0)          # (2, T): flip channel dim
            stems = stems.flip(1)        # (S, 2, T): flip channel dim

        return mix, stems
