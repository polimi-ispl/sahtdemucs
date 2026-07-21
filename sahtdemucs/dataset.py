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
* Random crops are drawn to avoid (near-)silent segments: the mixture RMS
  of each candidate must clear ``min_rms`` (see ``_choose_start``).
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

__all__ = ["MusdbSpatialDataset", "load_audio"]

# Default source order matches the Demucs convention
DEFAULT_SOURCES: List[str] = ["drums", "bass", "other", "vocals"]


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """Load a WAV file and return a ``(2, T)`` stereo tensor at ``sample_rate``.

    Mono files are duplicated to stereo; files with more than two channels are
    truncated to the first two.  Resampling is performed on-the-fly if the file
    sample rate differs from ``sample_rate``.  Falls back to ``soundfile`` when
    torchaudio's backend cannot decode the file (e.g. missing FFmpeg libs).
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        # Fall back to soundfile when torchaudio's backend (torchcodec)
        # cannot load the file (e.g. missing FFmpeg shared libraries).
        data, sr = sf.read(str(path), always_2d=True)   # (T, C)
        wav = torch.from_numpy(data.T).float()           # (C, T)

    # Ensure exactly 2 channels
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)       # mono -> duplicate to stereo
    elif wav.shape[0] > 2:
        wav = wav[:2]                # keep first two channels only

    # Resample on the fly if the file rate differs from the target rate
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    return wav

class MusdbSpatialDataset(Dataset):
    """Random-segment dataset over a MUSDB18-HQ style directory.

    Each item is a ``(mix, targets)`` tuple:
        * ``mix``    — ``(2, segment_len)`` stereo mixture
        * ``targets``— ``(S, 2, segment_len)`` per-source stereo stems

    Args:
        root:            path to the dataset root (contains ``train/`` and/or ``test/``)
        split:           ``"train"`` or ``"test"``
        sources:         ordered list of source stem names
        segment_len:     number of samples per training segment
        sample_rate:     target sample rate; files are resampled on the fly if needed
        augment:         if ``True``, apply random gain and channel-flip augmentation
        crops_per_track: number of independent random crops drawn from each track per
                         epoch (default 1).  Setting this to *k* multiplies the
                         effective dataset size by *k* at no I/O cost — the audio
                         file is loaded once per ``__getitem__`` call and a fresh
                         random start offset is drawn each time.
        min_rms:         minimum mixture RMS a random crop must have to be
                         accepted, to skip (near-)silent segments (default
                         ``1e-4``).  Set to ``0`` to disable the check.
        max_crop_attempts: number of random offsets tried to satisfy ``min_rms``
                         before falling back to the loudest candidate (default 10).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sources: List[str] = DEFAULT_SOURCES,
        segment_len: int = 44100 * 6,   # 6 seconds by default
        sample_rate: int = 44100,
        augment: bool = True,
        crops_per_track: int = 1,
        min_rms: float = 1e-4,
        max_crop_attempts: int = 10,
    ) -> None:
        self.root              = Path(root) / split
        self.sources           = sources
        self.segment_len       = segment_len
        self.sample_rate       = sample_rate
        self.augment           = augment
        self.crops_per_track   = max(1, crops_per_track)
        self.min_rms           = min_rms
        self.max_crop_attempts = max(1, max_crop_attempts)

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
        return len(self.tracks) * self.crops_per_track

    def subset(
        self,
        tracks: List[Path],
        *,
        crops_per_track: Optional[int] = None,
        augment: Optional[bool] = None,
    ) -> "MusdbSpatialDataset":
        """Return a shallow clone restricted to ``tracks``, without re-scanning disk.

        The clone shares all configuration (``root``, ``sample_rate``,
        ``segment_len``, ``sources``, …) with the parent dataset but serves only
        the given track list, so a train/valid split can be made at the *track*
        level — no crop of the same song ever leaks across the split (which
        ``torch.utils.data.random_split`` cannot guarantee, since it splits at
        the crop level).  ``crops_per_track`` and ``augment`` can be overridden
        for the clone, e.g. a single, non-augmented crop per track for
        validation.

        Args:
            tracks:          subset of track directories the clone should serve.
            crops_per_track: override the crops per track (default: keep parent).
            augment:         override augmentation (default: keep parent).

        Returns:
            A new ``MusdbSpatialDataset`` sharing this instance's configuration.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.tracks = list(tracks)
        if crops_per_track is not None:
            clone.crops_per_track = max(1, crops_per_track)
        if augment is not None:
            clone.augment = augment
        return clone

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track_dir = self.tracks[idx % len(self.tracks)]

        # Load mixture and all source stems at the target sample rate
        mix   = self._load(track_dir / "mixture.wav")   # (2, T)
        stems = torch.stack(
            [self._load(track_dir / f"{src}.wav") for src in self.sources]
        )  # (S, 2, T)

        # ── Random crop ───────────────────────────────────────────────────────
        T = mix.shape[-1]                               # songs length in samples
        if T > self.segment_len:
            # Pick a random, non-silent start so the model sees diverse temporal
            # positions with actual content (see _choose_start).
            start = self._choose_start(mix, T)
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

    def _choose_start(self, mix: torch.Tensor, n: int) -> int:
        """Pick a random crop start whose mixture segment is not (near-)silent.

        MUSDB tracks contain silent intros/outros and quiet passages; a fully
        silent crop wastes a training step and — worse for the spatial terms —
        makes the ILD (a log-ratio of L/R energy) and the GCC-PHAT ITD
        ill-defined, injecting noise into those gradients.

        Draws up to ``max_crop_attempts`` random offsets and returns the first
        whose mixture RMS clears ``min_rms``.  If none clears it (e.g. an almost
        entirely silent track) the loudest candidate seen is returned, so the
        method always makes progress and never loops forever.  With
        ``min_rms <= 0`` the check is disabled and a single random offset is
        drawn (the original behaviour).

        The check uses the *mixture* only: an individual stem may legitimately be
        silent over the segment (the model must still learn to output silence),
        but the segment as a whole is guaranteed to carry content.
        """
        hi = n - self.segment_len
        if self.min_rms <= 0.0 or self.max_crop_attempts <= 1:
            return random.randint(0, hi)

        best_start, best_rms = 0, -1.0
        for _ in range(self.max_crop_attempts):
            start = random.randint(0, hi)
            seg   = mix[:, start: start + self.segment_len]
            rms   = seg.pow(2).mean().sqrt().item()
            if rms >= self.min_rms:
                return start
            if rms > best_rms:
                best_rms, best_start = rms, start
        return best_start

    def _load(self, path: Path) -> torch.Tensor:
        """Load a WAV file and return a ``(2, T)`` tensor at ``self.sample_rate``."""
        return load_audio(path, self.sample_rate)

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
        # Random gain: uniform in log scale -> multiplicative in linear scale
        gain  = 10.0 ** (random.uniform(-0.3, 0.3))   # ±6 dB range
        mix   = mix   * gain
        stems = stems * gain

        # Random channel swap: flip L and R channels in both mix and all stems
        if random.random() < 0.5:
            mix   = mix.flip(0)          # (2, T): flip channel dim
            stems = stems.flip(1)        # (S, 2, T): flip channel dim

        return mix, stems