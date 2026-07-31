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
* ``__len__`` returns ``len(tracks) * crops_per_track``.  Each item draws a
  fresh random crop from its track, so with ``crops_per_track = k`` an epoch
  sees *k* independent segments per track at no extra I/O cost.
* Random crops are drawn to avoid (near-)silent segments: the mixture RMS
  of each candidate must clear ``min_rms`` (see ``_choose_start``).
* Only the crop is read from disk: the mixture header gives the track length,
  and each of the five files is then seeked into for ``segment_len`` samples.
  Decoding whole songs to keep 8 s of each costs ~40x more I/O and dominates
  the epoch time on a network filesystem.  Files that ``soundfile`` cannot seek,
  or that are not already at ``sample_rate`` (resampling needs the full signal),
  fall back to decoding the whole track — same crops either way.
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

__all__ = ["MusdbSpatialDataset", "load_audio", "load_audio_segment"]

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


def _to_stereo(wav: torch.Tensor) -> torch.Tensor:
    """Force a ``(C, T)`` tensor to exactly two channels."""
    if wav.shape[0] == 1:
        return wav.repeat(2, 1)      # mono -> duplicate to stereo
    if wav.shape[0] > 2:
        return wav[:2]               # keep first two channels only
    return wav


def load_audio_segment(
    path: Path,
    start: int,
    frames: int,
) -> torch.Tensor:
    """Read ``frames`` samples of a WAV starting at ``start``, as ``(2, frames)``.

    Seeks straight to the crop instead of decoding the whole file, which is what
    makes training on a network filesystem viable: a segment costs a few hundred
    kB rather than the tens of MB of a full song.  The samples are identical to
    the corresponding slice of :func:`load_audio` — both decode PCM to float32 in
    ``[-1, 1)`` — so a crop read this way matches one read the long way.

    The file must already be at the target sample rate (the caller checks it with
    ``soundfile.info``); resampling needs the surrounding context and therefore
    the full-file path.  Short reads at the end of a file are zero-padded on the
    right, matching the padding :meth:`MusdbSpatialDataset.__getitem__` applies to
    tracks shorter than one segment.

    Args:
        path:   WAV file to read from.
        start:  first sample of the crop.
        frames: how many samples to read.

    Returns:
        ``(2, frames)`` float32 stereo tensor.
    """
    data, _ = sf.read(str(path), start=start, frames=frames,
                      always_2d=True, dtype="float32")     # (T, C)
    wav = _to_stereo(torch.from_numpy(data.T.copy()))      # (2, T')
    if wav.shape[-1] < frames:
        wav = torch.nn.functional.pad(wav, (0, frames - wav.shape[-1]))
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
        self._probe_cache: dict = {}       # mixture path -> length, or None

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
        mix_path  = track_dir / "mixture.wav"

        # Reading the crop directly beats decoding five whole songs to keep 8 s
        # of each, by more than an order of magnitude — but it needs the file to
        # be seekable and already at the target rate.  ``sf.info`` answers both
        # questions from the header alone; anything else takes the slow path.
        seekable = self._probe(mix_path)

        if seekable is None:
            mix, stems = self._load_whole(track_dir)
        else:
            mix, stems = self._load_crop(track_dir, mix_path, n_samples=seekable)

        # ── Data augmentation (train split only) ──────────────────────────────
        if self.augment:
            mix, stems = self._augment(mix, stems)

        return mix, stems

    def _load_crop(
        self,
        track_dir: Path,
        mix_path: Path,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one crop out of the track, touching only the samples it needs."""
        if n_samples <= self.segment_len:
            # Nothing to choose from: take the track whole and pad on the right.
            mix   = load_audio_segment(mix_path, 0, self.segment_len)
            stems = torch.stack([load_audio_segment(track_dir / f"{src}.wav", 0,
                                                    self.segment_len)
                                 for src in self.sources])
            return mix, stems

        # Candidate crops are read one at a time; the winner is already in hand,
        # so an accepted crop costs exactly one read of the mixture.
        crops: dict = {}

        def read_mix(start: int) -> torch.Tensor:
            if start not in crops:
                crops[start] = load_audio_segment(mix_path, start, self.segment_len)
            return crops[start]

        start = self._choose_start(read_mix, n_samples)
        mix   = read_mix(start)
        stems = torch.stack([
            load_audio_segment(track_dir / f"{src}.wav", start, self.segment_len)
            for src in self.sources
        ])                                              # (S, 2, segment_len)
        return mix, stems

    def _load_whole(self, track_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the full track, then crop — for files ``_load_crop`` can't seek."""
        mix   = self._load(track_dir / "mixture.wav")   # (2, T)
        stems = torch.stack(
            [self._load(track_dir / f"{src}.wav") for src in self.sources]
        )  # (S, 2, T)

        T = mix.shape[-1]                               # song length in samples
        if T > self.segment_len:
            # Pick a random, non-silent start so the model sees diverse temporal
            # positions with actual content (see _choose_start).
            start = self._choose_start(
                lambda s: mix[:, s: s + self.segment_len], T)
            mix   = mix[:, start: start + self.segment_len]
            stems = stems[:, :, start: start + self.segment_len]
        else:
            # Zero-pad on the right if the track is shorter than the segment
            pad   = self.segment_len - T
            mix   = torch.nn.functional.pad(mix,   (0, pad))
            stems = torch.nn.functional.pad(stems, (0, pad))
        return mix, stems

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _probe(self, mix_path: Path) -> Optional[int]:
        """Track length in samples when the crop can be read directly, else None.

        ``None`` means the header could not be read or the file is not already at
        ``self.sample_rate`` — resampling needs the whole signal, so those tracks
        go through :meth:`_load_whole`.

        The answer is memoised per track: a static dataset is probed once instead
        of once per crop, which on a network filesystem saves one metadata
        round-trip per item.
        """
        if mix_path in self._probe_cache:
            return self._probe_cache[mix_path]
        try:
            info = sf.info(str(mix_path))
            n = info.frames if info.samplerate == self.sample_rate else None
        except Exception:
            n = None                        # not seekable by soundfile
        self._probe_cache[mix_path] = n
        return n

    def _choose_start(self, read_segment, n: int) -> int:
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

        Args:
            read_segment: ``start -> (2, segment_len)`` mixture crop.  Slicing an
                          already-decoded track and seeking into the file are
                          both valid; the RNG is consumed identically either way,
                          so the two paths select the same crops from the same
                          seed.
            n:            track length in samples.
        """
        hi = n - self.segment_len
        if self.min_rms <= 0.0 or self.max_crop_attempts <= 1:
            return random.randint(0, hi)

        best_start, best_rms = 0, -1.0
        for _ in range(self.max_crop_attempts):
            start = random.randint(0, hi)
            rms   = read_segment(start).pow(2).mean().sqrt().item()
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