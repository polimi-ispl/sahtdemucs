"""
SA-HTDemucs — Stereo source separation with spatial cue preservation (ILD/ITD).

Package overview
----------------
The package is organized into the following modules:

    model.py        SAHTDemucs: the main separation model (HT-Demucs + per-source SpatialCueModules).

    cue_module.py   SpatialCueModule: lightweight spectro-temporal CNN that estimates and applies per-source,
                    per-band ILD corrections.

    spatial.py      Low-level spatial cue utilities:
                    - mel_bin_assignment
                    - compute_ild
                    - compute_ild_bands
                    - compute_ild_bands_mel
                    - compute_itd_samples
                    - compute_itd_bands
                    - compute_itd_bands_mel
                    - apply_itd

    losses.py       SpatialLoss: λ_si·SIDegradationLoss + λ_ild·ILD_MSE + λ_itd·ITD_MSE.

    dataset.py      MusdbSpatialDataset: random-segment DataLoader for MUSDB18-HQ style directories.

    metrics.py      Inference-time metrics: si_sdr, ild_bands_mae, itd_bands_mae.

    train.py        Training CLI: ``python -m sahtdemucs.train`` — one run per
                    process, each in its own output directory.

    separate.py     Inference / evaluation CLI: ``python -m sahtdemucs.separate``.
                    ``notebook/TrainSAHTDemucs.ipynb`` does the same
                    interactively, comparing several runs at once.
"""

from .model       import SAHTDemucs
from .losses      import SpatialLoss
from .spatial     import compute_ild, compute_ild_bands
from .cue_module  import SpatialCueModule, SpatialCueModule2D, build_spatial_module
from .dataset     import load_audio
from .metrics     import si_sdr, ild_bands_mae, itd_bands_mae

__all__ = [
    "SAHTDemucs",
    "SpatialLoss",
    "compute_ild",
    "compute_ild_bands",
    "SpatialCueModule",
    "SpatialCueModule2D",
    "build_spatial_module",
    "load_audio",
    "si_sdr",
    "ild_bands_mae",
    "itd_bands_mae",
]

__version__ = "0.1.0"
