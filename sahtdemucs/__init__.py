"""
msslnet — Stereo source separation with spatial cue preservation (ILD/ITD).

Package overview
----------------
The package is organised into the following modules:

    model.py        MSSLNet — the main separation model (U-Net + BiLSTM +
                    per-source SpatialCueModules).

    pretrained.py   HTDemucsWithSpatial — wraps a frozen pre-trained HTDemucs
                    model from the ``demucs`` package and attaches trainable
                    SpatialCueModules on top.

    blocks.py       ConvBlock / ConvTransposeBlock — 1-D encoder/decoder
                    building blocks shared by the U-Net backbone.

    cue_module.py   SpatialCueModule — lightweight temporal CNN that estimates
                    and applies per-source, per-band ILD corrections.

    spatial.py      Low-level spatial cue utilities: compute_ild,
                    compute_itd_samples (GCC-PHAT soft-argmax), apply_itd
                    (frequency-domain fractional delay).

    losses.py       SpatialLoss — SI-SNR + λ_ild·ILD_MSE + λ_itd·ITD_MSE.
                    SISNRLoss — standalone scale-invariant SNR loss.

    solver.py       Solver — training/validation loop, checkpoint I/O.

    dataset.py      MusdbSpatialDataset — random-segment DataLoader for
                    MUSDB18-HQ style directories.

    train.py        Command-line entry point: ``python -m msslnet.train``.

    separate.py     Inference CLI: ``python -m msslnet.separate``.

Quick start
-----------
.. code-block:: python

    from msslnet import MSSLNet, SpatialLoss, Solver

    model  = MSSLNet(sources=["drums", "bass", "other", "vocals"])
    loss   = SpatialLoss(lambda_ild=1.0, lambda_itd=0.5)
    solver = Solver(model, optimizer, loss_fn=loss)
"""

from .model       import HTDemucsWithSpatial
from .losses      import SpatialLoss, SISNRLoss
from .spatial     import compute_ild, compute_ild_bands
from .cue_module  import SpatialCueModule, SpatialCueModule2D, build_spatial_module

__all__ = [
    "HTDemucsWithSpatial",
    "SpatialLoss",
    "SISNRLoss",
    "compute_ild",
    "compute_ild_bands",
    "SpatialCueModule",
    "SpatialCueModule2D",
    "build_spatial_module",
]

__version__ = "0.1.0"
