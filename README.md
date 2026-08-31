# SA-HTDemucs

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)

**Spatially-Aware HT-Demucs** (**SA-HTDemucs**) extends the pre-trained
__[HT-Demucs](https://github.com/facebookresearch/demucs)__ music source separator with explicit preservation of spatial
cues. From an input mixture $x_{gt}$, HT-Demucs separates $S=4$ sources - `drums`, `bass`, `other`, `vocals`, in the
Demucs output order. For each of the separated sources $\widehat{s}$, a **SpatialCueModule** estimates the
**Interaural Level Difference** in the time-frequency domain, which is related to the perceived left/right balance,
and predicts a time-frequency correction to be applied to $\widehat{s}$ to replicate the ILD of the groundtruth
source $s_{gt}$.
While the frozen HT-Demucs backbone provides strong separation quality for free, only the Spatial Cue Modules are trained,
making the approach practical on a small spatially-annotated dataset. SA-HTdemucs has been trained and evaluated on
binauralMUSDB18-HQ, a novel binaural dataset synthesized by convolving each source stem in
[MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) dataset with open-source Head-Related
Transfer Functions (HRTFs) at randomized horizontal positions (HRIR from
[SADIE II](https://www.york.ac.uk/sadie-project/database.html) dataset).
Samples are available for listening tests [on our sample page](https://polimi-ispl.github.io/sahtdemucs/).

---

## Architecture

![SAHTDemucs Architecture](docs/images/architecture.svg)

`SA-HTDemucs` wraps a frozen, pre-trained HT-Demucs model and attaches one `SpatialCueModule` per source ($S=4$).
Only the spatial heads are updated during training; the HT-Demucs backbone (~42 M parameters) stays frozen.

| Component            | Parameters         | Role |
|----------------------|--------------------|---|
| HT-Demucs backbone   | ~42 M (frozen)     | Music source separation |
| SpatialCueModule × S | ~700 K (trainable) | Per-source ILD correction |

The ~700 K figure is that of the **shipped training configuration** - `spatial_arch="cnn2d"`, `hidden=64`,
`n_bands=64`, `n_fft=4096`, `ild_scale=15.0`, `band_scale="mel"`, i.e. the defaults of `sahtdemucs.train` and
`sahtdemucs.separate`. The library-level defaults of `SAHTDemucs` build lighter heads:

| Configuration                                                   | Trainable parameters ($S=4$) |
|-----------------------------------------------------------------|-----------------------------:|
| `sahtdemucs.train` defaults - `cnn2d`, `hidden=64`, `n_bands=64` |                      697 092 |
| `SAHTDemucs` default - `cnn1d`, `hidden=64`, `n_bands=32`        |                      180 864 |
| `SpatialCueModule2D` class default - `hidden=32`, `n_bands=32`   |                      176 516 |

---

## SpatialCueModule

The `SpatialCueModule` analyses the **per-sub-band, per-frame ILD** of each separated source  $\widehat{ILD}^{(s)}_k(t)$
and applies a learned time-frequency-resolved ILD correction $\Delta ILD_k(t)$ so that each source preserves a meaningful
stereo position.

Two architectures are available, selectable via `spatial_arch`:

### `"cnn1d"` - Temporal CNN (default)

Conv1d layers operate along the time axis only; each frequency band is processed independently.

```
ILD map (B, n_bands, T_frames)
    → Conv1d(n_bands → hidden, k=kernel_size) → ReLU
    → Conv1d(hidden  → hidden, k=kernel_size) → ReLU
    → Conv1d(hidden  → n_bands, k=1)          → Tanh   ∈ [−1, +1]
    → × ild_scale                                        dB
```

Default parameters: `hidden=64`, `n_fft=2048`, `hop_length=512`, `n_bands=32`, `ild_scale=6.0`, `kernel_size=7`.

### `"cnn2d"` - Spectro-temporal CNN

Conv2d layers jointly model frequency and time, capturing cross-band patterns (e.g. "apply a larger correction at low
frequencies when high frequencies show a consistent ILD offset").  A **global context branch** (temporal mean per
frequency band) captures DC ILD offsets, freeing the local branch to focus on fine spectro-temporal variations.
The output projection is zero-initialised so corrections start near zero at the beginning of training.

```
ILD map (B, n_bands, T_frames)
    → unsqueeze(1) → (B, 1, n_bands, T_frames)

Local branch (3 × Conv2d with GroupNorm, internal residual layer1 → layer3):
    layer1: Conv2d(1→hidden, freq_k×time_k) → GroupNorm → ReLU    → h1
    layer2: Conv2d(hidden→hidden, ...)       → GroupNorm → ReLU
    layer3: Conv2d(hidden→hidden, ...)       → GroupNorm → ReLU + h1 → h3

Global context branch (collapses time axis):
    mean over T → Conv2d(1→hidden, freq_k×1) → ReLU → broadcast over T

Fusion + projection:
    (h3 + global) → Conv2d(hidden→1, 1×1) → Tanh ∈ [−1, +1]
    → squeeze(1) → (B, n_bands, T_frames)
    → × ild_scale                             dB
```

![SpatialCueModule (cnn2d)](docs/images/spatial_cue_module_cnn2d.svg)

Default parameters: `hidden=32`, `n_fft=2048`, `hop_length=512`, `n_bands=32`, `ild_scale=6.0`, `freq_kernel=3`,
`time_kernel=7`.

### Gain application (shared by both architectures)

At each forward pass both modules:

1. Compute the STFT of the separated source and partition the frequency bins into `n_bands` sub-bands, either by splitting
the linear frequency axis into equal-width intervals (`band_scale`=`linear`) or by splitting the mel-frequency axis into
equal-width intervals and mapping each STFT bin to the mel interval containing its center frequency (`band_scale`=`mel`).
Derive a per-band ILD trajectory $\widehat{ILD}^{(s)}_k(t) \in \mathbb{R}^{B \times K \times T_{frames}}$, where
$K$=`n_bands`.
2. Feed the ILD map into the CNN to predict a correction $\delta_{ILD} \in [−1, +1] \in \mathbb{R}^{B \times K \times T_{frames}}$
per band per frame.
3. Scale $\delta_{ILD}$ by $\alpha_{ILD}$ = `ild_scale` to obtain $\Delta_{ILD}$ in dB, then apply a **symmetric time-varying per-bin
gain** in the STFT domain:
   - Left channel: $g_L = 10^{+\Delta/40}$
   - Right channel: $g_R = 10^{-\Delta/40}$
   - $g_L \cdot g_R = 1$ ⟹ the correction is a pure rebalancing: it shifts the ILD by exactly $\Delta$ dB while
     leaving the geometric mean of the two channel gains unchanged
4. ISTFT reconstructs the corrected waveform.

The entire pipeline (STFT → CNN → gain → ISTFT) is fully differentiable.

> **Note on ITD** - the spatial modules correct **ILD only**.  The STFT-domain gain they apply is real and positive,
> so it cannot alter interaural phase; ITD *correction* is therefore disabled in `SpatialCueModule` /
> `SpatialCueModule2D` (`max_lag` is accepted but unused).
>
> ITD *estimation*, on the other hand, is fully implemented in `spatial.py`: broadband (`compute_itd_samples`,
> GCC-PHAT + soft-argmax), and per-sub-band, per-frame (`compute_itd_bands`, `compute_itd_bands_mel`, band-limited
> GCC-PHAT sharing the ILD band layout), plus `apply_itd` (fractional delay via FFT phase shift).  The per-band
> estimators are wired into `SpatialLoss` (`lambda_itd`) and into `metrics.itd_bands_mae`.  Because PHAT whitening
> discards all magnitude information, a magnitude-only head produces no ITD gradient, so `lambda_itd` defaults to
> `0.0` here; the term only does useful work for models that can change interaural phase, such as the full-backbone
> fine-tune in [`htdemucsspatial/`](htdemucsspatial/README.md).

---

## Installation

```bash
git clone https://github.com/polimi-ispl/sahtdemucs
cd sahtdemucs
pip install demucs torchaudio soundfile librosa tqdm matplotlib
```

`demucs` pulls in `torch`, `torchaudio` and `numpy`.  `librosa` and `tqdm` are used by `binaural_synth.py`;
`matplotlib` by the training and evaluation CLIs (`sahtdemucs.train`, `sahtdemucs.separate`,
`htdemucsspatial.compare_ablation`).  The MoisesDB mode of `binaural_synth.py` additionally needs the optional
`moisesdb` package, which is imported lazily.

---

## Binaural Dataset Generation (binauralMUSDB18-HQ)

SA-HTDemucs is trained and evaluated on **binauralMUSDB18-HQ**, synthesized from
[MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) by convolving each source stem with a
Head-Related Impulse Response (HRIR) at a randomized horizontal position, then summing the binaural stems into a
normalized binaural mixture. The script `sahtdemucs/binaural_synth.py` performs this synthesis.

The HRIRs are those of the Neumann KU100 dummy head from the
[SADIE II](https://www.york.ac.uk/sadie-project/database.html) database (subject D1), diffuse-field compensated (DFC),
at 44.1 kHz. HRIR files are expected to be named `azi_{angle}_ele_0_DFC.wav`.

### What it does, per track

1. Down-mixes each stereo stem to mono, then convolves it with the left/right HRIR for the chosen azimuth (elevation
   fixed at 0°).
2. Assigns each of the 4 stems (`vocals`, `bass`, `drums`, `other`) a **distinct** azimuth, drawn *without replacement*
   from the frontal grid `{0, 10, …, 90} ∪ {270, …, 350}` degrees, so sources do not directly overlap - unless a
   metadata file fixes the angles.
3. Writes the binaural stems, the normalized `mixture.wav` (peak-normalized sum of the binaural stems), and a
   `metadata.json` recording the per-stem azimuth angles.

All audio must be at 44.1 kHz (the script raises if a stem has a different sample rate). The output tree matches what
`MusdbSpatialDataset` expects (see [Module Reference](#sahtdemucsdatasetpy---musdb18-hq-dataloader)).

### Arguments

The script has two modes: **MUSDB mode** (the default, driven by `--input_dir`) and **MoisesDB mode**
(`--moisesdb_dir`).  Exactly one of the two must be given; `--output_dir` and `--hrir_dir` are required in both.

| Argument           | Required      | Description                                                                             |
|--------------------|:-------------:|------------------------------------------------------------------------------------------|
| `--input_dir`      | MUSDB mode    | MUSDB18-HQ root, containing `train/` and `test/`, each with one folder per track.        |
| `--output_dir`     |      yes      | Target root; `train/` and `test/` are created, mirroring the input track structure.      |
| `--hrir_dir`       |      yes      | Folder with the SADIE II HRIR WAVs named `azi_{angle}_ele_0_DFC.wav`.                    |
| `-m`, `--metadata` |      no       | JSON with the per-track stem azimuths, to reproduce an exact dataset version.            |
| `--moisesdb_dir`   | MoisesDB mode | MoisesDB root; switches the script to MoisesDB mode (`--input_dir` and `-m` are ignored).|
| `--test_frac`      |      no       | MoisesDB mode: fraction of tracks held out into `test/` (default `0.15`).                |
| `--seed`           |      no       | MoisesDB mode: RNG seed for the train/test split (default `0`).                          |
| `--limit`          |      no       | MoisesDB mode: process only the first N tracks (debugging).                              |

When `--metadata` is omitted, a new random angle assignment is generated (and saved per track as `metadata.json`).
A reusable metadata file for the published dataset version is provided at `data/binaural_musdb_metadata.json`.  It is
keyed first by split, then by track name (100 train + 50 test entries, matching MUSDB18-HQ):

```json
{
  "train": { "<track name>": {"vocals": 300, "bass": 10, "drums": 80, "other": 70}, "...": {} },
  "test":  { "...": {} }
}
```

Every track directory found under `--input_dir` must have an entry in the file, otherwise the run stops with a
`KeyError`.

### Usage

```bash
python sahtdemucs/binaural_synth.py \
    --input_dir="path/to/MUSDB18-HQ" \
    --output_dir="path/to/binauralMUSDB18-HQ" \
    --hrir_dir="path/to/SADIE_II/Subject_001_Wav/DFC/44K_16bit"
```

On Windows PowerShell, call the interpreter explicitly and keep the command on a single line:

```powershell
python .\sahtdemucs\binaural_synth.py --input_dir="D:\Dataset\MUSDB18HQ" --output_dir="D:\Dataset\binauralMUSDB18HQ" --hrir_dir="D:\Dataset\SADIEII\Subject_001_Wav\DFC\44K_16bit"
```

To reproduce a fixed dataset version, pass the metadata JSON:

```bash
python sahtdemucs/binaural_synth.py \
    --input_dir="path/to/MUSDB18-HQ" \
    --output_dir="path/to/binauralMUSDB18-HQ" \
    --hrir_dir="path/to/SADIE_II/.../DFC/44K_16bit" \
    -m data/binaural_musdb_metadata.json
```

### Extending the dataset with MoisesDB

The same script can binauralize [MoisesDB](https://github.com/moises-ai/moises-db) with the same HRIRs and the same
random azimuth assignment, merging the result into an existing dataset root.  MoisesDB's top-level taxonomy is
collapsed onto the four Demucs stems (anything that is not `vocals`/`bass`/`drums` becomes `other`) and sources absent
from a track are written as silence, so every track still ends up with all four stems.  MoisesDB has no official
split, so `--test_frac` of the tracks are held out into `test/` and the rest go to `train/`; track directories are
prefixed `moisesdb_` to stay identifiable inside the merged dataset, and tracks already synthesized are skipped, so
the job is resumable.

```bash
python sahtdemucs/binaural_synth.py \
    --moisesdb_dir="path/to/moisesdb" \
    --output_dir="path/to/binauralMUSMOISESDB" \
    --hrir_dir="path/to/SADIE_II/.../DFC/44K_16bit" \
    --test_frac=0.15 --seed=0
```

This mode requires the optional `moisesdb` package.  Both the flat (`<root>/<track_id>/data.json`) and the nested
(`<root>/<provider>/<track_id>/data.json`) release layouts are detected automatically.

---

## Quick Start

### Python API

```python
import torch
from demucs.pretrained import get_model
from sahtdemucs.model import SAHTDemucs

bag   = get_model("htdemucs")
base  = bag.models[0] if hasattr(bag, "models") else bag

# Default: cnn1d spatial heads
model = SAHTDemucs(base, sources=base.sources)

# Spectro-temporal variant
model = SAHTDemucs(base, sources=base.sources, spatial_arch="cnn2d")

# Only SpatialCueModule weights are updated
optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)

mix = torch.randn(1, 2, 44100 * 6)        # (B, 2, T)
estimates, raw_estimates, deltas = model(mix)
# estimates:     (B, S, 2, T)  spatially corrected sources
# raw_estimates: (B, S, 2, T)  raw HTDemucs output (no spatial correction)
# deltas:        [S × (B, n_bands, T_frames)]  CNN corrections in [−1, +1]
```

### Full-track inference

```python
# Chunked overlap-add backbone + full-signal spatial correction
stems = model.separate(wav)   # (2, T) → (S, 2, T)
```

### Training the spatial heads

```python
from sahtdemucs.losses import SpatialLoss

loss_fn = SpatialLoss(lambda_si=0.0, lambda_ild=1.0)  # ILD supervision only

for mix, targets in train_loader:          # mix: (B,2,T)  targets: (B,S,2,T)
    estimates, raw_estimates, _ = model(mix)
    loss, loss_si, loss_ild, loss_itd = loss_fn(estimates, targets, raw_estimates)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
```

In practice training is run headless - `python -m sahtdemucs.train` - and
`notebook/TestSAHTDemucs.ipynb` evaluates the resulting runs against the frozen
HT-Demucs baseline on the test split; `python -m sahtdemucs.separate` does the same headless, for a single run.

---

## Loss Function

`SpatialLoss` is an objective that supervises both **sub-band** spatial cue fidelity and separation quality for each of
the $S$ sources

$$
\mathcal{L} = \frac{1}{S} \sum_{s=1}^{S} \left(
\lambda_{\text{ILD}} \cdot \mathcal{L}_{\text{ILD}}^{(s)}
+ \lambda_{\text{SI}} \cdot \mathcal{L}_{\text{SI-SDR}}^{(s)}
+ \lambda_{\text{ITD}} \cdot \mathcal{L}_{\text{ITD}}^{(s)}
\right).
$$

$\mathcal{L}_{\text{ILD}}^{(s)}$ is the MSE between the corrected source time-frequency ILD and the ground-truth one,
defined as

$$
\mathcal{L}_{\text{ILD}}^{(s)} =
  \frac{1}{K \cdot T_f} \sum_{k=1}^{K} \sum_{t=1}^{T_f}
  \left(
    \widehat{\text{ILD}}_k^{(s)}(t) - \text{ILD}_{k,\text{gt}}^{(s)}(t)
  \right)^2,
$$

where $K$ = `n_bands` and $T_f$ is the number of STFT frames.

$\mathcal{L}_{\text{ITD}}^{(s)}$ has the same form on the per-sub-band, per-frame **ITD** (in samples), estimated with
the band-limited, differentiable GCC-PHAT of `compute_itd_bands` / `compute_itd_bands_mel`, which shares its band
layout with the ILD term:

$$
\mathcal{L}_{\text{ITD}}^{(s)} =
  \frac{1}{K \cdot T_f} \sum_{k=1}^{K} \sum_{t=1}^{T_f}
  \left(
    \widehat{\text{ITD}}_k^{(s)}(t) - \text{ITD}_{k,\text{gt}}^{(s)}(t)
  \right)^2 .
$$

It is **disabled by default** ($\lambda_{\text{ITD}}=0$): an ILD-gain head cannot change interaural phase, so the term
yields no useful gradient for SA-HTDemucs (see the note on ITD above) - it is wired in for the full-backbone fine-tune
in `htdemucsspatial/`.  Being in samples², it sits on a far larger numerical scale than the ILD's dB², so a non-zero
$\lambda_{\text{ITD}}$ is typically one to two orders of magnitude below $\lambda_{\text{ILD}}$.

$\mathcal{L}_{\text{SI-SDR}}^{(s)}$ is not a plain SI-SDR loss but a one-sided **degradation penalty**: it penalises
the spatial correction only when it lowers the SI-SDR of the corrected source $\hat{s}$ below that of the frozen HT-Demucs
output $\bar{s}$ by more than a tolerated margin $m_{dB}$ = `si_margin_db` (in dB). With $\text{SI-SDR}(\cdot)$ evaluated
against the ground-truth source $s_{gt}$,

$$
\mathcal{L}_{\text{SI-SDR}}^{(s)} =
  \text{ReLU}\left(
    \text{SI-SDR}(\bar{s}, s_{gt}) - \text{SI-SDR}(\hat{s}, s_{gt}) - m_{dB}
  \right),
$$

where

$$
\text{SI-SDR}(\hat{s}, s_{gt}) = 10 \cdot \log_{10} \left( \frac{\left\| \dfrac{\langle \hat{s},
s_{gt} \rangle}{\|s_{gt}\|^2} \cdot s_{gt} \right\|^2} {\left\| \hat{s} - \dfrac{\langle \hat{s}, s_{gt} \rangle}
{\|s_{gt}\|^2} \cdot s_{gt}\right\|^2} \right) \quad \text{[dB]}
$$

is the **Scale-invariant Signal-To-Distortion Ratio**.

The $\mathcal{L}_{\text{SI-SDR}}^{(s)}$ term is always non-negative and is zero (no gradient) as long as the spatial head does not hurt separation beyond the
margin, letting it improve ILD freely. Setting $\lambda_{\text{SI}}=0$ recovers a purely spatial loss (and lets
`raw_estimates` be omitted in the forward call).

The forward pass takes the raw HT-Demucs output and returns the total loss together with its three (already weighted)
components for logging:

```python
total, si_part, ild_part, itd_part = loss_fn(estimates, targets, raw_estimates)
total.backward()
```

`raw_estimates` is required whenever `lambda_si > 0`; if `None`, the SI-SDR degradation term is skipped.

### Loss hyperparameters

|         Symbol          | Parameter      | Default  | Description                                                                   |
|:-----------------------:|----------------|:--------:|-------------------------------------------------------------------------------|
| $\lambda_{\text{SI}}$   | `lambda_si`    |  `1.0`   | Weight of the SI-SDR degradation penalty                                      |
| $\lambda_{\text{ILD}}$  | `lambda_ild`   |  `1.0`   | Weight of the sub-band ILD penalty                                            |
| $\lambda_{\text{ITD}}$  | `lambda_itd`   |  `0.0`   | Weight of the sub-band ITD penalty (disabled by default)                      |
|        $m_{dB}$         | `si_margin_db` |  `0.5`   | Tolerated SI-SDR degradation (dB) before it is penalised                      |
|          $K$            | `n_bands`      |   `32`   | Number of frequency sub-bands                                                 |
|           -             | `n_fft`        |  `2048`  | STFT FFT size                                                                 |
|           -             | `hop_length`   |  `512`   | STFT hop size                                                                 |
|           -             | `band_scale`   | `linear` | Sub-band spacing - `linear` (equal-width in Hz) or `mel` (equal-width in mel)  |
|           -             | `sample_rate`  | `44100`  | Sample rate, used only when `band_scale="mel"`                                |
|           -             | `itd_max_lag`  |   `64`   | GCC-PHAT search range in samples (±1.45 ms @ 44.1 kHz)                        |
|           -             | `itd_beta`     |  `20.0`  | Soft-argmax temperature of the ITD estimator                                  |

`band_scale`, `n_fft`, `hop_length` and `n_bands` must match the `SpatialCueModule` configuration, otherwise the loss
supervises a band layout different from the one the head can act on.

These are the `SpatialLoss` **class** defaults; the shipped training recipe overrides several of them.
`sahtdemucs.train` defaults to `--lambda-si 10.0`, `--lambda-ild 1.0`, `--lambda-itd 0.0`, `--si-margin-db 0.2`, with
`--n-bands 64`, `--n-fft 4096` and `--band-scale mel`.

```python
from sahtdemucs.losses import SpatialLoss

# ILD supervision only - raw_estimates not needed
loss_fn = SpatialLoss(lambda_si=0.0, lambda_ild=1.0)
total, si_part, ild_part, itd_part = loss_fn(estimates, targets)   # (B,S,2,T), (B,S,2,T) → 4 scalars

# Joint ILD + SI-SDR-degradation guard (raw_estimates required when lambda_si > 0);
# these are the weights sahtdemucs.train uses
loss_fn = SpatialLoss(lambda_si=10.0, lambda_ild=1.0, si_margin_db=0.2)
total, si_part, ild_part, itd_part = loss_fn(estimates, targets, raw_estimates)
```

---

## Repository Map

```
sahtdemucs/
├── sahtdemucs/               ← Python package (frozen backbone + spatial heads)
│   ├── __init__.py           ← public API
│   ├── model.py              ← SAHTDemucs
│   ├── cue_module.py         ← SpatialCueModule (cnn1d), SpatialCueModule2D (cnn2d), build_spatial_module
│   ├── spatial.py            ← mel_bin_assignment, compute_ild, compute_ild_bands[_mel],
│   │                            compute_itd_samples, compute_itd_bands[_mel], apply_itd
│   ├── losses.py             ← SpatialLoss (sub-band ILD MSE + SI-SDR degradation guard + sub-band ITD MSE)
│   ├── dataset.py            ← MusdbSpatialDataset, load_audio, load_audio_segment
│   ├── metrics.py            ← si_sdr, ild_bands_mae, itd_bands_mae (inference-time)
│   ├── train.py              ← training CLI: python -m sahtdemucs.train
│   ├── separate.py           ← inference/evaluation CLI: python -m sahtdemucs.separate
│   ├── export_torchscript.py ← TorchScript export for the C++ CLI and the JUCE/VST plugin
│   └── binaural_synth.py     ← binaural dataset synthesis from MUSDB18-HQ (or MoisesDB) + SADIE II HRIRs
├── htdemucsspatial/          ← full-backbone spatial fine-tune (see its own README)
│   ├── __init__.py
│   ├── train.py              ← training CLI: python -m htdemucsspatial.train
│   ├── freeze.py             ← freeze-strategy grammar (which blocks stay trainable)
│   ├── losses.py             ← HTDemucsSpatialLoss (time-domain L1 + ILD/ITD MSE)
│   └── compare_ablation.py   ← table + curves over all runs of a sweep
├── cpp/sahtdemucs_cli/       ← C++/LibTorch CLI running the exported TorchScript model with the
│                                plugin's chunking strategy (see its own README)
├── matlab/                   ← perceptual evaluation: PEASS, goniometer, stereo metrics
├── data/
│   └── binaural_musdb_metadata.json   ← per-track stem azimuths (reproducible dataset)
├── notebook/
│   ├── TestSAHTDemucs.ipynb        ← SA-HTDemucs evaluation & baseline comparison
│   ├── TestHTDemucsSpatial.ipynb   ← freeze-strategy comparison of the backbone fine-tunes
│   └── PrepareOnlineDemo.ipynb     ← generate demo page audio
└── docs/                     ← GitHub Pages listening demo + figures (see docs/README.md)
    ├── index.html, style.css, main.js
    ├── audio/song{1,2,3}/    ← mixture + reference, HT-Demucs, SA-HTDemucs and fine-tune stems
    └── images/               ← architecture.{svg,png}, spatial_cue_module_cnn.svg,
                                 spatial_cue_module_cnn2d.{svg,png}, ispl_logo.png
```

Created locally and not tracked in git (see `.gitignore`): `runs/` (checkpoints), `plot/` (loss and per-band MAE
figures), `checkspatial/` (REAPER project for informal spatial QA) and `matlab/figures/`.

---

## Module Reference

### `sahtdemucs/binaural_synth.py` - Binaural tracks synthesis

Synthesizes **binauralMUSDB18-HQ**, from [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) by convolving each source stem with a
Head-Related Impulse Response (HRIR) at a randomized horizontal position, then summing the binaural stems into a
normalized binaural mixture.

---

### `sahtdemucs/model.py` - SAHTDemucs

```python
model = SAHTDemucs(
    base_model,
    sources       = base.sources,
    spatial_arch  = "cnn1d",   # or "cnn2d"
    max_lag       = 64,        # accepted for ITD support; unused by the ILD heads
    hidden        = 64,        # SpatialCueModule2D's own class default is 32
    n_fft         = 2048,
    hop_length    = 512,
    n_bands       = 32,
    ild_scale     = 6.0,
    band_scale    = "linear",  # or "mel"
    sample_rate   = 44100,
    freeze_base   = True,
    use_gb        = True,      # cnn2d only: enable the global context branch
)

estimates, raw_estimates, deltas = model(mix)
# estimates:     (B, S, 2, T)  spatially corrected sources
# raw_estimates: (B, S, 2, T)  raw HTDemucs output (no spatial correction)
# deltas:        [S × (B, n_bands, T_frames)]  CNN corrections in [−1, +1]

stems = model.separate(wav)       # full-track (2, T) → (S, 2, T)

optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-4)
print(model.count_trainable())    # number of trainable parameters
```

### `sahtdemucs/cue_module.py` - Spatial correction heads

```python
from sahtdemucs.cue_module import SpatialCueModule, SpatialCueModule2D, build_spatial_module

# Temporal CNN (cnn1d)
mod = SpatialCueModule(hidden=64, n_bands=32, ild_scale=6.0)
corrected, delta = mod(source_estimate)
# corrected: (B, 2, T)
# delta:     (B, n_bands, T_frames) in [−1, +1]

# Spectro-temporal CNN (cnn2d)
mod = SpatialCueModule2D(hidden=32, n_bands=32, ild_scale=6.0)

# Factory
mod = build_spatial_module("cnn2d", n_bands=32, ild_scale=6.0)
```

`build_spatial_module` drops keywords belonging to the *other* registered architecture (`use_gb`, `freq_kernel` and
`time_kernel` are `cnn2d`-only, `kernel_size` is `cnn1d`-only), so a single config dict can be reused across both; a
keyword no architecture accepts still raises `TypeError`.

---

### `sahtdemucs/spatial.py` - Low-level spatial cue primitives

| Function | Input | Output | Notes |
|---|---|---|---|
| `mel_bin_assignment(n_fft, n_bands, sample_rate)` | - | `(n_fft//2+1,)` long | STFT bin → mel band index; cached, shared by analysis and synthesis |
| `compute_ild(left, right)` | `(B, T)` | `(B,)` dB | Broadband RMS ratio L/R |
| `compute_ild_bands(left, right, n_fft, hop, n_bands)` | `(B, T)` | `(B, n_bands, T_frames)` dB | Per-sub-band, per-frame ILD via STFT, equal-width **linear** bands |
| `compute_ild_bands_mel(left, right, n_fft, hop, n_bands, sample_rate)` | `(B, T)` | `(B, n_bands, T_frames)` dB | Same, with equal-width **mel** bands (rectangular filterbank) |
| `compute_itd_samples(left, right, max_lag)` | `(B, T)` | `(B,)` samples | Broadband GCC-PHAT + soft-argmax (differentiable) |
| `compute_itd_bands(left, right, n_fft, hop, n_bands, max_lag, beta)` | `(B, T)` | `(B, n_bands, T_frames)` samples | Band-limited GCC-PHAT, **linear** bands |
| `compute_itd_bands_mel(left, right, n_fft, hop, n_bands, sample_rate, max_lag, beta)` | `(B, T)` | `(B, n_bands, T_frames)` samples | Band-limited GCC-PHAT, **mel** bands |
| `apply_itd(signal, delay_samples)` | `(B, T)`, `(B,)` | `(B, T)` | Fractional delay via FFT phase shift |

All functions are fully differentiable.  A positive ILD means the left channel is louder; the ITD sign convention is
the lag that maximises the PHAT cross-correlation of $L \cdot \overline{R}$.

---

### `sahtdemucs/losses.py` - Loss functions

| Class | Formula                                                                                                           |
|---|-------------------------------------------------------------------------------------------------------------------|
| `SpatialLoss` | $$ \mathcal{L} = \frac{1}{S} \sum_{s=1}^{S} \left(\lambda_{\text{SI}} \cdot \mathcal{L}_{\text{SI-SDR}}^{(s)} + \lambda_{\text{ILD}} \cdot \mathcal{L}_{\text{ILD}}^{(s)} + \lambda_{\text{ITD}} \cdot \mathcal{L}_{\text{ITD}}^{(s)}\right)$$ where $\mathcal{L}_{\text{SI-SDR}}^{(s)}$ is the one-sided SI-SDR degradation penalty (see [Loss Function](#loss-function)), $\mathcal{L}_{\text{ILD}}^{(s)}$ the sub-band ILD MSE and $\mathcal{L}_{\text{ITD}}^{(s)}$ the sub-band ITD MSE ($\lambda_{\text{ITD}}=0$ by default). Returns `(total, si_part, ild_part, itd_part)`, all already weighted. |

---

### `sahtdemucs/dataset.py` - MUSDB18-HQ DataLoader

Reads mixture and source stems from a MUSDB18-HQ style directory tree. Returns random fixed-length segments with optional
gain (±6 dB) and channel-flip augmentation.

```
musdb18hq/
└── train/
    └── <track name>/
        ├── mixture.wav
        ├── drums.wav  ├── bass.wav  ├── other.wav  └── vocals.wav
```

```python
from sahtdemucs.dataset import MusdbSpatialDataset
from torch.utils.data import DataLoader

ds     = MusdbSpatialDataset("musdb18hq/", split="train", segment_len=44100*6)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
mix, targets = next(iter(loader))   # (4, 2, 264600),  (4, 4, 2, 264600)
```

Crops are read straight out of the WAVs (`soundfile` seek) instead of decoding whole songs, near-silent crops are
rejected via `min_rms`, and `subset()` clones the dataset at the *track* level, so a train/valid split never leaks
crops of the same song across the two.

---

### `sahtdemucs/metrics.py` - Inference-time metrics

| Function | Output | Notes |
|---|---|---|
| `si_sdr(est, tgt)` | `float` dB | Both channels flattened into one signal - one scalar per source |
| `ild_bands_mae(est, tgt, ...)` | `(n_bands,)` dB | Per-sub-band ILD MAE |
| `itd_bands_mae(est, tgt, ...)` | `(n_bands,)` samples | Per-sub-band ITD MAE |

All three take `(2, T)` tensors and accept the same band configuration as `SpatialLoss` (`n_fft`, `hop_length`,
`n_bands`, `scale="linear"` or `"mel"`, `sample_rate`), so metric and objective can be driven from one config.

---

### `sahtdemucs/export_torchscript.py` - TorchScript export

Traces `SAHTDemucs.forward` at a **fixed** input length into a TorchScript checkpoint for the C++ CLI in
[`cpp/sahtdemucs_cli/`](cpp/sahtdemucs_cli/README.md) and for the JUCE/VST plugin.  The trace bakes shape-dependent
branches of HT-Demucs in as constants, so the consumer's chunk length must equal the traced
`EXAMPLE_LENGTH_SAMPLES` - see the C++ README for the details.

<!-- ---

## Citation

If you build on this work, please also cite the original Demucs papers:

```bibtex
@inproceedings{rouard2022hybrid,
  title     = {Hybrid Transformers for Music Source Separation},
  author    = {Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle = {ICASSP 2023},
  year      = {2023}
}
```
-->