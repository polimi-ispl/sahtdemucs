"""
export_torchscript.py
=====================

Exports the SA-HTDemucs model (Elettromedia / Politecnico di Milano ISPL) to a
TorchScript file that can be loaded from C++ via LibTorch's torch::jit::load(),
with no dependency on the Python source code or the `sahtdemucs` package.

The resulting .pt file is a *traced* or *scripted* module that can be called
directly with:

    auto module = torch::jit::load("sahtdemucs_traced.pt");
    auto output  = module.forward({input_tensor}).toTensor();

------------------------------------------------------------------------------
CUSTOMISATION POINTS (search for "# >>> CUSTOMISE")
------------------------------------------------------------------------------

This script makes the following ASSUMPTIONS about the model interface, which
are the standard Demucs v4 / HTDemucs convention. Adjust if your SA-HTDemucs
forward signature differs:

  Input:  torch.Tensor, shape [batch, 2, T]   (stereo waveform, float32)
          sample rate = model.samplerate (typically 44100 Hz)

  Output: torch.Tensor, shape [batch, 4, 2, T]
          dim 1 = source index: [drums, bass, other, vocals]  (Demucs order)
          dim 2 = stereo channels (L, R)

If your model's forward() returns a dict or a different ordering, adjust the
ExportWrapper.forward() method below accordingly.

------------------------------------------------------------------------------
MODEL LOADING
------------------------------------------------------------------------------

build_model() constructs SAHTDemucs by wrapping a pretrained HTDemucs
instance (via demucs.pretrained.get_model) — so backbone weights are
already loaded at construction time, exactly as in the training script
(`bag = get_model("htdemucs"); base = bag.models[0]; model = SAHTDemucs(base, ...)`).

load_weights() then loads the spatial cue module weights from your
`spatial_modules_<timestamp>.pt` file on top of the pretrained backbone.
"""

from pathlib import Path
import argparse
import warnings

import torch
import torch.nn as nn

# ==============================================================================
# COMMAND-LINE ARGUMENTS
# ==============================================================================
#
# _run_ts identifies which training run's spatial module checkpoint to
# export. It determines:
#   CHECKPOINT_PATH_SPATIAL = RUNS_DIR / f"spatial_modules_{_run_ts}.pt"
#   OUTPUT_PATH             = RUNS_DIR / f"sahtdemucs_traced_{_run_ts}.pt"
#
# Usage:
#   python export_torchscript.py                       # uses DEFAULT_RUN_TS below
#   python export_torchscript.py 20260531_155238       # exports that run
#   python export_torchscript.py --run-ts 20260531_155238
#
DEFAULT_RUN_TS = "20260531_155238"

# Default repo root — the folder that CONTAINS the importable `sahtdemucs`
# package (i.e. it has a subfolder `sahtdemucs/` with `__init__.py`, NOT the
# `runs/` output folder). Override with --repo-path if running from a
# different machine / checkout.
DEFAULT_REPO_PATH = r"H:\Il mio Drive\Polimi\PhD\Progetto di Ricerca\MSS & SSL\sahtdemucs"

_arg_parser = argparse.ArgumentParser(
    description="Export a SAHTDemucs spatial_modules checkpoint to TorchScript.")
_arg_parser.add_argument(
    "run_ts", nargs="?", default=DEFAULT_RUN_TS,
    help=f"Timestamp of the run to export (default: {DEFAULT_RUN_TS}).")
_arg_parser.add_argument(
    "--run_ts", dest="run_ts_flag", default=None,
    help="Alternative way to pass the timestamp (overrides the positional argument).")
_arg_parser.add_argument(
    "--repo-path", dest="repo_path", default=DEFAULT_REPO_PATH,
    help=f"Root of the sahtdemucs repository, containing the importable "
         f"`sahtdemucs` package and the `runs/` folder "
         f"(default: {DEFAULT_REPO_PATH})")
_args = _arg_parser.parse_args()

_run_ts = _args.run_ts_flag if _args.run_ts_flag is not None else _args.run_ts

# Suppress TracerWarning messages emitted by torch.jit.trace when it
# encounters Python control flow based on tensor values (asserts, shape
# checks, etc. inside demucs/hdemucs.py, htdemucs.py, transformer.py,
# cue_module.py). These are expected and harmless as long as
# EXAMPLE_LENGTH_SAMPLES matches the actual input length used at inference
# time (see comment on EXAMPLE_LENGTH_SAMPLES below) — the traced graph
# bakes in the branch taken for that specific length.
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

# Suppress the FutureWarning about torch.load's default weights_only=False.
# We trust CHECKPOINT_PATH_SPATIAL (our own training output) — this only
# affects the spatial checkpoint load below (the backbone is loaded via
# demucs.pretrained.get_model(), not torch.load directly).
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# ==============================================================================
# CONFIG — edit these paths and settings for your setup
# ==============================================================================

# Root of the sahtdemucs repository — the folder that CONTAINS the importable
# `sahtdemucs` package (e.g. it has a subfolder `sahtdemucs/` with
# `__init__.py`, NOT the `runs/` output folder). Set via --repo-path
# (default: DEFAULT_REPO_PATH above).
SAHTDEMUCS_REPO_PATH = Path(_args.repo_path)

# Checkpoints directory (commonly used as a base for the paths below)
RUNS_DIR = SAHTDEMUCS_REPO_PATH / "runs"

# Backbone is loaded via demucs.pretrained.get_model(), not a file.
# Set this to the model name you used as the base during training, e.g.:
#   "htdemucs"     - standard 4-source HTDemucs (most common)
#   "htdemucs_ft"  - fine-tuned variant (4 separate models, slower to load)
#   "htdemucs_6s"  - 6-source variant (guitar, piano added)
# Check your training script for the exact name passed to get_model().
BACKBONE_MODEL_NAME = "htdemucs"

# Checkpoint path — spatial cue module weights
CHECKPOINT_PATH_SPATIAL = RUNS_DIR / f"spatial_modules_{_run_ts}.pt"

# SAHTDemucs constructor hyperparameters — copy these EXACTLY
# from your training script (the values used when the spatial_modules
# checkpoint was produced). A mismatch here loads weights into the wrong
# tensors (silently, if shapes happen to match) or raises shape errors.
#
# Per sahtdemucs/model.py SAHTDemucs.__init__ signature:
SPATIAL_ARCH      = "cnn2d" # "cnn1d" (default) or "cnn2d"
N_FFT             = 2048    # n_fft (model.py default 2048)
HOP_LEN           = 512     # hop_length (model.py default 512)
N_BANDS           = 64      # n_bands (model.py default 32)
ILD_SCALE         = 15.0    # ild_scale (model.py default 6.0)
BAND_SCALE        = "mel"   # "linear" (default) or "mel"
USE_GLOBAL_BRANCH = True    # False (default)

# Output path for the traced model
OUTPUT_PATH = RUNS_DIR / f"sahtdemucs_{_run_ts}.pt"

# Export method: "trace" is more robust for CNNs with fixed-size operations,
# "script" supports more dynamic control flow but requires the model code to
# be TorchScript-compatible (no unsupported Python constructs).
# Start with "trace"; switch to "script" only if tracing fails or if your
# model has data-dependent control flow that tracing can't capture correctly.
EXPORT_METHOD = "trace"

# Example input length for tracing, in samples.
# This should match (or be a representative multiple of) the chunk size used
# by the VST plugin. HTDemucs is convolutional and handles variable lengths,
# but tracing fixes the shape used during trace — test with the actual
# plugin chunk size to be safe (e.g. 4 seconds @ 44100 Hz = 176400 samples).
EXAMPLE_LENGTH_SAMPLES = 44100 * 4  # 4 seconds

# Device for export. CPU is recommended even if you trained on GPU — the VST
# plugin will run inference on CPU (LibTorch CPU build is much simpler to
# integrate into JUCE/CMake than the CUDA build, and avoids requiring an
# NVIDIA GPU on the end user's machine). If you need GPU inference in the
# plugin, export with device="cuda" and link the CUDA LibTorch build instead.
DEVICE = "cpu"

# Print detailed type/shape information for the model's raw output (before
# unwrapping tuples/dicts in ExportWrapper). Set to True on the first run to
# discover the correct output index/key, then set back to False once
# ExportWrapper.forward() is configured correctly.
VERBOSE_SHAPES = False

# ==============================================================================
# Model loading
# ==============================================================================

import sys
sys.path.insert(0, str(SAHTDEMUCS_REPO_PATH))

# Import path to your model class.Adjust this import to match your
# repository's module structure.
from sahtdemucs.model import SAHTDemucs

def build_model() -> nn.Module:
    """
    Instantiate SA-HTDemucs by wrapping a pretrained HTDemucs backbone,
    exactly as in the training script:

        bag  = get_model("htdemucs")
        base = bag.models[0] if hasattr(bag, "models") else bag
        model = SAHTDemucs(base, spatial_arch=..., sources=base.sources, ...)

    Because `base` is the pretrained backbone INSTANCE (not just the
    architecture), its weights are already loaded at construction time.
    Only the spatial cue module weights need to be loaded afterwards.
    """
    from demucs.pretrained import get_model as get_demucs_model

    print(f"  Loading pretrained backbone '{BACKBONE_MODEL_NAME}' "
          f"via demucs.pretrained.get_model() ...")
    print(f"  (first run downloads and caches the checkpoint, "
          f"subsequent runs are instant)")

    bag = get_demucs_model(BACKBONE_MODEL_NAME)
    base = bag.models[0] if hasattr(bag, "models") else bag

    if hasattr(bag, "models") and len(bag.models) > 1:
        print(f"  [WARN] '{BACKBONE_MODEL_NAME}' is a BagOfModels with "
              f"{len(bag.models)} members; using member 0 (base.models[0]), "
              f"matching the training script convention.")

    model = SAHTDemucs(base,
                       spatial_arch=SPATIAL_ARCH,
                       sources=base.sources,
                       n_fft=N_FFT,
                       hop_length=HOP_LEN,
                       n_bands=N_BANDS,
                       ild_scale=ILD_SCALE,
                       band_scale=BAND_SCALE,
                       sample_rate=base.samplerate,
                       use_gb=USE_GLOBAL_BRANCH)
    return model

def load_weights(model: nn.Module):
    """
    Load the spatial cue module weights from CHECKPOINT_PATH_SPATIAL on top
    of `model` (whose HTDemucs backbone weights are already loaded — see
    build_model()).
    """
    # ---------------------------------------------------------------
    # Backbone weights are ALREADY loaded — `model` was built in
    # build_model() by wrapping a pretrained HTDemucs instance (`base`),
    # so model.state_dict() already contains correct backbone values.
    # We only need to load the spatial cue module weights from your
    # checkpoint on top.
    # ---------------------------------------------------------------
    spatial_ckpt = torch.load(CHECKPOINT_PATH_SPATIAL, map_location=DEVICE, weights_only=True)
    spatial_state = spatial_ckpt.get("state_dict", spatial_ckpt)

    # ── Key check ─────────────────────────────────────────────────────────
    # Print a small sample of keys to help verify that the spatial
    # checkpoint's key names line up with model.state_dict() — i.e. that
    # load_state_dict will actually overwrite the right tensors.
    model_keys = set(model.state_dict().keys())
    print(f"  Model has {len(model_keys)} parameter/buffer tensors total.")
    print(f"  Spatial checkpoint provides {len(spatial_state)} tensors, "
          f"e.g.: {list(spatial_state.keys())[:5]}")
    print(f"  Model's own keys, e.g.: {list(model_keys)[:5]}")

    # If the spatial checkpoint's keys need a prefix to match
    # the submodule name inside SAHTDemucs (e.g. your model has
    # `self.spatial_cue_module = ...`, so keys need a
    # "spatial_cue_module." prefix), remap here
    spatial_state = {f"spatial_modules.{k}": v for k, v in spatial_state.items()}

    missing, unexpected = model.load_state_dict(spatial_state, strict=False)

    # `missing` here is EXPECTED to be large: it's every backbone tensor not
    # present in the spatial checkpoint (those were already loaded via `base`
    # in build_model() and remain correctly populated — load_state_dict
    # (strict=False) does not zero them out).
    # What matters is `unexpected`: any spatial checkpoint key that doesn't
    # match a tensor name in `model` was silently NOT loaded.
    print(f"  {len(spatial_state) - len(unexpected)} / {len(spatial_state)} "
          f"spatial checkpoint tensors matched a model tensor by name.")

    if unexpected:
        print(f"[WARN] {len(unexpected)} spatial checkpoint key(s) did NOT "
              f"match any tensor in the model (NOT loaded): "
              f"{unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
        print(f"[INFO] These likely need a prefix remap — see the "
              f">>> CUSTOMISE comment above. Compare these names against "
              f"the model's own keys printed above to find the correct prefix.")
    else:
        print(f"  OK - all spatial checkpoint tensors were loaded.")

    return model

# ==============================================================================
# Export wrapper
# ==============================================================================

class ExportWrapper(nn.Module):
    """
    Thin wrapper around the trained model to:
      1. Fix the forward() signature to a single Tensor in, single Tensor out
         (TorchScript / LibTorch work best with simple tensor I/O, not dicts
         or custom objects).
      2. Set eval() mode and disable gradient tracking permanently.
      3. Optionally apply any pre/post-processing (normalisation, etc.) that
         your training pipeline expects, so the C++ side only has to feed raw
         audio samples.

    If your model's forward() already returns a plain Tensor of
    shape [batch, 4, 2, T], you can remove this wrapper entirely and trace
    `model` directly. The wrapper is provided in case your forward() returns
    a dict, a namedtuple, or requires extra arguments.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: [batch, 2, T] stereo waveform, float32, range [-1, 1]

        Returns:
            stems: [batch, S, 2, T] — ILD-corrected separated sources
                   (the spatially-corrected `estimates` output of
                   SAHTDemucs.forward(), which returns
                   (estimates, raw_estimates, deltas) — see model.py).

        Source order S follows `base.sources` from the HTDemucs backbone
        (typically ["drums", "bass", "other", "vocals"] for "htdemucs").
        """
        out = self.model(mix)

        # SAHTDemucs.forward() returns a 3-tuple:
        #   out[0] = estimates     (B, S, 2, T)  <- ILD-corrected (what we want)
        #   out[1] = raw_estimates (B, S, 2, T)  <- pre-correction HTDemucs output
        #   out[2] = deltas        list[Tensor]  <- per-source CNN correction maps
        #
        # We keep only `estimates`. `raw_estimates` and `deltas` are still
        # computed during the traced forward pass (they're cheap byproducts
        # of the same spatial-module calls), but are dropped from the
        # TorchScript output.
        if VERBOSE_SHAPES:
            print(f"  [DEBUG] model output is a {type(out).__name__} "
                  f"with {len(out)} element(s):")
            for i, el in enumerate(out):
                if isinstance(el, torch.Tensor):
                    print(f"    [{i}] Tensor, shape={tuple(el.shape)}, dtype={el.dtype}")
                elif isinstance(el, (list, tuple)):
                    shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                              for t in el]
                    print(f"    [{i}] {type(el).__name__} of {len(el)}: {shapes}")
                else:
                    print(f"    [{i}] {type(el).__name__}: {el}")

        return out[0]


# ==============================================================================
# Main export routine
# ==============================================================================

def main():
    print(f"Building model...")
    model = build_model()

    # model.sources comes from base.sources (HTDemucs backbone), forwarded
    # through SAHTDemucs.__getattr__. This is the stem order of the output
    # tensor's S dimension — must match `busToStem` in PluginProcessor.h.
    sources = list (model.sources)
    print(f"Model sources (output dim 1 order): {sources}")

    print(f"Loading weights...")
    model = load_weights(model)

    model = model.to(DEVICE)
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    # Example input for tracing
    example_input = torch.randn(1, 2, EXAMPLE_LENGTH_SAMPLES, device=DEVICE)

    print(f"Running a test forward pass with input shape {tuple(example_input.shape)}...")
    with torch.no_grad():
        test_output = wrapper(example_input)
    print(f"  -> output shape: {tuple(test_output.shape)}")

    expected_shape = (1, len(sources), 2, EXAMPLE_LENGTH_SAMPLES)
    if tuple(test_output.shape) != expected_shape:
        print(f"[WARN] Output shape {tuple(test_output.shape)} does not match "
              f"expected {expected_shape}. Check ExportWrapper.forward() and "
              f"the source ordering assumption in the C++ plugin.")
    else:
        print(f"  Shape OK. Reminder: PluginProcessor.h's `busToStem` array "
              f"must map bus order -> index in {sources}.")

    print(f"Exporting via {EXPORT_METHOD}...")
    if EXPORT_METHOD == "trace":
        traced = torch.jit.trace(wrapper, example_input, strict=False)
    elif EXPORT_METHOD == "script":
        traced = torch.jit.script(wrapper)
    else:
        raise ValueError(f"Unknown EXPORT_METHOD: {EXPORT_METHOD!r}")

    # NOTE: torch.jit.optimize_for_inference() is intentionally NOT used here.
    # It applies a freezing pass that, in several PyTorch versions, produces
    # prim::Constant nodes with Tensor values that fail to deserialize
    # ("RuntimeError: required keyword attribute 'value' is undefined" on
    # torch.jit.load). The plain traced module is fully usable from LibTorch
    # via torch::jit::load() + module.forward(); if desired, call
    # torch::jit::optimize_for_inference() on the C++ side AFTER loading
    # (it does not need to survive a save/load round trip).
    traced.save(OUTPUT_PATH)
    print(f"Saved TorchScript model to: {OUTPUT_PATH}")

    # ── Sanity check: reload and compare ─────────────────────────────────────
    print("Reloading exported model for a sanity check...")
    reloaded = torch.jit.load(OUTPUT_PATH, map_location=DEVICE)
    with torch.no_grad():
        reloaded_output = reloaded(example_input)

    max_diff = (reloaded_output - test_output).abs().max().item()
    print(f"Max abs difference (original vs reloaded): {max_diff:.3e}")
    if max_diff > 1e-4:
        print("[WARN] Difference is larger than expected — verify the export.")
    else:
        print("OK - exported model matches the original.")

if __name__ == "__main__":
    main()
