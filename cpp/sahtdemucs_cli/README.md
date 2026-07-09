# sahtdemucs_cli

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![LibTorch CPU](https://img.shields.io/badge/LibTorch-CPU-orange.svg)](https://pytorch.org/cppdocs/)

Standalone C++/LibTorch command-line harness for the **[SA-HTDemucs](../../README.md)** model. It loads a
TorchScript checkpoint exported by [`sahtdemucs/export_torchscript.py`](../../sahtdemucs/export_torchscript.py),
runs chunked overlap-add inference on a stereo WAV file, and writes the four separated stems to disk.

Its purpose is **validation, not production**: it reproduces the exact chunking strategy used by the JUCE/VST plugin
(`PluginProcessor.h`) so that the *exported model* and the *plugin's real-time chunking* can be tested end-to-end —
outside a DAW, with plain WAV I/O and a measurable per-chunk timing report — before they are wired into the plugin.

```
export_torchscript.py  ──►  sahtdemucs_<run_ts>.pt  ──►  sahtdemucs_cli  ──►  drums/bass/other/vocals.wav
   (Python / PyTorch)        (TorchScript trace)         (C++ / LibTorch)        (per-stem WAV @ 44.1 kHz)
                                                              ▲
                                                              └── same chunking as the VST plugin
```

---

## How it differs from `model.separate()`

The Python `SAHTDemucs.separate()` uses `demucs.apply.apply_model` for the HTDemucs backbone, which performs its **own**
internal overlap-add chunking (chunk ≈ `base.segment`, ~7.8 s for `htdemucs`), then applies the spatial correction
**once** on the full-length signal.

The TorchScript export wraps `model.forward(mix)` directly: it calls the backbone on whatever-length input it is given —
**without** `apply_model`'s internal chunking — and applies the spatial correction on that same chunk. This CLI therefore
performs its **own** chunked overlap-add on top of the traced `forward()`, mirroring what the plugin does in real time.

Results are very close to, but **not bit-identical** with, `model.separate()`. This tool validates the **exported model**
and the **plugin's chunking strategy**, not a numerically identical reproduction of the notebook's evaluation.

> ### ⚠️ Critical: chunk length must match the trace
> `export_torchscript.py` traces the model with a **fixed** input length (`EXAMPLE_LENGTH_SAMPLES`). Several branches
> inside HTDemucs / HDemucs / `transformer.py` depend on tensor shapes and get baked into the traced graph as constants
> (hence the `TracerWarning`s during export). Running the traced model on a **different** input length may silently take
> the wrong branch and produce incorrect output.
>
> `kChunkSeconds` in [`sahtdemucs_cli.cpp`](sahtdemucs_cli.cpp) **must** satisfy
> `kChunkSeconds × 44100 == EXAMPLE_LENGTH_SAMPLES`. As exported so far: `EXAMPLE_LENGTH_SAMPLES = 44100 × 4 = 176400`
> (4 s), so `kChunkSeconds = 4.0`. If you re-export with a different length, update `kChunkSeconds` accordingly.

---

## Usage

```text
sahtdemucs_cli <model.pt> <input.wav> <output_dir>
```

| Argument       | Description                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------|
| `<model.pt>`   | TorchScript checkpoint from `export_torchscript.py` (e.g. `runs/sahtdemucs_20260531_155238.pt`). |
| `<input.wav>`  | Input mixture. **Must already be 44.1 kHz** — no resampling is performed.                         |
| `<output_dir>` | Created if missing. Receives `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`.                  |

Example:

```powershell
.\x64\Release\sahtdemucs_cli.exe `
    "..\..\runs\sahtdemucs_20260531_155238.pt" `
    "..\..\docs\song1\mixture.wav" `
    ".\output"
```

The output stems follow the model's source order (`base.sources` for `htdemucs`): **`drums`, `bass`, `other`,
`vocals`** — matching the notebook's `SOURCES` list and file names, for direct diffing against the Python pipeline's
output.

### Output WAV format

Stems are written as **32-bit IEEE-float** WAV (canonical RIFF/WAVE), chosen over 16-bit PCM to avoid
clipping/quantisation when comparing against `torchaudio.save(..., dtype=float32)`. Values outside `[-1, +1]` are
written as-is (float WAV has no hard clipping).

### Console output

The tool prints input properties, the chunk plan (chunk / overlap / hop in samples), per-chunk inference time, and a
**real-time verdict**: it compares average inference time per chunk against the hop duration and reports whether the
exported model is fast enough to keep up in the plugin (`avg < hop`) or would cause underruns.

---

## Pipeline details

1. **Load** the TorchScript module (`torch::jit::load`, `eval`, `NoGradGuard`).
2. **Read** the input WAV via the dependency-free [`WavIO.h`](WavIO.h) (PCM 16/24/32-bit or 32/64-bit float in,
   float out). Mono is duplicated to stereo; >2 channels are truncated to the first two — matching `load_stem()` in the
   notebook.
3. **Chunk** the signal into `kChunkSeconds` windows with `kOverlapRatio` (25 %) overlap, zero-padding the tail so the
   last chunk is full-length.
4. **Infer** each chunk through `forward()` → `[1, S, 2, T]`.
5. **Overlap-add** with a linear crossfade ramp over the overlap region (identical to the plugin's `PluginProcessor.h`).
6. **Trim** to the original length and **write** one WAV per stem.

### Configuration constants

Defined at the top of [`sahtdemucs_cli.cpp`](sahtdemucs_cli.cpp) — **must match** `export_torchscript.py` and
`PluginProcessor.h`:

| Constant              | Value     | Meaning                                                          |
|-----------------------|-----------|------------------------------------------------------------------|
| `kExpectedSampleRate` | `44100`   | Model sample rate (`base.samplerate` for `htdemucs`).            |
| `kChunkSeconds`       | `4.0`     | Chunk length; `× 44100` must equal `EXAMPLE_LENGTH_SAMPLES`.     |
| `kOverlapRatio`       | `0.25`    | Crossfade overlap fraction between consecutive chunks.           |
| `kNumStems`           | `4`       | Number of separated sources.                                     |
| `kStemNames`          | drums, bass, other, vocals | Output dim-1 order (`base.sources`).            |

---

## Building

The project targets **Visual Studio 2022** (toolset `v143`, C++17) and links against a **CPU LibTorch** build. Only the
**x64** configurations are set up for LibTorch (Win32 is left unconfigured).

### Prerequisites

- Visual Studio 2022 with the *Desktop development with C++* workload.
- [LibTorch (CPU, Release)](https://pytorch.org/get-started/locally/) extracted to `C:\Librerie\cpp\libtorch`.

The `.vcxproj` references LibTorch by absolute path:

| Setting                  | Value                                                                                          |
|--------------------------|------------------------------------------------------------------------------------------------|
| Include directories      | `C:\Librerie\cpp\libtorch\include`, `…\include\torch\csrc\api\include`                          |
| Library directories      | `C:\Librerie\cpp\libtorch\lib`                                                                  |
| Linker dependencies      | `torch.lib`, `torch_cpu.lib`, `c10.lib`                                                         |
| Post-build event         | `xcopy` of `…\libtorch\lib\*.dll` into the output directory                                     |

> If your LibTorch lives elsewhere, update the *Additional Include/Library Directories* and the post-build `xcopy` path
> in [`sahtdemucs_cli.vcxproj`](sahtdemucs_cli.vcxproj) (or override them via a property sheet).

### Build & run

```powershell
# From a Developer PowerShell for VS 2022
msbuild sahtdemucs_cli.sln /p:Configuration=Release /p:Platform=x64

# The post-build step copies the LibTorch DLLs next to the .exe
.\x64\Release\sahtdemucs_cli.exe <model.pt> <input.wav> <output_dir>
```

Or open `sahtdemucs_cli.sln` in Visual Studio and build the **Release | x64** configuration.

---

## Files

| File                                                 | Role                                                              |
|------------------------------------------------------|------------------------------------------------------------------|
| [`sahtdemucs_cli.cpp`](sahtdemucs_cli.cpp)           | Entry point: load, chunk, infer, overlap-add, write.             |
| [`WavIO.h`](WavIO.h)                                 | Minimal, dependency-free WAV reader/writer (header-only).        |
| `sahtdemucs_cli.vcxproj` / `.sln`                    | Visual Studio 2022 project and solution.                         |

---

## Related

- [SA-HTDemucs — main README](../../README.md) — model architecture, spatial cue modules, training, dataset.
- [`sahtdemucs/export_torchscript.py`](../../sahtdemucs/export_torchscript.py) — produces the `.pt` consumed here.
  The `SPATIAL_ARCH`, `N_FFT`, `N_BANDS`, `ILD_SCALE`, … constants there define the exported model; the source order it
  prints (`base.sources`) must match `kStemNames` above.

---

<sub>Part of the SA-HTDemucs research project — Politecnico di Milano (ISPL).</sub>
