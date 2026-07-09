/**
 * sahtdemucs_cli.cpp
 * ===================
 * Standalone command-line equivalent of the notebook's testset inference
 * loop:
 *
 *   wav   = load_stem(mix_path)                 # (2, T) @ SAMPLE_RATE
 *   stems = model.separate(wav, progress=False) # (S, 2, T)
 *   for i, src in enumerate(SOURCES):
 *       torchaudio.save(f"{src}.wav", stems[i], SAMPLE_RATE)
 *
 * -- Important difference from model.separate() --
 * The Python `model.separate()` uses `demucs.apply.apply_model` for the
 * HTDemucs backbone, which performs its OWN overlap-add chunking internally
 * (chunk length = `base.segment`, ~7.8s for "htdemucs"), THEN applies the
 * spatial correction modules ONCE on the full-length signal.
 *
 * The TorchScript export (sahtdemucs_traced_<run_ts>.pt) wraps
 * `model.forward(mix)` directly — it calls the backbone on whatever-length
 * input it's given, WITHOUT apply_model's internal chunking, and applies
 * spatial correction on that same chunk.
 *
 * This CLI tool therefore performs its own chunked overlap-add on top of 
 * the traced `forward()`. Results will be very close to, but not bit-identical
 * with, `model.separate()` - this tool validates the EXPORTED MODEL and the 
 * PLUGIN'S CHUNKING STRATEGY, not a numerically identical reproduction of the
 * notebook's evaluation.
 *
 * --- CRITICAL: chunk length must match the trace ---
 * export_torchscript.py traces the model with a FIXED input length
 * (EXAMPLE_LENGTH_SAMPLES). Several branches inside HTDemucs/Hdemucs/
 * transformer.py are based on tensor shapes and get baked into the traced
 * graph as constants (see the TracerWarnings during export). Running the
 * traced model on a DIFFERENT input length than EXAMPLE_LENGTH_SAMPLES may
 * silently take the wrong branch and produce incorrect output.
 *
 * kChunkSamples below MUST equal EXAMPLE_LENGTH_SAMPLES used at export time.
 * As exported so far: EXAMPLE_LENGTH_SAMPLES = 44100 * 4 = 176400 (4 seconds).
 * If you re-export with a different length, update kChunkSeconds accordingly.
 *
 * --- Usage ---
 *   sahtdemucs_cli <model.pt> <input.wav> <output_dir>
 *
 * Produces in <output_dir>: drums.wav, bass.wav, other.wav, vocals.wav
 * (model.sources order — matches the notebook's SOURCES list and file names,
 * for direct diffing against the Python pipeline's output).
 */

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "WavIO.h"

namespace fs = std::filesystem;

// ============================================================================
// Configuration — MUST match export_torchscript.py and PluginProcessor.h
// ============================================================================

/// Sample rate expected by the model (base.samplerate for "htdemucs" = 44100).
/// The input WAV must already be at this rate — no resampling is performed.
constexpr int kExpectedSampleRate = 44100;

/// Chunk length in seconds. MUST equal EXAMPLE_LENGTH_SAMPLES / sample_rate
/// from export_torchscript.py (as exported so far: 176400 / 44100 = 4.0s).
constexpr double kChunkSeconds = 4.0;

/// Overlap ratio for crossfade between consecutive chunks (same as
/// PluginProcessor.h: 0.25 -> 25% of the chunk is crossfaded).
constexpr double kOverlapRatio = 0.25;

/// Number of separated stems and their order in the model's output tensor
/// (dim 1 of the (1, S, 2, T) output) — matches base.sources for "htdemucs".
constexpr int kNumStems = 4;
constexpr std::array<const char*, kNumStems> kStemNames
{ "drums", "bass", "other", "vocals" };

// ============================================================================

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model.pt> <input.wav> <output_dir>\n";
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string inputPath = argv[2];
    const std::string outputDir = argv[3];

    // 1. Load the TorchScript model
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(modelPath);
        module.eval();
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error loading model '" << modelPath << "': " << e.what() << "\n";
        return 1;
    }
    torch::NoGradGuard noGrad;

    // 2. Load input WAV
    WavData wav;
    try
    {
        wav = readWav(inputPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error reading WAV '" << inputPath << "': " << e.what() << "\n";
        return 1;
    }

    std::cout << "Input: " << inputPath << "\n"
        << "  sample rate : " << wav.sampleRate << " Hz\n"
        << "  channels    : " << wav.numChannels << "\n"
        << "  length      : " << wav.channels[0].size() << " samples ("
        << static_cast<double> (wav.channels[0].size()) / wav.sampleRate << " s)\n";

    if (wav.sampleRate != kExpectedSampleRate)
    {
        std::cerr << "[WARN] input sample rate (" << wav.sampleRate
            << ") != expected model sample rate (" << kExpectedSampleRate
            << "). No resampling is performed by this tool — results will "
            "be incorrect. Resample the input to "
            << kExpectedSampleRate << " Hz first (e.g. with ffmpeg/sox).\n";
    }

    // Mono -> duplicate to stereo; >2 channels -> take first two
    // (matches load_stem() in the notebook).
    std::vector<float> L, R;
    if (wav.numChannels == 1)
    {
        L = wav.channels[0];
        R = wav.channels[0];
    }
    else
    {
        L = wav.channels[0];
        R = wav.channels[1];
    }

    const int64_t numSamples = static_cast<int64_t> (L.size());

    // 3. Chunking setup
    const int64_t chunkSamples = static_cast<int64_t> (std::llround(kChunkSeconds * kExpectedSampleRate));
    const int64_t overlapSamples = static_cast<int64_t> (std::llround(chunkSamples * kOverlapRatio));
    const int64_t hopSamples = chunkSamples - overlapSamples;

    std::cout << "Chunking:\n"
        << "  chunk   : " << chunkSamples << " samples (" << kChunkSeconds << " s)\n"
        << "  overlap : " << overlapSamples << " samples\n"
        << "  hop     : " << hopSamples << " samples\n";

    // Number of chunks needed to cover numSamples with the given hop, such
    // that the LAST chunk's end (chunkStart + chunkSamples) >= numSamples.
    //   chunkStart_k = k * hop
    //   need: (numChunks-1)*hop + chunkSamples >= numSamples
    //   =>    numChunks >= (numSamples - chunkSamples) / hop + 1
    int64_t numChunks = 1;
    if (numSamples > chunkSamples)
        numChunks = 1 + static_cast<int64_t> (std::ceil(
            static_cast<double> (numSamples - chunkSamples) / static_cast<double> (hopSamples)));

    const int64_t paddedLength = (numChunks - 1) * hopSamples + chunkSamples;

    std::cout << "  numChunks = " << numChunks
        << "  (padded length = " << paddedLength << " samples, "
        << "input = " << numSamples << " samples)\n";

    // Zero-pad input to paddedLength
    L.resize(static_cast<size_t> (paddedLength), 0.f);
    R.resize(static_cast<size_t> (paddedLength), 0.f);

    // 4. Output buffers (one per stem, L/R), overlap-add crossfaded ???????
    // outputL[s][i], outputR[s][i] — sized to paddedLength, trimmed to
    // numSamples before writing.
    std::array<std::vector<float>, kNumStems> outputL, outputR;
    for (int s = 0; s < kNumStems; ++s)
    {
        outputL[s].assign(static_cast<size_t>(paddedLength), 0.f);
        outputR[s].assign(static_cast<size_t>(paddedLength), 0.f);
    }

    // Linear crossfade ramp over the overlap region.
    // fadeIn[i] = i / overlapSamples, used to blend the new chunk's start
    // with the previous chunk's tail (same as PluginProcessor.h).
    std::vector<float> fadeIn(static_cast<size_t>(overlapSamples));
    for (int64_t i = 0; i < overlapSamples; ++i)
        fadeIn[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(overlapSamples);

    // 5. Process each chunk
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    double totalInferenceMs = 0.0;

    for (int64_t c = 0; c < numChunks; ++c)
    {
        const int64_t chunkStart = c * hopSamples;

        std::cout << "Chunk " << (c + 1) << "/" << numChunks
            << "  [" << chunkStart << ", " << (chunkStart + chunkSamples) << ")"
            << " ... " << std::flush;

        // Build input tensor [1, 2, chunkSamples]
        torch::Tensor left = torch::from_blob(L.data() + chunkStart, { 1, 1, chunkSamples }, options);
        torch::Tensor right = torch::from_blob(R.data() + chunkStart, { 1, 1, chunkSamples }, options);
        torch::Tensor input = torch::cat({ left, right }, /*dim=*/1).clone();  // [1, 2, T], owns its memory

        // Run inference
        const auto t0 = std::chrono::high_resolution_clock::now();
        torch::Tensor output = module.forward({ input }).toTensor();  // [1, S, 2, T]
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalInferenceMs += ms;
        std::cout << ms << " ms\n";

        output = output.contiguous();
        const float* outData = output.data_ptr<float>();

        // ?? Crossfade into output buffers ???????????????????????????????????
        // Layout of `output`: [1, S, 2, T] -> index (s, ch, t) = outData[(s*2+ch)*T + t]
        const int64_t T = chunkSamples;

        for (int64_t r = 0; r < chunkSamples; ++r)
        {
            const int64_t absIdx = chunkStart + r;
            const bool inOverlap = (r < overlapSamples) && (c > 0);

            for (int s = 0; s < kNumStems; ++s)
            {
                const float newL = outData[(s * 2 + 0) * T + r];
                const float newR = outData[(s * 2 + 1) * T + r];

                if (inOverlap)
                {
                    const float a = fadeIn[static_cast<size_t>(r)];
                    outputL[s][static_cast<size_t>(absIdx)] =
                        outputL[s][static_cast<size_t>(absIdx)] * (1.f - a) + newL * a;
                    outputR[s][static_cast<size_t>(absIdx)] =
                        outputR[s][static_cast<size_t> (absIdx)] * (1.f - a) + newR * a;
                }
                else
                {
                    outputL[s][static_cast<size_t> (absIdx)] = newL;
                    outputR[s][static_cast<size_t> (absIdx)] = newR;
                }
            }
        }
    }

    std::cout << "Total inference time: " << totalInferenceMs << " ms over "
        << numChunks << " chunk(s), avg " << (totalInferenceMs / static_cast<double> (numChunks))
        << " ms/chunk\n";

    const double hopMs = (static_cast<double> (hopSamples) / kExpectedSampleRate) * 1000.0;
    std::cout << "Hop duration: " << hopMs << " ms  -- "
        << ((totalInferenceMs / static_cast<double> (numChunks)) < hopMs
            ? "real-time capable (avg < hop)"
            : "TOO SLOW for real-time (avg >= hop) -- underruns expected in the plugin")
        << "\n";

    // ?? 6. Trim to original length and write output WAVs ????????????????????
    fs::create_directories(outputDir);

    for (int s = 0; s < kNumStems; ++s)
    {
        outputL[s].resize(static_cast<size_t>(numSamples));
        outputR[s].resize(static_cast<size_t>(numSamples));

        const std::string outPath = (fs::path(outputDir) / (std::string(kStemNames[s]) + ".wav")).string();
        writeWavFloat(outPath, { outputL[s], outputR[s] }, kExpectedSampleRate);
        std::cout << "Wrote: " << outPath << "\n";
    }

    return 0;
}