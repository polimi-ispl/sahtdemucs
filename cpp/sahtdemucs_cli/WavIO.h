#pragma once
/**
 * WavIO.h
 * =======
 * Minimal, dependency-free WAV file reader/writer.
 *
 * Supports reading:
 *   - PCM 16/24/32-bit integer (audioFormat = 1)
 *   - IEEE float 32-bit (audioFormat = 3)
 *   - Any channel count (mono/stereo handled specially by the caller)
 *
 * Writes:
 *   - IEEE float 32-bit WAV (audioFormat = 3), canonical RIFF/WAVE layout.
 *     Chosen over 16-bit PCM to avoid clipping/quantisation when comparing
 *     against Python's torchaudio.save(..., dtype=float32) output.
 *
 * This is intentionally simple — it does NOT handle extensible fmt chunks
 * (WAVE_FORMAT_EXTENSIBLE), non-PCM compressed formats, or chunks other than
 * "fmt " and "data". Sufficient for WAV files produced by torchaudio,
 * soundfile, ffmpeg, Audacity, Reaper, etc.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct WavData
{
    int sampleRate{ 0 };
    int numChannels{ 0 };
    /// channels[c][i] = sample i of channel c, normalised to [-1, +1]
    std::vector<std::vector<float>> channels;
};

namespace wavio_detail
{
    inline uint32_t readU32(std::ifstream& f)
    {
        uint8_t b[4];
        f.read(reinterpret_cast<char*> (b), 4);
        return static_cast<uint32_t> (b[0]) | (static_cast<uint32_t> (b[1]) << 8)
            | (static_cast<uint32_t> (b[2]) << 16) | (static_cast<uint32_t> (b[3]) << 24);
    }

    inline uint16_t readU16(std::ifstream& f)
    {
        uint8_t b[2];
        f.read(reinterpret_cast<char*> (b), 2);
        return static_cast<uint16_t> (b[0]) | (static_cast<uint16_t> (b[1]) << 8);
    }

    inline void writeU32(std::ofstream& f, uint32_t v)
    {
        uint8_t b[4] = { static_cast<uint8_t> (v), static_cast<uint8_t> (v >> 8),
                         static_cast<uint8_t> (v >> 16), static_cast<uint8_t> (v >> 24) };
        f.write(reinterpret_cast<const char*> (b), 4);
    }

    inline void writeU16(std::ofstream& f, uint16_t v)
    {
        uint8_t b[2] = { static_cast<uint8_t> (v), static_cast<uint8_t> (v >> 8) };
        f.write(reinterpret_cast<const char*> (b), 2);
    }
}

/**
 * Reads a WAV file into a WavData struct. Samples are normalised to
 * [-1, +1] regardless of the source bit depth / format.
 *
 * @throws std::runtime_error on malformed/unsupported files.
 */
inline WavData readWav(const std::string& path)
{
    using namespace wavio_detail;

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("WavIO: cannot open file: " + path);

    char tag[4];
    f.read(tag, 4);
    if (std::strncmp(tag, "RIFF", 4) != 0)
        throw std::runtime_error("WavIO: not a RIFF file: " + path);

    readU32(f); // overall file size (ignored)

    f.read(tag, 4);
    if (std::strncmp(tag, "WAVE", 4) != 0)
        throw std::runtime_error("WavIO: not a WAVE file: " + path);

    WavData out;
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    bool haveFmt = false;

    while (f.good() && !f.eof())
    {
        char chunkId[4];
        f.read(chunkId, 4);
        if (f.gcount() < 4) break;

        uint32_t chunkSize = readU32(f);
        std::streampos chunkStart = f.tellg();

        if (std::strncmp(chunkId, "fmt ", 4) == 0)
        {
            audioFormat = readU16(f);
            numChannels = readU16(f);
            sampleRate = readU32(f);
            readU32(f);              // byte rate (ignored)
            readU16(f);              // block align (ignored)
            bitsPerSample = readU16(f);
            haveFmt = true;
        }
        else if (std::strncmp(chunkId, "data", 4) == 0)
        {
            if (!haveFmt)
                throw std::runtime_error("WavIO: 'data' chunk before 'fmt ': " + path);

            const int bytesPerSample = bitsPerSample / 8;
            if (bytesPerSample <= 0 || numChannels == 0)
                throw std::runtime_error("WavIO: invalid fmt chunk: " + path);

            const uint32_t numFrames = chunkSize / (static_cast<uint32_t> (bytesPerSample) * numChannels);

            out.sampleRate = static_cast<int> (sampleRate);
            out.numChannels = static_cast<int> (numChannels);
            out.channels.assign(numChannels, std::vector<float>(numFrames));

            std::vector<uint8_t> raw(chunkSize);
            f.read(reinterpret_cast<char*> (raw.data()), chunkSize);

            const uint8_t* p = raw.data();
            for (uint32_t i = 0; i < numFrames; ++i)
            {
                for (int c = 0; c < numChannels; ++c)
                {
                    float sample = 0.f;

                    if (audioFormat == 1) // PCM integer
                    {
                        switch (bitsPerSample)
                        {
                        case 16:
                        {
                            int16_t v;
                            std::memcpy(&v, p, 2);
                            sample = static_cast<float> (v) / 32768.f;
                            break;
                        }
                        case 24:
                        {
                            int32_t v = (static_cast<int32_t> (p[0]))
                                | (static_cast<int32_t> (p[1]) << 8)
                                | (static_cast<int32_t> (p[2]) << 16);
                            if (v & 0x800000) v |= 0xFF000000; // sign-extend
                            sample = static_cast<float> (v) / 8388608.f;
                            break;
                        }
                        case 32:
                        {
                            int32_t v;
                            std::memcpy(&v, p, 4);
                            sample = static_cast<float> (v) / 2147483648.f;
                            break;
                        }
                        default:
                            throw std::runtime_error("WavIO: unsupported PCM bit depth: "
                                + std::to_string(bitsPerSample));
                        }
                    }
                    else if (audioFormat == 3) // IEEE float
                    {
                        if (bitsPerSample == 32)
                        {
                            float v;
                            std::memcpy(&v, p, 4);
                            sample = v;
                        }
                        else if (bitsPerSample == 64)
                        {
                            double v;
                            std::memcpy(&v, p, 8);
                            sample = static_cast<float> (v);
                        }
                        else
                        {
                            throw std::runtime_error("WavIO: unsupported float bit depth: "
                                + std::to_string(bitsPerSample));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("WavIO: unsupported audioFormat code: "
                            + std::to_string(audioFormat));
                    }

                    out.channels[static_cast<size_t> (c)][i] = sample;
                    p += bytesPerSample;
                }
            }
        }

        // Move to the next chunk (chunks are word-aligned: pad byte if odd size)
        const std::streamoff offset = static_cast<std::streamoff>(chunkSize + (chunkSize % 2));

        f.seekg(chunkStart + offset);
    }

    if (out.channels.empty())
        throw std::runtime_error("WavIO: no 'data' chunk found in: " + path);

    return out;
}

/**
 * Writes a multi-channel float buffer as a 32-bit IEEE float WAV file.
 *
 * @param path        Output file path
 * @param channels    channels[c][i] = sample i of channel c, range [-1, +1]
 *                     (values outside this range are written as-is — float
 *                     WAV has no hard clipping, unlike PCM)
 * @param sampleRate  Sample rate in Hz
 */
inline void writeWavFloat(const std::string& path,
    const std::vector<std::vector<float>>& channels,
    int sampleRate)
{
    using namespace wavio_detail;

    if (channels.empty())
        throw std::runtime_error("WavIO: writeWavFloat called with no channels");

    const uint32_t numChannels = static_cast<uint32_t> (channels.size());
    const uint32_t numFrames = static_cast<uint32_t> (channels[0].size());
    const uint32_t bitsPerSample = 32;
    const uint32_t byteRate = static_cast<uint32_t> (sampleRate) * numChannels * (bitsPerSample / 8);
    const uint16_t blockAlign = static_cast<uint16_t> (numChannels * (bitsPerSample / 8));
    const uint32_t dataSize = numFrames * numChannels * (bitsPerSample / 8);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("WavIO: cannot create file: " + path);

    // ---- RIFF header ----
    f.write("RIFF", 4);
    writeU32(f, 36 + dataSize); // overall size
    f.write("WAVE", 4);

    // ---- fmt chunk ----
    f.write("fmt ", 4);
    writeU32(f, 16);              // fmt chunk size
    writeU16(f, 3);                // audioFormat = 3 (IEEE float)
    writeU16(f, static_cast<uint16_t> (numChannels));
    writeU32(f, static_cast<uint32_t> (sampleRate));
    writeU32(f, byteRate);
    writeU16(f, blockAlign);
    writeU16(f, static_cast<uint16_t> (bitsPerSample));

    // ---- data chunk ----
    f.write("data", 4);
    writeU32(f, dataSize);

    for (uint32_t i = 0; i < numFrames; ++i)
    {
        for (uint32_t c = 0; c < numChannels; ++c)
        {
            float v = channels[c][i];
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
    }
}