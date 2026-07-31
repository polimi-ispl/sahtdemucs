"""
spatial.py — Low-level spatial cue utilities (ILD, ITD).

All functions are differentiable and operate on batched tensors so they
can be used both inside the model (SpatialCueModule) and inside the loss
(SpatialLoss).

Spatial cue primer
------------------
The human auditory system localizes sound using two main binaural cues:

    ILD (Interaural Level Difference)
        The ratio of RMS energy at the left and right ears, expressed in dB.
        Dominant at high frequencies (> ~1.5 kHz).  A positive ILD means the
        left channel is louder (source to the left).

    ITD (Interaural Time Difference)
        The difference in arrival time between the two ears, expressed in
        samples (or seconds).  Dominant at low frequencies (< ~1.5 kHz).
        A positive ITD means the left channel leads (source to the left).

Both cues are estimated here.
"""

import functools
import math
import torch
import torch.nn.functional as F


__all__ = [
    "mel_bin_assignment",
    "compute_ild",
    "compute_ild_bands",
    "compute_ild_bands_mel",
    "compute_itd_samples",
    "compute_itd_bands",
    "compute_itd_bands_mel",
    "apply_itd",
]

# ──────────────────────────────────────────────────────────────────────────────
# ILD — broadband scalar
# ──────────────────────────────────────────────────────────────────────────────

def compute_ild(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Interaural Level Difference in dB: ``ILD = 20·log10(RMS_L / RMS_R)``.

    A value of 0 dB means equal energy in both channels.
    Positive values indicate the left channel is louder.

    Args:
        left:  ``(B, T)`` or ``(B, F, T)``
        right: ``(B, T)`` or ``(B, F, T)``
        eps:   small constant for numerical stability (avoids log(0))

    Returns:
        ild: ``(B,)`` or ``(B, F)``
    """
    # RMS over the time dimension (last dim)
    rms_l = left.pow(2).mean(dim=-1).clamp(min=eps).sqrt()
    rms_r = right.pow(2).mean(dim=-1).clamp(min=eps).sqrt()
    return 20.0 * torch.log10(rms_l / rms_r + eps)

# ──────────────────────────────────────────────────────────────────────────────
# ILD — frequency-resolved, per sub-band via STFT
# ──────────────────────────────────────────────────────────────────────────────

def compute_ild_bands(
    left: torch.Tensor,
    right: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sub-band ILD via STFT magnitude spectrum.

    The STFT magnitude spectrum is divided into *n_bands* equal-width frequency
    bands.  The ILD of each band is the ratio of mean left-channel power to
    mean right-channel power (in dB) for that band.

    Unlike the broadband scalar returned by :func:`compute_ild`, this gives a
    frequency-resolved spatial cue vector that captures the fact that different
    frequency regions can have different left/right energy balance (e.g. a
    panned guitar may have a strong ILD at 1–3 kHz while being centered at low
    frequencies).

    The function is fully differentiable — gradients flow through the STFT
    magnitudes and the log operation — so it can be used both inside the model
    (``SpatialCueModule``) and inside the loss (``SpatialLoss``).

    Args:
        left:       ``(B, T)`` left-channel waveform
        right:      ``(B, T)`` right-channel waveform
        n_fft:      FFT size (default 2048 → ~46 ms @ 44 100 Hz)
        hop_length: STFT hop in samples (default 512 → ~11.6 ms @ 44 100 Hz)
        n_bands:    number of equal-width frequency sub-bands (default 32)
        eps:        numerical stability constant

    Returns:
        ild_bands: ``(B, n_bands, T_frames)`` — ILD in dB per sub-band and STFT
                   frame.  Positive values indicate the left channel is louder.
    """
    window = torch.hann_window(n_fft, device=left.device)

    # STFT: (B, F_bins, T_frames),  F_bins = n_fft // 2 + 1
    L = torch.stft(left,  n_fft, hop_length, window=window, return_complex=True)
    R = torch.stft(right, n_fft, hop_length, window=window, return_complex=True)

    B, F_bins, T_frames = L.shape
    # Trim to the largest multiple of n_bands (drops at most n_bands−1 edge bins)
    F_trim = (F_bins // n_bands) * n_bands
    bpb    = F_bins // n_bands      # bins per band

    # Power spectrum grouped into bands: (B, n_bands, bpb, T_frames)
    # .abs() produces a new contiguous tensor, so .reshape() is safe.
    pw_l = L[:, :F_trim, :].abs().pow(2).reshape(B, n_bands, bpb, T_frames)
    pw_r = R[:, :F_trim, :].abs().pow(2).reshape(B, n_bands, bpb, T_frames)

    # Mean power per band (over frequency bins only) → RMS → ILD in dB
    # dim=2 averages over the bpb frequency bins within each band;
    # T_frames is kept so the output captures temporal ILD variation.
    rms_l = pw_l.mean(dim=2).clamp(min=eps).sqrt()   # (B, n_bands, T_frames)
    rms_r = pw_r.mean(dim=2).clamp(min=eps).sqrt()
    return 20.0 * torch.log10(rms_l / rms_r + eps)   # (B, n_bands, T_frames)


# ──────────────────────────────────────────────────────────────────────────────
# ILD — frequency-resolved, per sub-band via STFT + Mel-scale bands
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def mel_bin_assignment(n_fft: int, n_bands: int, sample_rate: int) -> torch.Tensor:
    """Map each STFT bin to a Mel-scale band index in ``[0, n_bands)``.

    The Mel axis from 0 Hz to Nyquist is divided into *n_bands* equal-width
    intervals; each STFT bin is assigned to the interval that contains its
    centre frequency.  The result is cached (keyed on the three integers) so
    the tensor is built only once per unique ``(n_fft, n_bands, sample_rate)``
    combination.

    Returns:
        band_idx: ``(n_fft // 2 + 1,)`` CPU LongTensor, values in
                  ``[0, n_bands)``.
    """
    F_bins  = n_fft // 2 + 1
    nyquist = sample_rate / 2.0

    # Centre frequency of each bin in Hz
    bin_hz = torch.arange(F_bins, dtype=torch.float64) * (nyquist / (F_bins - 1))

    # Convert to Mel (O'Shaughnessy formula, consistent with librosa/torchaudio)
    mel = 2595.0 * torch.log10(1.0 + bin_hz / 700.0)

    # Uniformly partition [0, mel_max] into n_bands equal intervals
    mel_max  = mel[-1]
    band_idx = (mel / mel_max * n_bands).long().clamp(0, n_bands - 1)

    return band_idx   # CPU LongTensor


def compute_ild_bands_mel(
    left: torch.Tensor,
    right: torch.Tensor,
    n_fft: int        = 2048,
    hop_length: int   = 512,
    n_bands: int      = 32,
    sample_rate: int  = 44100,
    eps: float        = 1e-8,
) -> torch.Tensor:
    """Per-sub-band ILD via STFT with **Mel-scale** frequency bands.

    Unlike :func:`compute_ild_bands`, which divides the STFT spectrum into
    *n_bands* equal-width **linear-frequency** bands (coarse at low
    frequencies, fine at high), this function partitions the Mel axis into
    *n_bands* equal-width intervals.  Because the Mel scale is compressed at
    high frequencies, each Mel band covers a **narrower Hz range at low
    frequencies** than at high ones.  This gives the network finer-grained
    ILD information in the perceptually important bass region, where equal-
    linear-width bands lump too many octaves into a single band.

    The analysis is a rectangular Mel filterbank: every STFT bin is assigned
    to exactly one Mel band (no overlap), and the mean power within each band
    is computed.  The function is fully differentiable — gradients flow
    through the STFT magnitudes and the log.

    Args:
        left:        ``(B, T)`` left-channel waveform
        right:       ``(B, T)`` right-channel waveform
        n_fft:       FFT size (default 2048 → ~46 ms @ 44 100 Hz)
        hop_length:  STFT hop in samples (default 512 → ~11.6 ms @ 44 100 Hz)
        n_bands:     number of Mel-scale frequency bands (default 32)
        sample_rate: audio sample rate in Hz (default 44 100)
        eps:         numerical stability constant

    Returns:
        ild_bands:  ``(B, n_bands, T_frames)`` — ILD in dB per Mel band and
                    STFT frame.  Band 0 is the lowest-frequency band.
                    Positive values indicate the left channel is louder.
    """
    window = torch.hann_window(n_fft, device=left.device)

    L = torch.stft(left,  n_fft, hop_length, window=window, return_complex=True)
    R = torch.stft(right, n_fft, hop_length, window=window, return_complex=True)

    B, F_bins, T_frames = L.shape

    # ── Build rectangular Mel filterbank on the right device ─────────────────
    # band_idx[f] = Mel band that STFT bin f belongs to
    band_idx = mel_bin_assignment(n_fft, n_bands, sample_rate).to(left.device)

    # One-hot encode bin → band membership, then normalise rows to mean power
    oh = torch.zeros(F_bins, n_bands, dtype=left.dtype, device=left.device)
    oh.scatter_(1, band_idx.unsqueeze(1), 1.0)        # (F_bins, n_bands)
    fb = oh.t()                                        # (n_bands, F_bins)
    fb = fb / fb.sum(dim=1, keepdim=True).clamp(min=1)  # normalise → mean

    # ── Mean power per Mel band ───────────────────────────────────────────────
    pw_l = L.abs().pow(2)   # (B, F_bins, T_frames)
    pw_r = R.abs().pow(2)

    mean_l = torch.einsum("nf,bft->bnt", fb, pw_l)    # (B, n_bands, T_frames)
    mean_r = torch.einsum("nf,bft->bnt", fb, pw_r)

    rms_l = mean_l.clamp(min=eps).sqrt()
    rms_r = mean_r.clamp(min=eps).sqrt()
    return 20.0 * torch.log10(rms_l / rms_r + eps)


# ──────────────────────────────────────────────────────────────────────────────
# ITD via GCC-PHAT
# ──────────────────────────────────────────────────────────────────────────────

def compute_itd_samples(
    left: torch.Tensor,
    right: torch.Tensor,
    max_lag: int = 64,
) -> torch.Tensor:
    """Estimate ITD via GCC-PHAT (generalized cross-correlation) in the time domain.

    GCC-PHAT computes the cross-correlation between the two channels after
    whitening in the frequency domain (PHAT = Phase Transform).  This makes
    the peak sharper and more robust to reverberation than plain
    cross-correlation.

    Differentiability
        The hard ``argmax`` over the lag axis is replaced by a **soft-argmax**
        (weighted average of lag indices under a softmax with temperature 10).
        This keeps the output differentiable w.r.t. the input waveforms so that
        gradients can flow back through the ITD loss term.

    Args:
        left:       ``(B, T)``
        right:      ``(B, T)``
        max_lag:    maximum lag in samples to search over (search range:
                    [−max_lag, +max_lag])

    Returns:
        itd:        ``(B,)`` in samples (float, differentiable)
    """
    B, T = left.shape
    beta = 10.0

    # Cap the FFT size to avoid cuFFT failures on very long signals.
    # ITD estimation only needs a short excerpt; 2^17 = 131072 samples
    # (~3 s @ 44100 Hz) is more than sufficient and stays within cuFFT limits.
    MAX_FFT = 2 ** 17  # 131 072
    if T > MAX_FFT // 2:
        left  = left[:, :MAX_FFT // 2]
        right = right[:, :MAX_FFT // 2]
        T = MAX_FFT // 2

    # Zero-pad to 2T so that circular correlation does not wrap around
    n_fft = 2 * T
    # ── Step 1: FFT of both channels ─────────────────────────────────────────
    L = torch.fft.rfft(left,  n=n_fft)   # (B, n_fft//2 + 1), complex
    R = torch.fft.rfft(right, n=n_fft)

    # ── Step 2: Cross-spectrum L · conj(R) ───────────────────────────────────
    cross = L * R.conj()                  # (B, F), complex

    # ── Step 3: PHAT whitening — normalize by magnitude ──────────────────────
    # Dividing by |cross| sets all frequency bins to unit amplitude, keeping
    # only phase information.  This sharpens the correlation peak.
    cross = cross / (cross.abs() + 1e-8)

    # ── Step 4: Inverse FFT → generalized cross-correlation ──────────────────
    cc = torch.fft.irfft(cross, n=n_fft)  # (B, n_fft), real

    # ── Step 5: Extract lags in [−max_lag, +max_lag] ─────────────────────────
    # Positive lags occupy cc[:, 1 : max_lag+1], negative lags are stored at
    # the end of the circular buffer: cc[:, n_fft-max_lag : n_fft].
    # Negative lags are stored at the end of the circular buffer
    # (cc[:, n_fft-max_lag:] = lags -max_lag...-1).  They must come *first*
    # so that the order matches lag_values = arange(-max_lag, max_lag+1).
    lags = torch.cat(
        [cc[:, n_fft - max_lag:], cc[:, :max_lag + 1]], dim=1
    )  # (B, 2*max_lag+1)  — order: [-max_lag, ..., -1, 0, 1, ..., max_lag]

    # ── Step 6: Soft-argmax to maintain differentiability ────────────────────
    # Temperature=10 sharpens the softmax so it approximates argmax while
    # keeping the gradient non-zero.
    weights    = torch.softmax(lags * beta, dim=-1)             # (B, 2*max_lag+1)
    lag_values = torch.arange(
        -max_lag, max_lag + 1, dtype=left.dtype, device=left.device
    )                                                            # (2*max_lag+1,)
    itd = (weights * lag_values).sum(dim=-1)                    # (B,)
    return itd


# ──────────────────────────────────────────────────────────────────────────────
# ITD — frequency-resolved, per sub-band via band-limited GCC-PHAT
# ──────────────────────────────────────────────────────────────────────────────

def _itd_bands_from_assignment(
    left: torch.Tensor,
    right: torch.Tensor,
    band_idx: torch.Tensor,
    n_bands: int,
    n_fft: int,
    hop_length: int,
    max_lag: int,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Per-sub-band, per-frame ITD via band-limited GCC-PHAT.

    Shared core for :func:`compute_itd_bands` (linear bands) and
    :func:`compute_itd_bands_mel` (Mel bands).  The two public wrappers differ
    only in how STFT bins are grouped into bands, encoded by ``band_idx``.

    The cross-power spectrum of the two channels is PHAT-whitened (so only the
    interaural *phase* survives), then — independently for every band and every
    STFT frame — the generalised cross-correlation is reconstructed over the lag
    range ``[-max_lag, +max_lag]`` and a **soft-argmax** picks the lag of the
    peak.  This mirrors the broadband :func:`compute_itd_samples` but yields a
    frequency- *and* time-resolved cue, exactly like :func:`compute_ild_bands`.

    Args:
        left:       ``(B, T)`` left-channel waveform
        right:      ``(B, T)`` right-channel waveform
        band_idx:   ``(n_fft // 2 + 1,)`` LongTensor mapping each STFT bin to a
                    band in ``[0, n_bands)``; bins with value ``-1`` are ignored.
        n_bands:    number of frequency sub-bands
        n_fft:      FFT size
        hop_length: STFT hop in samples
        max_lag:    maximum lag searched, in samples (range ±max_lag)
        beta:       soft-argmax temperature; larger → closer to a hard argmax
        eps:        numerical-stability constant for the PHAT whitening

    Returns:
        itd_bands:  ``(B, n_bands, T_frames)`` — ITD in **samples** per sub-band
                    and STFT frame.  The sign convention matches
                    :func:`compute_itd_samples`: a *positive* value is the lag
                    that maximises the PHAT cross-correlation of ``L·conj(R)``,
                    i.e. the right channel leads (the left channel is delayed).
    """
    window = torch.hann_window(n_fft, device=left.device)

    # STFT: (B, F_bins, T_frames), complex
    L = torch.stft(left,  n_fft, hop_length, window=window, return_complex=True)
    R = torch.stft(right, n_fft, hop_length, window=window, return_complex=True)

    B, F_bins, T_frames = L.shape

    # ── Cross-power spectrum + PHAT whitening ────────────────────────────────
    # Dividing by the magnitude keeps only the interaural phase (the ITD cue),
    # which sharpens the cross-correlation peak just like broadband GCC-PHAT.
    cross = L * R.conj()                       # (B, F_bins, T_frames), complex
    cross = cross / (cross.abs() + eps)

    # ── Inverse-DFT basis restricted to the lag range of interest ────────────
    # A delay of `lag` samples maps STFT bin f to phase exp(+j·2π·f·lag/n_fft);
    # reconstructing only the ±max_lag lags (instead of a full irfft) is cheaper.
    lag_values = torch.arange(
        -max_lag, max_lag + 1, dtype=left.dtype, device=left.device
    )                                          # (Lg,)
    freqs = torch.arange(F_bins, dtype=left.dtype, device=left.device)  # (F_bins,)
    phase = (2.0 * math.pi / n_fft) * freqs.unsqueeze(1) * lag_values.unsqueeze(0)
    basis = torch.exp(1j * phase)              # (F_bins, Lg), complex

    # ── Per-band band-limited GCC-PHAT + soft-argmax ─────────────────────────
    bands = []
    for k in range(n_bands):
        mask  = band_idx == k
        n_in  = int(mask.sum())
        if n_in == 0:
            # Empty band (can happen with Mel spacing at very low n_fft): ITD 0.
            bands.append(left.new_zeros(B, T_frames))
            continue

        cross_k = cross[:, mask, :]            # (B, n_in, T_frames), complex
        basis_k = basis[mask, :]               # (n_in, Lg), complex

        # gcc_k[b, lag, t] = Re( Σ_f cross_k[b, f, t] · basis_k[f, lag] )
        gcc = torch.einsum("bft,fl->blt", cross_k, basis_k).real  # (B, Lg, T_frames)
        gcc = gcc / n_in                        # scale-invariant: coherent peak ≈ 1

        weights = torch.softmax(gcc * beta, dim=1)                # over the lag axis
        itd_k   = (weights * lag_values.view(1, -1, 1)).sum(dim=1)  # (B, T_frames)
        bands.append(itd_k)

    return torch.stack(bands, dim=1)            # (B, n_bands, T_frames)


def compute_itd_bands(
    left: torch.Tensor,
    right: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    max_lag: int = 64,
    beta: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sub-band ITD via band-limited GCC-PHAT, **linear** frequency bands.

    The STFT spectrum is split into *n_bands* equal-width (linear-frequency)
    bands — the same partition used by :func:`compute_ild_bands` — and a
    band-limited GCC-PHAT estimates the interaural time difference per band and
    per STFT frame.  Fully differentiable (soft-argmax), so it can be used both
    inside a model and inside a loss alongside the ILD term.

    Args:
        left:       ``(B, T)`` left-channel waveform
        right:      ``(B, T)`` right-channel waveform
        n_fft:      FFT size (default 2048 → ~46 ms @ 44 100 Hz)
        hop_length: STFT hop in samples (default 512)
        n_bands:    number of equal-width frequency sub-bands (default 32)
        max_lag:    maximum lag searched, in samples (default 64 → ±1.45 ms
                    @ 44 100 Hz, comfortably covering the human ITD range)
        beta:       soft-argmax temperature (default 20.0)
        eps:        numerical-stability constant

    Returns:
        itd_bands:  ``(B, n_bands, T_frames)`` — ITD in samples per sub-band and
                    STFT frame.  Same sign convention as
                    :func:`compute_itd_samples` (positive → right channel leads).
    """
    F_bins = n_fft // 2 + 1
    bpb    = F_bins // n_bands                  # bins per band
    F_trim = bpb * n_bands                      # drop the top remainder bins

    band_idx = torch.full((F_bins,), -1, dtype=torch.long, device=left.device)
    band_idx[:F_trim] = torch.arange(F_trim, device=left.device) // bpb

    return _itd_bands_from_assignment(
        left, right, band_idx, n_bands, n_fft, hop_length, max_lag, beta, eps,
    )


def compute_itd_bands_mel(
    left: torch.Tensor,
    right: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 32,
    sample_rate: int = 44100,
    max_lag: int = 64,
    beta: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sub-band ITD via band-limited GCC-PHAT, **Mel-scale** frequency bands.

    Identical to :func:`compute_itd_bands` except STFT bins are grouped with the
    Mel partition from :func:`mel_bin_assignment` — the same band layout used by
    :func:`compute_ild_bands_mel`.  Because Mel bands are narrower at low
    frequencies, the bass region (where ITD perception dominates) gets finer
    resolution.

    Args:
        left:        ``(B, T)`` left-channel waveform
        right:       ``(B, T)`` right-channel waveform
        n_fft:       FFT size (default 2048)
        hop_length:  STFT hop in samples (default 512)
        n_bands:     number of Mel-scale frequency bands (default 32)
        sample_rate: audio sample rate in Hz (default 44 100)
        max_lag:     maximum lag searched, in samples (default 64)
        beta:        soft-argmax temperature (default 20.0)
        eps:         numerical-stability constant

    Returns:
        itd_bands:   ``(B, n_bands, T_frames)`` — ITD in samples per Mel band and
                     STFT frame.  Band 0 is the lowest-frequency band; same sign
                     convention as :func:`compute_itd_samples` (positive → right
                     channel leads).
    """
    band_idx = mel_bin_assignment(n_fft, n_bands, sample_rate).to(left.device)
    return _itd_bands_from_assignment(
        left, right, band_idx, n_bands, n_fft, hop_length, max_lag, beta, eps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fractional delay via phase shift
# ──────────────────────────────────────────────────────────────────────────────

def apply_itd(
    signal: torch.Tensor,
    delay_samples: torch.Tensor,
) -> torch.Tensor:
    """Apply a fractional delay to a signal via frequency-domain phase shift.

    A delay of *d* samples corresponds to multiplying each frequency bin *f*
    by `exp(−j · 2π · f · d)`.  This is exact for integer delays and gives
    a smooth interpolation for fractional delays (sinc interpolation in the
    time domain).

    Args:
        signal:         `(B, T)`
        delay_samples:  `(B,)` delay in samples (may be fractional and
                        differentiable)

    Returns:
        shifted:        `(B, T)` — signal delayed by *delay_samples*
    """
    B, T = signal.shape

    # Cap chunk size to avoid cuFFT failures on very long signals.
    MAX_FFT = 2 ** 17  # 131 072 samples (~3 s @ 44100 Hz)

    def _shift_chunk(chunk: torch.Tensor, n: int) -> torch.Tensor:
        freqs   = torch.fft.rfftfreq(n, device=chunk.device)
        S       = torch.fft.rfft(chunk)
        phase   = -2.0 * math.pi * freqs.unsqueeze(0) * delay_samples.unsqueeze(1)
        S_shift = S * torch.exp(1j * phase)
        return torch.fft.irfft(S_shift, n=n)

    if T <= MAX_FFT:
        return _shift_chunk(signal, T)

    # Process in non-overlapping chunks; boundary artefacts are limited to
    # ±max_delay samples (~64) per chunk edge, negligible for long signals.
    output = torch.zeros_like(signal)
    pos = 0
    while pos < T:
        end        = min(pos + MAX_FFT, T)
        chunk_len  = end - pos
        output[:, pos:end] = _shift_chunk(signal[:, pos:end], chunk_len)
        pos = end
    return output