function S = stereometrics(in, wLen, hop)
% STEREO_METRICS - Per-frame stereo descriptors emulating GonioMeter side-info.
%
%   Given a stereo signal as input it computes Pearson correlation, 
%   RMS balance and stereo width, per each time frame. Frame center time and
%   index are also stored. GonioMeter VST documentation is available at
%   https://www.toneboosters.com/tb_goniometer_v1.html.
%
%   Inputs:
%   in:     input stereo signal
%   wLen:   window length (used for STFT computation)
%   hop:    hop size (used for STFT computation)
%
%   Output:
%   S:      structure containing:tes, for each time frame:
%           .corr     : Pearson correlation between L and R in [-1, +1]
%           .balance  : RMS balance (L-R)/(L+R), in [-1, +1], + = left
%           .width    : stereo width = RMS(L-R) / (RMS(L+R)+eps)
%           .t        : frame center time, seconds (relative to first sample)
%           .frameIdx : [nFrames x 2] sample indices [start end] of each frame

if nargin < 2
    wLen = 1024;
end
if nargin < 3
    hop  = wLen/2;
end

win = hann(wLen, 'symmetric');
N = size(in, 1);
nFrames = floor((N - wLen) / hop) + 1;

eps_ = 1e-12;

S.corr     = zeros(1, nFrames);
S.balance  = zeros(1, nFrames);
S.width    = zeros(1, nFrames);
S.t        = zeros(1, nFrames);
S.frameIdx = zeros(nFrames, 2);

for k = 1:nFrames
    n0 = (k-1)*hop + 1;
    n1 = n0 + wLen - 1;
    L = in(n0:n1, 1) .* win;
    R = in(n0:n1, 2) .* win;

    % Pearson correlation (zero-mean, normalized)
    Lc = L - mean(L);
    Rc = R - mean(R);
    denom = sqrt(sum(Lc.^2) * sum(Rc.^2)) + eps_;
    S.corr(k) = sum(Lc .* Rc) / denom;

    rL = sqrt(mean(L.^2));
    rR = sqrt(mean(R.^2));
    S.balance(k) = (rL - rR) / (rL + rR + eps_);    % + = left

    M = (L + R) / sqrt(2);
    Sd = (L - R) / sqrt(2);
    S.width(k) = sqrt(mean(Sd.^2)) / (sqrt(mean(M.^2)) + eps_);

    S.frameIdx(k, :) = [n0, n1];
    S.t(k) = (n0 + n1 - 2) / 2;     % center sample (0-based-ish)
end