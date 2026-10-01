% sofa2hrir.m
%
% Converts SOFA HRIR/DTF files (e.g. the ARI HRTF database) into the
% SADIE II-like layout expected by sahtdemucs/binaural_synth.py:
%
%   <outRoot>/<subject>/azi_{phi}_ele_0_DFC.wav
%
% one stereo WAV (col 1 = left ear, col 2 = right ear) per azimuth phi of
% the binaural_synth.py grid, elevation 0 deg, resampled to 44.1 kHz.
%
% Angle convention: SOFA spherical coordinates (azimuth counter-clockwise,
% 0 = front, 90 = left, 270 = right), which is the same convention as
% SADIE II - no remapping is needed.
%
% The "DFC" suffix is kept only so that binaural_synth.py finds the files;
% the equalization is whatever the SOFA file carries (use the ARI DTF files
% for the closest match to SADIE's diffuse-field compensated HRIRs).
%
% Dependencies:
% - SOFA Toolbox (https://github.com/sofacoustics/SOFAtoolbox), in the path
% - Signal Processing Toolbox (resample)
%
% Usage (afterwards, from the repository root):
%   python sahtdemucs/binaural_synth.py --input_dir=<stems root> ...
%          --output_dir=<out> --hrir_dir=<outRoot>/<subject> -m <metadata.json>

clear
close all
clc

% Add SOFA Toolbox to path
addpath(['/Users/matteo-a/Documents/MATLAB/toolboxes/SOFAtoolbox-2.6.0/' ...
    'SOFAtoolbox'])
savepath

%% Configuration

% Dataset root on the external disk: drive letter on Windows, mount point
% under /Volumes on macOS
if ispc
    datasetRoot = 'D:\Polimi\PhD\Dataset';
else
    datasetRoot = '/Volumes/Elements/Polimi/PhD/Dataset';
end

% SOFA files to convert (one output sub-folder per file, named after it)
sofaDir   = fullfile(datasetRoot, 'ARI');
sofaFiles = {'dtf_nh2.sofa'};

% Output root: <outRoot>/<subject>/azi_{phi}_ele_0_DFC.wav
outRoot   = fullfile(datasetRoot, 'ARI_44k');

% Target sample rate (binaural_synth.py SAMPLE_RATE)
fsOut     = 44100;

% Azimuth grid of binaural_synth.py RANDOM_ANGLES (frontal half-plane)
azGrid    = [0:10:90, 270:10:350];
elTarget  = 0;

% Matching tolerance (deg): beyond it the nearest measured direction is
% used and a warning is printed
tolDeg    = 0.5;

% Bits per sample of the output WAVs: 32 -> IEEE float, no clipping of
% HRIR samples beyond +-1 (soundfile reads it transparently)
bitsOut   = 32;

%% SOFA Toolbox

if ~exist('SOFAstart', 'file')
    error(['SOFA Toolbox not found: add SOFAtoolbox/SOFAtoolbox to the ' ...
        'MATLAB path (addpath) first.']);
end
SOFAstart;

%% Conversion loop

for f = 1:numel(sofaFiles)
    sofaPath = fullfile(sofaDir, sofaFiles{f});
    [~, subject] = fileparts(sofaPath);
    subject = strrep(subject, ' ', '_');        % 'dtf b_nh2' -> 'dtf_b_nh2'
    outDir  = fullfile(outRoot, subject);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    fprintf('\n=== %s -> %s\n', sofaPath, outDir);
    Obj = SOFAload(sofaPath);

    % ---------------------------------------------------------------------
    % Source positions in spherical coordinates (deg)
    % ---------------------------------------------------------------------
    pos = Obj.SourcePosition;
    if strcmpi(Obj.SourcePosition_Type, 'cartesian')
        [az, el, r] = cart2sph(pos(:,1), pos(:,2), pos(:,3));
        pos = [rad2deg(az), rad2deg(el), r];
    end
    az = mod(pos(:,1), 360);                       % [-180,180) -> [0,360)
    el = pos(:,2);

    % ---------------------------------------------------------------------
    % IRs: Data.IR is (M measurements x R receivers x N samples);
    % receiver 1 = left ear, receiver 2 = right ear (SOFA convention)
    % ---------------------------------------------------------------------
    IR   = Obj.Data.IR;
    fsIn = Obj.Data.SamplingRate;
    if size(IR, 2) ~= 2
        error('%s: expected 2 receivers (L/R), found %d', sofaFiles{f}, ...
            size(IR, 2));
    end
    fprintf('fs = %g Hz, %d measurements, IR length = %d samples\n', ...
        fsIn, size(IR, 1), size(IR, 3));

    % Measurements on the target elevation
    onPlane = find(abs(el - elTarget) <= tolDeg);
    if isempty(onPlane)
        [~, iNear] = min(abs(el - elTarget));
        error('%s: no measurement at elevation %g deg (nearest: %g deg)', ...
            sofaFiles{f}, elTarget, el(iNear));
    end

    % Resampling factors (44100/48000 = 147/160)
    [p, q] = rat(fsOut / fsIn);

    fprintf('%8s %8s %8s %10s\n', 'phi', 'azSOFA', 'elSOFA', 'ILD (dB)');
    for a = azGrid
        % Nearest measured azimuth on the plane (circular distance)
        dAz = abs(mod(az(onPlane) - a + 180, 360) - 180);
        [dMin, k] = min(dAz);
        m = onPlane(k);
        if dMin > tolDeg
            warning(['%s: azimuth %d not measured, using %.2f deg ' ...
                '(|d| = %.2f)'], sofaFiles{f}, a, az(m), dMin);
        end

        h = squeeze(IR(m, :, :)).';         % (N x 2): [left, right]
        if fsIn ~= fsOut
            h = resample(h, p, q);          % column-wise, delay-compensated FIR
        end

        % Sanity check: broadband ILD, positive = left louder (sources at
        % 10..90 deg must be > 0, at 270..350 deg < 0)
        ild = 10 * log10(sum(h(:,1).^2) / sum(h(:,2).^2));

        outFile = fullfile(outDir, sprintf('azi_%d_ele_0_DFC.wav', a));
        audiowrite(outFile, h, fsOut, 'BitsPerSample', bitsOut);
        fprintf('%8d %8.2f %8.2f %+10.2f\n', a, az(m), el(m), ild);
    end
end

fprintf('\nDone.\n');