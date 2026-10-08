% goniometer.m
%
% Emulates the behaviour of the ToneBoosters GonioMeter VST in MATLAB to
% compare, frame by frame, the stereo panning of:
%   - groundtruth (ref_*)
%   - baseline    (htdemucs_*)
%   - proposed    (sahtdemucs_*)
%
% For each frame a 3x3 figure is shown:
%   row 1: Mid/Side vectorscope (frame samples in the (S,M) plane)
%   row 2: panning index psi(band), in [-1,+1], log-frequency axis
%   row 3: textual panel with correlation, balance, width, azimuth
%
% Sign convention (consistent with ild.m):
%   psi = +1 / phi = +90  -> hard LEFT
%   psi =  0 / phi =   0  -> CENTER
%   psi = -1 / phi = -90  -> hard RIGHT

clear
close all
clc

% Paths relative to this script, so it runs from any working directory
% and on any OS (fullfile picks the right separator)
% (Run Section / Evaluate Selection execute a temporary copy of the code,
% so mfilename points to a temp folder: fall back to the file open in the
% Editor in that case)
scriptDir   = fileparts(mfilename('fullpath'));
if ~isfolder(fullfile(scriptDir, 'helpers'))
    scriptDir = fileparts(matlab.desktop.editor.getActiveFilename);
end
addpath(fullfile(scriptDir, 'helpers'))

%% Parameters
datapath    = fullfile(scriptDir, '..', 'docs', 'audio');
nSongs      = 3;
sources     = {'vocals', 'drums', 'bass', 'other'};
fs          = 44100;
nfft        = 2048;
wLen        = 2048;
nBands      = 64;

% Range of frames to plot for each song (NB: one figure per frame!).
% To avoid opening hundreds of figures, by default only a few frames
% spread along the track are shown. Set frameSel = [] for all of them.
frameSel = 'sparse';       % 'sparse' | 'all' | explicit vector of indices
nFramesShown = 20;         % used only if frameSel = 'sparse'

% Linear frequency axis
f = (0:floor(nfft/2)).' * (fs / nfft);

% Mel-band frequency axis
[~, band_hz] = melbinassignment(fs, nfft, nBands);

%% Loop over songs and over sources
for i = 1:nSongs
    song_path    = fullfile(datapath, ['song', num2str(i)]);

    for s = 1:length(sources)

        % Import audio files
        gt_s_path   = fullfile(song_path, ['ref_',          sources{s}, '.wav']);
        bl_s_path   = fullfile(song_path, ['htdemucs_',     sources{s}, '.wav']);
        sp1_s_path  = fullfile(song_path, ['htdemucsspft_', sources{s}, '.wav']);
        sp2_s_path  = fullfile(song_path, ['sahtdemucs_',   sources{s}, '.wav']);
        [gt_s, fs]  = audioread(gt_s_path);
        bl_s        = audioread(bl_s_path);
        sp1_s       = audioread(sp1_s_path);
        sp2_s       = audioread(sp2_s_path);

        % Time-frequency panning index
        [~, PSI_gt, phi_gt, X_gt]  = ild(fs, gt_s, 'mode', 'stft', ...
            'nfft', nfft, 'wLen', wLen);
        [~, PSI_bl, phi_bl, X_bl]  = ild(fs, bl_s, 'mode', 'stft', ...
            'nfft', nfft, 'wLen', wLen);
        [~, PSI_sp1, phi_sp1, X_sp1]  = ild(fs, sp1_s, 'mode', 'stft', ...
            'nfft', nfft, 'wLen', wLen);
        [~, PSI_sp2, phi_sp2, X_sp2]  = ild(fs, sp2_s, 'mode', 'stft', ...
            'nfft', nfft, 'wLen', wLen);

        % Panning index per Mel band (weighted aggregation)
        PSI_mel_gt  = melbandaggregate(PSI_gt, fs, nBands, X_gt.PL + X_gt.PR);
        PSI_mel_bl  = melbandaggregate(PSI_bl, fs, nBands, X_bl.PL + X_bl.PR);
        PSI_mel_sp1 = melbandaggregate(PSI_sp1, fs, nBands, X_sp1.PL + X_sp1.PR);
        PSI_mel_sp2 = melbandaggregate(PSI_sp2, fs, nBands, X_sp2.PL + X_sp2.PR);

        % Frame metrics (correlation, balance, width)
        M_gt    = stereometrics(gt_s, nfft, nfft/2);
        M_bl    = stereometrics(bl_s, nfft, nfft/2);
        M_sp1   = stereometrics(sp1_s, nfft, nfft/2);
        M_sp2   = stereometrics(sp2_s, nfft, nfft/2);

        % Frames selection for plot visualization
        nFrames = min([size(PSI_gt,2), size(PSI_bl,2), size(PSI_sp1,2), ...
            size(PSI_sp2,2)]);
        if isnumeric(frameSel)
            kList = frameSel(frameSel >= 1 & frameSel <= nFrames);
        elseif strcmpi(frameSel, 'all')
            kList = 1:nFrames;
        else  % 'sparse'
            kList = unique(round(linspace(1, nFrames, nFramesShown)));
        end

        % Plot per selected frame
        fig = figure;
        for k = kList
            clf(fig);
            figure(fig);
            sgtitle(sprintf(['Song %d - %s.wav - frame %d/%d ' ...
                '(t = %.2f s)'], i, sources{s}, k, nFrames, ...
                M_gt.t(k)/fs), 'FontWeight', 'bold');

            % Time-domain sub-blocks (for the time-domain vectorscope)
            n0      = M_gt.frameIdx(k,1);
            n1      = M_gt.frameIdx(k,2);
            win     = X_gt.win;
            blk_gt  = gt_s(n0:n1, :) .* win;
            blk_bl  = bl_s(n0:n1, :) .* win;
            blk_sp1 = sp1_s(n0:n1, :) .* win;
            blk_sp2 = sp2_s(n0:n1, :) .* win;

            % --- Row 1: Mid/Side vectorscope -----------------------------
            ax1 = subplot(3,4,1);
            plotvectorscope(blk_gt, 'ax', ax1);
            title(ax1, {'\fontsize{12}Groundtruth', 'Vectorscope'});
            ax2 = subplot(3,4,2); 
            plotvectorscope(blk_bl, 'ax', ax2);
            title(ax2, {'\fontsize{12}HTDemucs', 'Vectorscope'});
            ax3 = subplot(3,4,3);
            plotvectorscope(blk_sp1, 'ax', ax3);
            title(ax3, {'\fontsize{12}HTDemucs SP Finetune', 'Vectorscope'});
            ax4 = subplot(3,4,4);
            plotvectorscope(blk_sp2, 'ax', ax4);
            title(ax4, {'\fontsize{12}SA-HTDemucs', 'Vectorscope'});

            % --- Row 2: panning index per band ---------------------------
            ax5 = subplot(3,4,5);
            plotpanbands(ax5, band_hz, PSI_mel_gt(:,k));
            ax6 = subplot(3,4,6);
            plotpanbands(ax6, band_hz, PSI_mel_bl(:,k));
            ax7 = subplot(3,4,7);
            plotpanbands(ax7, band_hz, PSI_mel_sp1(:,k));
            ax8 = subplot(3,4,8);
            plotpanbands(ax8, band_hz, PSI_mel_sp2(:,k));

            % --- Row 3: textual panels with metrics ----------------------
            ax9 = subplot(3,4,9);
            plotstereometricspanel(M_gt, phi_gt, k, 'ax', ax9);
            ax10 = subplot(3,4,10);
            plotstereometricspanel(M_bl, phi_bl, k, 'ax', ax10);
            ax11 = subplot(3,4,11);
            plotstereometricspanel(M_sp1, phi_sp1, k, 'ax', ax11);
            ax12 = subplot(3,4,12);
            plotstereometricspanel(M_sp2, phi_sp2, k, 'ax', ax12);

            drawnow;
            pause(0.5);
        end
        pause(0.5)
    end
    pause(0.5)
end

% Local plot functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotpanbands(ax, band_hz, psi_col)
% Panning index per band in the current frame.
semilogx(ax, band_hz, psi_col, 'LineWidth', 1.4);
hold(ax, 'on');
yline(ax, 0, '--', 'Color', [0.6 0.6 0.6]);   % center
yline(ax,  1, ':', 'Color', [0.8 0.8 0.8]);   % hard L
yline(ax, -1, ':', 'Color', [0.8 0.8 0.8]);   % hard R
grid(ax, 'on');
xlim(ax, [20 2e4]);
ylim(ax, [-1.05 1.05]);
xlabel(ax, 'Frequency [Hz]');
ylabel('\Psi (-1 R / +1 L)');
title(ax, 'Panning index per Mel band');
end