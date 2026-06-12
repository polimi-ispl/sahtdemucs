% goniometer_compare.m
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

%% Parameters
datapath = '..\docs\audio';
nSongs   = 3;
sources  = {'bass', 'drums', 'other', 'vocals'};
nfft     = 1024;
fs       = 44100;
nBands   = 64;

% Range of frames to plot for each song (NB: one figure per frame!).
% To avoid opening hundreds of figures, by default only a few frames
% spread along the track are shown. Set frameSel = [] for all of them.
frameSel = 'sparse';       % 'sparse' | 'all' | explicit vector of indices
nFramesShown = 6;          % used only if frameSel = 'sparse'

% Frequency axis (band-center Hz) for the per-band plots
bin_hz = (0:floor(nfft/2)).' * (fs / nfft);
ba_ref = melbinassignment(fs, nfft, nBands);
sum_hz = accumarray(ba_ref, bin_hz, [nBands 1], @sum, 0);
cnt_hz = accumarray(ba_ref, ones(size(bin_hz)), [nBands 1], @sum, 0);
band_hz = zeros(nBands, 1);
nz = cnt_hz > 0;
band_hz(nz) = sum_hz(nz) ./ cnt_hz(nz);

%% Loop over songs and over sources
for i = 1:nSongs
    gt_path      = fullfile(datapath, ['song', num2str(i), '\ref_']);
    bl_path      = fullfile(datapath, ['song', num2str(i), '\htdemucs_']);
    spatial_path = fullfile(datapath, ['song', num2str(i), '\sahtdemucs_']);

    for s = 3:length(sources)

        % Import audio files
        gt_s_path      = [gt_path,      sources{s}, '.wav'];
        bl_s_path      = [bl_path,      sources{s}, '.wav'];
        spatial_s_path = [spatial_path, sources{s}, '.wav'];

        [gt_s, fs] = audioread(gt_s_path);
        bl_s       = audioread(bl_s_path);
        spatial_s  = audioread(spatial_s_path);

        % Time-frequency panning index
        [~, PSI_gt, phi_gt, Xg]  = ild(fs, gt_s, 'stft', nfft);
        [~, PSI_bl, phi_bl, Xb]  = ild(fs, bl_s, 'stft', nfft);
        [~, PSI_sp, phi_sp, Xs]  = ild(fs, spatial_s, 'stft', nfft);

        % stft is two-sided -> keep only the upper half [0, Nyquist]
        F_one = floor(nfft/2) + 1;
        sel   = (size(Xg.PL,1) - F_one + 1) : size(Xg.PL,1);

        PSI_gt = PSI_gt(sel, :);   Wgt = Xg.PL(sel, :) + Xg.PR(sel, :);
        PSI_bl = PSI_bl(sel, :);   Wbl = Xb.PL(sel, :) + Xb.PR(sel, :);
        PSI_sp = PSI_sp(sel, :);   Wsp = Xs.PL(sel, :) + Xs.PR(sel, :);

        % Panning index per Mel band (weighted aggregation)
        PSI_mel_gt = panindextomelbands(PSI_gt, Wgt, nfft, fs, nBands);
        PSI_mel_bl = panindextomelbands(PSI_bl, Wbl, nfft, fs, nBands);
        PSI_mel_sp = panindextomelbands(PSI_sp, Wsp, nfft, fs, nBands);

        % Frame metrics (correlation, balance, width)
        Mgt = stereometrics(gt_s, nfft, nfft/2);
        Mbl = stereometrics(bl_s, nfft, nfft/2);
        Msp = stereometrics(spatial_s, nfft, nfft/2);

        % Frames selection for plot visualization
        nFrames = min([size(PSI_gt,2), size(PSI_bl,2), size(PSI_sp,2)]);
        if isnumeric(frameSel)
            kList = frameSel(frameSel >= 1 & frameSel <= nFrames);
        elseif strcmpi(frameSel, 'all')
            kList = 1:nFrames;
        else  % 'sparse'
            kList = unique(round(linspace(1, nFrames, nFramesShown)));
        end

        % Plot per selected frame
        for k = kList
            close
            fig = figure('Color', 'w', 'Position', [80 80 1200 780]);
            sgtitle(sprintf('Song %d - %s.wav - frame %d/%d  (t = %.2f s)', ...
                i, sources{s}, k, nFrames, Mgt.t(k)/fs), ...
                'FontWeight', 'bold');

            % Time-domain sub-blocks (for the time-domain vectorscope)
            n0 = Mgt.frameIdx(k,1); n1 = Mgt.frameIdx(k,2);
            blk_gt = gt_s(n0:n1, :)      .* Xb.win;
            blk_bl = bl_s(n0:n1, :)      .* Xb.win;
            blk_sp = spatial_s(n0:n1, :) .* Xb.win;

            % --- Row 1: Mid/Side vectorscope ---------------------------
            ax1 = subplot(3,3,1); plot_vectorscope(ax1, blk_gt, ...
                'Groundtruth');
            ax2 = subplot(3,3,2); plot_vectorscope(ax2, blk_bl, ...
                'HT-Demucs baseline');
            ax3 = subplot(3,3,3); plot_vectorscope(ax3, blk_sp, ...
                'SA-HTDemucs');

            % --- Row 2: panning index per band -------------------------
            ax4 = subplot(3,3,4);
            plot_pan_bands(ax4, band_hz, PSI_mel_gt(:,k), 'GT');
            ax5 = subplot(3,3,5);
            plot_pan_bands(ax5, band_hz, PSI_mel_bl(:,k), 'baseline');
            ax6 = subplot(3,3,6);
            plot_pan_bands(ax6, band_hz, PSI_mel_sp(:,k), 'SA-HTDemucs');

            % --- Row 3: textual panels with metrics --------------------
            ax7 = subplot(3,3,7);
            plot_metrics_panel(ax7, Mgt, phi_gt, k);
            ax8 = subplot(3,3,8);
            plot_metrics_panel(ax8, Mbl, phi_bl, k);
            ax9 = subplot(3,3,9);
            plot_metrics_panel(ax9, Msp, phi_sp, k);
        end
        pause(0.5)
    end
    pause(0.5)
end

% Local plot functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_vectorscope(ax, frameBlock, ttl)
% GonioMeter-style vectorscope: X axis = Side (L-R)/sqrt(2),
% Y axis = Mid (L+R)/sqrt(2). A perfectly mono signal -> vertical line.
L = frameBlock(:,1);
R = frameBlock(:,2);
M = (L + R) / sqrt(2);
Sd = (L - R) / sqrt(2);

% Auto-scaling based on the frame peak (with a floor)
r = max([abs(M); abs(Sd); 1e-3]);
r = 1.1 * r;

hold(ax, 'on');
% Reference circle (relative unit peak)
th = linspace(0, 2*pi, 200);
plot(ax, r*cos(th), r*sin(th), 'Color', [0.85 0.85 0.85]);
% +-45 deg diagonals (correspond to hard-L and hard-R in M/S coords)
plot(ax, [-r r], [-r r], ':',  'Color', [0.7 0.7 0.7]);
plot(ax, [-r r], [ r -r], ':', 'Color', [0.7 0.7 0.7]);
% Axes
plot(ax, [-r r], [0 0], 'Color', [0.6 0.6 0.6]);
plot(ax, [0 0], [-r r], 'Color', [0.6 0.6 0.6]);

% Frame samples cloud
scatter(ax, Sd, M, 6, 'filled', ...
    'MarkerFaceColor', [0.10 0.45 0.80], 'MarkerFaceAlpha', 0.6);

axis(ax, 'equal');
xlim(ax, [-r r]); ylim(ax, [-r r]);
grid(ax, 'on');
xlabel(ax, 'Side  (L-R)/\surd2');
ylabel(ax, 'Mid  (L+R)/\surd2');
title(ax, sprintf('Vectorscope - %s', ttl));
% L / R labels on the diagonals, NOT on the horizontal sides:
% hard-L  => L>0, R=0  =>  Side=+ , Mid=+  -> upper-right diagonal
% hard-R  => L=0, R>0  =>  Side=- , Mid=+  -> upper-left diagonal
text(ax,  0.72*r,  0.72*r, 'L', 'FontWeight', 'bold', ...
    'Color', [0.55 0.30 0.05], ...
    'BackgroundColor', [1 1 1], 'Margin', 1, ...
    'HorizontalAlignment', 'center');
text(ax, -0.72*r,  0.72*r, 'R', 'FontWeight', 'bold', ...
    'Color', [0.05 0.25 0.55], ...
    'BackgroundColor', [1 1 1], 'Margin', 1, ...
    'HorizontalAlignment', 'center');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_pan_bands(ax, band_hz, psi_col, tag)
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
ylabel(ax, '\Psi   (+1 L / -1 R)');
title(ax, sprintf('Panning index per band - %s', tag));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_metrics_panel(ax, M, phi, k)
% Textual panel with correlation, balance, width and broadband azimuth.
cla(ax); axis(ax, 'off');
box(ax, 'on');
txt = {
    sprintf('Correlation: %+0.2f', M.corr(k))
    sprintf('Balance:     %+0.2f', M.balance(k))   % + = L
    sprintf('Width  S/M:  %0.2f',  M.width(k))
    sprintf('Azimuth:     %+0.1f deg', phi(k))     % + = L
    ' '
    '(+ = LEFT, - = RIGHT)'
    };
text(ax, 0.05, 0.5, txt, 'FontName', 'Consolas', ...
    'FontSize', 11, 'VerticalAlignment', 'middle');
% Mini balance bar at the bottom
hold(ax, 'on');
plot(ax, [0 1], [0.08 0.08], 'Color', [0.85 0.85 0.85], 'LineWidth', 4);
bx = 0.5 + 0.5 * max(-1, min(1, M.balance(k)));
plot(ax, [0.5 bx], [0.08 0.08], 'Color', [0.10 0.45 0.80], ...
    'LineWidth', 4);
plot(ax, [0.5 0.5], [0.05 0.11], 'k-');
text(ax, 0.0, 0.02, 'R', 'FontSize', 9);
text(ax, 0.97, 0.02, 'L', 'FontSize', 9);
xlim(ax, [-0.05 1.05]); ylim(ax, [0 1]);
end