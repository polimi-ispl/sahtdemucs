% evaluate_perceptual_metrics.m
%
% Script comparing, per each song and each source of the test set:
%   - groundtruth   (ref_*)
%   - baseline      (htdemucs_*)
%   - spatial       (sahtdemucs_*)
%
% Computing per each couple (baseline vs GT) and (proposed vs GT) the PEASS
% metrics:
%   - OPS: Overall Perceptual Score
%   - TPS: Target-related Perceptual Score
%   - IPS: Interference-related Perceptual Score
%   - APS: Artifacts-related Perceptual Score
%
% Results are saved into a table (one row per song x source x model) and
% into an aggregate struct for the resume plots.
%
% Dependences:
% - PEASS-Software into the path (addpath(genpath('PEASS-Software')))
%   with compiled MEX (see PEASS-Software/compile.m)

clear
close all
clc

addpath(genpath('helpers\PEASS-Software'));

%% Configuration

% Audio sorces per each song of the test set, for:
% - groundtruth
% - HTDemucs baseline
% - SA-HTDemucs
gt_datapath = 'D:\Polimi\PhD\Dataset\binauralMUSDB18HQ\test';
bl_datapath = 'D:\Polimi\PhD\Dataset\HTDemucs\binauralMUSDB18HQ\test';
sp_datapath = 'D:\Polimi\PhD\Dataset\SAHTDemucs\estimates_20260531_155238';
gt_files = dir(fullfile(gt_datapath, '*'));
gt_files = gt_files(~ismember({gt_files.name}, {'.', '..'}));
bl_files = dir(fullfile(bl_datapath, '*'));
bl_files = bl_files(~ismember({bl_files.name}, {'.', '..'}));
sp_files = dir(fullfile(sp_datapath, '*'));
sp_files = sp_files(~ismember({sp_files.name}, {'.', '..'}));

% Number of songs
nSongs = numel(gt_files);
fprintf('Number of songs: %d\n', nSongs);

% Parameters
sources     = {'bass', 'drums', 'other', 'vocals'};
fs          = 44100;
snippetSec  = 30;       % duration of the song excerpt passed to PEASS (s)

% STFT parameters for sub-band IC
nfft    = 1024;
hop     = nfft/2;
nBands  = 64;

% PEASS options
peassOptions                    = struct();
peassOptions.destDir            = fullfile(pwd, 'peass\');
peassOptions.segmentationFactor = 2; % change if "run out of memory"
peassOptions.fs                 = fs;
if ~exist(peassOptions.destDir, 'dir')
    mkdir(peassOptions.destDir);
end

snippetWorkDir = fullfile(pwd, 'peass_snippets');
if ~exist(snippetWorkDir, 'dir')
    mkdir(snippetWorkDir);
end

% Results tables
% One row per (song, source, model) with model in {'htdemucs','sahtdemucs'}
allRows = {};

%% PRIMARY LOOP: song x source
for i = 1:nSongs
    gt_path      = fullfile(gt_datapath, gt_files(i).name);
    bl_path      = fullfile(bl_datapath, bl_files(i).name);
    spatial_path = fullfile(sp_datapath, sp_files(i).name);

    % Original files of all sources for current song -> needed by PEASS
    % for the target/interference/artifacts decomposition
    gt_files_i = cell(1, numel(sources));
    for s = 1:numel(sources)
        gt_files_i{s} = fullfile(gt_path, [sources{s},'.wav']);
    end

    for s = 1:length(sources)
        fprintf('=== Song %d - %s ===\n', i, sources{s});

        bl_s_path   = fullfile(bl_path, [sources{s}, '.wav']);
        sp_s_path   = fullfile(spatial_path, [sources{s}, '.wav']);

        % Extract snippet for PEASS (to reduce computation time)
        info = audioinfo(gt_files_i{s});
        fs   = info.SampleRate;

        [gt_snip, ~] = extract_snippet(gt_files_i{s}, snippetSec);
        [bl_snip, ~] = extract_snippet(bl_s_path, snippetSec);
        [sp_snip, ~] = extract_snippet(sp_s_path, snippetSec);

        % Guarantee same length
        nMin = min([size(gt_snip,1), size(bl_snip,1), size(sp_snip,1)]);
        gt_snip = gt_snip(1:nMin, :);
        bl_snip = bl_snip(1:nMin, :);
        sp_snip = sp_snip(1:nMin, :);

        % Verificy/guarantee that last PEASS internal audio segmente 
        % (according to aux_cutWav logic) is not too short for the chosen
        % segmentationFactor
        nSafe = safe_trim_for_segmentation(nMin, fs, ...
            peassOptions.segmentationFactor, 1.0);
        if nSafe < nMin
            gt_snip = gt_snip(1:nSafe, :);
            bl_snip = bl_snip(1:nSafe, :);
            sp_snip = sp_snip(1:nSafe, :);
        end

        % Write temporarely on local disk
        tagBase = sprintf('song%d_%s', i, sources{s});

        originalSnipPaths = cell(1, numel(sources));
        for ss = 1:numel(sources)
            thisGtPath = [gt_path, '\', sources{ss}, '.wav'];
            [thisSnip, ~] = extract_snippet(thisGtPath, snippetSec);
            thisSnip = thisSnip(1:min(nSafe, size(thisSnip,1)), :);
            outPath = fullfile(snippetWorkDir, sprintf('%s_orig_%s.wav', ...
                tagBase, sources{ss}));
            audiowrite(outPath, thisSnip, fs);
            originalSnipPaths{ss} = outPath;
        end

        blSnipPath = fullfile(snippetWorkDir, ...
            sprintf('%s_htdemucs.wav', tagBase));
        spSnipPath = fullfile(snippetWorkDir, ...
            sprintf('%s_sahtdemucs.wav', tagBase));
        audiowrite(blSnipPath, bl_snip, fs);
        audiowrite(spSnipPath, sp_snip, fs);

        % -----------------------------------------------------------------
        % PEASS on the snippets
        % -----------------------------------------------------------------
        peass_bl = PEASS_ObjectiveMeasure(originalSnipPaths, blSnipPath, ...
            peassOptions);
        peass_sp = PEASS_ObjectiveMeasure(originalSnipPaths, spSnipPath, ...
            peassOptions);

        % Clean temporary snippet fles
        for ss = 1:numel(sources)
            delete(originalSnipPaths{ss});
        end
        delete(blSnipPath);
        delete(spSnipPath);

        % Clean PEASS output files
        delete(fullfile(peassOptions.destDir, '*'));

        % Aggregate results row by row
        allRows(end+1, :) = { i, sources{s}, "htdemucs", peass_bl.OPS, ...
            peass_bl.TPS, peass_bl.IPS, peass_bl.APS};

        allRows(end+1, :) = { i, sources{s}, "sahtdemucs", peass_sp.OPS, ...
            peass_sp.TPS, peass_sp.IPS, peass_sp.APS};

         fprintf('  PEASS OPS  bl=%.1f  sp=%.1f\n', peass_bl.OPS, ...
             peass_sp.OPS);
    end
end

% Remove directories
rmdir('peass');
rmdir('peass_snippets');

%% Save results table

resultsTable = cell2table(allRows, 'VariableNames', ...
    {'song', 'source', 'model', 'OPS', 'TPS', 'IPS', 'APS'});

disp(resultsTable);

% Saving on disk
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
writetable(resultsTable, ['pmetrics_' timestamp '.csv']);

%% Load results table
resultsTable = readtable('pmetrics_20260623_093320.csv');

%% Final plots and summary

% Comparing HT-Demucs baseline and SA-HTDemucs
summaryTable = groupsummary(resultsTable, {'source', 'model'}, 'mean', ...
    {'OPS', 'TPS', 'IPS', 'APS'});
disp(summaryTable);

metrics   = {'OPS','TPS','APS','IPS'};
nSrc      = numel(sources);
nMetrics  = numel(metrics);

% Row indexes per model
idxBL = strcmp(resultsTable.model, 'htdemucs');
idxSP = strcmp(resultsTable.model, 'sahtdemucs');

tBL = resultsTable(idxBL, :);   % baseline rows
tSP = resultsTable(idxSP, :);   % spatial rows

songLabels   = string(tBL.song);
sourceLabels = string(tBL.source);

% -------------------------------------------------------------------------
%  Figure 1 - Boxplot
%  ------------------------------------------------------------------------
figure
priorityMetrics = {'APS','TPS','OPS','IPS'};

colorBL = '#D95319';   % orange - HT-Demucs (baseline)
colorSP = '#0072BD';   % blu - SA-HTDemucs (proposed)

for m = 1:nMetrics
    mn = priorityMetrics{m};

    bl_vals = tBL.(mn);
    sp_vals = tSP.(mn);
    blMat = nan(nSongs, nSrc);
    spMat = nan(nSongs, nSrc);
    for s = 1:nSrc
        idx = strcmp(sourceLabels, sources{s});
        blMat(:, s) = bl_vals(idx);
        spMat(:, s) = sp_vals(idx);
    end

    ax = subplot(1, nMetrics, m);
    axes(ax);
    hold(ax, 'on');

    % HT-Demucs box
    customboxplot(blMat, 'vertical', colorBL, colorBL, 'k', 'k', ...
        colorBL, 'o', 5, 9, sources, -0.18);

    % SA-HTDemucs box
    customboxplot(spMat, 'vertical', colorSP, colorSP, 'k', 'k', ...
        colorSP, 'o', 5, 9, sources, +0.18);

    set(ax, 'XTick', 1:nSrc, 'XTickLabel', sources, 'XTickLabelRotation', 25);
    xlim(ax, [0.5, nSrc + 0.5]);
    ylim(ax, [0 100]);
    % ylabel(ax, mn);
    title(ax, mn, 'FontWeight', 'bold');
    grid(ax, 'on');
    box(ax, 'on');

    if ismember(mn, {'APS','TPS'})
        set(ax, 'Color', [0.97 0.97 1.0]);
    end

    if m == 1
        h1 = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', colorBL, ...
            'MarkerEdgeColor','none', 'MarkerSize', 10);
        h2 = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', colorSP, ...
            'MarkerEdgeColor','none', 'MarkerSize', 10);
        legend(ax, [h1 h2], {'HT-Demucs','SA-HTDemucs'}, 'Location',...
            'best');
    end
end
sgtitle('PEASS scores','FontSize', 1, 'FontWeight', 'bold');

%  ------------------------------------------------------------------------
%  Figure 2: Delta (SA-HTDemucs − HT-Demucs) per each song, separated per
%  source
%  Dashed line = 0 (no difference). Upside = improvement.
%  ------------------------------------------------------------------------
figure
mainMetrics = {'APS','TPS'};

for m = 1:numel(mainMetrics)
    mn = mainMetrics{m};
    subplot(numel(mainMetrics), 1, m);
    hold on;

    colors = lines(nSrc);
    for s = 1:nSrc
        idx  = strcmp(sourceLabels, sources{s});
        delt = sp_vals(idx) - bl_vals(idx); % delta per each song
        nS   = sum(idx);
        plot(1:nS, delt, 'o-', 'Color', colors(s,:), ...
            'LineWidth', 1.6, 'MarkerFaceColor', colors(s,:), ...
            'MarkerSize', 7);
    end
    yline(0, '--k', 'LineWidth', 1.2);      % zero reference

    set(gca,'XTick', 1:nSongs, 'XTickLabel', ...
        arrayfun(@(k) sprintf('Song %d',k), 1:nSongs, 'UniformOutput', false));
    ylabel(sprintf('\\Delta%s  (SA − HT)', mn));
    title(sprintf('\\Delta%s per song', mn), 'FontWeight','bold');
    legend(sources, 'Location','best');
    grid on;
end

%  ------------------------------------------------------------------------
%  Figure 3: Scatter APS_{baseline} vs APS_{proposed} (and TPS), per song
%  Se points are above the diagonal -> SA-HTDemucs performs better
%  ------------------------------------------------------------------------
figure
for m = 1:numel(mainMetrics)
    mn = mainMetrics{m};
    ax = subplot(1, 2, m);
    hold(ax,'on');

    colors = lines(nSrc);
    bl_v = tBL.(mn);
    sp_v = tSP.(mn);

    for s = 1:nSrc
        idx = strcmp(sourceLabels, sources{s});
        scatter(ax, bl_v(idx), sp_v(idx), 70, colors(s,:), ...
            'filled', 'DisplayName', sources{s});
    end

    % Reference diagonal (equivalent to 'no change')
    lims = [0 100];
    plot(ax, lims, lims, '--k', 'LineWidth',1.0, 'HandleVisibility','off');

    axis(ax,'equal'); xlim(ax,lims); ylim(ax,lims);
    xlabel(ax, sprintf('%s - HT-Demucs', mn));
    ylabel(ax, sprintf('%s - SA-HTDemucs', mn));
    title(ax, sprintf(mn), 'FontWeight', 'bold');
    legend(ax, 'Location','southeast');
    grid(ax,'on');
end
sgtitle('Scatter HT-Demucs vs SA-HTDemucs (per song × source)', ...
    'FontSize', 11);

%  ------------------------------------------------------------------------
%  Print overall summary - number of (song, source) couple where SA-HTDemucs
%  outperforms HT-Demucs on APS e TPS
%  ------------------------------------------------------------------------
fprintf('\n=== SUMMARY: SA-HTDemucs outperforms HT-Demucs on ===\n');
for m = 1:numel(mainMetrics)
    mn = mainMetrics{m};
    delta = tSP.(mn) - tBL.(mn);
    nWin  = sum(delta > 0);
    nTot  = numel(delta);
    fprintf('  %s: %d / %d song × source couples (%.0f%%)\n', ...
        mn, nWin, nTot, 100*nWin/nTot);
end
fprintf('\n  (>50%% = SA-HTDemucs improves that metrics)\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y, fs] = extract_snippet(filePath, snippetSec, startSec)
% EXTRACT_SNIPPET - Estrae un segmento di lunghezza fissa da un file audio,
% a partire da un punto di inizio dato (o dal centro del brano se non
% specificato). Pensato per ridurre drasticamente il costo computazionale
% di PEASS su brani interi, in linea con la metodologia originale del
% paper (segmenti di 5s usati per il training/validazione di PEASS).
%
% Inputs:
%   filePath   : path al file audio
%   snippetSec : durata desiderata del segmento, in secondi (es. 15)
%   startSec   : (opzionale) istante di inizio in secondi. Se omesso,
%                viene preso un estratto centrato sul brano.
%
% Outputs:
%   y, fs      : segnale estratto e sample rate

info = audioinfo(filePath);
fs = info.SampleRate;
totalSec = info.TotalSamples / fs;

if nargin < 3 || isempty(startSec)
    % Estratto centrato: parte dal centro meno meta' della durata
    % desiderata, cosi' da catturare una sezione "tipica" del brano
    % evitando intro/outro spesso silenziose o atipiche
    startSec = max(0, totalSec/2 - snippetSec/2);
end

endSec = min(startSec + snippetSec, totalSec);
sampleStart = max(1, round(startSec*fs)+1);
sampleEnd   = round(endSec*fs);

y = audioread(filePath, [sampleStart, sampleEnd]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nSamplesUse = safe_trim_for_segmentation(nSamplesTotal, fs, ...
    segmentationFactor, minSegSec)
% SAFE_TRIM_FOR_SEGMENTATION - Calcola il numero di campioni da usare
% (a partire dall'inizio del file) in modo che, applicando la stessa
% logica di segmentazione di PEASS (aux_cutWav, overlap 50%), l'ULTIMO
% segmento non risulti più corto di minSegSec secondi.
%
% Questo evita il fallimento silenzioso di extractDistortionComponents
% sui segmenti finali troppo corti (causa dell'errore "file not found"
% in aux_mergeWav), mantenendo la segmentazione necessaria per la memoria.
%
% Uso tipico: trim/zero-pad i tuoi 3 file (gt, baseline, proposed) alla
% stessa nSamplesUse PRIMA di scriverli su disco / passarli a PEASS.

if nargin < 4
    minSegSec = 1.0;   % lunghezza minima di sicurezza per l'ultimo segmento
end

TCut = ceil(nSamplesTotal/segmentationFactor)/fs;
N = 2*round(TCut*fs/2);
step = N/2;

Istart = 1:step:(nSamplesTotal-step+1);
Iend   = min(Istart+N-1, nSamplesTotal);

lastSegLenSec = (Iend(end) - Istart(end) + 1) / fs;

if lastSegLenSec >= minSegSec
    % Già sicuro, nessun trim necessario
    nSamplesUse = nSamplesTotal;
    return
end

% Altrimenti: accorciamo il file totale di un numero di campioni pari
% al "buco" mancante, cosi' che il calcolo di TCut/N cambi leggermente
% e l'ultimo segmento risulti piu' bilanciato. Strategia iterativa
% semplice: riduciamo nSamplesTotal di piccoli passi finche' l'ultimo
% segmento torna sopra la soglia minima.
nSamplesUse = nSamplesTotal;
stepReduce = round(0.25*fs);  % riduci di 0.25s alla volta
maxIter = 200;
it = 0;
while lastSegLenSec < minSegSec && it < maxIter
    nSamplesUse = nSamplesUse - stepReduce;
    if nSamplesUse <= 0
        error('safe_trim_for_segmentation:tooShort', ...
            'Il file e'' troppo corto per il segmentationFactor richiesto.');
    end
    TCut = ceil(nSamplesUse/segmentationFactor)/fs;
    N = 2*round(TCut*fs/2);
    step = N/2;
    Istart = 1:step:(nSamplesUse-step+1);
    Iend   = min(Istart+N-1, nSamplesUse);
    lastSegLenSec = (Iend(end) - Istart(end) + 1) / fs;
    it = it + 1;
end
end