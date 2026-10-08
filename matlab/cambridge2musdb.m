% cambridge2musdb.m
%
% Converts the raw Cambridge-MT multitrack library (one ZIP per song, as
% downloaded from https://www.cambridge-mt.com/ms/mtk/) into a dataset with
% the MUSDB18-HQ layout expected by sahtdemucs/binaural_synth.py:
%
%   <outRoot>/train/                                     (left empty)
%   <outRoot>/test/<song>/{mixture,vocals,bass,drums,other}.wav
%   <outRoot>/cambridge2musdb_songs.csv                  one row per song
%   <outRoot>/cambridge2musdb_tracks.csv                 one row per track
%
% Every song lands in test/ (binaural_synth.py "test-only set" mode), so the
% binaural version is then generated with, from the repository root:
%
%   python sahtdemucs/binaural_synth.py --input_dir=<outRoot> ...
%          --output_dir=<datasetRoot>/binauralcambridge-mt --hrir_dir=<hrirs>
%
% Per song:
%   1. Version: the full multitrack (<song>_Full.zip, or all the parts
%      <song>_Full_1.zip, <song>_Full_2.zip, ...) is preferred; otherwise
%      the pre-mixed stems (<song>_Stems.zip); otherwise, if useExcerpts,
%      the ~30 s excerpt multitrack (<song>.zip). Stereo mixes/masters only
%      (_Mixes, _Masters) and non-song archives (e.g. 3D-MARCo) are ignored.
%   2. Songs already in MUSDB18-HQ (train or test) are excluded,
%      automatically from the musdb18hq folder names: same normalized title
%      whatever the artist (Cambridge-MT renamed some bands, e.g. Kangoro =
%      ANiMAL), or same artist and one title a prefix of the other
%      ('OnceMore' = 'Once More (With Feeling)').
%   3. Every track (WAV/FLAC) is assigned to vocals/bass/drums/other from
%      its file name (see classifyTrack); reference/rough mixes, previews
%      and click tracks are dropped, and so are alternate/raw/unedited
%      takes when the main take of the same part is also in the session.
%      If requireAllStems, songs where any of vocals/drums/bass/other gets
%      no track are dropped (and their output folder deleted, if a
%      previous run wrote it).
%   4. Tracks are resampled to fsOut, mono tracks duplicated to stereo,
%      zero-padded to the longest track and summed per stem; the mixture is
%      the exact sum of the four stems (absent stems are written as
%      silence). One gain per song brings the loudest of mixture and stems
%      to peakdBFS, so nothing clips at 16 bit and mixture = sum of stems.
%
% Notes:
% - Cambridge-MT sessions are raw, unmixed recordings: the stems are plain
%   unity-gain sums of the tracks (no balance, panning or processing), so
%   they sound less polished than the MUSDB18-HQ stems. Panning is
%   irrelevant downstream, since binaural_synth.py down-mixes every stem to
%   mono before the HRIR convolution.
% - The name-based mapping is heuristic: dryRun = true writes the stem of
%   every track to cambridge2musdb_tracks_dryrun.csv without extracting
%   any audio.
% - The job is resumable: songs whose five WAVs already exist are skipped
%   when skipExisting is true.
%
% Dependencies:
% - Signal Processing Toolbox (resample)

clear
close all
clc

%% Configuration

% Dataset root on the external disk: drive letter on Windows, mount point
% under /Volumes on macOS
if ispc
    datasetRoot = 'D:\Polimi\PhD\Dataset';
else
    datasetRoot = '/Volumes/Elements/Polimi/PhD/Dataset';
end

% Raw Cambridge-MT ZIPs, MUSDB18-HQ (for the overlap check), output root
camRoot   = fullfile(datasetRoot, 'cambridge-mt');
musdbRoot = fullfile(datasetRoot, 'musdb18hq');
outRoot   = fullfile(datasetRoot, 'cambridge-mt-vdbo');

% Split the songs are written to (binaural_synth.py needs both train/ and
% test/ to exist; the other one is created empty)
outSplit  = 'test';

% Dry run: only list the ZIP contents and write the two CSVs (no audio is
% extracted or written) - use it to review the stem assignment
dryRun    = false;

% Output format: MUSDB18-HQ is 44.1 kHz, 16-bit PCM, stereo
fsOut     = 44100;
bitsOut   = 16;

% Peak level (dBFS) of the loudest among mixture and stems, per song
peakdBFS  = -1;

% Fall back to the ~30 s excerpt multitrack when no full version exists
useExcerpts  = true;

% Exclude the songs that are also in MUSDB18-HQ
excludeMusdb = true;

% Stem for percussion (congas, shakers, tambourine, claps, ...): 'drums'
% (as in MUSDB18-HQ, where percussion is part of the drums stem) or 'other'
% (as in the MoisesDB mapping of binaural_synth.py)
percussionStem = 'drums';

% Drop the songs where a stem gets no track at all (e.g. instrumental
% electronic pieces without vocals), whose WAV would be silent: all of
% vocals, drums, bass and other must be non-empty. The output folder of a
% dropped song is deleted if a previous run already wrote it
requireAllStems = true;

% Skip songs whose output WAVs already exist (resumable job)
skipExisting = true;

% Temporary extraction folder (one song at a time, deleted afterwards)
tmpRoot   = fullfile(tempdir, 'cambridge2musdb');

stems     = {'vocals', 'drums', 'bass', 'other'};

%% Song list and version selection

zips  = dir(fullfile(camRoot, '*.zip'));
zips  = {zips(~[zips.isdir]).name};
songs = selectVersions(zips, useExcerpts);
fprintf('%d ZIPs -> %d songs\n', numel(zips), numel(songs));

% MUSDB18-HQ keys (normalized artist + title), for the overlap check
musdbKeys  = {};
musdbNames = {};
if excludeMusdb
    for split = {'train', 'test'}
        d = dir(fullfile(musdbRoot, split{1}));
        d = d([d.isdir] & ~startsWith({d.name}, '.'));
        for k = 1:numel(d)
            parts = strsplit(d(k).name, ' - ');
            musdbKeys{end+1}  = songKey(parts{1}, strjoin(parts(2:end), ' ')); %#ok<SAGROW>
            musdbNames{end+1} = sprintf('%s/%s', split{1}, d(k).name); %#ok<SAGROW>
        end
    end
    fprintf('%d MUSDB18-HQ tracks loaded for the overlap check\n', ...
        numel(musdbKeys));
end

%% Output folders

for split = {'train', 'test'}
    if ~exist(fullfile(outRoot, split{1}), 'dir')
        mkdir(fullfile(outRoot, split{1}));
    end
end

songRows  = {};
trackRows = {};

%% Conversion loop

for s = 1:numel(songs)
    song   = songs(s);
    outDir = fullfile(outRoot, outSplit, song.name);
    fprintf('\n[%d/%d] %s (%s)\n', s, numel(songs), song.name, song.version);

    row = struct('song', song.name, 'version', song.version, ...
        'zips', strjoin(song.zips, ';'), 'status', '', 'musdb', '', ...
        'nVocals', 0, 'nBass', 0, 'nDrums', 0, 'nOther', 0, ...
        'nSkipped', 0, 'durationSec', NaN, 'gaindB', NaN);

    % ---------------------------------------------------------------------
    % Songs that cannot / must not be converted
    % ---------------------------------------------------------------------
    if isempty(song.zips)
        row.status = song.version;      % reason, e.g. 'no full version'
        fprintf('  skipped: %s\n', row.status);
        songRows{end+1} = row; %#ok<SAGROW>
        continue
    end

    if excludeMusdb
        m = find(cellfun(@(k) sameSong(k, songKey(song.artist, ...
            song.title)), musdbKeys), 1);
        if ~isempty(m)
            row.status = 'excluded (in MUSDB18-HQ)';
            row.musdb  = musdbNames{m};
            fprintf('  excluded: in MUSDB18-HQ as %s\n', row.musdb);
            songRows{end+1} = row; %#ok<SAGROW>
            continue
        end
    end

    % Songs with an empty stem, from the ZIP listing (no extraction)
    if requireAllStems
        try
            files = {};
            for z = 1:numel(song.zips)
                files = [files, listZip(fullfile(camRoot, song.zips{z}))]; %#ok<AGROW>
            end
            [~, names, ext] = cellfun(@fileparts, audioFiles(files), ...
                'UniformOutput', false);
            cls = classifyTracks(strcat(names, ext), percussionStem);
        catch
            cls = stems;        % unreadable listing: let unzip report it
        end
        missing = stems(~ismember(stems, cls));
        if ~isempty(missing)
            row.status = ['skipped (no ' strjoin(missing, ', ') ')'];
            fprintf('  skipped: no %s tracks\n', strjoin(missing, '/'));
            if ~dryRun && exist(outDir, 'dir')
                rmdir(outDir, 's');
                fprintf('  removed %s\n', outDir);
            end
            songRows{end+1} = row; %#ok<SAGROW>
            continue
        end
    end

    if ~dryRun && skipExisting && all(cellfun(@(x) isfile(fullfile( ...
            outDir, [x '.wav'])), [stems, {'mixture'}]))
        row.status = 'exists';
        fprintf('  already converted\n');
        songRows{end+1} = row; %#ok<SAGROW>
        continue
    end

    try
        % -----------------------------------------------------------------
        % Track list: listing (dry run) or extraction of the ZIPs
        % -----------------------------------------------------------------
        files = {};
        if dryRun
            for z = 1:numel(song.zips)
                files = [files, listZip(fullfile(camRoot, song.zips{z}))]; %#ok<AGROW>
            end
        else
            tmpDir = fullfile(tmpRoot, song.name);
            if exist(tmpDir, 'dir')
                rmdir(tmpDir, 's');
            end
            for z = 1:numel(song.zips)
                files = [files, unzip(fullfile(camRoot, song.zips{z}), tmpDir)]; %#ok<AGROW>
            end
        end
        files = audioFiles(files);
        if isempty(files)
            error('no WAV/FLAC tracks in the archive');
        end

        % -----------------------------------------------------------------
        % Stem assignment
        % -----------------------------------------------------------------
        [~, names, ext] = cellfun(@fileparts, files, 'UniformOutput', false);
        names = strcat(names, ext);
        [cls, note] = classifyTracks(names, percussionStem);

        % -----------------------------------------------------------------
        % Read, resample, sum per stem
        % -----------------------------------------------------------------
        acc = cell(1, numel(stems));
        for i = 1:numel(stems)
            acc{i} = zeros(0, 2, 'single');
        end
        for k = 1:numel(files)
            fsIn = NaN; nCh = NaN; dur = NaN;
            if ~dryRun && ~strcmp(cls{k}, 'skip')
                info = audioinfo(files{k});
                fsIn = info.SampleRate; nCh = info.NumChannels;
                dur  = info.Duration;
                if nCh > 2
                    cls{k}  = 'skip';
                    note{k} = sprintf('%d channels', nCh);
                else
                    x = audioread(files{k});
                    if fsIn ~= fsOut
                        [p, q] = rat(fsOut / fsIn);
                        x = resample(x, p, q);
                    end
                    if nCh == 1
                        x = [x, x];         % mono -> stereo
                    end
                    i = find(strcmp(stems, cls{k}));
                    acc{i} = addPadded(acc{i}, single(x));
                end
            end
            trackRows{end+1} = struct('song', song.name, 'file', names{k}, ...
                'stem', cls{k}, 'note', note{k}, 'fs', fsIn, ...
                'channels', nCh, 'durationSec', dur); %#ok<SAGROW>
        end

        row.nVocals  = sum(strcmp(cls, 'vocals'));
        row.nBass    = sum(strcmp(cls, 'bass'));
        row.nDrums   = sum(strcmp(cls, 'drums'));
        row.nOther   = sum(strcmp(cls, 'other'));
        row.nSkipped = sum(strcmp(cls, 'skip'));
        fprintf(['  tracks: %d vocals, %d bass, %d drums, %d other, ' ...
            '%d skipped\n'], row.nVocals, row.nBass, row.nDrums, ...
            row.nOther, row.nSkipped);

        if dryRun
            row.status = 'dry run';
            songRows{end+1} = row; %#ok<SAGROW>
            continue
        end

        % -----------------------------------------------------------------
        % Common length, mixture, gain, write
        % -----------------------------------------------------------------
        N = max(cellfun(@(a) size(a, 1), acc));
        if N == 0
            error('all tracks were skipped');
        end
        if requireAllStems && any(cellfun(@isempty, acc))
            % Tracks dropped while reading (e.g. > 2 channels)
            error('no %s track left after reading', ...
                strjoin(stems(cellfun(@isempty, acc)), '/'));
        end
        for i = 1:numel(stems)
            acc{i}(end+1:N, :) = 0;
        end
        mixture = acc{1} + acc{2} + acc{3} + acc{4};

        pk = max(abs(mixture(:)));
        for i = 1:numel(stems)
            pk = max(pk, max(abs(acc{i}(:))));
        end
        gain = 10^(peakdBFS / 20) / pk;

        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        for i = 1:numel(stems)
            audiowrite(fullfile(outDir, [stems{i} '.wav']), ...
                double(gain * acc{i}), fsOut, 'BitsPerSample', bitsOut);
        end
        audiowrite(fullfile(outDir, 'mixture.wav'), double(gain * mixture), ...
            fsOut, 'BitsPerSample', bitsOut);

        row.status      = 'ok';
        row.durationSec = N / fsOut;
        row.gaindB      = 20 * log10(gain);
        fprintf('  written: %.1f s, gain %+.1f dB\n', row.durationSec, ...
            row.gaindB);
    catch err
        row.status = ['error: ' err.message];
        fprintf(2, '  ERROR: %s\n', err.message);
    end

    if ~dryRun && exist(fullfile(tmpRoot, song.name), 'dir')
        rmdir(fullfile(tmpRoot, song.name), 's');
    end
    songRows{end+1} = row; %#ok<SAGROW>

    % Logs rewritten after every song, so they are up to date if the job
    % is interrupted
    writeLogs(outRoot, songRows, trackRows, dryRun);
end

writeLogs(outRoot, songRows, trackRows, dryRun);

status = cellfun(@(r) r.status, songRows, 'UniformOutput', false);
fprintf(['\nDone: %d converted, %d already present, %d excluded ' ...
    '(MUSDB18-HQ), %d with an empty stem, %d skipped/errors.\n'], ...
    sum(strcmp(status, 'ok')), sum(strcmp(status, 'exists')), ...
    sum(startsWith(status, 'excluded')), ...
    sum(startsWith(status, 'skipped (no ')), ...
    sum(~ismember(status, {'ok', 'exists', 'dry run'}) & ...
    ~startsWith(status, {'excluded', 'skipped (no '})));

%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function songs = selectVersions(zips, useExcerpts)
% SELECTVERSIONS
%   Groups the ZIPs by song and picks the version to convert. A song with
%   empty .zips is reported but not converted (.version holds the reason).

base = regexprep(zips, '\.zip$', '', 'ignorecase');
kind = regexp(base, '_(Full(_\d+)?|Stems|Mixes|Masters)$', 'match', 'once');
root = regexprep(base, '_(Full(_\d+)?|Stems|Mixes|Masters)$', '');

names = unique(root, 'stable');
songs = struct('name', {}, 'artist', {}, 'title', {}, 'version', {}, ...
    'zips', {});
for n = 1:numel(names)
    idx = find(strcmp(root, names{n}));
    k   = kind(idx);
    z   = zips(idx);
    tok = regexp(names{n}, '^([A-Za-z0-9]+)_(.+)$', 'tokens', 'once');

    if isempty(tok) || contains(names{n}, ' ')
        % Not an <Artist>_<Title> song archive (e.g. 3D-MARCo samples)
        version = 'not a song archive';  sel = {};
    elseif any(strcmp(k, '_Full'))
        version = 'full';                sel = z(strcmp(k, '_Full'));
    elseif any(startsWith(k, '_Full_'))
        parts = sort(str2double(erase(k(startsWith(k, '_Full_')), '_Full_')));
        if isequal(parts(:)', 1:numel(parts))
            version = 'full (multi-part)';
            sel = z(startsWith(k, '_Full_'));
        else
            version = 'incomplete multi-part';  sel = {};
        end
    elseif any(strcmp(k, '_Stems'))
        version = 'stems';               sel = z(strcmp(k, '_Stems'));
    elseif any(strcmp(k, ''))
        if useExcerpts
            version = 'excerpt';         sel = z(strcmp(k, ''));
        else
            version = 'excerpt only';    sel = {};
        end
    else
        version = 'mixes/masters only';  sel = {};
    end
    if isempty(sel) && strcmp(version, 'incomplete multi-part') && ...
            useExcerpts && any(strcmp(k, ''))
        version = 'excerpt';             sel = z(strcmp(k, ''));
    end

    if isempty(tok), tok = {names{n}, ''}; end
    songs(end+1) = struct('name', names{n}, 'artist', tok{1}, ...
        'title', tok{2}, 'version', version, 'zips', {sel}); %#ok<AGROW>
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function key = songKey(artist, title)
% SONGKEY
%   Normalized "artist|title": lower case, alphanumerics only, no leading
%   "the" (e.g. 'The Doppler Shift' == 'DopplerShift')

f   = @(s) regexprep(regexprep(lower(s), '[^a-z0-9]', ''), '^the', '');
key = [f(artist) '|' f(title)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tf = sameSong(keyA, keyB)
% SAMESONG
%   Same title, whatever the artist (bands renamed in Cambridge-MT, e.g.
%   Kangoro = ANiMAL); or same artist and one title a prefix of the other
%   ('OnceMore' == 'Once More (With Feeling)')

a = strsplit(keyA, '|');
b = strsplit(keyB, '|');
n = min(strlength(a{2}), strlength(b{2}));
tf = strcmp(a{2}, b{2}) || (strcmp(a{1}, b{1}) && n >= 4 && ...
    (startsWith(a{2}, b{2}) || startsWith(b{2}, a{2})));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function names = listZip(zipPath)
% LISTZIP - Entry names of a ZIP without extracting it. 
% 
%   CP437 is the charset of non-UTF-8 entries. UTF-8-flagged entries are 
%   still decoded as UTF-8.

zf = java.util.zip.ZipFile(java.io.File(zipPath), ...
    java.nio.charset.Charset.forName('IBM437'));
c  = onCleanup(@() zf.close());
en = zf.entries();
names = {};
while en.hasMoreElements()
    e = en.nextElement();
    if ~e.isDirectory()
        names{end+1} = char(e.getName()); %#ok<AGROW>
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function files = audioFiles(files)
% AUDIOFILES
%   WAV/FLAC tracks only, without the macOS resource forks (__MACOSX/, ._*)

files = strrep(files, '\', '/');
[~, n, e] = cellfun(@fileparts, files, 'UniformOutput', false);
keep  = ismember(lower(e), {'.wav', '.flac'}) & ...
    ~contains(files, '__MACOSX/') & ~startsWith(n, '._');
files = sort(files(keep));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function acc = addPadded(acc, x)
% ADDPADDED - acc + x, zero-padding the shorter of the two

n = max(size(acc, 1), size(x, 1));
acc(end+1:n, :) = 0;
x(end+1:n, :)   = 0;
acc = acc + x;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tok = nameTokens(name)
% NAMETOKENS - Lower-case word tokens
%   Lower-case word tokens of a track file name, without the leading track
%   number: the words split at non-letters, then also at camelCase
%   boundaries ('LeadVox' -> lead, vox; 'LowBVs' -> low, bvs) and before the
%   last capital of an acronym ('ACGtr' -> ac, gtr), plus the unsplit words
%   ('leadvox')

[~, n] = fileparts(name);
n     = regexprep(n, '^\d+[\s_\-\.]*', '');
whole = regexp(lower(n), '[a-z]+', 'match');
n     = regexprep(n, '([a-z])([A-Z])', '$1 $2');
camel = regexp(lower(n), '[a-z]+', 'match');
n     = regexprep(n, '([A-Z]+)([A-Z][a-z])', '$1 $2');
split = regexp(lower(n), '[a-z]+', 'match');
tok   = unique([whole, camel, split]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cls, note] = classifyTracks(names, percussionStem)
% CLASSIFYTRACKS - Assigns every track of a song to vocals/bass/drums/other
% (VDBO) or 'skip'.
%
%   The rules are applied in order, the first match wins:
%   1. skip: reference/rough mixes, previews, click tracks
%   2. skip: alternate/raw/unedited/uncomped/untuned/scratch takes, only
%      when a main take of the same part (same words) is in the session
%   3. vocals: vocal words, unless a synth/keys word ('SynthChoir') or a
%      guitar word without lead/backing ('ElecGtr_VoxM160', Vox amp) is
%      also there
%   4. drums: drum kit words (kick, snare, tom, hat, overheads, ...)
%   5. percussion -> percussionStem (congas, shakers, tambourine, ...);
%      then, in sessions with a drum kit, bare room mics ('Rooms') ->
%      drums; in choir sessions (soprano/tenor/alto sections), vocals for
%      every track without an instrument word, 'Basses' included
%   6. bass: bass words, unless a wind instrument word is also there
%      ('BassClarinet')
%   7. other: everything else

drumW  = {'kick','kik','kicks','snare','snares','sanre','snr','tom', ...
    'toms','hat','hats','hihat','hh','hhat','overhead','overheads', ...
    'oh','ohs','cymbal','cymbals','crash','crashes','ride','china', ...
    'splash','drum','drums','drumkit','kit','drumloop','drumloops', ...
    'drummachine','rim','rimshot','sidestick','kicksnare','brushes', ...
    'beater','beat','beats','breaks','fill','fills','trash','rototom', ...
    'rototoms','sn','kck','kic','rde','cymb'};
percW  = {'perc','percs','percussion','conga','congas','bongo', ...
    'bongos','shaker','shakers','tambourine','tambourines','tamb', ...
    'tambo','tamborine','clap','claps','handclap','handclaps', ...
    'cowbell','cajon','djembe','timbale','timbales','maracas', ...
    'cabasa','guiro','guira','claves','clave','woodblock','snaps', ...
    'fingerclicks','stomp','stomps','taiko','pandero','guasa', ...
    'bombo','llamador','guache','shekere','vibraslap','rainstick', ...
    'timpani','timp','gong','triangle','tabla','darbuka','zills', ...
    'anvil','paliteo','cuica','agogo','surdo','sleigh','sleighbells', ...
    'maraca','rainmaker'};
vocW   = {'vox','voxl','voxr','vocal','vocals','voc','vocs','voice', ...
    'voices','choir','choirs','bv','bvs','bgv','bgvs','leadvox', ...
    'backingvox','adlib','adlibs','libs','whisper','whispers', ...
    'scream','screams','shout','shouts','speech','narration', ...
    'spoken','rap','singer','sing','harmonies','falsetto','beatbox', ...
    'soprano','sopranos','laugh','giggles'};
synthW = {'synth','synths','electro','pad','pads','mellotron','keys', ...
    'keyboard','piano','organ','vocoder','strings'};
sectW  = {'soprano','sopranos','alto','altos','tenor','tenors', ...
    'baritone','baritones','bass','basses'};
instrW = {'sax','saxophone','saxes','flute','flutes','clarinet', ...
    'clarinets','oboe','oboes','bassoon','bassoons','trumpet', ...
    'trumpets','trombone','trombones','horn','horns','tuba','tubas', ...
    'recorder','violin','violins','viola','violas','cello','cellos', ...
    'guitar','gtr','gtrs','ukulele','ukelele','banjo','mandolin', ...
    'harp','piano','organ','synth','harmonica','brass','strings'};
windW  = {'clarinet','clarinets','sax','saxophone','trombone', ...
    'trombones','flute','harmonica','recorder','tuba','horn'};
gtrW   = {'gtr','gtrs','guitar','guitars'};
leadW  = {'lead','backing','leadvox','backingvox','bg','bgv','bv', ...
    'vocal','vocals'};
roomW  = {'room','rooms','amb'};
bassW  = {'bass','basses','bassdi','bassamp','subbass','sub', ...
    'doublebass','doublebasses','contrabass','contrabasses', ...
    'synthbass','basssynth','bassgtr'};
varW   = {'alt','alternate','alttake','take','takes','uncomped', ...
    'unedited','untuned','raw','scratch'};

n    = numel(names);
tok  = cellfun(@nameTokens, names, 'UniformOutput', false);
has  = @(t, w) any(ismember(t, w));
cls  = repmat({''}, 1, n);
note = repmat({''}, 1, n);

% Main-take signatures: split words without variant words and numbers
sig = cell(1, n);
for k = 1:n
    [~, b] = fileparts(names{k});
    b = regexprep(regexprep(b, '^\d+[\s_\-\.]*', ''), ...
        '([a-z])([A-Z])', '$1 $2');
    w = regexp(lower(b), '[a-z]+', 'match');
    sig{k} = strjoin(sort(setdiff(w, varW)), ' ');
end
isVar = cellfun(@(t) has(t, varW), tok);

% Choir session: soprano, or both tenor and alto, sections recorded
% without an instrument word (not 'TenorSax', 'AltoSax')
voiceTok = @(w) any(cellfun(@(t) has(t, w) && ~has(t, instrW), tok));
choirSession = voiceTok({'soprano','sopranos'}) || ...
    (voiceTok({'tenor','tenors'}) && voiceTok({'alto','altos'}));

% Session with a drum kit: bare room mics go to drums
drumSession = any(cellfun(@(t) has(t, drumW), tok));

for k = 1:n
    t = tok{k};
    if (has(t, 'mix') && has(t, {'reference','ruff','rough'})) || ...
            has(t, 'preview') || ...
            (has(t, {'click','clicks'}) && ~has(t, 'finger'))
        cls{k} = 'skip';  note{k} = 'mix/preview/click';
    elseif isVar(k) && ~isempty(sig{k}) && ...
            any(strcmp(sig(~isVar), sig{k}))
        cls{k} = 'skip';  note{k} = 'alternate take';
    elseif has(t, vocW) && ~has(t, synthW) && ...
            ~(has(t, gtrW) && ~has(t, leadW))
        cls{k} = 'vocals';
    elseif has(t, drumW)
        cls{k} = 'drums';
    elseif has(t, percW)
        cls{k} = percussionStem;  note{k} = 'percussion';
    elseif drumSession && has(t, roomW) && ~has(t, [instrW, vocW, bassW])
        cls{k} = 'drums';  note{k} = 'room mic';
    elseif choirSession && ~has(t, setdiff(instrW, sectW))
        cls{k} = 'vocals';  note{k} = 'choir session';
    elseif has(t, bassW) && ~has(t, windW)
        cls{k} = 'bass';
    else
        cls{k} = 'other';
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function writeLogs(outRoot, songRows, trackRows, dryRun)
% WRITELOGS - Song- and track-level CSVs (the dry run writes *_dryrun.csv)

sfx = '';
if dryRun
    sfx = '_dryrun';
end
if ~isempty(songRows)
    writetable(struct2table([songRows{:}], 'AsArray', true), ...
        fullfile(outRoot, ['cambridge2musdb_songs' sfx '.csv']));
end
if ~isempty(trackRows)
    writetable(struct2table([trackRows{:}], 'AsArray', true), ...
        fullfile(outRoot, ['cambridge2musdb_tracks' sfx '.csv']));
end
end
