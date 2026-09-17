clear
close all
clc

% Separated sources (VDBO)
sources = ["bass","drums","vocals","other"];

% MUSDB18HQ folder path
musdb18hq_path = 'D:\Polimi\PhD\Dataset\MUSDB18HQ\train';

% Audio file name
song = 'A Classic Education - NightOwl';

% HRIR path - SADIE II -> Subject_001 -> DFC -> 44K_16bit
hrir_path = 'D:\Polimi\PhD\Dataset\SADIEII\Subject_001_Wav\DFC\44K_16bit';
hrirs = dir(fullfile(hrir_path,'*ele_0*.wav'));
hrirs = hrirs(~[hrirs.isdir]);

% Loop over sources and convolved by impulse response at random azimuth
for s=1:length(sources)
    % Select random azimuth angle
    idx     = randi(numel(hrirs));
    hrir    = hrirs(idx);
    azi     = hrir.name(5:7);
    if strcmp(azi(end),"_")
        azi = azi(1:2);
    elseif strcmp(azi(end-1),"_")
        azi = azi(1);
    end

    % Import related impulse response
    [hrir, fs] = audioread(fullfile(hrir.folder, hrir.name));

    % Import current source .wav file
    [source, fs_rir] = audioread(fullfile(musdb18hq_path, song, ...
        strcat(sources(s), '.wav')));

    if fs ~= fs_rir
        waring(['BINAURAL_SYNTH: Input song and HRIR must have same ' ...
            'sample rate.'])
        [p, q] = rat(fs / fs_rir);    
        hrir = resample(hrir, p, q);
    end

    % Convolve hrir L wit source L
    binSource_L = filter(hrir(:,1),1,source(:,1));

    % Convolve hrir R wit source R
    binSource_R = filter(hrir(:,2),1,source(:,2));

    % Aggregate into one stereo audio file
    binSource = [binSource_L, binSource_R];

    figure
    sgtitle(strcat("Binaural synthesis: ", sources(s), ...
        ".wav, hrir azimuth = ", num2str(azi), " deg"), 'FontWeight','bold')
    subplot(2,3,1)
    plot(source(:,1))
    ylabel('[Amplitude]')
    title('Input left channel')
    grid

    subplot(2,3,2)
    plot(hrir(:,1))
    title('HRIR - left channel')
    grid

    subplot(2,3,3)
    plot(binSource(:,1))
    title('Output left channel')
    grid

    subplot(2,3,4)
    plot(source(:,2))
    xlabel('[Samples]')
    ylabel('[Amplitude]')
    title('Input right channel')
    grid

    subplot(2,3,5)
    plot(hrir(:,2))
    xlabel('[Samples]')
    title('HRIR - right channel')
    grid

    subplot(2,3,6)
    plot(binSource(:,2))
    xlabel('[Samples]')
    title('Output right channel')
    grid
    
    % Write output binaural audio
    audiowrite(strcat(sources(s),'_azi_',num2str(azi),'.wav'),binSource,fs);
end