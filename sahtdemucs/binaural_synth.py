import os
import json
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# Important: HRIRs
# For this dataset, we use the Head-Related Impulse Responses (HRIRs) associated
# with the Neumann KU100 Binaural Head as documented on the [SADIE II website].
# These measurements correspond to subject D1.

# Constants
SAMPLE_RATE = 44100
STEMS = ["vocals", "bass", "drums", "other"]

# Get possible angles in front of the listener for random distribution of instruments
RANDOM_ANGLES = np.concatenate((np.arange(0, 91, 10), np.arange(270, 351, 10))).tolist()


def make_binaural(y, angle, ir_dir):
    """
    Turn a monophonic signal into a binaural 2-channel signal by
    convolving it with the left and right HRIRs for a given angle
    on the horizontal plane. The elevation for all locations is
    0 degrees.

    Args:
        y:          Monophonic input signal (np.ndarray)
        angle:      Target location of the source along the azimuth (int)
        hrir_dir:   Path to directory containing the HRIRs (str)

    Returns:
    binaural:       2-dimensional array with the binaural left and right
                    channels (np.ndarray)
    """
    # Load HRIR
    hrir_path = os.path.join(ir_dir, f'azi_{angle}_ele_0_DFC.wav')
    hrir, sr = sf.read(hrir_path)

    # Convolve each channel with mono signal
    left = np.convolve(y, hrir[:, 0])
    right = np.convolve(y, hrir[:, 1])

    # Combine into array
    binaural = np.vstack((left, right))
    
    return binaural

def process_song(song_dir, ir_dir, output_dir, angles=None):
    """
    Turn all of the stems from a song in the MUSDB18 dataset
    into binaural 2-channel signals. The resulting binaural
    mixture is the normalized sum of each binaural stem.

    Args:
        song_dir :  Path to directory containing the song's
                    original stems (str or Path)
        ir_dir :    Path to directory containing the HRIRs
                    (str or Path)
        output_dir: Path to target directory where the binaural
                    stems and mixture will be saved (str or Path)
        angles :    (optional) Dictionary mapping each source
                    to a desired azimuth angle. If None, random
                    angles will be assigned (dict or None)
    """
    # make the output directory, if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set angles
    if angles:
        # Load angles if metadata is provided
        stem_loc = []
        for s in STEMS:
            stem_loc.append(angles[s])
    else:
        # Otherwise, assign a random angle without replacement
        # to ensure there is no direct overlap
        stem_loc = random.sample(RANDOM_ANGLES, k=len(STEMS))

    # Save metadata of stem locations
    metadata = dict(zip(STEMS, stem_loc))

    # Initialize mixture
    mixture = None

    for i in range(len(STEMS)):
        # Load stem
        # (samples, channels)
        in_file = os.path.join(song_dir, f"{STEMS[i]}.wav")
        orig_stem, sr = sf.read(in_file)

        # Check that sample rates match
        if not sr == SAMPLE_RATE:
            raise ValueError("The file has the incorrect sample rate!")

        # Convert to mono first
        # (channels, samples)
        mono_stem = librosa.to_mono(orig_stem.T)

        # Make binaural
        # (channels, samples)
        binaural_stem = make_binaural(mono_stem, stem_loc[i], ir_dir)

        # Save
        # (samples, channels)
        out_file = os.path.join(output_dir, f"{STEMS[i]}.wav")
        sf.write(out_file, binaural_stem.T, SAMPLE_RATE)

        # Create mixture by summing stems
        if mixture is None:
            mixture = binaural_stem
        else:
            mixture += binaural_stem

    # Normalize mixture to -1/+1
    mixture_norm = mixture / np.max(np.abs(mixture))

    # Save mixture
    # (samples, channels)
    out_file = os.path.join(output_dir, "mixture.wav")
    sf.write(out_file, mixture_norm.T, SAMPLE_RATE)

    # Dump json
    out_file = os.path.join(output_dir, "metadata.json")
    with open(out_file, 'w') as f:
        json.dump(metadata, f)

def main():
    """
    Parse command-line arguments and run the binaural synthesis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create binaural versions of MUSDB18-HQ tracks using HRIRs.")

    parser.add_argument("--input_dir", dest="input_dir", required=True,
                        help="Path to MUSDB18-HQ directory")
    parser.add_argument("--output_dir", dest="output_dir", required=True,
                        help="Path to directory for saving synthesized data")
    parser.add_argument("--hrir_dir", dest="hrir_dir", required=True,
                        help="Path to directory containing SADIE II HRIR WAV files")
    parser.add_argument("-m", "--metadata", type=str, default=None, metavar="",
                        help="Path to JSON metadata to reproduce Binaural MUSDB (default: None, \
                        a new random version of the dataset will be generated)")

    args = parser.parse_args()

    # Set directory constants
    IN_TRAIN_DIR    = os.path.join(args.input_dir, "train")
    IN_TEST_DIR     = os.path.join(args.input_dir, "test")
    OUT_TRAIN_DIR   = os.path.join(args.output_dir, "train")
    OUT_TEST_DIR    = os.path.join(args.output_dir, "test")
    HRIR_DIR        = args.hrir_dir

    # Validate directory
    if not (os.path.isdir(HRIR_DIR) or os.path.exists(HRIR_DIR)):
        msg = "Provided HRIR database is not a directory or does not exist"
        raise NotADirectoryError(msg)

    # Load metadata, if provided. Metadata are included in the file binaural_musdb_metadata.json available in the
    # subfolder \data. Per each song of the dataset provided as input, it contains aziuth angles per each stem
    # (bass, drums, vocals, others)
    if args.metadata:
        if not (os.path.exists(args.metadata) and args.metadata.endswith(".json")):
            msg = "Metadata file not found or is incorrect file type"
            raise FileNotFoundError(msg)
        with open(args.metadata, "r") as f:
            angle_dict = json.load(f)
    else:
        angle_dict = None

    # Make output directories
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_TEST_DIR, exist_ok=True)

    train_songs = [f for f in os.listdir(IN_TRAIN_DIR) if os.path.isdir(os.path.join(IN_TRAIN_DIR, f))]
    test_songs = [f for f in os.listdir(IN_TEST_DIR) if os.path.isdir(os.path.join(IN_TEST_DIR, f))]

    print("Synthesizing binaural training data")
    for song in tqdm(train_songs):
        if angle_dict:
            song_angles = angle_dict['train'][song]
        else:
            song_angles = None
        input_dir = os.path.join(IN_TRAIN_DIR, song)
        output_dir = os.path.join(OUT_TRAIN_DIR, song)
        process_song(input_dir, HRIR_DIR, output_dir, song_angles)

    print("Synthesizing binaural test data")
    for song in tqdm(test_songs):
        if angle_dict:
            song_angles = angle_dict['test'][song]
        else:
            song_angles = None
        input_dir = os.path.join(IN_TEST_DIR, song)
        output_dir = os.path.join(OUT_TEST_DIR, song)
        process_song(input_dir, HRIR_DIR, output_dir, song_angles)

    print("Binaural synthesis complete.")

if __name__ == '__main__':
    main()

