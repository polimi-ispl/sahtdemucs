import os
import json
import random
import shutil
import tempfile
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

# MoisesDB top-level taxonomy -> Demucs 4 stems (STEMS). Anything that is not
# vocals/bass/drums collapses into "other" (the Demucs convention). MoisesDB,
# unlike Slakh2100, provides real vocals, so it can extend all four stems.
MOISESDB_CATEGORY_TO_STEM = {
    "vocals":        "vocals",
    "bass":          "bass",
    "drums":         "drums",
    "guitar":        "other",
    "piano":         "other",
    "other_keys":    "other",
    "bowed_strings": "other",
    "wind":          "other",
    "percussion":    "other",
    "other_plucked": "other",
    "other":         "other",
}


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


def _to_stereo_cs(a):
    """Normalize an audio array to a ``(2, samples)`` channels-first tensor."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a])            # mono -> duplicate to stereo
    # For music, samples >> channels; put the smaller axis first as "channels".
    if a.shape[0] > a.shape[1]:
        a = a.T
    if a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    return a[:2]


def build_moisesdb_stems(track):
    """Collapse a MoisesDB track's category stems into the 4 Demucs sources.

    Returns ``{stem_name: (2, T)}`` for every name in ``STEMS``; sources absent
    from the track are returned as silence so ``process_song`` always finds all
    four stem files.

    Args:
        track: a ``moisesdb`` track whose ``.stems`` maps each top-level
               taxonomy category to a blended audio array.
    """
    arrs = {cat: _to_stereo_cs(a) for cat, a in track.stems.items()}
    length = max((a.shape[-1] for a in arrs.values()), default=1)
    buffers = {s: np.zeros((2, length), dtype=np.float32) for s in STEMS}
    for cat, a in arrs.items():
        dst = MOISESDB_CATEGORY_TO_STEM.get(cat, "other")
        t = min(a.shape[-1], length)
        buffers[dst][:, :t] += a[:, :t]
    return buffers


def process_moisesdb_track(track, ir_dir, output_dir, tmp_root):
    """Binauralize one MoisesDB track into ``output_dir``.

    Writes the collapsed dry 4-stems to a temporary MUSDB-style folder, then
    reuses :func:`process_song` so the HRIRs and random angle assignment match
    exactly how binauralMUSDB18HQ was built.

    Args:
        track:      a ``moisesdb`` track (needs ``.stems`` and ``.id``).
        ir_dir:     path to the SADIE II HRIR directory.
        output_dir: target directory for this track's binaural mixture + stems.
        tmp_root:   directory under which the temporary dry stems are written.
    """
    dry = os.path.join(tmp_root, track.id)
    os.makedirs(dry, exist_ok=True)
    try:
        for name, buf in build_moisesdb_stems(track).items():
            # soundfile expects (samples, channels)
            sf.write(os.path.join(dry, f"{name}.wav"), buf.T, SAMPLE_RATE)
        process_song(dry, ir_dir, output_dir)
    finally:
        shutil.rmtree(dry, ignore_errors=True)


def process_moisesdb(moisesdb_dir, ir_dir, out_root, test_frac=0.15, seed=0, limit=None):
    """Binauralize a whole MoisesDB into ``out_root/{train,test}/moisesdb_<id>/``.

    MoisesDB has no official train/test split, so ``test_frac`` of the tracks are
    held out into ``out_root/test`` and the rest go to ``out_root/train`` — merged
    alongside the existing binauralMUSDB18HQ tracks. The ``moisesdb_`` name prefix
    keeps the added tracks identifiable inside the merged dataset.

    Args:
        moisesdb_dir: path to the extracted MoisesDB root.
        ir_dir:       SADIE II HRIR directory (``azi_*_ele_0_DFC.wav``).
        out_root:     dataset root that already holds ``train/`` and ``test/``.
        test_frac:    fraction of tracks held out into ``test/`` (default 0.15).
        seed:         RNG seed for the train/test split (default 0).
        limit:        if set, process only the first ``limit`` tracks (debug).
    """
    from moisesdb.dataset import MoisesDB   # optional dependency; import lazily

    db = MoisesDB(data_path=moisesdb_dir, sample_rate=SAMPLE_RATE)
    n = len(db)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = round(n * test_frac)
    test_idx = set(idx[:n_test])
    if limit is not None:
        idx = idx[:limit]

    os.makedirs(os.path.join(out_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "test"), exist_ok=True)
    print(f"MoisesDB: {n} tracks -> {n - n_test} train / {n_test} test "
          f"(seed={seed})")

    tmp_root = tempfile.mkdtemp(prefix="moisesdb_dry_")
    try:
        for done, i in enumerate(idx, 1):
            track = db[i]
            split = "test" if i in test_idx else "train"
            out_dir = os.path.join(out_root, split, f"moisesdb_{track.id}")
            if os.path.exists(os.path.join(out_dir, "mixture.wav")):
                print(f"[{done}/{len(idx)}] skip (exists) {os.path.basename(out_dir)}")
                continue
            print(f"[{done}/{len(idx)}] {split:5s} {track.id} ...", flush=True)
            process_moisesdb_track(track, ir_dir, out_dir, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("MoisesDB binaural synthesis complete.")


def main():
    """
    Parse command-line arguments and run the binaural synthesis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create binaural versions of MUSDB18-HQ tracks using HRIRs.")

    parser.add_argument("--input_dir", dest="input_dir", required=False,
                        help="Path to MUSDB18-HQ directory (MUSDB mode)")
    parser.add_argument("--output_dir", dest="output_dir", required=True,
                        help="Path to dataset root for saving synthesized data "
                             "(holds train/ and test/); shared by both modes")
    parser.add_argument("--hrir_dir", dest="hrir_dir", required=True,
                        help="Path to directory containing SADIE II HRIR WAV files")
    parser.add_argument("-m", "--metadata", type=str, default=None, metavar="",
                        help="Path to JSON metadata to reproduce Binaural MUSDB (default: None, \
                        a new random version of the dataset will be generated)")
    # ── MoisesDB mode (extend the dataset with MoisesDB tracks) ──────────────
    parser.add_argument("--moisesdb_dir", default=None,
                        help="Path to a MoisesDB root. If given, run in MoisesDB "
                             "mode instead of MUSDB mode, writing the binauralized "
                             "tracks under --output_dir/{train,test}.")
    parser.add_argument("--test_frac", type=float, default=0.15,
                        help="MoisesDB fraction held out into test/ (default 0.15)")
    parser.add_argument("--seed", type=int, default=0,
                        help="MoisesDB train/test split seed (default 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="MoisesDB: process only the first N tracks (debug)")

    args = parser.parse_args()

    # ── MoisesDB mode ────────────────────────────────────────────────────────
    if args.moisesdb_dir:
        if not os.path.isdir(args.hrir_dir):
            raise NotADirectoryError(
                "Provided HRIR database is not a directory or does not exist"
            )
        process_moisesdb(
            args.moisesdb_dir, args.hrir_dir, args.output_dir,
            test_frac=args.test_frac, seed=args.seed, limit=args.limit,
        )
        return

    # ── MUSDB18-HQ mode (default) ─────────────────────────────────────────────
    if not args.input_dir:
        parser.error("--input_dir is required for MUSDB mode "
                     "(or pass --moisesdb_dir to run in MoisesDB mode)")

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

