# HTDemucsWithSpatial — Audio Demo Site

Static demo page for the paper **"Preserving Spatial Cues in Music Source Separation
via Lightweight ILD Correction Heads"**.

Live URL (after deployment): `https://<your-username>.github.io/msslnet/`

---

## Project structure

```
msslnet-demo/
├── index.html          ← main page
├── style.css           ← stylesheet
├── main.js             ← minimal JS (audio coordination, copy button)
└── audio/
    ├── song1/
    │   ├── mixture.wav
    │   ├── ref_drums.wav
    │   ├── ref_bass.wav
    │   ├── ref_vocals.wav
    │   ├── ref_other.wav
    │   ├── htdemucs_drums.wav
    │   ├── htdemucs_bass.wav
    │   ├── htdemucs_vocals.wav
    │   ├── htdemucs_other.wav
    │   ├── spatial_drums.wav
    │   ├── spatial_bass.wav
    │   ├── spatial_vocals.wav
    │   └── spatial_other.wav
    ├── song2/          ← same layout
    └── song3/          ← same layout
```

---

## Step 1 — Prepare your audio files

### File format
Use **WAV** (stereo, 44 100 Hz, 16-bit or 32-bit float).
Each clip should be **10–30 seconds** — long enough to judge spatial
qualities, short enough to keep GitHub repo size manageable.
A good rule of thumb: 20 s × 13 files per song × 3 songs ≈ 30–50 MB total.

### How to export the clips
Run your separation pipeline on each test track and export:

```bash
# Reference stems come from MUSDB18-HQ directly:
cp musdb18hq/test/<track>/drums.wav  audio/song1/ref_drums.wav
cp musdb18hq/test/<track>/bass.wav   audio/song1/ref_bass.wav
cp musdb18hq/test/<track>/vocals.wav audio/song1/ref_vocals.wav
cp musdb18hq/test/<track>/other.wav  audio/song1/ref_other.wav

# HTDemucs stems (using the official demucs CLI):
python -m demucs --two-stems=drums --out separated/ test_mixture.wav
# … then copy each output .wav to audio/song1/htdemucs_drums.wav etc.

# HTDemucsWithSpatial stems (using your msslnet inference script):
python -m msslnet.separate test_mixture.wav \
    --model checkpoints/best.pt \
    --out-dir separated_spatial/
# … then copy each output .wav to audio/song1/spatial_drums.wav etc.
```

### Trim to 30 seconds with ffmpeg (optional but recommended)
```bash
ffmpeg -ss 30 -t 30 -i original.wav clip.wav
```

---

## Step 2 — Edit the HTML to match your tracks

Open `index.html` and update:

1. **Author name / venue badge** — search for `Anonymous Author` and `ICASSP 2025`.
2. **Track titles** — find `<h3 class="track-title">` in each song block.
3. **License** — update `<span class="track-license">`.
4. **BibTeX** — update the `<pre id="bibtex">` block at the bottom.
5. **Number of songs** — copy/paste a `<div class="track-block">` block to add more songs.

---

## Step 3 — Deploy on GitHub Pages (free, permanent URL)

### 3a — Put the demo inside your existing repo

The cleanest layout is to place the demo files in a `docs/` subfolder
of your existing `msslnet` repository:

```bash
cd /path/to/msslnet          # your existing git repo
mkdir -p docs/audio/song1 docs/audio/song2 docs/audio/song3

cp /path/to/msslnet-demo/index.html docs/
cp /path/to/msslnet-demo/style.css  docs/
cp /path/to/msslnet-demo/main.js    docs/

# Copy your audio files:
cp -r /path/to/audio/* docs/audio/

git add docs/
git commit -m "add: audio demo page"
git push
```

### 3b — Enable GitHub Pages

1. Go to your repo on GitHub → **Settings** → **Pages**.
2. Under *Build and deployment* → *Source*, choose **Deploy from a branch**.
3. Branch: `main`, folder: `/docs`.
4. Click **Save**.

After ~60 seconds the site will be live at:
```
https://atzerbi.github.io/msslnet/
```

### 3c — Pin the URL

Copy that URL and paste it into:
- Your paper's footnote or header.
- The repo's **About** section (top-right on GitHub → ⚙ → Website).
- Your README's badge or links section.

---

## Alternative deployment: Netlify (drag-and-drop, 30 seconds)

1. Go to <https://app.netlify.com/drop>.
2. Drag the entire `msslnet-demo/` folder onto the page.
3. Netlify gives you a URL like `https://random-name.netlify.app`.
4. Rename it under *Site settings* → *Site name*.

This is useful for a quick preview before committing to GitHub.

---

## Audio file size tips

GitHub Pages has a **100 MB per file** and **1 GB total repo** soft limit.
To stay well under:

| Strategy | Command |
|---|---|
| Trim to 20 s clips | `ffmpeg -ss 10 -t 20 -i in.wav out.wav` |
| Convert to 16-bit | `ffmpeg -i in.wav -c:a pcm_s16le out.wav` |
| Use MP3 (smaller, some browsers) | `ffmpeg -i in.wav -q:a 2 out.mp3` (update `<source type>` in HTML) |
| Use Opus in WebM (best quality/size) | `ffmpeg -i in.wav -c:a libopus out.webm` |

For paper submission demos, WAV is safest for reviewer trust.
MP3 (320 kbps) is a good middle ground if size is an issue.

---

## Citing

When you reference the demo page in your paper, use a footnote like:

> Audio samples available at \url{https://atzerbi.github.io/msslnet/}

Or as a BibTeX `misc` entry:

```bibtex
@misc{atzerbi2025spatialDemo,
  title  = {{HTDemucsWithSpatial} Audio Demo},
  author = {Atzerbi, [Author Names]},
  year   = {2025},
  url    = {https://atzerbi.github.io/msslnet/}
}
```
