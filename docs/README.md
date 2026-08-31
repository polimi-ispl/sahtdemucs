# SA-HTDemucs — listening demo page

Static site with the audio examples for **"Preserving Spatial Cues in Music Source Separation via Lightweight ILD
Correction Heads"**.

Live URL: <https://polimi-ispl.github.io/sahtdemucs/>

It is served straight from this `docs/` folder by GitHub Pages (repo → *Settings* → *Pages* → *Deploy from a branch*,
branch `main`, folder `/docs`), so pushing to `main` publishes it.

---

## Structure

```
docs/
├── index.html          ← the page itself (all content is inline)
├── style.css           ← stylesheet
├── main.js             ← single-player audio coordination + copy-BibTeX button
├── images/             ← figures, shared with the root README (+ ispl_logo.png for the topbar)
└── audio/
    ├── song1/
    │   ├── mixture.wav                                  ← binaural mixture (binauralMUSDB18-HQ)
    │   ├── ref_{drums,bass,other,vocals}.wav            ← ground-truth binaural stems
    │   ├── htdemucs_{drums,bass,other,vocals}.wav       ← frozen HT-Demucs baseline
    │   ├── sahtdemucs_{drums,bass,other,vocals}.wav     ← SA-HTDemucs (frozen backbone + ILD heads)
    │   └── htdemucsspft_{drums,bass,other,vocals}.wav   ← HTDemucs spatial fine-tune (htdemucsspatial/)
    ├── song2/          ← same layout
    └── song3/          ← same layout
```

Three tracks × 13 clips, one `<div class="track-block" id="song<N>">` per track in `index.html`.
The `.wav.reapeaks` files next to the audio are REAPER peak caches; they are unused by the page and harmless.

`main.js` does two things only: it pauses every other `<audio>` element when one starts playing (so A/B comparison is
always single-source), and it wires the *Copy* button next to the BibTeX block.

---

## Regenerating the audio

`notebook/PrepareOnlineDemo.ipynb` renders every clip: it loads a trained checkpoint, separates the selected test
tracks with the three systems and writes the files into `docs/audio/song<N>/` under the names above.

Clips are 44.1 kHz stereo WAV. Keep them short (~20–30 s): long enough to judge the spatial image, short enough that
the folder stays well inside the GitHub Pages limits (100 MB per file, ~1 GB per repository). Trimming by hand:

```bash
ffmpeg -ss 30 -t 30 -i original.wav clip.wav      # 30 s starting at 0:30
ffmpeg -i in.wav -c:a pcm_s16le out.wav           # 32-bit float → 16-bit, halves the size
```

WAV is kept deliberately: lossy codecs alter the very inter-channel cues the page is meant to demonstrate.

---

## Editing the page

Everything lives in `index.html`:

| What | Where |
|---|---|
| Page title | `<h1 class="topbar-title">` in the topbar (the hero `<h1>` below it is commented out) |
| Intro paragraph | `.subtitle` |
| Venue badge and author list | `.venue-badge`, `.authors` — both commented out for anonymous submission |
| Method summary boxes | `.method-card` / `.method-text` |
| Track titles and credits | `.track-title`, `.track-meta`, `.track-license` inside each `.track-block` |
| Citation | `<pre id="bibtex">` at the bottom |

Adding a song means copying one `.track-block` (updating its `id`, title and audio paths) and dropping the matching
`audio/song<N>/` folder next to the others.
