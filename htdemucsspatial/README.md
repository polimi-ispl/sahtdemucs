# HTDemucs spatial fine-tune - headless training & freeze-strategy ablation

Unlike SA-HTDemucs (frozen backbone + small `SpatialCueModule` heads), this
package fine-tunes the **HTDemucs backbone itself** under a spatial objective:
the original time-domain L1 separation loss plus a masked sub-band ILD term (and
optionally a sub-band ITD term).  A *freeze strategy* decides which blocks stay
trainable.

A notebook is fine for a single interactive run, but not for a sweep: one kernel
holds one strategy, `clear_output()` throws the log away, and a dropped
SSH/Jupyter connection can take the run with it.  Training therefore lives in a
command-line script, one process per strategy; the notebooks only evaluate.

| file | role |
|---|---|
| `train.py`            | one run = one process = one freeze strategy |
| `freeze.py`           | the strategy grammar (which blocks stay trainable) |
| `losses.py`           | `HTDemucsSpatialLoss` = λ_td·L1 + λ_ild·masked ILD (Huber/MSE) + λ_itd·ITD MSE |
| `compare_ablation.py` | table + curves over all runs of a sweep, from `history.csv` (works while running) |
| `notebook/TestHTDemucsSpatial.ipynb`     | test-set evaluation of **one** run vs the frozen baseline |
| `notebook/AblationHTDemucsSpatial.ipynb` | test-set comparison of **several** runs of a sweep |

The dataset, the sub-band cue primitives and the metrics are shared with the
`sahtdemucs` package (`sahtdemucs.dataset`, `sahtdemucs.spatial`,
`sahtdemucs.metrics`), so training and evaluation use the exact same ILD/ITD
definitions and band layout.

## Training objective

For a batch of $B$ crops, $S = 4$ sources (`drums`, `bass`, `other`, `vocals`)
and stereo (binaural) stems $\hat{s}, s \in \mathbb{R}^{2 \times T}$, the loss
is the source average of three weighted terms:

$$
\mathcal{L} \;=\; \frac{1}{S}\sum_{s=1}^{S}\Big[\,\lambda_{\text{td}}\,\mathcal{L}_{\text{td}}^{(s)}
\;+\; \lambda_{\text{ild}}\,\mathcal{L}_{\text{ILD}}^{(s)}
\;+\; \lambda_{\text{itd}}\,\mathcal{L}_{\text{ITD}}^{(s)}\Big]
$$

`history.csv` logs the three weighted parts (`td`, `ild`, `itd`) and their sum
(`total`), each already divided by $S$.  The loss is always evaluated in FP32,
even under AMP (log, division and soft-argmax are fragile in FP16).

### Separation term - time-domain L1

Exactly the HTDemucs training objective (Rouard et al., 2022):

$$
\mathcal{L}_{\text{td}}^{(s)} = \frac{1}{2BT}\sum_{b,c,t}\big|\hat{s}^{(s)}_{b,c}(t) - s^{(s)}_{b,c}(t)\big|
$$

### Level cue - masked sub-band ILD

Both channels are analysed with an STFT (`--ild-n-fft` 4096, `--ild-hop` 512,
Hann window) and the bins are grouped into $K$ = `--ild-n-bands` (64) bands,
equal-width on the Mel axis (`--ild-band-scale mel`, rectangular, no overlap) or
on the linear axis.  With $P_c(k,t) = \frac{1}{|k|}\sum_{f\in k}|X_c(f,t)|^2$
the mean power of channel $c$ in band $k$:

$$
\mathrm{ILD}(k,t) = 10\log_{10}\frac{P_L(k,t)}{P_R(k,t)} \quad [\text{dB}]
$$

The ILD of a (near-)silent cell is the ratio of two noise floors, so the term is
restricted to the **audible** cells of the target.  Per crop $b$ and band $k$
the reference is the loudest source's peak over the crop,
$\Pi_b(k) = \max_{s', t'} 10\log_{10}\big(P^{(s')}_{L}+P^{(s')}_{R}\big)(k,t')$, and

$$
M^{(s)}_b(k,t) = \mathbb{1}\Big[\,10\log_{10}\big(P_L+P_R\big)^{(s)}_b(k,t) > \Pi_b(k) + \texttt{floor}\Big]\cdot
\mathbb{1}\Big[\,10\log_{10}\big(P_L+P_R\big)^{(s)}_b(k,t) > -60\ \text{dB}\Big]
$$

with `floor` = `--ild-floor-db` (−40 dB; 0 disables the mask).  Using the
loudest source as reference masks a stem that is *silent* in the crop entirely,
while a stem that is only *quieter* keeps its cells.  The term is the masked
mean of a robust error $\rho$:

$$
\mathcal{L}_{\text{ILD}}^{(s)} = \frac{\sum_{b,k,t} M^{(s)}_b(k,t)\;\rho\big(\widehat{\mathrm{ILD}}^{(s)}_b(k,t) - \mathrm{ILD}^{(s)}_b(k,t)\big)}{\sum_{b,k,t} M^{(s)}_b(k,t)},
\qquad
\rho(x) = \begin{cases} \tfrac12 x^2 & |x| < 1\ \text{dB}\\ |x| - \tfrac12 & \text{otherwise}\end{cases}
$$

$\rho$ is the Huber loss with $\beta = 1$ dB (`--ild-criterion huber`, default),
or $\rho(x) = x^2$ with `--ild-criterion mse`.  Huber (≈ dB) is about an order
of magnitude smaller than MSE (dB²): rescale `--lambda-ild` when switching.

### Time cue - sub-band ITD (optional)

Same STFT and band layout.  The cross-spectrum is PHAT-whitened, a band-limited
generalised cross-correlation is rebuilt over the lags
$\tau \in [-\tau_{\max}, \tau_{\max}]$ (`--itd-max-lag`, 64 samples = ±1.45 ms),
and a soft-argmax with temperature $\beta$ (`--itd-beta`, 20) keeps the lag
estimate differentiable:

$$
C(f,t) = \frac{X_L(f,t)\,X_R^*(f,t)}{|X_L(f,t)\,X_R^*(f,t)|},\qquad
g_k(\tau,t) = \frac{1}{|k|}\,\mathrm{Re}\sum_{f\in k} C(f,t)\,e^{\,j2\pi f\tau/N},
$$

$$
\mathrm{ITD}(k,t) = \sum_{\tau} \tau\,\operatorname{softmax}_\tau\!\big(\beta\,g_k(\tau,t)\big)\ [\text{samples}],\qquad
\mathcal{L}_{\text{ITD}}^{(s)} = \frac{1}{BKT_f}\sum_{b,k,t}\big(\widehat{\mathrm{ITD}}^{(s)}_b(k,t) - \mathrm{ITD}^{(s)}_b(k,t)\big)^2
$$

The ITD term is **not** masked and is in samples², so errors can reach ~10³:
`--lambda-itd` must be much smaller than `--lambda-ild`.  It is off by default
(`--lambda-itd 0`).  Above ~1.5 kHz the band-limited correlation has several
peaks inside ±`max_lag`, so the gradient there is mostly noise.

### Balancing the weights

The terms live on very different scales: the waveform L1 is ~10⁻², the Huber
ILD a few dB, the ITD MSE up to 10³ samples².  The CLI defaults
(`--lambda-td 0.9 --lambda-ild 1e-4 --lambda-itd 0`) keep the separation term
dominant.  Larger ILD weights (e.g. `--lambda-td 1 --lambda-ild 1e-2`) trade a
little separation quality for spatial fidelity; the checkpoint rule below stops
that trade from going too far.  Read the first `history.csv` rows to check the
relative size of `td` and `ild` before a sweep.

## Training recipe

* **Model.** Pre-trained `htdemucs` (single model of the bag), or the
  `model_state` of `--init-ckpt` (e.g. the best checkpoint of a td-only run).
  Trainable blocks are selected by `--freeze-strategy` (see below).
* **Data.** `MusdbSpatialDataset` on the `train/` split of `--dataset-root`.
  Random crops of the model segment length (`model.segment` ≈ 7.8 s @ 44.1 kHz),
  `--crops-per-track` (4) per track per epoch, re-drawn until the mixture RMS
  clears `--min-rms`.
* **Split.** Track-level: the last `--valid-split` (20 %) of the tracks is the
  validation set, with one fixed, non-augmented crop per track.  Validation runs
  with a frozen RNG, so every epoch (and every run) scores the same crops.
* **Augmentation** (train only): random gain ±6 dB and a random L/R swap applied
  to the mix and all stems together.  The swap mirrors the azimuth, which is a
  valid binaural scene because the KU100 head is symmetric.
* **Optimisation.** Adam (β = 0.9, 0.999, no weight decay - HTDemucs recipe),
  `--lr` (1e-3; use ≤ 1e-5 for `all`) with cosine annealing to `lr/20` over
  `--epochs`.  Batch `--batch-size` (4) × `--accum-steps` (4) = 16 effective,
  gradient-norm clipping at `--clip-grad` (5), FP16 autocast unless `--no-amp`.
* **Checkpoint selection.** Before training, the initial model's validation
  SI-SDR is measured (reference $\text{SI-SDR}_0$).  An epoch becomes the best
  checkpoint only if it has the lowest validation total loss **and**
  $\text{SI-SDR} \ge \text{SI-SDR}_0 - $ `--max-si-sdr-drop` (0.2 dB), so a
  spatial gain cannot be bought with a separation loss.  `--max-si-sdr-drop inf`
  restores the plain lowest-loss rule.
* **Reproducibility.** All runs share `--seed` (1234), so the split and the crops
  are identical across strategies; `config.json` stores every argument plus
  host, GPU, torch version and git commit.

## Strategy names

A strategy is `"all"` (full fine-tune) or `+`-joined selectors
`<module>[_<range>]`; every block not selected is frozen.

| token | meaning |
|---|---|
| `enc` / `zenc` | frequency (spectrogram) encoder - HTDemucs `model.encoder` |
| `dec` / `zdec` | frequency decoder - `model.decoder` |
| `tenc` / `tdec` | time (waveform) encoder / decoder |
| `_all` (or no range) | every block of that branch |
| `_first<k>` / `_last<k>` | the first / last *k* blocks |
| `_<i>` | a single block, by index from the input side |

Binaural cues are output-side phenomena, so the useful recipes train the last
decoder block(s):

| strategy | trains |
|---|---|
| `dec_last1`, `dec_last2` | last 1–2 frequency-decoder blocks |
| `tdec_last1` | last time-decoder block |
| `dec_last1+tdec_last1` | last decoder block of **both** branches |
| `dec_last2+tenc_first1` | last two decoder blocks + first tencoder (useful when `--lambda-itd > 0`) |
| `all` | everything (full fine-tune) |

See the `freeze.py` docstring for the full grammar.

## Command-line reference

| argument | default | meaning |
|---|---|---|
| `--dataset-root` | *(required)* | dataset root, containing `train/` and `test/` |
| `--out-root` | *(required)* | parent of the run directories |
| `--freeze-strategy` | *(required)* | trainable blocks, see above |
| `--tag` | `""` | suffix of the run directory: `<strategy>__<tag>` |
| `--init-ckpt` | pre-trained | start from this checkpoint's `model_state` |
| `--epochs` | 100 | |
| `--batch-size` / `--accum-steps` | 4 / 4 | effective batch = product |
| `--lr` | 1e-3 | peak learning rate (cosine to `lr/20`) |
| `--valid-split` | 0.2 | fraction of tracks held out for validation |
| `--crops-per-track` / `--min-rms` | 4 / 1e-4 | training crops per track, silence rejection |
| `--clip-grad` | 5.0 | gradient-norm clipping |
| `--lambda-td` / `--lambda-ild` / `--lambda-itd` | 0.9 / 1e-4 / 0 | loss weights |
| `--ild-criterion` | `huber` | `huber` (β = 1 dB) or `mse` |
| `--ild-floor-db` | −40 | audibility mask of the ILD term (0 = off) |
| `--ild-n-fft` / `--ild-hop` | 4096 / 512 | STFT of both cue terms |
| `--ild-n-bands` / `--ild-band-scale` | 64 / `mel` | band layout of both cue terms |
| `--itd-max-lag` / `--itd-beta` | 64 / 20 | GCC-PHAT lag range and soft-argmax temperature |
| `--max-si-sdr-drop` | 0.2 | checkpoint rule, dB below the initial valid SI-SDR |
| `--device` | `auto` | `auto` (most free VRAM), `cuda:N` or `cpu` |
| `--no-amp` / `--no-save-last` / `--resume` | | disable FP16 / skip `last.pt` / continue from `last.pt` |
| `--limit-train-batches` / `--limit-valid-batches` | 0 | cap batches per epoch (smoke tests) |
| `--log-every` | 25 | progress line every N training batches |
| `--workers` / `--seed` | 0 / 1234 | DataLoader workers, global seed |

## Workflow on the remote machine

```bash
cd /nas/home/macerbi/sahtdemucs
git pull

# 0. smoke test (~minutes): verifies dataset, loss, freeze and checkpointing
python -m htdemucsspatial.train \
    --dataset-root /nas/home/macerbi/Dataset/binauralMUSMOISESDB \
    --out-root     /tmp/smoke --freeze-strategy dec_last2 \
    --epochs 1 --limit-train-batches 4 --limit-valid-batches 2

# 1. launch one run per strategy (one process each; `--device cuda:N` or
#    CUDA_VISIBLE_DEVICES pins the GPU, `nohup ... &` detaches the job)
for s in dec_last1 dec_last2 dec_last1+tdec_last1; do
  CUDA_VISIBLE_DEVICES=0 nohup python -m htdemucsspatial.train \
      --dataset-root /nas/home/macerbi/Dataset/binauralMUSMOISESDB \
      --out-root     /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial \
      --freeze-strategy "$s" --epochs 100 --device cuda:0 \
      > /dev/null 2>&1 &
done

# full fine-tune with a stronger ILD weight, tagged to keep it apart
python -m htdemucsspatial.train \
    --dataset-root /nas/home/macerbi/Dataset/binauralMUSMOISESDB \
    --out-root     /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial \
    --freeze-strategy all --lr 1e-5 --lambda-td 1 --lambda-ild 1e-2 \
    --tag td1_ild1e-2_itd0_lr_1e-5

# 2. follow / compare
tail -f /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial/dec_last2/train.log
python htdemucsspatial/compare_ablation.py \
       /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial \
       --plot ablation.png --csv ablation.csv
```

Each run writes to `<out-root>/<strategy>[__<tag>]/`:

```
htdmcs_sp_<strategy>.pt  best checkpoint (see "Checkpoint selection"); also carries
                         epoch, valid loss / ILD / ITD / SI-SDR, reference SI-SDR
                         and "freeze_strategy"
last.pt                  latest epoch + optimizer/scheduler/scaler, for --resume
                         (--no-save-last to skip)
config.json              every hyper-parameter + host, GPU, torch, git commit,
                         trainable-parameter count per group
history.csv              one row per epoch, appended live
train.log                full log
```

`history.csv` columns: `epoch`, `train_{total,td,ild,itd}`,
`valid_{total,td,ild,itd}`, `valid_si_sdr` (dB, stems silent in the crop are
skipped), `lr`, `seconds`.

## Test-set evaluation

`notebook/TestHTDemucsSpatial.ipynb` loads the best checkpoint of one run
(`RUN` under `RUNS_ROOT` = the `--out-root` above), separates every test track
**whole** with both the fine-tune and the frozen pre-trained baseline
(`apply_model`, `SHIFTS = 0` for reproducible numbers), and scores each stem
against its ground truth.  The band layout (`n_fft`, hop, bands, scale,
`max_lag`, `beta`) is read from the run's `config.json`, so the metrics use the
same bands as the loss.

| metric | unit | definition |
|---|---|---|
| SI-SDR | dB | scale-invariant SDR, both channels flattened into one signal |
| ILD MAE | dB, per band | $\lvert\widehat{\mathrm{ILD}} - \mathrm{ILD}\rvert$ |
| ITD MAE | samples (plotted in µs), per band | $\lvert\widehat{\mathrm{ITD}} - \mathrm{ITD}\rvert$, same GCC-PHAT as the loss |
| IPD MAE | rad, per band | wrapped phase error of the cross-spectrum, $\lvert\angle(\hat X_L\hat X_R^*\,(X_LX_R^*)^*)\rvert$, averaged over the bins of the band with weights $\lvert X_LX_R^*\rvert$ of the target |
| ΔIC | -, per band | $\lvert\widehat{\mathrm{IC}} - \mathrm{IC}\rvert$ with $\mathrm{IC} = \lvert\langle X_LX_R^*\rangle\rvert / \sqrt{\langle\lvert X_L\rvert^2\rangle\langle\lvert X_R\rvert^2\rangle}$ over the band and `IC_FRAMES` frames |

Every band metric is computed per band and frame, then averaged over the
audible frames of the target (`FLOOR_DB`, −40 dB, relative to the band's own
peak over the track; `None` = all frames) and finally over tracks.  The
reference stems are single HRIR-rendered sources, so their IC is ≈ 1: a lower
estimated IC (a larger ΔIC) points to leakage from sources at other azimuths or
to diffuse artefacts.  ITD and IPD errors are only perceptually meaningful below
~1.5 kHz (marked on the plots).

The notebook reports an SI-SDR table and a ΔIC table per source (baseline,
fine-tune, delta), a per-track SI-SDR ranking, and per-band ILD / ITD / IPD MAE
curves.  Baseline metrics are cached in `RUNS_ROOT/baseline_metrics_test.json`
and recomputed automatically whenever the band layout, metric set, `FLOOR_DB`,
dataset or track list change.  `notebook/AblationHTDemucsSpatial.ipynb` runs the
same comparison over several runs at once (its own cache:
`baseline_metrics.json`).

## Notes on running in parallel

* **GPU pinning.** Export `CUDA_VISIBLE_DEVICES` per job and pass
  `--device cuda:0`, so two jobs never race for the same GPU (the default
  `--device auto`, "pick the GPU with the most free memory", does race when
  several runs start at once).
* **Comparability.** Every run uses the same `--seed` (default 1234), so the
  track split and the crops are identical across strategies; the only difference
  is which layers are trainable (and the loss weights, if changed).
* **Resume.** A killed run restarts where it stopped with `--resume`
  (needs `last.pt`, kept by default).  The reference SI-SDR is recomputed from
  the initial weights before resuming, so the checkpoint rule is unchanged.
