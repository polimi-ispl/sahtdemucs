# HTDemucs spatial fine-tune — headless training & freeze-strategy ablation

Unlike SA-HTDemucs (frozen backbone + small `SpatialCueModule` heads), this
package fine-tunes the **HTDemucs backbone itself** under a spatial objective:
the original time-domain L1 separation loss plus sub-band ILD (and optionally
ITD) MSE terms.  A *freeze strategy* decides which blocks stay trainable.

A notebook is fine for a single interactive run, but not for a sweep: one kernel
holds one strategy, `clear_output()` throws the log away, and a dropped
SSH/Jupyter connection can take the run with it.  Training therefore lives in a
command-line script, one process per strategy; the notebook only evaluates.

| file | role |
|---|---|
| `train.py`            | one run = one process = one freeze strategy |
| `freeze.py`           | the strategy grammar (which blocks stay trainable) |
| `losses.py`           | `HTDemucsSpatialLoss` = λ_td·L1 + λ_ild·ILD MSE + λ_itd·ITD MSE |
| `compare_ablation.py` | table + curves over all runs of a sweep (works while running) |
| `notebook/TestHTDemucsSpatial.ipynb` | test-set comparison of every run vs the frozen baseline |

The dataset, the sub-band cue primitives and the metrics are shared with the
`sahtdemucs` package (`sahtdemucs.dataset`, `sahtdemucs.spatial`,
`sahtdemucs.metrics`), so training and evaluation use the exact same ILD/ITD
definitions.

## Strategy names

A strategy is `"all"` (full fine-tune) or `+`-joined selectors
`<module>[_<range>]`; every block not selected is frozen.

| token | meaning |
|---|---|
| `enc` / `zenc` | frequency (spectrogram) encoder — HTDemucs `model.encoder` |
| `dec` / `zdec` | frequency decoder — `model.decoder` |
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

# 2. follow / compare
tail -f /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial/dec_last2/train.log
python htdemucsspatial/compare_ablation.py \
       /nas/home/macerbi/sahtdemucs/runs/htdemucsspatial \
       --plot ablation.png --csv ablation.csv
```

Each run writes to `<out-root>/<strategy>[__<tag>]/`:

```
htdmcs_sp_dec_last2.pt   best checkpoint (also carries "freeze_strategy" inside)
last.pt                  latest epoch, for --resume  (--no-save-last to skip)
config.json              every hyper-parameter + host, GPU, git commit
history.csv              one row per epoch, appended live
train.log                full log
```

## Notes on running in parallel

* **GPU pinning.** Export `CUDA_VISIBLE_DEVICES` per job and pass
  `--device cuda:0`, so two jobs never race for the same GPU (the default
  `--device auto`, "pick the GPU with the most free memory", does race when
  several runs start at once).
* **Comparability.** Every run uses the same `--seed` (default 1234), so the
  track split and the crops are identical across strategies; the only difference
  is which layers are trainable.
* **Resume.** A killed run restarts where it stopped with `--resume`
  (needs `last.pt`, kept by default).
* **Final test-set metrics** (SI-SDR / per-band ILD & ITD MAE) come from
  `notebook/TestHTDemucsSpatial.ipynb`: point its `RUNS_ROOT` at the `--out-root`
  used above and it discovers, loads and ranks every run automatically.
