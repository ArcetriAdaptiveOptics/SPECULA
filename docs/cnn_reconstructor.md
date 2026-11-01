# CNN reconstructor: training, testing and closed loop

How the convolutional reconstructor is built, trained and used to drive a loop
in SPECULA, with the SOUL non-modulated pyramid as the worked example
(`config/SOUL/params_pyr_soul*.yml`). It assumes familiarity with SPECULA
objects, YAML parameter files and the `_override` mechanism.

## Pipeline and objects

The network is trained *inside* a simulation. In the first round the loop is
closed by the linear reconstructor and the CNN objects only tap it for data;
afterwards the network drives the loop itself, and can keep training on the
data that loop produces. All configs below share the base `params_pyr_soul.yml`
and are given as a second (or third) file on the command line:

| Config | What it does |
|---|---|
| `..._cnn_train_data.yml` | linear loop + `Conv2dNetTrainer`: generates the data and trains |
| `..._cnn_test_data.yml` | linear loop + `Conv2dNetTester`: scores a checkpoint, and the linear reconstructor, on the same unseen frames |
| `..._cnn_closed_loop.yml` | `Conv2dNetRec` replaces `modalrec` and drives the loop |
| `..._cnn_selftrain.yml` | on top of the closed loop: the trainer keeps learning from the CNN-driven loop, in its own atmosphere |
| `..._linear_closed_loop.yml` | the same loop driven by the linear reconstructor: the reference to compare against |
| `..._cnn_physics_check.yml`, `..._cnn_fixed_seeing.yml` | short diagnostic runs that measure what is learnable in a given regime |

Objects involved:

- **`Slopes2D`** turns a `Slopes` object into its 2D map (`Slopes.get2d()`) as a
  plain `BaseValue`, so it can be buffered. With `slopes_from_intensity` the map
  is a single 120×120 intensity frame (the four pupil images, 7424 illuminated
  pixels); otherwise it is a pair of x/y slope maps.
- **`DataBuffer`** accumulates `buffer_size` steps (1000 ≈ 0.59 s at 1.7 kHz) and
  emits them as one batch. It drops inputs that are not fresh on a step and emits
  only step indices common to all inputs, so buffered signals stay row-aligned.
- **`Conv2dNetTrainer`** (`input_2d_batch`, `labels`, optional `baseline` and
  `gain_mod`) trains the network and writes the checkpoint.
- **`Conv2dNetTester`** runs the same forward pass without gradients and reports
  per-mode and per-group errors at the end of the run. Its optional `reference`
  input takes another estimate of the same modes — typically the linear
  reconstructor's `out_modes`, buffered alongside the labels — and scores it on
  exactly the same frames, which is how the two reconstructors are compared.
- **`Conv2dNetRec`** is a `BaseModalrec` drop-in (`in_slopes` → `out_modes`): it
  builds the map itself, stacks frames, normalizes, runs inference each step.
- **`ModalGainModulation`** feeds `control`'s `gain_mod` during training: one
  random factor per mode, redrawn periodically, times the loop-opening wave (see
  *Making the modes learnable*).
- Both `Conv2dNetTester` and `Conv2dNetRec` take the network from the
  checkpoint alone — architecture, `nmodes`, `input_channels`, `n_frames` — so
  their configs only name the file and cannot disagree with the weights.
- Support libraries: `lib/efficient_u_net.py` (the network), `lib/cnn_checkpoint.py`
  (save/load, architecture checks, calibration, chunked prediction),
  `lib/frame_stacker.py`, `lib/nn_training_diagnostics.py`, `lib/ridge_regression.py`.

The network's input is `slopes2d` fed from **`slopecNOSN`**, a second slope
computer in the base config: the same intensity map as the linear
reconstructor's `slopec`, without the slope-null subtraction. Every CNN config
uses it, and the reason is structural (see *Closed loop*).

Labels come from `modal_analysis.out_modes` (630 KL modes of the *residual*
phase); the network predicts all 630 (`label_offset: 0`), the same modes the
linear reconstructor outputs and the loop corrects.

## Network structure

`UNetRegressor` maps `(B, input_channels × n_frames, H, W)` to `(B, nmodes)`:

- **Encoder**, `depth` levels: a convolution block then `AvgPool2d(2)`, channels
  doubling from `base_channels`. With 120×120 maps and `depth: 4`: 120 → 60 → 30
  → 15 → 7.
- **Bottleneck**: conv block, `SelfAttention` over all positions, conv block.
- **Decoder**, `depth` levels: `ConvTranspose2d(2, stride 2)`, concatenation with
  the encoder skip, conv block. Odd sizes (15 → 7 → 14) are reconciled by
  interpolation; with the spatial head the upsampled map is resized *up* to the
  skip, so the last decoder output keeps the input resolution.
- **Convolution block** (`conv_block_type`): 0 plain 3×3 pair, 1 depthwise
  separable, 3 coordinate attention — all with a residual path and squeeze-and-excitation.
- **Head** (`head_type`):
  - `pooled` (original): every level is pooled to grids of at most 4×4 and fed to
    an MLP narrowing to 32 features. That caps the output at 32 independent
    directions and discards fine detail — unusable for hundreds of modes.
  - `spatial`: a 1×1 convolution reduces the last decoder map to 4 channels, which
    is resampled to `head_grid` × `head_grid` and flattened; the modes are a
    512-wide hidden layer plus a *direct* linear map from the same features, summed.
    `head_grid` must be fine enough to preserve the sampling that matters: at 60 it
    matches the 60×60 pupil images, while 32 averaged neighbouring subapertures and
    mixed adjacent pupils.

SOUL configuration: `channels: 64`, `depth: 4`, `conv_block_type: 1`,
`head_type: spatial`, `head_grid: 60`, `input_channels: 1`, `n_frames: 1`,
`nmodes: 630`.

## Training

Each `Conv2dNetTrainer` trigger receives one buffered batch and:

1. **Reads and filters it.** `n_frames` consecutive frames are stacked along the
   channel axis *before* any frame is dropped, so each sample keeps its true
   history. Then: samples with `gain_mod < gain_mod_threshold` (loop open) are
   dropped; the first `settle_frames: 50` after the loop returns to full gain are
   weighted by `transient_weight: 0.1`; samples whose label RMS exceeds
   `max_label_rms: 400` nm are dropped.
2. **Normalizes.** The input is normalized by one mean/std over all pixels,
   taken from the first batch and then kept: the network's first block ends in a
   GroupNorm, which rescales every sample itself, so the choice barely matters
   and tracking it would gain nothing (that same GroupNorm is why the input must
   keep the static illumination pattern, see *Closed loop*). The per-mode output
   statistics do matter —
   the network predicts in that space — but a few seconds of data don't
   represent an atmosphere that keeps changing, so they follow a moving average
   with time constant `norm_timescale: 30` s of simulated time (~15 seeing
   draws). A plain moving average would make the network chase a moving
   target, so every update also rescales the head's output layers so that
   the *denormalized* predictions stay exactly the same (prediction =
   std · o + mean, and o is a sum of linear layers), which makes the update rate
   irrelevant to stability. Training starts on the first batch. The loss is
   computed on the denormalized output, in nm.
3. **Splits** off the most recent `val_split: 0.2` of the batch, as one
   contiguous block, into a validation set kept as a FIFO of the most recent
   `val_size: 5000` samples (~17 s of the run, ~8 seeing draws; a window of a
   few seconds made the validation numbers follow the conditions). A random
   split would leave every validation frame next to nearly identical training
   frames (a few ms apart): the validation loss then
   rewards memorization, and in practice it sat ~10× below the loss on each new
   batch and even rose while the network was improving on new data. The step
   line reports both the validation loss and FVU (error variance / label
   variance): the loss, in nm, follows the seeing of the last few batches, the
   FVU much less, so progress, the best step and early stopping go by the FVU.
   The first ~5 steps of every run (fresh or resumed: the validation set is
   not saved) are unreliable, in either direction: the set is then one or two
   0.12 s blocks, with little variance of their own and correlated with the
   frames just trained on. After a resume it typically reads ~0.9 at step 1
   and is back to the previous level by step ~7; this is the measurement, not
   the network — it reads the same whether the first trigger takes 5 or 55
   minibatch steps.
4. **Trains from a replay buffer.** Training frames enter a ring buffer
   (`replay_size: 30000`, one of every `replay_subsample: 4` frames), and each
   trigger takes minibatches of `replay_batch: 64` drawn from everything
   collected so far — `replay_ratio: 20` draws per sample it added, ~55 steps.
   Training only on the newest batch means a fraction of a second of one
   atmospheric condition, forgotten when the next arrives.
5. **Validates, saves, reports.** Checkpoint every 10 steps: weights `.pth`,
   plus `_stats.json` with the network's full architecture (`network`), the
   normalization, the per-mode prediction gain (see below), the validation FVU
   and the learning rate. Resuming checks the configured network against the
   recorded one and names every difference, and continues the normalization's
   moving average and the learning rate's annealing. The replay buffer and the
   validation set are not saved: a resumed run rebuilds them. Diagnostics every
   `diag_interval`.

**Loss**: Huber in nm with `loss_delta: 10` (large residuals count linearly, so
occasional outliers don't dominate), plus `norm_loss_weight: 5` × the squared
error per mode normalized by that mode's std. Without the second term the
low-order modes, hundreds of times larger, absorb the whole gradient; too much
of it and capacity goes into modes that are pure noise. Optimizer Adam at 1e-3
with `lr_decay: 0.995` per trigger (~4e-4 at the end of a 100 s run; the learning
rate is saved in the checkpoint, so a resumed run continues the annealing
instead of restarting at 1e-3), `grad_clip_value: 20` (loose: the median
gradient norm is ~2, so only spikes are clipped), `dropout: 0.1`.

**Two settings that look harmless and are not.** Both were found the hard way,
by the training loss sitting far below the loss on each new batch:

- How fast the buffer grows must be compared with how fast training consumes
  it. With a fixed number of steps per trigger and `replay_subsample: 10`, each
  trigger added 80 rows while training drew 6400: four passes over the same
  samples every trigger, 80 by step 20, and the network memorized the buffer
  (training loss 37× below the loss on new batches). That is why the steps are
  now tied to the new data: `replay_ratio` is the number of draws per sample
  added, so a sample is drawn about that many times over its stay in the buffer,
  whatever `replay_subsample` or the filters leave.
- `transient_weight` weighs frames that carry 3–10× the residual of a settled
  one, and in a Huber loss a large sample counts far more than a small one. At
  0.3 they dominated the gradient and the settled regime — the one the loop
  actually runs in — got clearly worse (validation FVU 0.36–0.49 against
  0.11–0.30 at the same step with 0.1).

**Making the modes learnable.** The data-generation side matters as much as the
training side, and it all lives in the training config:

- The atmosphere varies during the run (`PeriodicRandomGenerator`: seeing
  redrawn every 2 s in 0.6–1.2", wind speed every 3 s). The wind direction is
  drawn once per run and held: `AtmoEvolution` rotates each layer by it, so a
  new direction rotates the whole phase map under the pupil at once — a step as
  large as the full turbulence. The Strehl fell to 0 at every such redraw, and
  one of them lost the loop for good; a new speed or seeing the loop rides out.
- The loop is opened 5% of every second (`gain_mod_wave`, a square wave with
  `duty_cycle: 0.95`), which diversifies the data; the trainer drops those frames
  and down-weights the re-convergence transient that follows. Those transients
  are also the only large-residual data the network ever sees, and therefore the
  only training signal for acquiring the loop from far away — which is why
  `max_label_rms` must not be set so low that it removes them. The opening falls
  at k + 0.25 s, away from every redraw of the conditions: a loop that re-closes
  into new, worse conditions can end up outside the pyramid's linear range and
  never recover.
- **Per-mode randomized loop gains.** At nominal gain the loop holds the modes
  above ~50 at ~2 nm each, below the noise: neither the CNN nor a linear map
  fitted on past data can predict them on the next batch. Scaling each mode's
  gain by its own random factor (`ModalGainModulation`, redrawn every 0.5 s)
  makes them move and become learnable. Two tiers (`tiers: [[50, 200, 0.15],
  [200, 630, 0.20]]`), both with a floor of `min_factor: 0.3` of nominal: at each
  redraw a mode of 50–199 is de-gained with probability 15%, one of 200–629 with
  20%, to a factor uniform in [0.3, 1). **This is bounded by loop stability**:
  de-gaining ~100 modes at once is safe, while 197 at once (or down to fully
  open) took the pyramid out of its linear range and the *low-order* loop
  diverged with it.

**Diagnostics** (`_diag.jsonl` next to the checkpoint, plus printed blocks). Per
mode group (0–4, 5–19, 20–49, 50–99, 100–199, 200–499, 500–629): label RMS, CNN
error, FVU, prediction gain, and the same for a ridge-regression baseline on
identical inputs. The number that matters is the **forward-in-time** FVU: both
are fitted on past samples and tested on the next batch, which is the only
honest test — a random split inside a batch leaks, since neighbouring frames are
nearly identical. It separates a network limit from a data limit: if the linear
map also fails, no architecture will help. The block also reports the
loop-transient table (residual and CNN error by frames since the loop
re-closed), the RMS of the labels in the window, the loss on each new batch
before training on it against the training loss, and gradient norms.

## Closed loop

`params_pyr_soul_cnn_closed_loop.yml` removes `rec` and defines `modalrec` as a
`Conv2dNetRec`, built from the checkpoint. Per step it builds the map from the
slopes, stacks the last `n_frames`, normalizes with the trained statistics, runs
the network and denormalizes the output into `out_modes`, which feeds `control`
exactly as a `Modalrec` would.

Two things decide whether this works:

- **The input must come from the same slope computer as training**, and that is
  `slopecNOSN`: the same intensity map as the linear reconstructor's `slopec`,
  *without* the slope-null subtraction. The static illumination pattern that the
  subtraction removes is what tells the network how large the residual is. The
  first block ends in a GroupNorm, with no bias before it (nor on its shortcut),
  so the network's output does not change when the whole input map is scaled —
  and a slope-null-subtracted map is zero-mean and scales with the residual.
  Tripling the residual changes the output by 0.4% on such a map, against 45%
  with the pattern kept: the network can only guess each frame's amplitude from
  secondary cues (the pyramid's nonlinearity, the noise). (Feeding
  a network the other slope computer than it was trained on is just as bad: the
  two maps differ by 1.24 standard deviations of the input normalization on
  every in-pupil pixel.)
- **The loop gains must account for the prediction shrinkage.** The network's
  predictions are shrunk towards the mean, by a factor that varies per mode, and
  in the loop that acts exactly like a lower gain on that mode. The *effective*
  loop gain is therefore `int_gain × prediction gain`, and that is what has to
  land where a loop is stable — for this delay, the linear loop's own gains.

  This is the sharpest edge in the whole pipeline, because the shrinkage is a
  property of the trained weights and changes with every retrain, in both
  directions:

  - with gains about three times too low, the loop crawled to a Strehl of 0.2,
    fell back to 0.02 and wandered without ever settling — each excursion took
    the residual outside the regime the network knows, which made its estimates
    worse in turn;
  - with gains left over from a *more shrunk* network, a retrain that shrank
    less (prediction gains 0.87/0.86/0.80/0.75/0.78/0.83/0.73 against the
    previous 0.70/0.68/0.54/0.47/0.52/0.57) pushed the effective gain to
    0.56–0.59, past what an integrator with two frames of delay tolerates, and
    the Strehl dropped to **0.387**. Dividing the same linear-loop targets by
    the *new* measurement gave **0.875** on the same 3 s of atmosphere. A better
    network made the loop worse, purely through stale gains.

  The closed-loop config therefore sets `calibrate_gain: true` on
  `Conv2dNetRec`: the trainer measures the per-mode prediction gain on each
  batch's training frames after training on them, smoothed across triggers,
  and stores it in the
  stats file as `mode_gain`; the reconstructor divides it out of its
  predictions. `int_gain` in the config is then the *effective* loop gain — the
  linear loop's own values — and it survives retraining. The correction is
  bounded: modes whose prediction gain is below 0.2 are left alone, since
  amplifying what the network barely predicts would mostly amplify noise, and
  the factor is capped at 3. A checkpoint without the measurement raises a clear
  error rather than silently doing nothing. The measurement is in-sample on
  purpose: on held-out frames the network shrinks the modes it predicts poorly
  much harder (0.2–0.4 early in training), and dividing by that would mostly
  amplify noise — at a gain of 0.3 and FVU 0.75 the calibrated prediction's
  noise is ~1.7× the signal.

  For the current checkpoint the gains run 0.94 / 0.96 / 0.84 / 0.74 / 0.71 /
  0.73 / 0.58 per group (lowest mode 0.46), i.e. calibration factors of
  1.04–1.7, and no mode falls below the threshold. Without
  the calibration the same tuning has to be divided by those numbers by hand
  after every retrain (`Conv2dNetTester` takes the same flag, off by default, so
  its numbers describe the raw estimator).

`Conv2dNetRec` runs inference on the CPU, ~150 ms per simulation step here —
about seven times the cost of the physics alone.

## Comparing against the linear reconstructor

`params_pyr_soul_linear_closed_loop.yml` runs the same loop with `rec`. Both
files leave the atmosphere and WFS to the base config and set the same `control`
groups and gains — the training loop's — and the same 630-mode DM, so the two
runs differ only in what turns slopes into modes. Identical gains are fair
because `calibrate_gain` makes them the CNN's effective gains too.

There are two levels of comparison, and both are worth running:

- **Per frame**, with `Conv2dNetTester`'s `reference` input, or offline by
  storing the maps (`DataStore` on `slopes2d.out_value`) together with
  `modal_analysis.out_modes` and `rec.out_modes` and replaying the network over
  them. Offline is the more flexible of the two: it allows restricting the
  comparison to settled frames and re-scoring an updated checkpoint without a
  new simulation.
- **In closed loop**, by Strehl over the same atmosphere.

An earlier network, measured on a settled linear loop, was the worse per-frame
estimator (total error 34.7 nm against the linear reconstructor's 31.0 nm) and
gave Strehl 0.875 against 0.888 in closed loop, when the linear run still
corrected only modes 0–499. The current network is well past it. Its
validation FVU per group, on frames it was not trained on or next to, is

| | 0–4 | 5–19 | 20–49 | 50–99 | 100–199 | 200–499 | 500–629 | total |
|---|---|---|---|---|---|---|---|---|
| CNN | 0.04 | 0.10 | 0.28 | 0.53 | 0.50 | 0.44 | 0.64 | 0.28 |
| linear (ridge) map, same inputs and data | 0.19 | 0.16 | 0.50 | 0.84 | 0.87 | 0.81 | 0.96 | 0.51 |

and it drives the closed loop and the self-training loop stably. About 85% of
its remaining error variance is in modes 50–629, where its predictions are
shrunk to half to three quarters of the true amplitude and the linear map
explains only 4–19% of the variance: a single frame carries little signal about
those modes compared with the noise.

Tuning the loop further is unlikely to help: with the earlier network, a scan at
0.7× the calibrated gains gave Strehl 0.861 against 0.875, i.e. slowing the loop
makes it *worse*. The
loop is lag-limited rather than noise-limited at this speed, which is also why
both reconstructors can leave a large per-frame FVU while the closed loop still
reaches 0.87–0.89. Both estimators are noise-limited in that regime —
even the linear one leaves FVU 0.58 overall — so reconstructing the *present*
frame better has limited headroom.

Beyond more training, the levers are ones a single-frame linear reconstructor
does not have:

- **Training on the network's own loop** (`..._cnn_selftrain.yml`). The first
  round trains on data from the linear loop, and the network then works in a
  loop driven by itself, whose residual statistics differ. In that config
  `Conv2dNetRec` drives with the weights it loaded at start-up while the trainer
  refines a copy and overwrites the checkpoint, so one run is one iteration
  (copy the checkpoint first to keep the previous one). It runs its own
  atmosphere (`seed: 2`) with the training config's varying seeing and wind:
  the base atmosphere (seed 1, fixed conditions) is where the two
  reconstructors are compared, and training on it would teach the network the
  very phase screens it is scored on, in a single condition.
- **Several frames** (`n_frames` > 1). The noise changes from frame to frame
  while the turbulence changes slowly, so stacking frames is the direct way to
  lift the noise-limited modes 50–629. It changes the architecture, so it means
  training from scratch.
- **Predicting ahead.** With a few frames of history the network can also
  estimate the residual at the time the correction actually lands (2 frames
  later), which attacks the lag error — the dominant term of a closed-loop
  residual at this speed. This needs a label shift in the trainer and is not
  implemented yet.
