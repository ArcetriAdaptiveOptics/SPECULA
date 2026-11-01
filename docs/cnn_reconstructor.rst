.. _cnn_reconstructor:

CNN and nonlinear reconstructors
================================

SPECULA can replace the linear modal reconstructor with a convolutional
neural network (CNN) trained inside the simulation, or with a linear
reconstructor plus a per-mode saturation correction. This page describes the
objects and the training recipe of the best CNN found for the SOUL
non-modulated pyramid (630 KL modes), from a study by G. Agapito and F. Rossi
(INAF - Osservatorio Astrofisico di Arcetri, 2026). The basic use of the CNN
classes is shown in the tutorial :doc:`tutorials/cnn_rms_tutorial`.

The CNN classes need PyTorch, an optional dependency: ``pip install specula[nn]``.

Objects
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Object
     - Role
   * - ``Slopes2D``
     - The 2D map of a ``Slopes`` object (``Slopes.get2d()``) as a ``BaseValue``:
       x/y slope maps, or one intensity map with ``slopes_from_intensity``.
   * - ``DataBuffer``
     - Collects ``buffer_size`` steps of its inputs and emits them as one batch.
       With ``align`` (default) it emits only the steps in which all the inputs
       were updated, so that maps and labels stay row-aligned.
   * - ``Conv2dNetTrainer``
     - Trains a U-Net (``iteras.lib.efficient_u_net``) on the batches, while the
       simulation runs, and writes the checkpoint: the weights (``.pth``) and a
       ``_stats.json`` with the architecture and the normalization.
   * - ``Conv2dNetTester``
     - Scores a checkpoint on new batches; with its ``reference`` input it also
       scores another reconstructor on the same frames.
   * - ``Conv2dNetRec``
     - A modal reconstructor (``in_slopes`` to ``out_modes``) that runs a
       checkpoint at every step, on the object's device.
   * - ``ModalGainModulation``
     - Random per-mode gain factors for the ``gain_mod`` input of a filter, to
       collect training data with modes at reduced gain.
   * - ``NonlinearResponseCalibrator``, ``NonlinearModalrec``
     - The nonlinear modal reconstructor, see `Nonlinear modal reconstructor`_.

The tester and the reconstructor read the network (architecture,
``nmodes``, ``input_channels``, ``n_frames``) from the checkpoint, so their
configuration cannot disagree with the weights. The labels units are the
loss units: with modes in nm, ``loss_delta`` and ``max_label_rms`` are in nm.

The SOUL recipe
---------------

The configuration files are in ``config/SOUL/cnn/``; they use the SOUL
calibrations of the Arcetri calibration root. The base file
``params_soul_mod0.yml`` is the SOUL loop with the non-modulated pyramid at
1.7 kHz, closed with the linear reconstructor calibrated in partial correction
(SIMPC). The CNN input is the intensity map of a second slope computer,
``slopecNOSN``, without slope-null subtraction: the network needs the static
illumination pattern to sense how large the residual is. The labels are the
630 modes of the residual phase (``ModalAnalysis``).

1. **Phase A, open-loop pretraining** (``phaseA_pushpull.yml``). Each frame is
   a partial-correction background (random turbulence scaled per mode, new
   every 2 frames, seeing 0.4"-1.4") plus a push or pull of one mode, at the
   ``amp100``, ``amp200`` and ``amp500`` amplitudes. The network learns the 630
   modes of the field entering the pyramid, with detector noise on and bright
   stars (magnitude 6.5-8.5)::

      specula params_soul_mod0.yml phaseA_pushpull.yml

   The best network also had "edge bumps" in this phase: blocks of bumps at
   the inner edge of the central obscuration, interleaved with the mode sweeps
   (a ``TimeHistoryGenerator`` instead of the push-pull generator).

2. **Phase B, self-training in closed loop** (``cnn_closed_loop.yml`` +
   ``phaseB_train.yml``). ``Conv2dNetRec`` closes the loop with a frozen copy of
   the checkpoint, while ``Conv2dNetTrainer`` keeps training a second copy on
   the frames the loop produces. Seeing and wind speed change during the run;
   the loop gain ramps from 0 to 1 in the first 0.5 s, and the ramp frames are
   kept in training with a small weight::

      cp trained_models/soul_cnn_m630.pth trained_models/soul_cnn_m630_frozen.pth
      cp trained_models/soul_cnn_m630_stats.json trained_models/soul_cnn_m630_frozen_stats.json
      specula params_soul_mod0.yml cnn_closed_loop.yml phaseB_train.yml

   One run is one iteration: the next run starts from the checkpoint the
   previous one wrote. The best network, ``T2_bright_run25``, is Phase A with
   edge bumps followed by 25 Phase B runs on bright stars only. The study
   accepted a run only if the loop stayed healthy: settled mean Strehl ratio
   above 0.6, no limit cycle in the second half of the run, and no residual
   runaway (1-s mean residual above 1000 nm). Otherwise the next run started
   again from the run's input checkpoint, since a network trained on a lost
   loop learns to lose it. The launch scripts with these rules are in the
   study repository.

3. **Closed loop** (``cnn_closed_loop.yml``), with the frozen checkpoint::

      specula params_soul_mod0.yml cnn_closed_loop.yml

   ``calibrate_gain`` undoes the per-mode shrinkage of the predictions
   measured during training, so that the integrator gains are the effective
   gains.

Never point ``Conv2dNetRec`` to the file a trainer is writing: it changes
every 10 training steps.

Results
^^^^^^^

Against the SIMPC calibrated at the true seeing, each reconstructor at its
own best integrator gains, on 24 paired cases (seeing 0.4"-1.4", magnitude
6.5-10.5, 3 seeds, plus 3 fresh seeds), ``T2_bright_run25``:

* reaches 1.00-1.06 times the SIMPC Strehl ratio in every case; the gain
  grows with the residual, from +0.1% at 0.4" to +3-6% at 1.4";
* converges in every case, acquiring the loop in less than 0.5 s;
* needs no seeing information, while a single SIMPC calibration (0.8") fails
  at 1.4";
* is robust to pupil shifts up to 0.2-0.25 subapertures and to rotations up
  to 1.4 degrees.

Training on bright stars only matters: the network input carries no flux, and
networks trained with all magnitudes respond only 0.5-0.7 to the residual at
magnitude 10.5.

The checkpoints are not distributed with SPECULA.

Acceleration
^^^^^^^^^^^^

The CNN objects run outside the CUDA graphs of ``--accelerate`` (they
override ``trigger()``, or have no graph of their own), and the rest of the
loop is accelerated as usual. The results of an accelerated CNN loop are
not identical to the normal ones: the float32 sums of the captured kernels
differ in the last bits (about 1e-7), cuDNN computes the convolutions in TF32
by default, which turns such a difference into about 1e-4 in the network
output, and the closed loop amplifies it over the run. The statistics of
the run are unaffected.

Nonlinear modal reconstructor
-----------------------------

``NonlinearModalrec`` is the linear ``Modalrec`` plus a per-mode correction of
the sensor saturation, for sensors whose response saturates mode by mode.
``NonlinearResponseCalibrator`` measures the response of each mode,
injected alone at a grid of amplitudes: the slopes projected on the mode's
small-signal interaction vector. The curves are saved as a
``NonlinearCalibration``. At each step the reconstructor inverts the curves on
the linear estimate, then adds one linear correction of the signal the
curves do not explain (see the class documentation).

The calibrator is driven one mode at a time, for example by a
``PushPullGenerator`` with the amplitude grid as ``pattern``:

.. code-block:: yaml

   pushpull:
     class:             'PushPullGenerator'
     nmodes:            54
     push_pull_type:    'PUSHPULL'
     amp:               1.0
     constant_amp:      true
     pattern:           [-200., -100., -50., 50., 100., 200.]   # [nm], no zero
     outputs: ['output']

   nonlinear_calibrator:
     class:             'NonlinearResponseCalibrator'
     nmodes:            54
     intmat_object:     'my_intmat'
     calib_tag:         'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
       in_commands:     'pushpull.output'

and the reconstructor replaces ``Modalrec``:

.. code-block:: yaml

   modalrec:
     class:             'NonlinearModalrec'
     recmat_object:     'my_recmat'
     intmat_object:     'my_intmat'
     nonlinear_calib_object: 'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs: ['out_modes']

The response curves must increase with the amplitude: the calibrator warns
about the modes whose curve rolls over, and the reconstructor refuses them.
Reduce the largest amplitudes, or calibrate fewer modes (``first_mode`` and
``nmodes`` select them). Modes outside the calibration are reconstructed
linearly. ``NonlinearModalrec`` can be accelerated.
