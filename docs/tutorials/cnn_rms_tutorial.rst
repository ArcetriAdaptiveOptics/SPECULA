.. _cnn_rms_tutorial:

Training a CNN on Simulated WFS Data: Wavefront RMS from Shack-Hartmann Slopes
==============================================================================

This tutorial trains a small convolutional neural network (CNN) inside a
SPECULA simulation. The network learns one number per frame, the rms of the
wavefront, from the slopes of a Shack-Hartmann (SH) wavefront sensor. It
is then tested on an independent simulation.

**What you'll learn:**

* How to feed simulated data to a network: the slope maps of the slope
  computer and :class:`DataBuffer`
* How to train while the simulation runs: :class:`Conv2dNetTrainer`
* How to follow the training
* How to test the trained network on new data: :class:`Conv2dNetTester`

**Prerequisites:**

* Completed the :ref:`scao_basic_tutorial`
* PyTorch installed: ``pip install specula[nn]``
* A GPU is recommended (on an NVIDIA L40S training takes ~1.5 minutes, the
  test less than 1 minute)

Background
----------

**The task.** The rms of the wavefront is a *quadratic* function of the
slopes (roughly, the sum of the squared modal coefficients). A linear
reconstructor cannot estimate it; a CNN can learn it from examples.

**The choices that make it work.** Each choice below avoids a failure we
observed while designing this tutorial:

* **Spots inside the subaperture field.** The SH pixels are ``lambda/(2d)``
  and each subaperture has 12 of them. A seeing-limited spot is then well
  sampled and stays inside the field over the whole seeing range, so the
  slopes are linear. With a field comparable to the spot size, the slopes
  saturate and no network can learn the rms.
* **Open loop.** In open loop the rms is dominated by the low orders, which
  the SH measures well: the label is observable from the input. (In closed
  loop the residual rms is dominated by the fitting error, which the slopes
  do not see directly.)
* **Independent phase screens for training.** A new screen at every frame
  (:class:`AtmoRandomPhase`) makes every frame a new sample. With a frozen
  screen moving with the wind (:class:`AtmoEvolution`) the same turbulence
  passes over the pupil several times during training: the network partly
  memorizes it, and its validation is optimistic. The price: the network
  learns nothing about the temporal evolution, which a per-frame estimate
  does not need.

Tutorial Overview
-----------------

The system is a 2 m telescope in open loop, with a 12x12 SH at 750 nm on a
magnitude 8 star, at 1 kHz. The seeing changes every 0.1 s, uniformly in
[0.5, 1.5]", so the rms covers a wide range.

At every frame:

* :class:`ModalAnalysis` computes the rms (std, piston removed) of the phase
  in the pupil: the **label**.
* :class:`ShSlopec` gives the slopes also as two 12x12 maps (x and y), its
  ``out_slopes_map`` output: the **network input**.
* :class:`DataBuffer` collects both. Every 500 frames it passes the batch to
  the trainer (or the tester).

Four files, all launched from the same folder:

1. ``params_sh_cnn.yml``: the system and the data chain above
2. ``calib_subaps.yml``: subaperture selection (once)
3. ``train_rms.yml``: training
4. ``test_rms.yml``: test on an independent run

Part 1: System
--------------

Create ``params_sh_cnn.yml``:

.. code-block:: yaml

   main:
     class:             'SimulParams'
     root_dir:          './calib/CNN_TUTORIAL'
     pixel_pupil:       120                    # 10 pixels per subaperture
     pixel_pitch:       0.0166667              # [m] D = 2 m
     total_time:        1.000                  # [s] set by the training/test files
     time_step:         0.001                  # [s] 1 kHz

   # New seeing every 0.1 s, uniform in [0.5, 1.5]" (UNIFORM: constant +/- amp/2)
   seeing:
     class:             'PeriodicRandomGenerator'
     update_interval:   0.1
     distribution:      'UNIFORM'
     constant:          [1.0]
     amp:               [1.0]
     seed:              1
     output_size:       1

   wind_speed:
     class:             'WaveGenerator'
     constant:          [20.]                  # [m/s]

   wind_direction:
     class:             'WaveGenerator'
     constant:          [0.]                   # [deg]

   on_axis_source:
     class:             'Source'
     polar_coordinates: [0.0, 0.0]
     magnitude:         8
     wavelengthInNm:    750

   pupilstop:
     class:             'Pupilstop'
     simul_params_ref:  'main'

   atmo:
     class:             'AtmoEvolution'
     simul_params_ref:  'main'
     L0:                40                     # [m]
     heights:           [119.]                 # [m]
     Cn2:               [1.00]
     fov:               0.0
     seed:              1
     inputs:
       seeing:          'seeing.output'
       wind_speed:      'wind_speed.output'
       wind_direction:  'wind_direction.output'
     outputs: ['layer_list']

   prop:                                       # open loop: no DM
     class:             'AtmoPropagation'
     simul_params_ref:  'main'
     source_dict_ref:   ['on_axis_source']
     inputs:
       atmo_layer_list:   ['atmo.layer_list']
       common_layer_list: ['pupilstop']
     outputs: ['out_on_axis_source_ef']

   sh:
     class:             'SH'
     subap_wanted_fov:  5.568                  # [arcsec] 12 pixels
     sensor_pxscale:    0.464                  # [arcsec/pix] lambda/(2d), d = 0.167 m
     subap_npx:         12
     subap_on_diameter: 12
     wavelengthInNm:    750
     inputs:
       in_ef:           'prop.out_on_axis_source_ef'
     outputs: ['out_i']

   detector:
     class:             'CCD'
     simul_params_ref:  'main'
     size:              [144,144]              # 12 subapertures x 12 pixels
     dt:                0.001
     bandw:             300
     photon_noise:      True
     readout_noise:     True
     readout_level:     1.0
     quantum_eff:       0.32
     inputs:
       in_i:            'sh.out_i'
     outputs: ['out_pixels']

   slopec:
     class:             'ShSlopec'
     subapdata_object:  'cnn_tutorial_subaps_n12_th0.5'   # from calib_subaps.yml
     inputs:
       in_pixels:       'detector.out_pixels'
     outputs: ['out_slopes', 'out_slopes_map', 'out_subapdata']

   # Label: rms of the phase in the pupil (piston removed), every frame
   modal_analysis:
     class:             'ModalAnalysis'
     type_str:          'zernike'
     npixels:           120
     obsratio:          0.0
     diaratio:          1.0
     nmodes:            105
     inputs:
       in_ef:           'prop.out_on_axis_source_ef'
     outputs: ['out_modes', 'rms']

   # Input: the x and y slope maps of slopec (2 channels, 12x12)
   data_buffer:
     class:             'DataBuffer'
     buffer_size:       500                    # frames per batch
     inputs:
       input_list: ['input_2d-slopec.out_slopes_map',
                    'rms-modal_analysis.rms']

:class:`DataBuffer` names its outputs after the prefixes in ``input_list``:
``data_buffer.input_2d_buffered`` and ``data_buffer.rms_buffered``.

Part 2: Subaperture Selection
-----------------------------

Create ``calib_subaps.yml``:

.. code-block:: yaml

   sh_subaps:
     class:             'ShSubapCalibrator'
     subap_on_diameter: 12
     output_tag:        'cnn_tutorial_subaps_n12_th0.5'
     energy_th:         0.5                    # keep subapertures with >= 50% illumination
     inputs:
       in_i:            'sh.out_i'

   main_override:
     total_time:        0.001

   # Flat wavefront; nothing downstream of the detector
   remove: ['atmo', 'seeing', 'wind_speed', 'wind_direction',
            'slopec', 'modal_analysis', 'data_buffer']

Run it once:

.. code-block:: bash

   specula params_sh_cnn.yml calib_subaps.yml

Part 3: Training
----------------

Create ``train_rms.yml``:

.. code-block:: yaml

   # Training atmosphere: a new, independent phase screen at every frame
   remove: ['atmo', 'prop', 'wind_speed', 'wind_direction']

   atmo_random:
     class:             'AtmoRandomPhase'
     simul_params_ref:  'main'
     L0:                40
     update_interval:   1                      # [frames]
     source_dict_ref:   ['on_axis_source']
     seed:              1
     inputs:
       seeing:          'seeing.output'
       pupilstop:       'pupilstop'
     outputs: ['out_on_axis_source_ef']

   sh_override:
     inputs:
       in_ef:           'atmo_random.out_on_axis_source_ef'

   modal_analysis_override:
     inputs:
       in_ef:           'atmo_random.out_on_axis_source_ef'

   nn_trainer:
     class:             'Conv2dNetTrainer'
     network_filename:  './trained_models/cnn_tutorial_rms.pth'
     # Network (saved with the weights; the tester reads it from there)
     nmodes:            1                      # one output: the rms
     input_channels:    2                      # x and y slope maps
     channels:          16
     depth:             2                      # 12x12 map -> 3x3 bottleneck
     head_type:         'pooled'
     conv_block_type:   0
     dropout:           0.0
     # Data
     label_offset:      0                      # the label is the rms itself
     val_split:         0.2
     val_size:          2000
     replay_size:       10000
     replay_subsample:  2
     replay_ratio:      10
     replay_batch:      32
     # Optimization
     loss_delta:        10.0                   # [nm] Huber loss transition
     lr_decay:          0.99
     inputs:
       input_2d_batch:  'data_buffer.input_2d_buffered'
       labels:          'data_buffer.rms_buffered'
     outputs: ['loss', 'val_loss']

   main_override:
     total_time:        30.0                   # [s] 60 training steps

Run it:

.. code-block:: bash

   specula params_sh_cnn.yml train_rms.yml

What happens at each training step (every 500 frames):

* Part of the new samples is held out for **validation** (the most recent
  ones, up to ``val_size``): consecutive frames are similar, so validating
  on frames mixed with the training ones would be optimistic.
* The other samples join a **replay buffer** (``replay_size``, one frame
  out of ``replay_subsample``). The network trains on minibatches drawn from
  the whole buffer, ``replay_ratio`` draws per new sample, so it does not
  forget the earlier batches.
* The loss is a Huber loss in the label units: quadratic below
  ``loss_delta``, linear above, so a few large errors do not dominate it.
* The checkpoint (``.pth``) and a statistics file (``_stats.json``, with the
  network architecture and the normalization of inputs and outputs) are
  saved regularly in ``trained_models/``.

At each step the trainer logs a line with the losses on the new samples
and on the validation samples. The last ones of the run:

.. code-block:: text

   [nn_trainer]: Step 30 | LR 7.40e-04 | Train 618.604213 | Val 908.517517 (FVU 0.218) | Best FVU 0.216
   [nn_trainer]: Step 60 | LR 5.47e-04 | Train 510.673458 | Val 753.400024 (FVU 0.161) | Best FVU 0.153

How to read it:

* ``Train`` and ``Val``: the Huber loss on the training and validation
  samples, in the label units.
* ``FVU`` (fraction of variance unexplained) = error variance / label
  variance on the validation samples. 0 is perfect; 1 is no better than
  always predicting the mean. Here the CNN reaches about 0.15. A linear map
  of the same slopes stays at FVU ~1: a linear function of the slopes cannot
  estimate the rms, which is quadratic.
* ``LR``: the learning rate, reduced when the validation FVU stops
  improving.

The training is not seeded, so the numbers change slightly from run to run.

Part 4: Test on an Independent Run
----------------------------------

The validation samples come from the same run as the training ones. A fair
test uses new data. Create ``test_rms.yml``:

.. code-block:: yaml

   # AtmoEvolution (as in params_sh_cnn.yml) with other seeds:
   # other atmosphere, other seeing sequence, and another atmosphere model
   atmo_override:
     seed:              2

   seeing_override:
     seed:              2

   nn_tester:
     class:             'Conv2dNetTester'
     network_filename:  './trained_models/cnn_tutorial_rms.pth'
     label_offset:      0
     inputs:
       input_2d_batch:  'data_buffer.input_2d_buffered'
       labels:          'data_buffer.rms_buffered'
     outputs: ['loss', 'prediction']

   main_override:
     total_time:        20.0                   # [s] 40 batches

Run it:

.. code-block:: bash

   specula params_sh_cnn.yml test_rms.yml

The tester predicts the rms of every batch and prints the statistics at
the end:

.. code-block:: text

   [nn_tester]: Test statistics over 20000 samples
   ...
   Per mode group (RMS in quadrature over the group, in the labels units):
       modes  label_rms   cnn_err  cnn_FVU
         0-0     278.20    112.44    0.163

On 20 000 new frames the network explains about 85% of the variance of the
rms (FVU 0.14-0.16, an error of 105-112 nm on a label spread of 278 nm),
close to its validation FVU: it generalizes to an atmosphere it has never
seen, from another atmosphere model.

.. image:: /_static/tutorial/cnn_rms_test.png
   :width: 100%
   :align: center

The prediction follows the true rms, but with a frame-to-frame jitter the
true rms does not have: each frame is estimated on its own, with the noise of
its slopes and no memory of the previous frames (the network was trained on
independent screens). Averaging the estimate over a few frames reduces it.
