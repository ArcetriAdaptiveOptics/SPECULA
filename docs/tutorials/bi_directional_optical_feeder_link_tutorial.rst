.. _bi_directional_optical_feeder_link_tutorial:

Bi-directional Optical Feeder Link Tutorial
=========================================

This tutorial guides you through creating and running a bi-directional optical feeder link simulation in SPECULA.

**What you'll learn:**

* How to make a propagation from and to a LEO satellite including time-delays.
* Applying a Point-Ahead-Angle (PAA) and telescope slewing.
* Using the infinite phase screen method in SPECULA to deal with fast evolving turbulence due to telescope slewing.
* Calibration of a Pyramid WFS and interaction and reconstruction matrices.
* Post-compensation of the downlink IR laser using a Pyramid WFS, modal reconstruction and DM.
* Pre-compensation of the uplink IR laser beam using the downlink.
* Computation of downlink Strehl ratio and uplink power loss.

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* Basic understanding of propagation concepts and Fresnel propagation (see corresponding tutorial)
* Python and YAML familiarity

Tutorial Overview
-----------------

We'll simulate a simple bi-directional optical feeder link setting with:

* Circular pupil (1m class)
* 3 atmospheric layers
* Upwards and downwards Fresnel propagation to and from a LEO satellite at 400 km with a PAA of 5 arcsec
* Including telescope slewing via setting the effective wind speed of atmospheric layers
* Post-compensation of the downlink and pre-compensation of the uplink
* Calibration of a Pyramid WFS, the interaction matrix and reconstructor
* Storing downlink Strehl ratio and uplink power loss


Part 1: System configuration
-----------------
In order to calculate the PAA, time delays, effective wind speeds and directions the SPECULA functions ``calc_paa``, ``calc_timing_uplink_downlink`` and ``calc_effective_wind_speed`` in ``lib/fsoc_lib.py`` can be used. Those functions require as input atmospheric parameters, such as Cn2, wind speed and directions, as well as space object specific parameters such as height and speed.

The calculated PAA is used in ``polar_coordinates`` of the ``Source`` class. The effective wind speeds and directions are set as input for the ``AtmoInfiniteEvolutionUpDown`` class and the time delays are set via ``extra_delta_time_up`` and ``extra_delta_time_down`` for the ``AtmoInfiniteEvolutionUpDown`` class.

To set up the main SPECULA parameter file, create a YAML configuration file, for example ``params_leo_satellite.yml``:

.. code-block:: yaml

   # main section with simulation parameters used by most of the components
    main:
      class:             'SimulParams'
      root_dir:          './calib/'            # Root directory for calibration manager
      pixel_pupil:       120                   # Linear dimension of pupil phase array
      pixel_pitch:       0.00833               # [m] Pitch of the pupil phase array
      total_time:        1.0                   # [s] Total simulation running time
      zenith_angle:      50.0
      time_step:         0.001                 # [s] Simulation time step

   # Atmospheric seeing (approx. inverse of Fried parameter r0).
    seeing:
      class:             'WaveGenerator'
      constant:          1.6                    # ["] seeing value (500nm and at zenith)

   # Wind speed and direction.
    wind_speed:
      class:             'WaveGenerator'
      constant:          [3.54,84.94,387.74]     # [m/s] Wind speed value
    wind_direction:
      class:             'WaveGenerator'
      constant:          [90.,270.,0.]           # [degrees] Wind direction value

   # Source definitions for upwards and downwards propagation. This defines the direction of the propagation the electromagnetic field
   # the flux intensity and the wavelength.
    source_up:
      class:             'Source'
      polar_coordinates: [0.0, 0.0]           # [arcsec, degrees] source polar coordinates
      magnitude:         0                    # source magnitude
      wavelengthInNm:    1550                 # [nm] wavelength
      height:            400000
    source_down:
      class:             'Source'
      polar_coordinates: [5.0, 0.0]           # [arcsec, degrees] source polar coordinates
      magnitude:         0                    # source magnitude
      wavelengthInNm:    1064                 # [nm] wavelength
      height:            400000

   # Pupil stop definition. This is the pupil geometry used in our simulation.
   # When no parameters are specified, a circular pupil is assumed, with the size
   # defined by the pixel_pupil parameter in the main section.
    pupilstop:
      class:             'Pupilstop'
      simul_params_ref:  'main'

   # Atmospheric layers generation and temporal evolution.
   # Here we define 3 atmospheric layer evolving using the infinite phase screen method.
   # The layer heights are defined in meters, and the Cn2 values must sum to 1.
   # The fov parameter defines the field-of-view in arcseconds.
   # The extra_delta_time specifies the time delays between upwards and downwards propagation for each layer,
   # which can be calculated using lib/fsoc_lib.py
   # The inputs are the seeing, wind speed, and wind direction defined above.
    atmo:
      class:                   'AtmoInfiniteEvolutionUpDown'
      simul_params_ref:        'main'
      L0:                      20.0
      heights:                 [0,4000,20000]
      Cn2:                     [0.95,0.01,0.04]
      extra_delta_time_down:   [0.0,0.000024,0.00012]
      extra_delta_time_up:     [0.00045,0.00042,0.00033]
      seed:                    1
      fov_in_m:                8
      inputs:
        seeing:                'seeing.output'
        wind_speed:            'wind_speed.output'
        wind_direction:        'wind_direction.output'
      outputs:                 ['layer_list_down', 'layer_list_up']

   # The propagation blocks simulate the propagation of the electromagnetic field
   # through the atmosphere, and the pupil stop. It takes the source and the atmospheric layers
   # as inputs and outputs the electric field at the pupil plane in all the directions corresponding
   # to the source polar coordinates.
   # To activate Fresnel propagation doFresnel is set to true. In this case also the wavelengthInNm must be provided.
   # In order to deal with the FFT and numerical issues a padding_factor is recommended.
   # To enable upwards propagation, the standard one is downwards, upwards has to be set to true.
   # The output is a list of electric fields, one for each source direction.
    prop_down:
      class:             'AtmoPropagation'
      simul_params_ref:  'main'
      source_dict_ref:   ['source_down']
      doFresnel:         true
      padding_factor:    4
      wavelengthInNm:    1064
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop', 'dm.out_layer:-1']
      outputs:           ['out_source_down_ef']
    prop_up:
      class:             'AtmoPropagation'
      simul_params_ref:  'main'
      source_dict_ref:   ['source_up']
      doFresnel:         true
      upwards:           true
      padding_factor:    4
      wavelengthInNm:    1550
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop', 'dm.out_layer:-1']
      outputs:           ['out_source_up_ef']

   # The Pyramid WFS block simulates the Pyramid wavefront sensor.
   # It takes the electric field from the propagation block and computes the intensity on the detector.
   # The full list of parameters can be found in the init method of the ModulatedPyramid class.
    pyramid:
      class:                    'ModulatedPyramid'
      simul_params_ref:         'main'
      pup_diam:                 40.
      pup_dist:                 72.
      fov:                      5.0
      fov_errinf:               0.1
      fov_errsup:               30.0
      mod_amp:                  2.4
      output_resolution:        240
      fft_res:                  3.0
      wavelengthInNm:           1064
      inputs:
        in_ef:                  'prop_down.out_source_down_ef'
      outputs:                  ['out_i']

   # The detector simulates the CCD sensor where the Pyramid WFS intensity is recorded.
   # Its integration time can be a multiple of the simulation time step.
    detector:
      class:                'CCD'
      simul_params_ref:     'main'
      size:                 [240,240]
      dt:                   0.0005
      bandw:                10
      photon_noise:         true
      excess_noise:         true
      readout_noise:        true
      readout_level:        0.5
      inputs:
        in_i:               'pyramid.out_i'
      outputs:              [ 'out_pixels' ]

   # The slope computer calculates the wavefront slopes from the detector frame.
   # It requires a pupil data object with the list of the valid sub-apertures (this
   # is computed from the pyramid WFS input during the calibration step).
   # The full list of parameters can be found in the init method of the PyrSlopec class.
    slopec:
      class:                'PyrSlopec'
      pupdata_object:        'pupdata'
      thr_value:             0.0
      inputs:
        in_pixels:          'detector.out_pixels'
      outputs:              ['out_slopes']

   # The modal reconstruction block reconstructs the wavefront slopes into modal coefficients.
   # It uses a reconstruction matrix that is computed during the calibration phase.
    modalrec:
      class:                'Modalrec'
      recmat_object:        'rec_mat'
      inputs:
        in_slopes:          'slopec.out_slopes'
      outputs:              ['out_modes']

   # The control block computes the control commands based on the differential modal coefficients.
   # The modal coefficients are differential because it operates in closed loop.
   # The full list of parameters can be found in the init method of the Integrator class.
    control:
      class:                'Integrator'
      delay:                1
      int_gain:             [0.,0.]
      n_modes:              [2,28]
      inputs:
        delta_comm:         'modalrec.out_modes'
      outputs:              ['out_comm']

   # The DM block simulates the deformable mirror.
   # In this case, it uses Zernike modes directly without generating a modal basis.
   # The Zernike modes are generated on the fly.
   # It can also use influence functions and modes-to-command matrix stored in files.
   # The DM height is set to 0, meaning it is at the pupil plane.
   # The full list of parameters can be found in the init method of the DM class.
    dm:
      class:                'DM'
      simul_params_ref:     'main'
      type_str:             'zernike'
      height:               0
      nmodes:               30
      inputs:
        in_command:         'control.out_comm'
      outputs:              ['out_layer']

   # The PSF block computes the point spread function (PSF) based on the electric field
   # at the pupil plane after the DM. It uses the wavelength and the padding coefficient
   # to compute the PSF.
    psf_down:
      class:                'PSF'
      simul_params_ref:     'main'
      wavelengthInNm:       1064               # [nm] Imaging wavelength
      nd:                   4                  # padding coefficient for PSF computation
      inputs:
        in_ef:              'prop_down.out_source_down_ef'
      outputs:              ['out_psf','out_sr']

   # The power loss block computes the power loss of the uplink laser beam in dB based on the electric field
   # at the satellite. It is only supported for upwards propagation.
    power_loss:
      class:                'PowerLoss'
      simul_params_ref:     'main'
      prop_ref:             'prop_up'
      inputs:
        in_ef:               'prop_up.out_source_up_ef'
      outputs:              ['out_power_loss']

   # Data store for saving the simulation results.
   # The data will be stored in a directory named with a timestamp (TN) located in 'output'.
    data_store:
      class:                'DataStore'
      store_dir:            './results/'
        input_list:         ['sr_down-psf_down.out_sr',
                            'powerloss-power_loss_up.out_power_loss']

.. note::
    * **Pupilstop**: The pupilstop block defines a circular aperture mask for both paths, though alternative geometries, such as a Gaussian profile for simulating uplink laser beams, can also be configured. If the beam for upwards propagation is not centered, it is required to set the ``beam_center`` parameter for the ``AtmoPropagation`` class.
    * **Fresnel propagation**: When ``doFresnel=True`` SPECULA employs the Angular Spectrum Method (ASM) for step-by-step propagation between consecutive phase screens. When evaluating beam propagation across long vacuum distances such as from the top of the atmospheric turbulence layer to a GEO or LEO satellite receiver the propagation distance satisfies the far-field condition. In this regime, SPECULA transitions from ASM to Fraunhofer diffraction. In this case also ``wavelegnthInNm`` has to be set. If the output of the propagation is used as input for a WFS, make sure that the wavelengths match.
    * **Phase wrapping**: While phase extraction following these propagation methods naturally yields a wrapped field, SPECULA incorporates a dedicated 2D phase unwrapping function ``unwrap_2d`` to reconstruct continuous, smooth phase maps in ``modal_analysis.py``. If you perform a modal analysis of a Fresnel propagated electric field, you have to set the ``wavelengthInNm`` to perform an automatic unwrapping.
    * **Zero padding**: To mitigate circular wrapping artifacts arising from the Fast Fourier Transforms (FFTs) inherent to ASM and Fraunhofer implementations, setting a large enough ``padding_factor`` is highly recommended. If the padding factor is too small, SPECULA will automatically reduce the propagation distance and output a warning.
    * **Power loss computation**: The ``power_loss`` block takes as input the upwards propagated electric field and computes the power loss in dB, normalized to the diffraction limited case, i.e., 0 dB is the diffraction limit. Negative values indicate a loss, i.e., smaller values correspond to a higher power loss.


Part 2: Running the Simulation
-----------------
Before running the full closed-loop simulation, we need to calibrate several components of the AO system.

The calibration process has two main steps:

Wavefront Sensor Geometry Calibration
~~~~~~~~~~~~~~~~~~~~~~
We need to identify which part of the Pyramid WFS contains enough light from the guide star to provide reliable slope measurements, excluding those outside the pupil or with insufficient illumination.

Create a YAML configuration file, for example ``calib_pyr_pupdata.yml``:

.. code-block:: yaml

    pyr_pupdata:
      class:          'PyrPupdataCalibrator'
      thr1:            0.1
      thr2:            0.25
      obs_thr:         0.95
      display_debug:   True
      output_tag:      'pupdata'
      overwrite:       True
      inputs:
        in_i:         'pyramid.out_i'

    prop_down_override:
      inputs:
        common_layer_list: ['pupilstop']

    main_override:
      total_time:  0.001

    remove: [
        #"main",
        "seeing",
        "wind_speed",
        "wind_direction",
        "atmo",
        "source_up",
        #"source_down",
        #"pupilstop",
        #"prop_down",
        "prop_up",
        #"pyramid,
        "slopec",
        "detector",
        "modalrec",
        "control",
        "dm",
        "power_loss",
        "psf_down",
        "data_store",
    ]

Now run the WFS calibration:

.. code-block:: bash

   specula params_leo_satellite.yml calib_pyr_pupdata.yml

Interaction Matrix and Reconstructor Calibration
~~~~~~~~~~~~~~~~~~~~~~
Create a YAML configuration file, for example ``calib_im_rec.yml``:

.. code-block:: yaml

    # Push-pull command generator
    pushpull:
      class:     'PushPullGenerator'
      nmodes:    30
      amp:       [50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                  50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                  50., 50., 50., 50., 50., 50., 50., 50., 50., 50.]
      outputs:   ['output']

    # Interaction matrix calibrator
    im_calibrator:
      class:          'ImCalibrator'
      nmodes:         30
      im_tag:         'int_mat'
      overwrite:      true
      inputs:
        in_slopes:    'slopec.out_slopes'
        in_commands:  'pushpull.output'

    # Reconstructor calibrator
    rec_calibrator:
      class:          'RecCalibrator'
      nmodes:         30
      rec_tag:        'rec_mat'
      overwrite:      true
      inputs:
        in_intmat:    'im_calibrator.out_intmat'

    # Override main simulation parameters
    main_override:
      time_step:         0.001
      total_time:        0.06               # 30 modes × 2 (push+pull) × 0.001s

    prop_down_override:
      source_dict_ref:   ['source_down']
      inputs:
        common_layer_list: ['pupilstop', 'dm.out_layer']
      outputs:             ['out_source_down_ef']

    # Override DM to use calibration commands
    dm_override:
      sign: 1
      inputs:
        in_command: 'pushpull.output'

    # Disable noise for clean measurements
    detector_override:
      dt:             0.001
      photon_noise:   false
      readout_noise:  false
      excess_noise:   false

    # Remove unnecessary objects during calibration
    remove: [
        #"main",
        "seeing",
        "wind_speed",
        "wind_direction",
        "atmo",
        "source_up",
        #"source_down",
        #"pupilstop",
        #"prop_down",
        "prop_up",
        #"pyramid,
        #"slopec",
        #"detector",
        "modalrec",
        "control",
        #"dm",
        "power_loss",
        "psf_down",
        "data_store",
    ]
Now run the interaction and reconstruction matrix calibration:

.. code-block:: bash

   specula params_leo_satellite.yml calib_im_rec.yml

The system is now fully calibrated and ready for closed-loop operation!

For more details on the calibration procedure also see the SPECULA SCAO tutorial.


Closed-Loop Simulation
~~~~~~~~~~~~~~~~~~~~~~

Now run the full closed-loop simulation:

.. code-block:: bash

   specula params_leo_satellite.yml

See also the :ref:`running_simulations` section for details on how to run the simulation.

SR is printed during the simulation at each iteration while time and iterations per seconds are displayed every 10 iterations.

Part 3: Results Analysis
------------------------
After running the closed-loop simulation, you can analyze the results stored inside the directory specified in ``store_dir`` of the ``data_store`` block in ``params_leo_satellite.yml``.

Create a script ``analyse_data.py``:

.. code-block:: python

    import os
    import glob
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt

    data_dirs = ["./results/LEO_satellite"]
    data = {}
    colors = ['blue']

    plt.figure()
    for di, data_dir in enumerate(data_dirs):
        for fname in glob.glob(os.path.join(data_dir, "*.fits")):
            key = os.path.splitext(os.path.basename(fname))[0]
            with fits.open(fname) as hdul:
                arr = hdul[0].data
            data[key] = arr
            print('key:', key, 'type:', type(data[key]))

        powerloss = data["powerloss"]
        counts, bins = np.histogram(powerloss, density=True)
        plt.stairs(counts, bins, color=colors[di], fill=True, alpha=0.7)

    plt.xlabel('Power loss in dB')
    plt.ylabel('Probability density')
    plt.yscale('log')
    plt.legend(['no AO', 'AO'])
    plt.show()

This file loads the ``powerloss`` file from the directory in `data_dirs` and plots a histogram. If you have performed several simulations, e.g., with and without using AO, you can simply add further directories to ``data_dirs`` and ``colors`` to compare the histograms.