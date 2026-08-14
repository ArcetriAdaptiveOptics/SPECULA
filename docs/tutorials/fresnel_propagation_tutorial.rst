.. _fresnel_propagation:

Fresnel Propagation Tutorial
=============================================================================

This tutorial guides you through creating and running a Fresnel propagation using SPECULA.

**What you'll learn:**

* Propagating an electric field through turbulence using Fresnel propagation
* Using upwards propagation to a source and downwards propagation from a source
* Storing the electric field after propagation

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* Basic understanding of propagation concepts
* Python and YAML familiarity

Tutorial Overview
-----------------

We'll simulate a simple Fresnel propagation with:

* Circular pupil (1m class)
* 3 atmospheric layer
* Upwards Fresnel propagation to a source on-axis
* Downwards Fresnel propagation from a source off-axis
* Storing the propagated electric fields


System Configuration and Running the Simulation
-----------------

Create a YAML configuration file, for example ``params_fresnel_propagation.yml``:

.. code-block:: yaml

   # main section with simulation parameters used by most of the components
   # Note: zenith angle is not specified, so it is assumed to be 0 (on-axis)
    main:
      class:             'SimulParams'
      root_dir:          './calib/'            # Root directory for calibration manager
      pixel_pupil:       120                   # Linear dimension of pupil phase array
      pixel_pitch:       0.00833               # [m] Pitch of the pupil phase array
      total_time:        1.0                   # [s] Total simulation running time
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
   # The extra_delta_time specifies the time delay between upwards and downwards propagation for each layer.
   # a dedicated utility function for calculting this can be found in lib/fsoc_lib.py
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
   # To activate Fresnel propagation doFresnel is set to true. In this case also the wavelenghtInNm must be provided.
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
        common_layer_list: ['pupilstop']
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
        common_layer_list: ['pupilstop']
      outputs:           ['out_source_up_ef']

   # Data store for saving the simulation results.
   # The data will be stored in a directory named with a timestamp (TN) located in 'output'.
   # In this case it saves the downwards and upwards propagated electric fields.
    data_store:
      class:            'DataStore'
      store_dir:         './results/'
      inputs:
        input_list:     ['ef_down-prop_down.out_source_down_ef',
                        'ef_up-prop_up.out_source_up_ef']

.. note::
    * **Fresnel propagation**: When ``doFresnel=True`` SPECULA employs the Angular Spectrum Method (ASM) for step-by-step propagation between consecutive phase screens. When evaluating beam propagation across long vacuum distances such as from the top of the atmospheric turbulence layer to a GEO or LEO satellite receiver the propagation distance satisfies the far-field condition. In this regime, SPECULA transitions from ASM to Fraunhofer diffraction.
    * **Pupilstop**: The ``pupilstop`` block defines a circular aperture mask for both paths, though alternative geometries, such as a Gaussian profile for simulating uplink laser beams, can also be configured.  If the beam for upwards propagation is not centered, it is required to set the ``beam_center`` parameter for the ``AtmoPropagation`` class.
    * **Zero padding**: To mitigate circular wrapping artifacts arising from the Fast Fourier Transforms (FFTs) inherent to ASM and Fraunhofer implementations, setting a large enough ``padding_factor`` is highly recommend. If the padding factor is too small, SPECULA will automatically reduce the propagation distance and output a warning.
    * **Phase wrapping**: While phase extraction following these propagation methods naturally yields a wrapped field, SPECULA incorporates a dedicated 2D phase unwrapping function ``unwrap_2d`` to reconstruct continuous, smooth phase maps in ``modal_analysis.py``. If you perform a modal analysis of a Fresnel propagated electric field, you have to set the ``wavelenghtInNm`` to perform an automatic unwrapping.

Now run the full propagation simulation:

.. code-block:: bash

   specula params_fresnel_propagation.yml

After running the simulation, you can analyze the results stored inside the directory specified in ``store_dir`` of the ``data_store`` block.
Note that if you extract the phase from the electric field it will be wrapped due to Fresnel propagation.

