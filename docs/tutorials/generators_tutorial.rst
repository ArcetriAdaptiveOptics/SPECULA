.. _generators_tutorial:

Using Generators for Vibrations and Time Histories
=================================================

SPECULA provides several generator classes to inject time-dependent signals into your simulation. Here we show two common approaches for adding vibrations, but the same logic applies to other types of signals.

**Note:** In SPECULA, the unit of the phase in an electric field is nanometers (nm) of optical path difference (OPD), i.e. wavefront. 

VibrationGenerator: Using a Power Spectrum Density (PSD)
--------------------------------------------------------

If you have a vibration PSD (e.g., from telemetry), you can use the ``VibrationGenerator`` class. This generator will synthesize a time history matching the given PSD for the specified number of modes (e.g., 2 for tip and tilt).

**YAML example:**

.. code-block:: yaml

    vibration:
      class:             'VibrationGenerator'
      simul_params_ref:  'main'
      nmodes:            2
      psd_data:          'PSD_LBT'         # load from file (unit [nm^2/Hz])
      freq_data:         'PSD_FREQ_LBT'    # load from file
      seed:              1987              # optional, but recommended
      start_from_zero:   False             # optional
      outputs:           ['output']

    dm_vibration:
      class:             'DM'
      simul_params_ref:  'main'
      ifunc_object:      'LBT_ASM_IFUNC'   # Influence function for LBT ASM
      nmodes:            2                 # Number of modes same as in VibrationGenerator
      height:            0
      sign:              1                 # it doesn't change sign to the aberration
      inputs:
          in_command:    'vibration.output'
      outputs:           ['out_layer']

To apply the vibration, add the new DM to the propagation step:

.. code-block:: yaml

    prop:
      class:                'AtmoPropagation'
      ...
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop',
                            'dm.out_layer:-1',
                            'dm_vibration.out_layer']
      ...

For details on the ``_data`` YAML option, see :ref:`special-yaml-options-data-and-object`.

TimeHistoryGenerator: Using a Precomputed Time History
------------------------------------------------------

Alternatively, you can generate the vibration time history offline (e.g., using ``get_vibrations_time_hist()``) and save it as a ``TimeHistory`` object. This is useful if you want to reuse the same realization or speed up simulation startup.

**YAML example:**

.. code-block:: yaml

    vibration:
      class:                'TimeHistoryGenerator'
      time_hist_object:     'VIBRATION_TIME_HIST_LBT'  # Precomputed TimeHistory object (unit [nm])
      outputs:              ['output']

**Python example to generate and save the time history:**

.. code-block:: python

    import specula
    import numpy as np
    specula.init(0)
    from specula.processing_objects.vibration_generator import get_vibrations_time_hist
    from specula.data_objects.time_history import TimeHistory

    # TODO: define nmodes, psd, freq, seed, samp_freq, niter, start_from_zero
    # psd unit: [nm^2/Hz], freq unit: [Hz]
    time_hist = get_vibrations_time_hist(
        nmodes, psd=psd, freq=freq, seed=seed,
        samp_freq=samp_freq, niter=niter, start_from_zero=start_from_zero,
        xp=np, dtype=np.float32, complex_dtype=np.complex64
    )

    time_hist_obj = TimeHistory(time_hist)
    dir = "PATH_TO_DATA/"  # Update this path
    time_hist_obj.save(dir + "VIBRATION_TIME_HIST_LBT.fits")

You can choose a file name that encodes relevant parameters (e.g., telemetry source, framerate, etc.).
Note that this approach requires to save a different time history file for each unique set of parameters (framerate, no. iterations, seed, ...).

Other Generators
----------------

SPECULA includes several other generator classes, such as:

- ``WaveGenerator``: for constant or sinusoidal signals
- ``RandomGenerator``: for white noise or random signals
- ``TimeHistoryGenerator``: for arbitrary precomputed time series

See the API documentation and parameter file examples for more details.
