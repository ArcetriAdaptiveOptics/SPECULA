.. _generators_tutorial:

Using Generators for Vibrations, Time Histories and Static Aberrations
======================================================================

SPECULA provides several generator classes to inject time-dependent signals into your simulation. Here we show two common approaches for adding vibrations, but the same logic applies to other types of signals. The last part shows how to add static aberrations: common path aberrations, non-common path aberrations and phase diversity.

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
                            'dm.out_layer:-1',  # use the previous-step DM correction; 1-frame causal feedback lag, separate from any explicit block delay
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

Static Aberrations
------------------

A static aberration is a phase that does not change during the simulation:
a telescope aberration, a non-common path aberration (NCPA), or a known
aberration added on purpose (phase diversity). There are two ways to add it.

**1. A generator and a DM.** A ``WaveGenerator`` with a ``constant`` value
sends a fixed command to a modal DM. With ``type_str: 'zernike'`` the command
is a vector of Zernike coefficients in nm RMS of wavefront, starting from tip
(Noll index 2):

.. code-block:: yaml

    aberr_coeff:
      class:             'WaveGenerator'
      constant:          [0.0, 0.0, 100.0, 0.0, 50.0]  # [nm RMS] tip, tilt, focus, astig., astig.
      outputs:           ['output']

    aberr_dm:
      class:             'DM'
      simul_params_ref:  'main'
      type_str:          'zernike'
      nmodes:            5                 # same length as the constant vector
      obsratio:          0.0               # central obstruction, as in the pupil stop
      height:            0
      inputs:
        in_command:      'aberr_coeff.output'
      outputs:           ['out_layer']

The DM modes are orthonormalized over the pupil defined by ``obsratio``: with
a central obstruction they are the annular versions of the Zernike
polynomials, with unit RMS over the annulus.

**2. A static layer.** Any phase map can be computed offline, saved as a
``Layer`` object in the ``layers/`` directory of the calibration root
(``root_dir`` in ``SimulParams``), and loaded with its tag. The layer must
have the same size and pixel pitch as the pupil:

.. code-block:: python

    import numpy as np
    import specula
    specula.init(-1)  # CPU
    from specula.data_objects.layer import Layer
    from specula.lib.zernike_generator import ZernikeGenerator

    pixel_pupil, pixel_pitch = 160, 0.05          # as in SimulParams
    zg = ZernikeGenerator(pixel_pupil, xp=np, dtype=np.float32)
    phase_nm = 200.0 * zg.getZernike(5) + 100.0 * zg.getZernike(7)  # [nm RMS]

    layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=0)
    layer.phaseInNm[:] = phase_nm
    layer.A[:] = 1.0
    layer.save('calib/layers/ncpa_example.fits')

.. code-block:: yaml

    ncpa_layer:
      class:             'Layer'
      tag:               'ncpa_example'

In both cases the result is a layer that is used like any other layer: the
DM output is ``aberr_dm.out_layer``, and the static layer is ``ncpa_layer``.

.. note::

   SPECULA runs an object only when one of its inputs has changed. A
   ``WaveGenerator`` output is refreshed at every step, even if it is
   constant, so it keeps the downstream objects running. A static layer is
   not: if nothing else changes (no turbulence, no AO loop), the propagation
   and the sensors run only at the first step. See
   :ref:`science_camera_tutorial`.

Common path aberrations
~~~~~~~~~~~~~~~~~~~~~~~

An aberration seen by all the sensors (e.g. a telescope aberration) goes in
the ``common_layer_list`` of the propagation, which is shared by the WFS and
the science camera:

.. code-block:: yaml

    prop:
      class:                'AtmoPropagation'
      ...
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop',
                            'aberr_dm.out_layer',
                            'dm.out_layer:-1']

Non-common path aberrations (NCPA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An NCPA is seen only by the science path, not by the WFS. There are two
ways to apply it, and they give the same electric field.

**Option A: a second propagation.** The WFS and the science camera use two
propagation objects; only the science one includes the NCPA:

.. code-block:: yaml

    prop_wfs:
      class:             'AtmoPropagation'
      simul_params_ref:  'main'
      source_dict_ref:   ['on_axis_source']
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop', 'dm.out_layer:-1']
      outputs: ['out_on_axis_source_ef']

    prop_sci:
      class:             'AtmoPropagation'
      simul_params_ref:  'main'
      source_dict_ref:   ['on_axis_source']
      inputs:
        atmo_layer_list: ['atmo.layer_list']
        common_layer_list: ['pupilstop', 'dm.out_layer:-1', 'ncpa_layer']
      outputs: ['out_on_axis_source_ef']

This is the most explicit option, but the turbulence is propagated twice. It
is also the option for **field-dependent aberrations**: an optical surface
that is not conjugated to the pupil is a layer at a non-zero height, and
each source sees a different part of it. For a modal DM, ``npixels`` sets the
size of the meta-pupil, which must cover the footprints of all the sources
(D + 2 h θ_max):

.. code-block:: yaml

    ncpa_dm:
      class:             'DM'
      simul_params_ref:  'main'
      type_str:          'zernike'
      nmodes:            3
      npixels:           164               # meta-pupil: 8 m + 2 * 500 m * 30 arcsec
      obsratio:          0.0
      height:            500               # [m] conjugation altitude
      inputs:
        in_command:      'ncpa_coeff.output'
      outputs:           ['out_layer']

The Zernike modes are now defined over the meta-pupil. With a focus on this
DM, a source at 30 arcsec sees the focus shifted by h θ = 7.3 cm, which is
the same focus plus a tilt.

**Option B: add the NCPA to the WFS field.** An ``ElectricFieldCombinator``
multiplies the amplitudes and adds the phases of its inputs. A layer is an
electric field with zero flux (``S0 = 0``), so it can be added directly to
the field propagated for the WFS, without changing the number of photons:

.. code-block:: yaml

    ef_sci:
      class:             'ElectricFieldCombinator'
      inputs:
        in_ef_list: ['prop.out_on_axis_source_ef', 'ncpa_layer']  # or 'ncpa_dm.out_layer'
      outputs: ['out_ef']

The science camera then uses ``ef_sci.out_ef`` as input. This option is
cheaper, but the layer is added as it is, in the pupil: it must have the same
size and pitch as the field, and its height, shift and rotation are ignored.

Phase diversity
~~~~~~~~~~~~~~~

Phase diversity estimates the wavefront from focal-plane images taken with a
known aberration added on purpose (the *diversity*). In a simulation the
diversity is a static aberration on the path of the camera that records the
diverse image, so it is added as an NCPA (option A or B above):

* **Classical phase diversity** uses two images, one in focus and one with a
  known defocus: use two cameras (see :ref:`science_camera_tutorial`), one
  with the field ``prop.out_on_axis_source_ef`` and one with the field
  combined with a defocus DM.
* **LIFT** (``Lift`` processing object) uses a single image with a known
  diversity, typically astigmatism. The camera image (``out_pixels`` of a
  ``CCD``) is the input of ``Lift``. The diversity is given to ``Lift`` in
  ``ref_zern_amp``, in radians of phase: for a DM command of *a* nm RMS at
  wavelength λ it is 2π *a* / λ, as long as ``Lift`` uses the same modal base
  as the diversity DM.

From focus in nm RMS to a focal plane shift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optical designs often give a defocus as a shift Δz of the detector along the
optical axis, for a beam with focal ratio N (f/#). For a small Δz the
wavefront is

.. math::

   W(\rho) = W_{020}\,\rho^2, \qquad W_{020} = \frac{\Delta z}{8 N^2},

with ρ the normalized pupil radius. A focus command :math:`a_4` [nm RMS]
(the third DM mode) gives the wavefront
:math:`a_4\,(\rho^2 - \langle\rho^2\rangle)/\sigma`, with
:math:`\sigma = (1-\varepsilon^2)/(2\sqrt{3})` the RMS of :math:`\rho^2` over a
pupil with central obstruction ε (``obsratio``). With the default DM
``sign``, a positive command gives a positive :math:`W_{020}`:
:math:`W_{020} = 2\sqrt{3}\,a_4/(1-\varepsilon^2)`, and

.. math::

   \Delta z = \frac{16\sqrt{3}\,N^2\,a_4}{1-\varepsilon^2}
   \approx 27.7\,\frac{N^2\,a_4}{1-\varepsilon^2}.

For example, with N = 10, ε = 0 and Δz = 1 mm, the focus coefficient is
:math:`a_4 = 10^{-3}/(16\sqrt{3}\cdot 100)` m ≈ 361 nm RMS.

A simple check: when the defocus is much larger than the wavelength, the
image of a star is a uniform disc of angular diameter
:math:`8 W_{020}/D = 16\sqrt{3}\,a_4/((1-\varepsilon^2) D)`, which does not
depend on N. With D = 8 m, ε = 0 and :math:`a_4` = 2000 nm the expected
diameter is 1.43 arcsec. In a simulated camera image (26 mas pixels), the
radius that contains half of the energy is r₅₀ = 0.50 arcsec, which gives a
disc diameter :math:`2\sqrt{2}\,r_{50}` = 1.42 arcsec.

Other Generators
----------------

SPECULA includes several other generator classes, such as:

- ``WaveGenerator``: for constant or sinusoidal signals
- ``RandomGenerator``: for white noise or random signals
- ``TimeHistoryGenerator``: for arbitrary precomputed time series

See the API documentation and parameter file examples for more details.
