.. _science_camera_tutorial:

Science Camera Images: a Single-Lenslet SH and a CCD
====================================================

This tutorial shows how to simulate the image recorded by a **science camera**:
a chosen pixel scale and field of view, the integration of the light over each
detector pixel, and the detector noise.

The ``PSF`` processing object is the right tool to measure image quality
(Strehl ratio, FWHM, encircled energy), but it does not give a detector image:

* the PSF is **sampled at points** (the centres of the pixels). With
  ``pixel_size_mas`` you choose the pixel scale, but the light is not
  integrated over the pixel area, so the pixel response is missing;
* the field of view is the whole FFT grid, and there is no field stop;
* the PSF is normalized to unit sum: there are no photons and no noise.

A **Shack-Hartmann (SH) with a single lenslet** covering the whole pupil is an
imaging camera: it computes the focal-plane image of the full aperture, with
the pixel scale and the field of view you ask for, and it integrates the PSF
over each detector pixel. A ``CCD`` object then converts the image into
electrons and adds the noise.

**Goal:** get a noisy, pixel-integrated image of a star with a given pixel
scale, field of view, integration time and detector.

Prerequisites
-------------

* SPECULA installed (see :ref:`installation`)
* Basic knowledge of SPECULA simulations (see :ref:`scao_basic_tutorial`)

Signal chain
------------

.. code-block:: text

    Source -> AtmoPropagation -> SH (1 lenslet) -> CCD -> DataStore
                                  |
                                  +-> intensity [ph/s/nm], noiseless
                                            CCD -> frames [e-], with noise

The SH output is an intensity in photons per second per nm of bandwidth. The
CCD integrates it over ``dt`` seconds and ``bandw`` nm, applies the quantum
efficiency, and adds the noise.

Configuration
-------------

Save this file as ``params_science_camera.yml``. The telescope is an 8 m
pupil with a 14% central obstruction, observed at 1 µm (λ/D = 25.8 mas). A
small static astigmatism (100 nm RMS) is added with a DM, as explained in
:ref:`generators_tutorial`.

.. code-block:: yaml

    main:
      class:             'SimulParams'
      root_dir:          './calib'
      pixel_pupil:       160                    # Linear dimension of pupil phase array
      pixel_pitch:       0.05                   # [m] Pitch of the pupil phase array (D = 8 m)
      total_time:        0.050                  # [s] Total simulation running time
      time_step:         0.001                  # [s] Simulation time step

    on_axis_source:
      class:             'Source'
      polar_coordinates: [0.0, 0.0]             # [arcsec, degrees] source polar coordinates
      magnitude:         14                     # source magnitude
      wavelengthInNm:    1000                   # [nm] wavelength

    pupilstop:
      class:             'Pupilstop'
      simul_params_ref:  'main'
      obs_diam:          0.14                   # central obstruction (fraction of the diameter)

    aberr_coeff:
      class:             'WaveGenerator'
      constant:          [0.0, 0.0, 0.0, 100.0] # [nm RMS] Zernike tip, tilt, focus, astigmatism
      outputs: ['output']

    aberr_dm:
      class:             'DM'
      simul_params_ref:  'main'
      type_str:          'zernike'
      nmodes:            4
      obsratio:          0.14
      height:            0
      inputs:
        in_command: 'aberr_coeff.output'
      outputs: ['out_layer']

    prop:
      class:             'AtmoPropagation'
      simul_params_ref:  'main'
      source_dict_ref:   ['on_axis_source']
      inputs:
        common_layer_list: ['pupilstop', 'aberr_dm.out_layer']
      outputs: ['out_on_axis_source_ef']

    # Science camera: a Shack-Hartmann with a single lenslet
    camera:
      class:             'SH'
      wavelengthInNm:    1000                   # [nm] imaging wavelength
      subap_on_diameter: 1                      # a single lenslet covering the whole pupil
      sensor_pxscale:    0.0125                 # [arcsec/pixel] detector pixel scale
      subap_npx:         128                    # [pixels] detector size
      subap_wanted_fov:  1.6                    # [arcsec] = sensor_pxscale * subap_npx
      inputs:
        in_ef: 'prop.out_on_axis_source_ef'
      outputs: ['out_i']

    detector:
      class:             'CCD'
      simul_params_ref:  'main'
      size:              [128, 128]             # [pixels] must be equal to subap_npx
      dt:                0.010                  # [s] integration time
      bandw:             100                    # [nm] optical bandwidth
      quantum_eff:       0.5                    # quantum efficiency * total transmission
      photon_noise:      True
      readout_noise:     True
      readout_level:     3.0                    # [e-/pixel/frame] read-out noise
      background_noise:  True
      background_level:  1.0                    # [e-/pixel/frame] background
      inputs:
        in_i: 'camera.out_i'
      outputs: ['out_pixels']

    # Reference: the PSF object, with the same pixel scale
    psf:
      class:             'PSF'
      simul_params_ref:  'main'
      wavelengthInNm:    1000                   # [nm] imaging wavelength
      pixel_size_mas:    12.5                   # [mas] same pixel scale as the camera
      inputs:
        in_ef: 'prop.out_on_axis_source_ef'
      outputs: ['out_psf']

    data_store:
      class:             'DataStore'
      store_dir:         './output'
      inputs:
        input_list: ['frames-detector.out_pixels',
                     'intensity-camera.out_i',
                     'psf-psf.out_psf']

Run it with:

.. code-block:: bash

   specula params_science_camera.yml

The camera parameters
~~~~~~~~~~~~~~~~~~~~~

With ``subap_on_diameter: 1`` the SH parameters become the parameters of a
camera:

* ``sensor_pxscale`` is the **pixel scale** in arcsec/pixel. It is used
  exactly: the SH integrates the PSF over pixels of this size, whatever the
  ratio between the pixel and λ/D. Here 12.5 mas is close to Nyquist
  sampling (λ/2D = 12.9 mas).
* ``subap_npx`` is the **detector size** in pixels, and ``size`` of the CCD
  must be equal to it.
* ``subap_wanted_fov`` is a square **field stop**. Set it to
  ``sensor_pxscale * subap_npx`` to use the whole detector.
* ``wavelengthInNm`` is the imaging wavelength. The SH is monochromatic: for a
  broad band, use ``PolyChromSH``, which sums several monochromatic SHs with
  their own wavelength and flux factor.

**How the pixel integration works.** The PSF of a pupil field sampled with
*n* pixels is band limited. Its samples from an FFT with 2 *n* points
therefore determine it everywhere, and its integral over any pixel is a
weighted sum of these samples. The SH applies this weighted sum, so each
detector pixel is the exact integral of the PSF over the pixel area. No finer
FFT and no rebinning are needed.

**The FFT period.** The FFT image is periodic, with period λ divided by the
pitch of the (oversampled) pupil field. The SH prints it in the log:

.. code-block:: text

    [camera]: -->     FFT pix. sc. [asec],   0.012891550390443523
    [camera]: -->     FFT period [asec],     8.250592249883855

Light that falls farther than half a period from the centre is folded back
into the image. This is not a problem for a diffraction-limited star with a
1.6 arcsec field, but it can be one for seeing-limited images or large
fields. Increase ``fov_ovs_coeff`` (default 2) to increase the period.

The detector parameters
~~~~~~~~~~~~~~~~~~~~~~~

* ``dt`` is the integration time. It must be a multiple of ``time_step``: the
  CCD sums the SH intensities of ``dt / time_step`` steps and returns one frame
  at the end of each integration.
* ``bandw`` is the optical bandwidth in nm. It only scales the number of
  photons: it does not make the image polychromatic.
* ``quantum_eff`` includes the quantum efficiency and the total transmission.
* The noise sources are switched on one by one (``photon_noise``,
  ``readout_noise``, ``background_noise``, ``darkcurrent_noise``,
  ``excess_noise`` for EMCCDs, ...). ``readout_level`` is in electrons per
  pixel and per frame; ``background_level`` and ``darkcurrent_level`` are in
  electrons per pixel and per frame (not per second).

The output frame is in electrons. The background is added before the photon
noise and it is not subtracted, so the mean level of an empty pixel is
``background_level``.

.. warning::

   **At least one input must change at every step.** SPECULA runs an object
   only when one of its inputs has changed. If the only layer is the pupil
   stop, the propagation, the SH and the CCD run only at the first step, and a
   CCD with ``dt`` longer than ``time_step`` never returns a frame (no
   ``frames.fits`` is saved). In this example the ``WaveGenerator`` output is
   refreshed at every step, even if it is constant, and this keeps the chain
   running. In a real simulation, the turbulence or the AO correction does the
   same.

Looking at the result
---------------------

This script compares the ``PSF`` object, the noiseless SH intensity and the
last CCD frame:

.. code-block:: python

    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    tn = sorted(glob.glob('output/*'))[-1]                 # last simulation
    frames = fits.getdata(f'{tn}/frames.fits')             # [e-], one frame per dt
    intensity = fits.getdata(f'{tn}/intensity.fits')[-1]   # noiseless, [ph/s/nm]
    psf = fits.getdata(f'{tn}/psf.fits')[-1]               # normalized to unit sum

    print(f'number of frames: {frames.shape[0]}')
    print(f'noiseless flux: {intensity.sum():.0f} ph/s/nm')
    print(f'mean frame sum: {frames.sum(axis=(1, 2)).mean():.0f} e-')

    def crop(img, half=32):
        n = img.shape[0] // 2
        return img[n-half:n+half, n-half:n+half]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, img, title in zip(ax[:2], [psf, intensity],
                             ['PSF object (point-sampled)', 'SH intensity (pixel-integrated)']):
        im = a.imshow(np.log10(crop(img) / img.max() + 1e-6), origin='lower')
        a.set_title(title)
        fig.colorbar(im, ax=a, label='log10 (normalized to peak)')
    im = ax[2].imshow(crop(frames[-1]), origin='lower')
    ax[2].set_title('CCD frame (with noise)')
    fig.colorbar(im, ax=ax[2], label='e-')
    plt.tight_layout()
    plt.show()

The expected values are:

* **5 frames** (50 ms of simulation, 10 ms per frame).
* **Flux.** A magnitude 14 star at 1 µm gives about 105 ph/s/m²/nm. With a
  collecting area of 49.3 m² this gives about 5170 ph/s/nm, which is the sum
  of the SH intensity.
* **Frame sum.** 5170 ph/s/nm × 0.01 s × 100 nm × 0.5 ≈ 2580 e⁻ from the
  star, plus 1 e⁻ × 128² pixels ≈ 16400 e⁻ of background, so about 19000 e⁻
  per frame, with a noise of about 400 e⁻.

.. note::

   **Centring.** For an even ``subap_npx`` the SH puts the optical axis at the
   corner of the 4 central pixels, as a SH does with its sub-apertures. The
   ``PSF`` object instead puts the axis at the centre of a pixel. So the two
   peaks cannot be compared directly: with the same pixel scale, the peak of
   the SH image is lower because of both the pixel integration and the
   centring.

What to change for your case
----------------------------

* **Undersampled or oversampled images:** change ``sensor_pxscale`` and keep
  ``subap_wanted_fov = sensor_pxscale * subap_npx``.
* **Turbulence or an AO residual:** add the atmosphere and the AO loop to the
  propagation (see :ref:`scao_basic_tutorial`); the camera does not change.
  Check the FFT period in the log for seeing-limited images.
* **A defocused or aberrated image:** add the static aberration with a DM or
  a layer, as shown in :ref:`generators_tutorial`.
* **Long exposures:** increase ``dt``. Remember that the simulation runs one
  step every ``time_step``, so a long exposure with a short ``time_step``
  costs many steps.
