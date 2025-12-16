.. _pwfs_calib_pc_tutorial:

Pyramid WFS Calibration in Partial Correction Regime
====================================================

This tutorial demonstrates how to calibrate a Pyramid WFS in the **partial correction regime**, accounting for the non-linear response of the sensor under realistic atmospheric turbulence conditions.
This approach was introduced by **Agapito et al. (2023)** [1]_ and is essential for achieving optimal performance with non-modulated or low-modulation Pyramid WFS configurations.

**What you'll learn:**

* Understanding why partial correction calibration is necessary
* Calibrating slope null reference in partial correction regime (with atmospheric background)
* Computing interaction matrices in partial correction regime (with atmospheric background)
* Setting appropriate correction levels and statistical sampling
* Comparing linear vs. non-linear calibration approaches

**Prerequisites:**

* Completed the :ref:`scao_basic_tutorial`
* Understanding of Pyramid WFS non-linearity
* Familiarity with SPECULA calibration workflow

Background: Why Partial Correction?
-----------------------------------

The Pyramid WFS exhibits **non-linear behavior** that depends on the optical aberrations present at the sensor input. In standard calibration approaches, the interaction matrix is measured under ideal conditions (flat wavefront), which does not accurately represent the sensor response when operating under real atmospheric turbulence with finite correction quality.

This limitation is addressed by performing calibration **in partial correction**, where:

1. A realistic level of residual atmospheric aberration is injected during calibration
2. The correction level matches the expected AO system performance
3. Statistical averaging over multiple turbulence realizations reduces noise

This approach is described in detail by **Agapito et al. (2023)** [1]_ for non-modulated Pyramid WFS.

Tutorial Overview
-----------------

Starting from the basic SCAO tutorial configuration (``params_scao_pyr_basic.yml``), we will:

1. Calibrate **slope null reference** in partial correction
2. Compute **interaction matrix** in partial correction
3. Compare results with linear calibration

Part 1: Slope Null Calibration
------------------------------

The slope null represents the WFS reference signal and here we calibrate it under residual turbulence. This calibration step computes the average WFS signal over many atmospheric realizations with a fixed correction level.

Step 1: Create Slope Null Calibration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``params_scao_pyr_calib_sn.yml``:

.. code-block:: yaml

   # Slope null calibrator - averages WFS response over atmospheric realizations
   pyr_sn:
     class: 'SnCalibrator'
     output_tag: 'scao_sn_pc_s1.0_c09'           # Output filename
     overwrite: true
     inputs:
       in_slopes: 'slopec.out_slopes'

   # Random seeing generator - varies around nominal value (here 1.0 arcsec)
   seeing_random:
     class:             'RandomGenerator'
     amp:               0.1                 # Variation amplitude [arcsec]
     constant:          1.0                 # Mean seeing [arcsec]
     distribution:      'UNIFORM'           # Uniform distribution in [0.9, 1.1]
     outputs: ['output']

   # Random atmospheric phase generator
   atmo_random:
     class:                'AtmoRandomPhase'
     simul_params_ref:  'main'
     L0:                   40               # [m] Outer scale
     source_dict_ref:      ['on_axis_source']
     inputs:
       seeing: 'seeing_random.output'
       pupilstop: 'pupilstop'
     outputs: ['out_on_axis_source_ef']

   # Modal decomposition of atmospheric phase
   modal_analysis_random:
     class: 'ModalAnalysis'
     type_str: 'zernike'
     nmodes: 54
     obsratio: 0.0
     inputs:
       in_ef: 'atmo_random.out_on_axis_source_ef'
     outputs: ['out_modes']

   # Scale modal coefficients to simulate partial correction
   # Note: In practice, this should be a mode-dependent vector
   scale_random:
     class:  'BaseOperation'
     constant_mul: 0.90                    # 90% correction level
     inputs:
       in_value1: 'modal_analysis_random.out_modes'
     outputs: ['out_value']

   # DM applying partial correction
   dm_random:
     class: 'DM'
     simul_params_ref: 'main'
     type_str: 'zernike'
     nmodes: 54
     obsratio: 0.0
     height: 0
     inputs:
       in_command: 'scale_random.out_value'
     outputs: ['out_layer']

   # Combine atmospheric turbulence with DM correction
   ef_combinator:
     class: 'ElectricFieldCombinator'
     inputs:
       in_ef1: 'atmo_random.out_on_axis_source_ef'    # Turbulence
       in_ef2: 'dm_random.out_layer'                  # Partial correction
     outputs: ['out_ef']

   # Override main parameters for sufficient statistical sampling
   main_override:
     total_time: 1.000                    # 1000 frames at 1kHz

   # Override Pyramid input to use combined field
   pyramid_override:
     inputs:
       in_ef: 'ef_combinator.out_ef'

   # Remove closed-loop components (not needed for slope null)
   remove: ['atmo', 'prop', 'rec', 'control', 'dm', 'psf', 'data_store']

**Key Parameters:**

- ``total_time: 1.000``: Collect 1000 frames for good statistics (more frames = better averaging)
- ``constant_mul: 0.90``: Assumes 90% correction level (residual = 10% of turbulence)
- ``seeing_random``: Varies seeing to capture realistic operational conditions

.. note::
   **Choosing the Correction Level:**
   
   The ``constant_mul`` parameter (here 0.90) represents the expected AO system performance. 
   In practice:
   
   - This value should be **iteratively refined**: run a full closed-loop simulation to estimate the actual correction level, then re-calibrate.
   - Use a **mode-dependent correction vector** rather than a scalar for better accuracy (low-order modes are typically better corrected than high-order).
   - Typical values range from 0.85 to 0.95 depending on system performance.

Step 2: Run Slope Null Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   specula params_scao_pyr_basic.yml params_scao_pyr_calib_sn.yml

This generates ``scao_sn_pc_s1.0_c09.fits`` containing the averaged slope null reference.

Part 2: Interaction Matrix Calibration
--------------------------------------

The interaction matrix is now computed by applying push-pull commands on top of the atmospheric background, similar to operational conditions.

Step 1: Create IM Calibration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``params_scao_pyr_calib_rec_pc.yml``:

.. code-block:: yaml

   # Push-pull command generator
   pushpull:
     class:     'PushPullGenerator'
     nmodes:    54
     ncycles:   100                          # 100 cycles per mode for averaging
     vect_amplitude: [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      50.0, 50.0, 50.0, 50.0]  # Push amplitude [nm]
     outputs:   ['output']

   # Interaction matrix calibrator
   im_calibrator:
     class:     'ImCalibrator'
     nmodes:    54
     im_tag:    'scao_pyr_im_pc_s1.0_c09'
     data_dir:  './calib/im'
     overwrite: true
     inputs:
       in_slopes:   'slopec.out_slopes'
       in_commands: 'pushpull.output'
     outputs: ['out_intmat']

   # Reconstructor calibrator
   rec_calibrator:
     class:     'RecCalibrator'
     nmodes:    54
     rec_tag:   'scao_pyr_rec_pc_s1.0_c09'
     data_dir:  './calib/rec'
     overwrite: true
     inputs:
       in_intmat: 'im_calibrator.out_intmat'

   # Random seeing for atmospheric background
   seeing_random:
     class:             'RandomGenerator'
     amp:               0.1                 # Variation amplitude [arcsec]
     constant:          1.0                 # Mean seeing [arcsec]
     distribution:      'UNIFORM'           # Uniform distribution in [0.9, 1.1]
     outputs: ['output']

   # Atmospheric phase generator (updated every 2 frames)
   atmo_random:
     class:                'AtmoRandomPhase'
     simul_params_ref:  'main'
     L0:                   40
     update_interval:      2                  # Update turbulence every 2ms
     source_dict_ref:      ['on_axis_source']
     inputs:
       seeing: 'seeing_random.output'
       pupilstop: 'pupilstop'
     outputs: ['out_on_axis_source_ef']

   # Modal analysis and partial correction (same as slope null)
   modal_analysis_random:
     class: 'ModalAnalysis'
     type_str: 'zernike'
     nmodes: 54
     obsratio: 0.0
     inputs:
       in_ef: 'atmo_random.out_on_axis_source_ef'
     outputs: ['out_modes']

   scale_random:
     class:  'BaseOperation'
     constant_mul: 0.90                     # Same correction level as slope null
     inputs:
       in_value1: 'modal_analysis_random.out_modes'
     outputs: ['out_value']

   dm_random:
     class: 'DM'
     simul_params_ref: 'main'
     type_str: 'zernike'
     nmodes: 54
     obsratio: 0.0
     height: 0
     inputs:
       in_command: 'scale_random.out_value'
     outputs: ['out_layer']

   # Combine turbulence background with push-pull DM commands
   prop_override:
     source_dict_ref: ['on_axis_source']
     inputs:
       common_layer_list: ['pupilstop', 'dm_random.out_layer', 'dm.out_layer']
     outputs: ['out_on_axis_source_ef']

   # Override DM to use push-pull commands
   dm_override:
     sign: 1
     inputs:
       in_command: 'pushpull.output'

   # Disable detector noise for clean IM measurement
   detector_override:
     photon_noise:   false
     readout_noise:  false

   # Override simulation time: 54 modes × 2 (push+pull) × 100 cycles × 0.001s
   main_override:
     total_time: 10.8

   # Remove closed-loop components
   remove: ['atmo', 'rec', 'control', 'psf', 'data_store']

**Key Differences from Linear Calibration:**

- ``ncycles: 100``: Multiple push-pull cycles average out atmospheric fluctuations
- ``update_interval: 2``: Atmospheric phase changes during calibration
- ``dm_random.out_layer`` in ``common_layer_list``: Adds atmospheric background to push-pull signal

Step 2: Run IM Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   specula params_scao_pyr_basic.yml params_scao_pyr_calib_rec_pc.yml

This generates:
- ``scao_pyr_im_pc_s1.0_c09.fits``: Interaction matrix in partial correction
- ``scao_pyr_rec_pc_s1.0_c09.fits``: Corresponding reconstruction matrix

Part 3: Using Partial Correction Calibration
-------------------------------------------

Update your main YAML to use the new calibration:

.. code-block:: yaml

   slopec:
     class:             'PyrSlopec'  
     pupdata_object:    'scao_pupdata'
     sn_object:         'scao_sn_pc_s1.0_c09'      # ← Partial correction slope null
     inputs:
       in_pixels:        'detector.out_pixels'

   rec:
     class:              'Modalrec'  
     recmat_object:      'scao_pyr_rec_pc_s1.0_c09' # ← Partial correction reconstructor
     inputs:
       in_slopes:        'slopec.out_slopes'

Comparison: Linear vs. Partial Correction
-----------------------------------------

**Linear Calibration** (flat wavefront):
- Simpler and faster
- Assumes WFS linearity
- May underestimate sensitivity at low spatial frequencies
- Suitable for low wavefront error regime or modulated PWFS

**Partial Correction Calibration**:
- More realistic for operational conditions
- Accounts for non-linear WFS response
- Requires more computation (many cycles/frames)
- Essential for non-modulated or low-modulation PWFS
- Improves performance for medium/high wavefront error regime

Best Practices
--------------

1. **Correction Level Tuning:**
   
   .. code-block:: python
   
      # Iterative approach:
      # 1. Run closed-loop with initial guess (e.g., 0.85)
      # 2. Measure actual residual variance
      # 3. Update correction level: 1 - (measured_residual/input_variance)
      # 4. Re-calibrate and repeat until convergence

2. **Statistical Sampling:**
   
   - Slope null: ≥ 500 frames minimum (1000+ recommended)
   - IM calibration: ≥ 50 cycles per mode (100+ for low SNR)

3. **Mode-Dependent Correction:**
   
   .. code-block:: yaml
   
      # Instead of scalar constant_mul, use vector:
      scale_random:
        class: 'BaseOperation'
        vect_mul_data: 'correction_vector_54modes'  # From file
        inputs:
          in_value1: 'modal_analysis_random.out_modes'

4. **Verification:**
   
   Compare IM singular values between linear and partial correction:
   
   .. code-block:: python
   
      import numpy as np
      from astropy.io import fits
      
      im_linear = fits.getdata('calib/im/scao_pyr_im.fits')
      im_pc = fits.getdata('calib/im/scao_pyr_im_pc_s1.0_c09.fits')
      
      U_lin, s_lin, Vt_lin = np.linalg.svd(im_linear)
      U_pc, s_pc, Vt_pc = np.linalg.svd(im_pc)
      
      # Plot singular value ratio
      import matplotlib.pyplot as plt
      plt.semilogy(s_lin/s_pc)
      plt.xlabel('Mode number')
      plt.ylabel('SV ratio (linear/PC)')
      plt.title('IM conditioning comparison')

Summary
-------

This tutorial demonstrated:

✓ Slope null calibration in partial correction
✓ Interaction matrix measurement in partial correction
✓ Parameter selection for realistic AO performance  
✓ Integration into closed-loop simulation  


.. [1] Agapito, G., et al. "Non-modulated pyramid wavefront sensor. Use in sensing and correcting atmospheric turbulence" 
       A&A, 677, A168 (2023). `ADS <https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.168A/abstract>`_

.. seealso::

   - :ref:`scao_basic_tutorial` for foundational SCAO concepts
   - :ref:`scao_tutorial` for advanced multi-layer simulations