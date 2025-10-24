.. _scao_tutorial:

SCAO Tutorial: Complete Walkthrough
====================================

This comprehensive tutorial guides you through creating, running, and analyzing a complete Single Conjugate Adaptive Optics (SCAO) simulation using SPECULA.
Note that a more basic SCAO simulation tutorial is available in the :ref:`scao_basic_tutorial`.

**What you'll learn:**

* Setting up a realistic SCAO system configuration
* Running calibration and closed-loop phases
* Analyzing performance results
* Optimizing system parameters
* Troubleshooting common issues

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* Basic understanding of adaptive optics concepts
* Python and YAML familiarity

Tutorial Overview
-----------------

We'll simulate a modern SCAO system similar to those used on 8-10m class telescopes:

**System Specifications:**

* 8.2m telescope (VLT-like) with 14% central obstruction
* Kolmogorov turbulence, r₀ = 15cm at 500nm
* 40×40 Shack-Hartmann WFS (1600 subapertures)
* 41x41 actuator deformable mirror
* 1 kHz control loop with integrator controller
* R-band natural guide star (magnitude 8)

**Performance Goals:**

* Strehl ratio > 60% in H-band
* RMS wavefront error < 150nm
* Stable closed-loop operation

Part 1: System Configuration
----------------------------

Calculate and save the influence functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before configuring the SCAO system, we need to compute and save the deformable mirror influence functions. These functions describe how each actuator affects the wavefront across the telescope pupil.

Create a script ``compute_influence_functions.py`` (inspired by ``test_modal_base.py``):

.. code-block:: python

  import specula
  specula.init(0)  # Use GPU device 0 (or -1 for CPU)

  import numpy as np
  import os
  from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
  from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
  from specula.data_objects.ifunc import IFunc
  from specula.data_objects.ifunc_inv import IFuncInv
  from specula.data_objects.m2c import M2C
  from specula.calib_manager import CalibManager
  from specula import cpuArray

  def compute_and_save_influence_functions():
      """
      Compute zonal influence functions and modal basis for the SCAO tutorial
      Follows the same approach as test_modal_basis.py
      """
      # create calibration directory if it doesn't exist
      root_dir = './calibration'
      os.makedirs(root_dir, exist_ok=True)

      # tags
      ifunc_tag = 'tutorial_ifunc'
      m2c_tag = 'tutorial_m2c'
      base_inv_tag = 'tutorial_base_inv'

      # initialize calibration manager
      calib_manager = CalibManager(root_dir)

      # DM and pupil parameters for VLT-like telescope
      pupil_pixels = 160           # Pupil sampling resolution
      n_actuators = 41             # 41x41 = 1681 total actuators
      telescope_diameter = 8.2     # meters (VLT Unit Telescope)
      
      # Pupil geometry
      obsratio = 0.14              # 14% central obstruction
      diaratio = 1.0               # Full pupil diameter
      
      # Actuator geometry - aligned with test_modal_basis.py
      circGeom = True              # Circular geometry (better for round pupils)
      angleOffset = 0              # No rotation
      
      # Mechanical coupling between actuators
      doMechCoupling = False       # Enable realistic coupling
      couplingCoeffs = [0.31, 0.05] # Nearest and next-nearest neighbor coupling
      
      # Actuator slaving (disable edge actuators outside pupil)
      doSlaving = True             # Enable slaving (very simple slaving)
      slavingThr = 0.1             # Threshold for master actuators
      
      # Modal basis parameters
      r0 = 0.15                    # Fried parameter at 500nm [m]
      L0 = 25.0                    # Outer scale [m] 
      zern_modes = 5               # Number of Zernike modes to include
      oversampling = 1             # No oversampling
      
      # Computation parameters
      dtype = specula.xp.float32   # Use current device precision
      
      print("Computing zonal influence functions...")
      print(f"Pupil pixels: {pupil_pixels}")
      print(f"Actuators: {n_actuators}x{n_actuators} = {n_actuators**2}")
      print(f"Telescope diameter: {telescope_diameter}m")
      print(f"Central obstruction: {obsratio*100:.1f}%")
      print(f"r0 = {r0}m, L0 = {L0}m")
      
      # Step 1: Generate zonal influence functions
      influence_functions, pupil_mask = compute_zonal_ifunc(
          pupil_pixels,
          n_actuators,
          circ_geom=circGeom,
          angle_offset=angleOffset,
          do_mech_coupling=doMechCoupling,
          coupling_coeffs=couplingCoeffs,
          do_slaving=doSlaving,
          slaving_thr=slavingThr,
          obsratio=obsratio,
          diaratio=diaratio,
          mask=None,
          xp=specula.xp,
          dtype=dtype,
          return_coordinates=False
      )
      
      # Print statistics
      n_valid_actuators = influence_functions.shape[0]
      n_pupil_pixels = specula.xp.sum(pupil_mask)
      
      print(f"\nZonal influence functions:")
      print(f"Valid actuators: {n_valid_actuators}/{n_actuators**2} ({n_valid_actuators/(n_actuators**2)*100:.1f}%)")
      print(f"Pupil pixels: {int(n_pupil_pixels)}/{pupil_pixels**2} ({float(n_pupil_pixels)/(pupil_pixels**2)*100:.1f}%)")
      print(f"Influence functions shape: {influence_functions.shape}")
      
      # Step 2: Generate modal basis (KL modes)
      print(f"\nGenerating KL modal basis...")
      
      kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
          pupil_mask=pupil_mask,
          diameter=telescope_diameter,
          influence_functions=influence_functions,
          r0=r0,
          L0=L0,
          zern_modes=zern_modes,
          oversampling=oversampling,
          if_max_condition_number=None,
          xp=specula.xp,
          dtype=dtype
      )
      
      print(f"KL basis shape: {kl_basis.shape}")
      print(f"Number of KL modes: {kl_basis.shape[0]}")
           
      kl_basis_inv = np.linalg.pinv(kl_basis)

      # Step 3: Create output directory
      os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
      os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

      # Step 4: Save using SPECULA data objects
      print(f"\nSaving influence functions and modal basis...")
      
      # Create IFunc object and save
      ifunc_obj = IFunc(
          ifunc=influence_functions,
          mask=pupil_mask
      )
      ifunc_filename = calib_manager.filename('ifunc', ifunc_tag)
      ifunc_obj.save(ifunc_filename)
      print("OK: " + ifunc_filename + " (zonal influence functions)")

      # Create M2C object for mode-to-command matrix and save
      m2c_obj = M2C(
          m2c=m2c
      )
      m2c_filename = calib_manager.filename('m2c', m2c_tag)
      m2c_obj.save(m2c_filename)
      print("OK: " + m2c_filename + " (KL modal basis)")

      # inverse influence function object for modal analysis
      print(f"\nSaving inverse modal base...")
      ifunc_inv_obj = IFuncInv(
          ifunc_inv=kl_basis_inv,
          mask=pupil_mask
      )
      base_inv_filename = calib_manager.filename('ifunc', base_inv_tag)
      ifunc_inv_obj.save(base_inv_filename)
      print("OK: " + base_inv_filename + " (inverse modal base)")

      # Step 5: Optional visualization
      try:
        import matplotlib.pyplot as plt

        print(f"\nGenerating visualization...")

        plt.figure(figsize=(10, 6))
        plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF Covariance')
        plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence Covariance')
        plt.xlabel('Mode number')
        plt.ylabel('Singular value')
        plt.title('Singular values of covariance matrices')
        plt.legend()
        plt.grid(True)

        # move to CPU / numpy for plotting if required
        kl_basis = cpuArray(kl_basis)
        pupil_mask = cpuArray(pupil_mask)

        # Plot some modes
        max_modes = min(16, kl_basis.shape[0])

        # Create a mask array for display
        mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))

        # Place each mode vector into the 2D pupil shape
        idx_mask = np.where(pupil_mask)
        for i in range(max_modes):
            mode_img = np.zeros(pupil_mask.shape)
            mode_img[idx_mask] = kl_basis[i]
            mode_display[i] = mode_img

        # Plot the reshaped modes
        n_rows = int(np.round(np.sqrt(max_modes)))
        n_cols = int(np.ceil(max_modes / n_rows))
        plt.figure(figsize=(18, 12))
        for i in range(max_modes):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(mode_display[i], cmap='viridis')
            plt.title(f'Mode {i+1}')
            plt.axis('off')
        plt.tight_layout()

        plt.show()
          
      except ImportError:
          print("Matplotlib not available - skipping visualization")
      
      print(f"\nInfluence functions and modal basis computation completed!")
      print(f"Files saved using CalibManager in: {calib_manager.root_dir}")
      print(f"\nFiles created:")
      print(f"  ifunc/tutorial_ifunc.fits      - Zonal influence functions ({n_valid_actuators} actuators)")
      print(f"  m2c/tutorial_m2c.fits          - KL modal basis ({kl_basis.shape[0]} modes)")
      print(f"  ifunc/tutorial_base_inv.fits   - Inverse modal base")
      
      # Step 6: Test loading the saved files
      print(f"\nTesting file loading...")
      
      try:
          # Test IFunc loading
          loaded_ifunc = IFunc.restore(ifunc_filename)
          assert loaded_ifunc.influence_function.shape == influence_functions.shape
          print("OK: IFunc loading test passed")

          # Test M2C loading
          loaded_m2c = M2C.restore(m2c_filename)
          assert loaded_m2c.m2c.shape == m2c.shape
          print("OK: M2C loading test passed")
          
      except Exception as e:
          print(f"⚠ File loading test failed: {e}")
      
      return ifunc_obj, m2c_obj

  if __name__ == "__main__":
      compute_and_save_influence_functions()

Run this script before starting the main simulation:

.. code-block:: bash

   python compute_influence_functions.py

Expected output:

.. code-block:: text

  Computing zonal influence functions...
  Pupil pixels: 160
  Actuators: 41x41 = 1681
  Telescope diameter: 8.2m
  Central obstruction: 14.0%
  r0 = 0.15m, L0 = 25.0m
  Actuators: 1141
  Master actuators: 1130
  Actuators to be slaved: 11

  Computation completed.

  Zonal influence functions:
  Valid actuators: 1130/1681 (67.2%)
  Pupil pixels: 19716/25600 (77.0%)
  Influence functions shape: (1130, 19716)

  Generating KL modal basis...
  KL basis shape: (1129, 19716)
  Number of KL modes: 1129

  Saving influence functions and modal basis...
  OK: ./calibration/ifunc/tutorial_ifunc.fits (zonal influence functions)
  OK: ./calibration/m2c/tutorial_m2c.fits (KL modal basis)
  Saving inverse modal base...
  OK: ./calibration/ifunc/tutorial_base_inv.fits (inverse modal base)

  Generating visualization...

  Influence functions and modal basis computation completed!
  Files saved using CalibManager in: ./calibration

  Files created:
    ifunc/tutorial_ifunc.fits      - Zonal influence functions (1130 actuators)
    m2c/tutorial_m2c.fits          - KL modal basis (1129 modes)
    ifunc/tutorial_base_inv.fits   - Inverse modal base

  Testing file loading...
  OK: IFunc loading test passed
  OK: M2C loading test passed

.. image:: /_static/tutorial/singular_values.png
   :width: 100%
   :align: center

.. image:: /_static/tutorial/DM_shapes.png
   :width: 100%
   :align: center

**What this does:**

1. **Defines the actuator geometry**: A 41×41 grid with a circular layout, optimized for round telescope pupils with a 14% obstruction, which removes the central actuators.

3. **Computes influence functions**: Each of the 1130 valid actuators produces a unique pattern of phase change across the ~19,000 pupil pixels

4. **Saves calibration data**: Files are saved in FITS format for use by the main simulation

5. **Generates visualization**: Example modes and singular values are plotted for inspection

This pre-computation step is essential because:
- Influence functions are expensive to calculate
- They're needed for interaction matrix calibration and closed-loop operation
- They can be reused for multiple simulations with the same geometry

The generated files will be automatically loaded by the DM configuration in the next steps.

Prepare the simulation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have computed the influence functions, we need to create the main simulation configuration file that uses them. We'll create a YAML parameter file inspired by the ERIS NGS configuration.

Create ``config/scao_tutorial.yml``:

.. code-block:: yaml

   # SCAO Tutorial Configuration
   # ===========================
   # VLT-like telescope with Shack-Hartmann NGS
   
   # Main simulation parameters
   main:
     class:             'SimulParams'
     root_dir:          './calibration'       # CalibManager root directory (all the data are in subdirectories)
     pixel_pupil:       160                   # Must match influence function computation
     pixel_pitch:       0.0513                # [m] 8.2m / 160 pixels = 0.0513 m/pixel
     total_time:        2.000                 # [s] 2 seconds simulation
     time_step:         0.001                 # [s] 1ms time steps (1 kHz)
     zenithAngleInDeg:  0.0                   # [deg] Zenith observation (no airmass)
     display_server:    false                 # Disable for batch runs
   
   # Atmospheric conditions
   seeing:
     class:             'WaveGenerator'
     constant:          0.65                  # [arcsec] Good seeing conditions (r0 about 15cm)
     outputs:           ['output']
   
   wind_speed:
     class:             'WaveGenerator'
     constant:          [10.0, 12.0, 8.0]    # [m/s] Multi-layer wind speeds
     outputs:           ['output']
   
   wind_direction:
     class:             'WaveGenerator'
     constant:          [45.0, 135.0, -30.0] # [deg] Wind directions for each layer
     outputs:           ['output']
   
   # Science target (on-axis)
   source_science:
     class:             'Source'
     polar_coordinates: [0.0, 0.0]            # [arcsec, deg] On-axis target
     height:            .inf                  # Infinite height (star)
     magnitude:         10.0                  # H-band magnitude
     wavelengthInNm:    1650                  # [nm] H-band center
   
   # Natural guide star for WFS
   source_ngs:
     class:             'Source'
     polar_coordinates: [0.0, 0.0]            # [arcsec, deg] On-axis NGS
     height:            .inf                  # Infinite height (star)
     magnitude:         8.0                   # R-band magnitude (bright NGS)
     wavelengthInNm:    800                   # [nm] R-band for WFS
   
   # Telescope pupil geometry
   pupilstop:
     class:             'Pupilstop'
     simul_params_ref:  'main'
     mask_diam:         1.0                   # Full pupil diameter
     obs_diam:          0.14                  # 14% central obstruction (VLT-like)
   
   # Multi-layer atmospheric model
   atmo:
     class:             'AtmoEvolution'
     simul_params_ref:  'main'
     L0:                25.0                  # [m] Outer scale
     # Simplified 3-layer model for tutorial
     heights:           [0.0, 4000.0, 12000.0]  # [m] Ground, mid, high layers
     Cn2:               [0.7, 0.2, 0.1]       # Cn2 fractions (sum = 1.0)
     fov:               60.0                   # [arcsec] Field of view
     inputs:
       seeing:          'seeing.output'
       wind_speed:      'wind_speed.output'
       wind_direction:  'wind_direction.output'
     outputs:           ['layer_list']
   
   # Atmospheric propagation
   prop:
     class:             'AtmoPropagation'
     simul_params_ref:  'main'
     source_dict_ref:   ['source_science', 'source_ngs']
     inputs:
       atmo_layer_list: ['atmo.layer_list']
       common_layer_list: ['pupilstop',       # Pupil
                          'dm.out_layer:-1']  # DM correction from last step
     outputs:           ['out_source_science_ef', 'out_source_ngs_ef']
   
   # Shack-Hartmann wavefront sensor
   sh:
     class:             'SH'
     subap_on_diameter: 40                    # 40x40 subapertures across pupil
     subap_wanted_fov:  2.4                   # [arcsec] Subaperture field of view
     sensor_pxscale:    0.4                   # [arcsec/pixel] Pixel scale
     subap_npx:         6                     # 8x8 pixels per subaperture
     wavelengthInNm:    800                   # [nm] R-band sensing
     inputs:
       in_ef:           'prop.out_source_ngs_ef'
     outputs:           ['out_i']
   
   # CCD detector simulation
   detector:
     class:             'CCD'
     simul_params_ref:  'main'
     size:              [240, 240]            # Total detector size (40x40 x 8x8)
     dt:                0.001                 # [s] Integration time (1ms)
     bandw:             400                   # [nm] R+I-band filter width 600-1000nm
     photon_noise:      true                  # Enable photon noise
     readout_noise:     true                  # Enable read noise
     excess_noise:      true                  # Enable excess noise
     readout_level:     0.2                   # [e-/pix/frame] Read noise level
     emccd_gain:        400                   # EMCCD gain factor
     quantum_eff:       0.3                   # QE x transmission
     inputs:
       in_i:            'sh.out_i'
     outputs:           ['out_pixels']
   
   # Slopes computation
   slopec:
     class:             'ShSlopec'
     thr_value:         0.1                   # Threshold for valid subapertures
     subapdata_object:  'tutorial_subaps'     # Will be generated during calibration
     sn_object:         null                  # No slope references initially
     inputs:
       in_pixels:       'detector.out_pixels'
     outputs:           ['out_slopes']
   
   # Modal reconstruction
   modalrec:
     class:             'Modalrec'
     recmat_object:     'tutorial_rec'        # Reconstruction matrix tag
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs:           ['out_modes']
   
   # Integrator controller
   integrator:
     class:             'Integrator'
     simul_params_ref:  'main'
     delay:             1                     # 1 frame delay (realistic)
     int_gain:          [0.30]
     n_modes:           [800]                 # Number of modes to control
     inputs:
       delta_comm:      'modalrec.out_modes'
     outputs:           ['out_comm']
   
   # Deformable mirror
   dm:
     class:             'DM'
     simul_params_ref:  'main'
     ifunc_object:      'tutorial_ifunc'      # Our computed influence functions
     m2c_object:        'tutorial_m2c'        # Modal-to-command matrix
     nmodes:            800                   # Number of controlled modes
     height:            0                     # Ground-conjugated DM
     inputs:
       in_command:      'integrator.out_comm'
     outputs:           ['out_layer']
   
   # Science PSF computation
   psf:
     class:             'PSF'
     simul_params_ref:  'main'
     wavelengthInNm:    1650                 # [nm] H-band science
     nd:                4                    # 4x padding for PSF
     start_time:        0.2                  # Start PSF integration after 200ms
     inputs:
       in_ef:           'prop.out_source_science_ef'
     outputs:           ['out_psf', 'out_sr']

   # modal analysis to compute modal residual
   modal_analysis:
     class:            'ModalAnalysis'
     ifunc_inv_object: 'tutorial_base_inv'   # Our computed ininverse modal base
     inputs:
       in_ef: 'prop.out_source_science_ef'
     outputs: ['out_modes']
   
   # Data store for results 
   data_store:
     class:             'DataStore'
     store_dir:         './output'            # Data result directory: 'store_dir'/TN/
     inputs:    
       input_list: ['comm-integrator.out_comm','sr-psf.out_sr','res-modal_analysis.out_modes']

**What we've created:**

1. **Main configuration file** (``scao_tutorial.yml``) that defines the complete AO system

The configuration is now ready to run the calibration step!

Note that the :class:`specula.processing_objects.data_store.DataStore` object can be configured to save more data, such as the slopes, the detector pixels, the PSF, etc.

Part 2: Running the Simulation
------------------------------

See the :ref:`running_simulations` section for details on how to run the simulation.

Calibration Phase
~~~~~~~~~~~~~~~~~

Before running the full closed-loop simulation, we need to calibrate several components of the AO system. The calibration process has three main steps:

Subaperture Geometry Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we need to identify which subapertures contain enough light from the guide star to provide reliable slope measurements.

Create ``calib_subaps.yml`` to measure the subaperture geometry:

.. code-block:: yaml

   # Subaperture Geometry Calibration
   # =================================
   
   # Subaperture calibrator
   sh_subaps:
     class: 'ShSubapCalibrator'
     subap_on_diameter: 40                   # 40×40 subapertures
     output_tag:        'tutorial_subaps'    # Output file tag
     energy_th:         0.5                  # 50% energy threshold
     inputs:
       in_i: 'sh.out_i'                     # WFS intensity input
   
   # Short calibration run
   main_override:
     total_time: 0.001                       # 0ms (just measure pupil)
   
   # Clean pupil measurement (no atmosphere)
   prop_override:
     inputs:
       common_layer_list: ['pupilstop']      # Only telescope pupil
   
   # Remove unnecessary objects
   remove: ['atmo', 'dm', 'slopec', 'modalrec', 'integrator', 'psf', 'modal_analysis', 'data_store']

Run the subaperture calibration:

.. code-block:: bash

   specula config/scao_tutorial.yml calib_subaps.yml

This step identifies approximately 1200 valid subapertures out of the 1600 total (40×40 grid), excluding those outside the pupil or with insufficient illumination.

Push-Pull Amplitude Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The interaction matrix calibration requires amplitude values for each actuator poke. Create ``prepare_pushpull_amplitudes.py``:

.. code-block:: python

  import os
  import numpy as np
  from astropy.io import fits

  import specula
  specula.init(0)  # Use GPU device 0 (or -1 for CPU)
  from specula.calib_manager import CalibManager 

  def create_scaled_amplitudes(n_actuators, base_amplitude=50):
      """
      Create amplitude vector with scaling pattern:
      [1, 1, 1/sqrt(2), 1/sqrt(2), 1/sqrt(2), 1/sqrt(3), 1/sqrt(3), 1/sqrt(3), 1/sqrt(3), ...]
      
      Parameters:
      -----------
      n_actuators : int
          Total number of actuators
      base_amplitude : float
          Base amplitude in nm (default: 50nm)
          
      Returns:
      --------
      amplitudes : ndarray
          Scaled amplitude vector
      """
      amplitudes = np.zeros(n_actuators)
      
      # Pattern: n repetitions of 1/sqrt(n)
      # Group 1: 2 actuators with factor 1 (1/sqrt(1))
      # Group 2: 3 actuators with factor 1/sqrt(2) 
      # Group 3: 4 actuators with factor 1/sqrt(3)
      # etc.
      
      idx = 0
      group = 1
      
      while idx < n_actuators:
          # Number of actuators in this group
          group_size = group + 1
          
          # Scale factor for this group
          scale_factor = 1.0 / np.sqrt(group)
          
          # Fill the group (up to remaining actuators)
          end_idx = min(idx + group_size, n_actuators)
          amplitudes[idx:end_idx] = scale_factor
          
          print(f"Group {group}: actuators {idx:4d}-{end_idx-1:4d} (size={end_idx-idx:2d}), factor=1/√{group} = {scale_factor:.4f}")
          
          idx = end_idx
          group += 1
      
      # Apply base amplitude
      amplitudes *= base_amplitude
      
      return amplitudes

  def main():
      root_dir = './calibration'

      # Initialize calibration manager
      calib_manager = CalibManager(root_dir)

      # tags
      data_filename = 'pushpull_1129modes_amp50'
      data_uniform_filename = 'pushpull_1129modes_amp50_uniform'

      # Create scaled amplitudes for all valid actuators
      n_actuators = 1129  # Number of valid actuators -1 (from influence functions)
      base_amplitude = 50  # 50nm
  
      print(f"Creating scaled amplitude vector for {n_actuators} actuators")
      print(f"Base amplitude: {base_amplitude:.1f} nm")
      print("")
      
      amplitudes = create_scaled_amplitudes(n_actuators, base_amplitude)
      
      # Print statistics
      print(f"\nAmplitude statistics:")
      print(f"  Minimum: {np.min(amplitudes):.2f} nm")
      print(f"  Maximum: {np.max(amplitudes):.2f} nm")
      print(f"  Mean:    {np.mean(amplitudes):.2f} nm")
      print(f"  Std:     {np.std(amplitudes):.2f} nm")
      
      # Show first and last few values
      print(f"\nFirst 10 amplitudes [nm]: {amplitudes[:10]}")
      print(f"Last 10 amplitudes [nm]:  {amplitudes[-10:]}")
      
      # Save amplitude vector
      os.makedirs(os.path.join(root_dir, 'data'), exist_ok=True)

      output_file = calib_manager.filename('data', data_filename)

      fits.writeto(output_file, amplitudes, overwrite=True)
      print(f"\nOK: Saved scaled amplitude vector: {output_file}")
      
      # Create comparison with uniform amplitudes
      uniform_amplitudes = np.full(n_actuators, base_amplitude)
      uniform_file = calib_manager.filename('data', data_uniform_filename)
      fits.writeto(uniform_file, uniform_amplitudes, overwrite=True)
      print(f"OK: Saved uniform amplitude vector: {uniform_file}")
      
      return amplitudes

  if __name__ == "__main__":
      amplitudes = main()

Run this script to generate the amplitude vector:

.. code-block:: bash

   python prepare_pushpull_amplitudes.py

**Performance note:** The 50nm amplitude is chosen as a compromise and scaling it for high order modes avoids saturation issues.

Interaction Matrix and Reconstructor Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now calibrate the interaction matrix (how actuators affect WFS measurements) and compute the reconstruction matrix (how to convert slopes to actuator commands).

Create ``calib_im_rec.yml``:

.. code-block:: yaml

   # Interaction Matrix and Reconstructor Calibration
   # ================================================
   
   # Push-pull command generator
   pushpull:
     class:     'PushPullGenerator'
     nmodes:    1129                         # Number of DM actuators
     vect_amplitude_data: 'pushpull_1129modes_amp50'  # Amplitude vector
     outputs:   ['output']
   
   # Interaction matrix calibrator
   im_calibrator:
     class:     'ImCalibrator'
     nmodes:    1129                         # Number of modes to calibrate
     im_tag:    'tutorial_im'                # Output IM filename
     overwrite: true                         # Overwrite existing files
     inputs:
       in_slopes:   'slopec.out_slopes'      # WFS slopes input
       in_commands: 'pushpull.output'        # Push-pull commands
   
   # Reconstructor calibrator
   rec_calibrator:
     class:     'RecCalibrator'
     nmodes:    800                          # Number of modes (reduced to keep noise propagation low and avoid numerical issues)
     rec_tag:   'tutorial_rec'               # Output REC filename
     overwrite: true                         # Overwrite existing files
     inputs:
       in_intmat:   'im_calibrator.out_intmat'  # Connect to IM output
   
   # Override main simulation parameters
   main_override:
     total_time: 2.258                        # 1129 modes × 2 (push+pull) × 0.001s
   
   # Disable atmosphere for clean calibration
   prop_override:
     source_dict_ref:   ['source_ngs']
     inputs:
       common_layer_list: ['pupilstop', 'dm.out_layer']  # Only pupil + DM
     outputs:           ['out_source_ngs_ef']

   # Override DM to use calibration commands
   dm_override:
     sign: 1                                 # Use positive sign for calibration (default is -1)
     nmodes: 1129                            # Use all 1129 modes for calibration
     inputs:
       in_command: 'pushpull.output'         # Connect to push-pull generator
   
   # Disable noise for clean measurements
   detector_override:
     photon_noise:   false                   # No photon noise
     readout_noise:  false                   # No read noise
     excess_noise:   false                   # No excess noise
   
   # Remove unnecessary objects during calibration
   remove: ['atmo', 'source_science', 'psf', 'modalrec', 'integrator', 'modal_analysis', 'data_store']

Run the interaction matrix calibration:

.. code-block:: bash

   specula config/scao_tutorial.yml calib_im_rec.yml

**What happens during calibration:**

1. **Push-pull sequence**: Each modes is poked +amp then -amp sequentially (amp starts at 50nm and scales down for higher modes)
2. **Slope measurement**: WFS measures the resulting slope changes
3. **Interaction matrix**: Built from the slope responses to each mode
4. **Reconstructor**: Computed as the pseudo-inverse of the interaction matrix

The system is now fully calibrated and ready for closed-loop operation!

Closed-Loop Simulation
~~~~~~~~~~~~~~~~~~~~~~

Now run the full closed-loop simulation:

.. code-block:: bash

   specula config/scao_tutorial.yml

SR is printed during the simulation at each iteration while time and iterations per seconds are displayed every 10 iterations.

Part 3: Results Analysis
------------------------

After running the closed-loop simulation, you can analyze the results using the following script.  
This script automatically finds the most recent output directory, loads all `.fits` and `.pickle` files, and plots the Strehl Ratio and RMS of turbulence, residuals, and commands.

Create a script ``analyse_data.py``:

.. code-block:: python

   import os
   import glob
   import pickle
   from astropy.io import fits
   import numpy as np
   import matplotlib.pyplot as plt

   # Find all directories in ./output starting with '20'
   dirs = [d for d in glob.glob("./output/20*") if os.path.isdir(d)]
   if not dirs:
       raise RuntimeError("No output directories found.")
   # Select the most recent one (by name, assuming timestamp format)
   data_dir = sorted(dirs)[-1]
   print(f"Using data directory: {data_dir}")

   data = {}

   # Load all .fits files in the directory
   for fname in glob.glob(os.path.join(data_dir, "*.fits")):
       key = os.path.splitext(os.path.basename(fname))[0]
       with fits.open(fname) as hdul:
           arr = hdul[0].data
       data[key] = arr
       print('key:', key, 'type:', type(data[key]))

   # Load all .pickle files in the directory
   for fname in glob.glob(os.path.join(data_dir, "*.pickle")):
       key = os.path.splitext(os.path.basename(fname))[0]
       with open(fname, "rb") as f:
           data[key] = pickle.load(f)
       print('key:', key, 'type:', type(data[key]))

   # Plot the sr.fits file if present (assumed to be a 1D vector)
   if "sr" in data:
       sr = data["sr"]
       print(f"The average Strehl Ratio after 50 iterations is: {sr[50:].mean():.4f}")
       plt.figure()
       plt.plot(sr, marker='o')
       plt.title("Strehl Ratio (sr.fits)")
       plt.xlabel("Frame")
       plt.ylabel("SR")
       plt.grid(True)
       plt.show()
   else:
       print("sr.fits file not found in the directory.")
       
   if "res" in data and "comm" in data:
       res = data["res"]
       comm = data["comm"]
       init = 50
       turb = res[init+1:, :].copy()
       # command is applied with a delay of 1 frame
       turb[:, :comm.shape[1]] += comm[init:-1, :]
       x = np.arange(turb.shape[1])+1
       
       # Plot RMS of residuals, commands and turbulence
       plt.figure(figsize=(12, 6))
       plt.plot(x,np.sqrt(np.mean(turb**2, axis=0)), label='Turbulence RMS', marker='o')
       plt.plot(x,np.sqrt(np.mean(res**2, axis=0)), label='Residuals RMS', marker='o')
       plt.plot(x[:comm.shape[1]],np.sqrt(np.mean(comm**2, axis=0)), label='Commands RMS', marker='o')
       plt.title("RMS of Turbulence, Residuals and Commands")
       plt.xlabel("Mode number")
       plt.ylabel("RMS")
       plt.xscale('log')
       plt.yscale('log')
       plt.legend()
       plt.grid(True)
       plt.show()

Save this script as ``analyse_data.py`` and run it after your simulation to visualize the results.

.. code-block:: bash

   python analyse_data.py

This will display the Strehl Ratio evolution and the RMS of turbulence, residuals, and commands for your simulation.

.. image:: /_static/tutorial/SR.png
   :width: 100%
   :align: center

.. image:: /_static/tutorial/modal_plot.png
   :width: 100%
   :align: center

Part 4: Parameter Optimization
------------------------------

TODO: Now that you have a working baseline, let's optimize the system performance.

Loop Gain Optimization
~~~~~~~~~~~~~~~~~~~~~~

A common task in AO system optimization is to find the best integrator gain for your controller.  
Here we show how to automate a **parameter sweep** over the integrator gain, running multiple simulations and analyzing the results.

**Step 1: Run simulations for each gain**

Create a script `gain_overrides.py` to modify the ``scao_tutorial.yml`` file, each time with a different gain value
and saving the result in a different output directory, using the ``overrides`` feature:

.. code-block:: python

    import specula
    import numpy as np

    # Range of gains to test
    gains = np.linspace(0.1, 1.0, 10)
    output_dir = "gain_overrides"
    base_config = "config/scao_tutorial.yml"

    for gain in gains:
        overrides = ("{"
                    f"integrator.int_gain: [{gain:.2f}], "
                    f"data_store.store_dir: ./output/gain_opt/gain_{gain:.2f}"
                    "}")

        specula.main_simul(yml_files=[base_config], overrides=overrides)

Run this file with the command ``python gain_overrides.py``

**Step 2: Analyze the results**

After all simulations are complete, you can plot the average Strehl Ratio as a function of the integrator gain.  
Each simulation output is stored in a separate directory (e.g., `./output/gain_opt/gain_0.10/`).

Example analysis script (`plot_gain_optimization.py`):

.. code-block:: python

    import os
    import glob
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    output_base = "./output/gain_opt"
    dirs = sorted(glob.glob(os.path.join(output_base, "gain_*/2*/")))

    gains = []
    mean_sr = []

    for d in dirs:
        # Find the YAML file to get the gain value
        yml_files = glob.glob(os.path.join(d, "*.yml"))
        gain = None
        for yml in yml_files:
            with open(yml, "r") as f:
                yml_data = yaml.safe_load(f)
                if "integrator" in yml_data:
                    gain = float(yml_data["integrator"]["int_gain"][0])
                    break
        if gain is None:
            # Fallback: parse from directory name
            gain = float(d.split("_")[-1].replace("/", ""))
        # Load sr.fits
        sr_file = os.path.join(d, "sr.fits")
        if os.path.exists(sr_file):
            with fits.open(sr_file) as hdul:
                sr = hdul[0].data
            mean_sr.append(sr[50:].mean())  # Ignore initial transient
            gains.append(gain)
            print(f"Gain {gain:.2f}: mean SR = {sr[50:].mean():.4f}")
        else:
            print(f"Warning: {sr_file} not found.")

    # Plot
    plt.figure()
    plt.plot(gains, mean_sr, marker='o')
    plt.xlabel("Integrator Gain")
    plt.ylabel("Mean Strehl Ratio")
    plt.title("Loop Gain Optimization")
    plt.grid(True)
    plt.show()

**Summary**

- You can automate parameter sweeps in SPECULA by generating override YAML files and running batch simulations.
- The results can be easily analyzed by loading the output files and plotting performance metrics as a function of the parameter of interest.

**Note:**
- A modal gain optimization can be done comparing the modal residuals across different gains.
- This approach can be generalized to optimize other parameters (e.g., number of modes, filter cutoff, etc.) by modifying the override YAML files accordingly.


Part 5: Advanced Topics
-----------------------
      
Guide Star Magnitude Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another important parameter in AO performance is the brightness of the guide star.  
Here we show how to automate a **parameter sweep** over the guide star magnitude, running multiple simulations and analyzing the results.

**Step 1: Run simulations for each magnitude**

Create a script `magnitude_overrides.py` to modify the ``scao_tutorial.yml`` file, each time with a different magnitude
and saving the result in a different output directory, using the ``overrides`` feature:

.. code-block:: python

    import specula
    import numpy as np

    # Range of magnitudes to test (e.g., from 6 to 12)
    magnitudes = np.arange(6, 13)
    output_dir = "magnitude_overrides"
    base_config = "config/scao_tutorial.yml"

    for mag in magnitudes:
        overrides = ("{"
                    f"source_ngs.magnitude: {mag}, "
                    f"data_store.store_dir: ./output/magnitude/mag_{mag}"
                    "}")

        specula.main_simul(yml_files=[base_config], overrides=overrides)


Run this file with the command ``python magnitude_overrides.py``

**Step 2: Analyze the results**

After all simulations are complete, you can plot the average Strehl Ratio as a function of the guide star magnitude.  
Each simulation output is stored in a separate directory (e.g., `./output/magnitude/mag_6/`).

Example analysis script (`plot_magnitude_effects.py`):

.. code-block:: python

    import os
    import glob
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    output_base = "./output/magnitude"
    dirs = sorted(glob.glob(os.path.join(output_base, "mag_*/2*/")))

    mean_sr = {}

    for d in dirs:
        # Find the YAML file to get the magnitude value
        params_file = os.path.join(d, "params.yml")
        with open(params_file, 'r') as f:
            yml_data = yaml.safe_load(f)
        mag = float(yml_data['source_ngs']['magnitude'])
        sr_file = os.path.join(d, "sr.fits")
        if os.path.exists(sr_file):
            with fits.open(sr_file) as hdul:
                sr = hdul[0].data
            mean_sr[mag] = sr[50:].mean()  # Ignore initial transient
            print(f"Magnitude {mag:.1f}: mean SR = {sr[50:].mean():.4f}")
        else:
            print(f"Warning: {sr_file} not found.")

    # Plot
    mag_to_plot = sorted(mean_sr.keys())
    sr_to_plot = [mean_sr[mag] for mag in mag_to_plot]
    plt.figure()
    plt.plot(mag_to_plot, sr_to_plot, marker='o')
    plt.xlabel("Guide Star Magnitude")
    plt.ylabel("Mean Strehl Ratio")
    plt.title("SR vs Guide Star Magnitude")
    plt.gca().invert_xaxis()  # Brighter stars (lower mag) on the left
    plt.grid(True)
    plt.show()

**Summary**

- You can automate magnitude sweeps in SPECULA by generating override YAML files and running batch simulations.
- The results can be easily analyzed by loading the output files and plotting performance metrics as a function of guide star magnitude.

Troubleshooting Common Issues
-----------------------------

TODO

Computational Issues
~~~~~~~~~~~~~~~~~~~~

TODO

Summary and Next Steps
----------------------

Congratulations! You've successfully:

✅ **Configured** a complete SCAO system
✅ **Calibrated** the interaction and reconstruction matrices  
✅ **Executed** a closed-loop simulation

TODO:

✅ **Analyzed** performance results
✅ **Optimized** system parameters

**Next Steps:**

1. **Experiment** with different atmospheric conditions
2. **Try** pyramid wavefront sensors
3. **Explore** laser guide star systems  
4. **Try** MCAO configurations
5. **Compute** off-axis PSFs

.. seealso::
   
   - :ref:`field_analyser_tutorial` for post-processing PSF, modal analysis, and phase cubes
   - TODO: Add links to relevant documentation sections for further reading
