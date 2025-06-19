SCAO Tutorial: Complete Walkthrough
====================================

This comprehensive tutorial guides you through creating, running, and analyzing a complete Single Conjugate Adaptive Optics (SCAO) simulation using SPECULA.

**What you'll learn:**

* Setting up a realistic SCAO system configuration
* Running calibration and closed-loop phases
* Analyzing performance results
* Optimizing system parameters
* Troubleshooting common issues

**Prerequisites:**

* SPECULA installed and working
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
  from specula.data_objects.m2c import M2C

  def compute_and_save_influence_functions():
      """
      Compute zonal influence functions and modal basis for the SCAO tutorial
      Follows the same approach as test_modal_basis.py
      """
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
      doMechCoupling = True        # Enable realistic coupling
      couplingCoeffs = [0.31, 0.05]  # Nearest and next-nearest neighbor coupling
      
      # Actuator slaving (disable edge actuators outside pupil)
      doSlaving = True             # Enable slaving
      slavingThr = 0.1             # Threshold for valid actuators
      
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
      
      kl_basis, _, _ = make_modal_base_from_ifs_fft(
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
      
      # Verify RMS normalization (like in test)
      for i in range(min(5, kl_basis.shape[0])):  # Check first 5 modes
          rms = float(specula.xp.sqrt(specula.xp.mean(kl_basis[i]**2)))
          print(f"Mode {i+1} RMS: {rms:.3f}")
      
      # Step 3: Create output directory
      os.makedirs('calibration', exist_ok=True)
      
      # Step 4: Save using SPECULA data objects
      print(f"\nSaving influence functions and modal basis...")
      
      # Create IFunc object and save
      ifunc_obj = IFunc(
          ifunc=influence_functions,
          mask=pupil_mask,
          target_device_idx=specula.current_device_idx,
          precision=specula.current_precision
      )
      ifunc_obj.save('calibration/tutorial_ifunc.fits')
      print("✓ tutorial_ifunc.fits (zonal influence functions)")
      
      # Create M2C object for mode-to-command matrix and save
      m2c_obj = M2C(
          m2c=kl_basis,
          target_device_idx=specula.current_device_idx,
          precision=specula.current_precision
      )
      m2c_obj.save('calibration/tutorial_m2c.fits')
      print("✓ tutorial_m2c.fits (KL modal basis)")
      
      # Step 5: Optional visualization
      try:
          import matplotlib.pyplot as plt
          
          print("\nGenerating visualization...")
          
          # Convert to CPU arrays for plotting
          if hasattr(influence_functions, 'get'):  # CuPy array
              influence_functions_cpu = influence_functions.get()
              pupil_mask_cpu = pupil_mask.get()
              kl_basis_cpu = kl_basis.get()
          else:  # NumPy array
              influence_functions_cpu = influence_functions
              pupil_mask_cpu = pupil_mask
              kl_basis_cpu = kl_basis
          
          # Function to reconstruct 2D functions for plotting
          def reconstruct_2d_function(func_1d, mask):
              func_2d = np.zeros(mask.shape)
              func_2d[mask] = func_1d
              return func_2d
          
          # Plot influence functions and KL modes
          fig, axes = plt.subplots(2, 4, figsize=(16, 8))
          
          # Top row: Example influence functions
          example_if_indices = [
              n_valid_actuators // 2,        # Center actuator
              n_valid_actuators // 4,        # Quarter point
              3 * n_valid_actuators // 4,    # Three quarters
              n_valid_actuators - 50,        # Near edge
          ]
          
          for i, act_idx in enumerate(example_if_indices):
              if act_idx < n_valid_actuators:
                  ifunc_2d = reconstruct_2d_function(influence_functions_cpu[act_idx], pupil_mask_cpu)
                  
                  im = axes[0, i].imshow(ifunc_2d, origin='lower', cmap='RdBu_r')
                  axes[0, i].set_title(f'Influence Function {act_idx}')
                  axes[0, i].set_xticks([])
                  axes[0, i].set_yticks([])
                  plt.colorbar(im, ax=axes[0, i], shrink=0.8)
          
          # Bottom row: First 4 KL modes
          for i in range(min(4, kl_basis.shape[0])):
              kl_2d = reconstruct_2d_function(kl_basis_cpu[i], pupil_mask_cpu)
              
              im = axes[1, i].imshow(kl_2d, origin='lower', cmap='RdBu_r')
              axes[1, i].set_title(f'KL Mode {i+1}')
              axes[1, i].set_xticks([])
              axes[1, i].set_yticks([])
              plt.colorbar(im, ax=axes[1, i], shrink=0.8)
          
          plt.tight_layout()
          plt.savefig('calibration/influence_functions_and_kl_modes.png', dpi=150, bbox_inches='tight')
          plt.show()
          
          print("✓ influence_functions_and_kl_modes.png")
          
      except ImportError:
          print("Matplotlib not available - skipping visualization")
      
      print(f"\nInfluence functions and modal basis computation completed!")
      print(f"Files saved in: {os.path.abspath('calibration/')}")
      print(f"\nFiles created:")
      print(f"  tutorial_ifunc.fits  - Zonal influence functions ({n_valid_actuators} actuators)")
      print(f"  tutorial_m2c.fits    - KL modal basis ({kl_basis.shape[0]} modes)")
      
      # Step 6: Test loading the saved files
      print(f"\nTesting file loading...")
      
      try:
          # Test IFunc loading
          loaded_ifunc = IFunc.restore('calibration/tutorial_ifunc.fits', target_device_idx=specula.current_device_idx)
          assert loaded_ifunc.influence_function.shape == influence_functions.shape
          print("✓ IFunc loading test passed")
          
          # Test M2C loading  
          loaded_m2c = M2C.restore('calibration/tutorial_m2c.fits', target_device_idx=specula.current_device_idx)
          assert loaded_m2c.m2c.shape == kl_basis.shape
          print("✓ M2C loading test passed")
          
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

   Results:
   Valid actuators: 1240/1681 (73.8%)
   Pupil pixels: 19847/25600 (77.5%)
   Influence functions shape: (1240, 19847)
   Memory usage: 98.5 MB

   Saving to calibration/ directory...
   ✓ influence_functions.fits
   ✓ pupil_mask.fits
   ✓ influence_functions_examples.png

   Influence functions computation completed!
   Files saved in: /path/to/your/simulation/calibration/

**What this does:**

1. **Defines the actuator geometry**: 41×41 grid with circular layout optimized for round telescope pupils

2. **Applies realistic constraints**:
   - **Mechanical coupling**: Actuators influence their neighbors (31% nearest, 5% next-nearest)
   - **Actuator slaving**: Edge actuators outside the pupil are disabled
   - **Central obstruction**: 14% obstruction removes central actuators

3. **Computes influence functions**: Each of the 1240 valid actuators produces a unique pattern of phase change across the 19,847 pupil pixels

4. **Saves calibration data**: Files are saved in FITS format for use by the main simulation

5. **Generates visualization**: Example influence functions showing the localized nature of actuator effects

**Key Parameters Explained:**

- **``pupil_pixels=160``**: Higher resolution than the basic 128 pixels, providing better sampling of actuator influence functions

- **``n_actuators=41``**: Chosen to give ~1240 valid actuators after removing those outside the pupil, typical for modern AO systems

- **``couplingCoeffs=[0.31, 0.05]``**: Realistic mechanical coupling between adjacent actuators in deformable mirrors

- **``slavingThr=0.1``**: Actuators with <10% overlap with the pupil are disabled (slaved to neighbors)

This pre-computation step is essential because:
- Influence functions are expensive to calculate
- They're needed for interaction matrix calibration
- They can be reused for multiple simulations with the same geometry

The generated files will be automatically loaded by the DM configuration in the next steps.

Prepare the simulation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have computed the influence functions, we need to create the main simulation configuration file that uses them. We'll create a YAML parameter file inspired by the ERIS NGS configuration.

Create ``config/scao_tutorial.yaml``:

.. code-block:: yaml

   # SCAO Tutorial Configuration
   # ===========================
   # VLT-like telescope with Shack-Hartmann NGS
   
   # Main simulation parameters
   main:
     class:             'SimulParams'
     root_dir:          './calibration'       # Directory containing influence functions
     pixel_pupil:       160                   # Must match influence function computation
     pixel_pitch:       0.0513                # [m] 8.2m / 160 pixels = 0.0513 m/pixel
     total_time:        2.000                 # [s] 2 seconds simulation
     time_step:         0.001                 # [s] 1ms time steps (1 kHz)
     zenithAngleInDeg:  0.0                   # [deg] Zenith observation (no airmass)
     display_server:    false                 # Disable for batch runs
   
   # Atmospheric conditions
   seeing:
     class:             'FuncGenerator'
     constant:          0.65                  # [arcsec] Good seeing conditions (r0 ≈ 15cm)
     outputs:           ['output']
   
   wind_speed:
     class:             'FuncGenerator'
     constant:          [10.0, 12.0, 8.0]    # [m/s] Multi-layer wind speeds
     outputs:           ['output']
   
   wind_direction:
     class:             'FuncGenerator'
     constant:          [45.0, 135.0, -30.0] # [deg] Wind directions for each layer
     outputs:           ['output']
   
   # Science target (on-axis)
   source_science:
     class:             'Source'
     polar_coordinates: [0.0, 0.0]           # [arcsec, deg] On-axis target
     magnitude:         10.0                  # H-band magnitude
     wavelengthInNm:    1650                  # [nm] H-band center
   
   # Natural guide star for WFS
   source_ngs:
     class:             'Source'
     polar_coordinates: [0.0, 0.0]           # [arcsec, deg] On-axis NGS
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
       common_layer_list: ['pupilstop', 'dm.out_layer:-1']  # Pupil + DM correction
     outputs:           ['out_source_science_ef', 'out_source_ngs_ef']
   
   # Shack-Hartmann wavefront sensor
   sh:
     class:             'SH'
     subap_on_diameter: 40                    # 40x40 subapertures across pupil
     subap_wanted_fov:  3.0                   # [arcsec] Subaperture field of view
     sensor_pxscale:    0.5                   # [arcsec/pixel] Pixel scale
     subap_npx:         8                     # 8x8 pixels per subaperture
     wavelengthInNm:    800                   # [nm] R-band sensing
     inputs:
       in_ef:           'prop.out_source_ngs_ef'
     outputs:           ['out_i']
   
   # CCD detector simulation
   detector:
     class:             'CCD'
     simul_params_ref:  'main'
     size:              [320, 320]            # Total detector size (40x40 × 8x8)
     dt:                0.001                 # [s] Integration time (1ms)
     bandw:             400                   # [nm] R+I-band filter width 600-1000nm
     photon_noise:      true                  # Enable photon noise
     readout_noise:     true                  # Enable read noise
     readout_level:     2.0                   # [e-/pix/frame] Read noise level
     quantum_eff:       0.8                   # QE × transmission
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
     gain:              [0.30]
     n_modes:           [1240]                # Number of modes to control
     inputs:
       delta_comm:      'modalrec.out_modes'
     outputs:           ['out_comm']
   
   # Deformable mirror
   dm:
     class:             'DM'
     simul_params_ref:  'main'
     ifunc_object:      'tutorial_ifunc'      # Our computed influence functions
     m2c_object:        'tutorial_m2c'        # Modal-to-command matrix
     nmodes:            1240                  # Number of controlled modes
     height:            0                     # Ground-conjugated DM
     inputs:
       in_command:      'integrator.out_comm'
     outputs:           ['out_layer']
   
   # Science PSF computation
   psf:
     class:             'PSF'
     wavelengthInNm:    1650                 # [nm] H-band science
     nd:                4                    # 4× padding for PSF
     start_time:        0.2                  # Start PSF integration after 200ms
     inputs:
       in_ef:           'prop.out_source_science_ef'
     outputs:           ['out_psf', 'out_sr']

**What we've created:**

1. **Main configuration file** (``scao_tutorial.yaml``) that defines the complete AO system

The configuration is now ready to run the calibration step!

Part 2: Running the Simulation
------------------------------

The basic way to run the simulation is to use the Simul class directly:

.. code-block:: python

    import specula
    specula.init(target_device_idx, precision=1)

    print(args)    
    from specula.simul import Simul
    simul = Simul(yml_file,
                  overrides=args.overrides,
                  diagram=args.diagram,
                  diagram_filename=args.diagram_filename,
                  diagram_title=args.diagram_title,
    )
    simul.run()

where target_device_idx is the GPU device number (or -1 for CPU), and yml_file is the path to your configuration file.

This is embedded in the main simulation script ``main_simul.py`` that can be found in the ``main/scao`` directory.

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
     energy_th:         0.25                 # 25% energy threshold
     inputs:
       in_i: 'sh.out_i'                     # WFS intensity input
   
   # Short calibration run
   main_override:
     total_time: 0.010                       # 10ms (just measure pupil)
   
   # Clean pupil measurement (no atmosphere)
   prop_override:
     inputs:
       common_layer_list: ['pupilstop']      # Only telescope pupil
   
   # Remove unnecessary objects
   remove: ['atmo', 'dm', 'slopec', 'modalrec', 'integrator', 'psf']

Run the subaperture calibration:

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml calib_subaps.yml

**Expected output:**

.. code-block:: text

   Subaperture calibration completed
   Valid subapertures: 1247/1600 (77.9%)
   Output file: calibration/tutorial_subaps.fits

This step identifies approximately 1247 valid subapertures out of the 1600 total (40×40 grid), excluding those outside the pupil or with insufficient illumination.

Push-Pull Amplitude Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The interaction matrix calibration requires amplitude values for each actuator poke. Create ``prepare_pushpull_amplitudes.py``:

.. code-block:: python

   import numpy as np
   from astropy.io import fits
   
   # Create 50nm poke amplitudes for all valid actuators
   n_actuators = 1240  # Number of valid actuators (from influence functions)
   amplitudes = np.full(n_actuators, 50e-9)  # 50nm in meters
   
   # Save amplitude vector
   fits.writeto('calibration/pushpull_1240modes_amp50.fits', amplitudes, overwrite=True)
   print(f"Created amplitude vector: {n_actuators} actuators, 50nm poke")

Run the preparation script:

.. code-block:: bash

   python prepare_pushpull_amplitudes.py

**Performance note:** The 50nm amplitude is chosen as a compromise:
   * **Too small** (< 20nm): Poor signal-to-noise ratio
   * **Too large** (> 100nm): Nonlinear WFS response
   * **50nm**: Good SNR while maintaining linearity

Interaction Matrix and Reconstructor Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now calibrate the interaction matrix (how actuators affect WFS measurements) and compute the reconstruction matrix (how to convert slopes to actuator commands).

Create ``calib_im_rec.yml``:

.. code-block:: yaml

   # Interaction Matrix and Reconstructor Calibration
   # ================================================
   
   # Push-pull command generator
   pushpull:
     class:     'FuncGenerator'
     func_type: 'PUSHPULL'
     nmodes:    1240                         # Number of DM actuators
     vect_amplitude_data: 'pushpull_1240modes_amp50'  # Amplitude vector
     outputs:   ['output']
   
   # Interaction matrix calibrator
   im_calibrator:
     class:     'ImCalibrator'
     nmodes:    1240                         # Number of modes to calibrate
     im_tag:    'tutorial_im'                # Output IM filename
     data_dir:  './calibration'              # Output directory
     overwrite: true                         # Overwrite existing files
     inputs:
       in_slopes:   'slopec.out_slopes'      # WFS slopes input
       in_commands: 'pushpull.output'        # Push-pull commands
   
   # Reconstructor calibrator
   rec_calibrator:
     class:     'RecCalibrator'
     nmodes:    1240                         # Number of modes
     rec_tag:   'tutorial_rec'               # Output REC filename
     data_dir:  './calibration'              # Output directory
     overwrite: true                         # Overwrite existing files
     inputs:
       in_intmat:   'im_calibrator.out_intmat'  # Connect to IM output
   
   # Override main simulation parameters
   main_override:
     total_time: 4.96                        # 1240 modes × 2 (push+pull) × 0.002s
   
   # Disable atmosphere for clean calibration
   prop_override:
     inputs:
       common_layer_list: ['pupilstop', 'dm.out_layer']  # Only pupil + DM
   
   # Override DM to use calibration commands
   dm_override:
     inputs:
       in_command: 'pushpull.output'         # Connect to push-pull generator
   
   # Disable noise for clean measurements
   detector_override:
     photon_noise:   false                   # No photon noise
     readout_noise:  false                   # No read noise
   
   # Remove unnecessary objects during calibration
   remove: ['atmo', 'source_science', 'psf', 'modalrec', 'integrator']

Run the interaction matrix calibration:

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml calib_im_rec.yml

**Expected output:**

.. code-block:: text

   Push-pull calibration progress:
   Mode 1/1240: Push +50nm, Pull -50nm
   Mode 2/1240: Push +50nm, Pull -50nm
   ...
   Mode 1240/1240: Push +50nm, Pull -50nm
   
   Interaction matrix: (2494, 1240) [slopes × modes]
   Condition number: 12.3
   
   Reconstructor: (1240, 2494) [modes × slopes]
   Reconstruction residual: 0.02%
   
   Files saved:
   ✓ calibration/tutorial_im.fits
   ✓ calibration/tutorial_rec.fits

**What happens during calibration:**

1. **Push-pull sequence**: Each actuator is poked +50nm then -50nm
2. **Slope measurement**: WFS measures the resulting slope changes
3. **Interaction matrix**: Built from the slope responses to each actuator
4. **Reconstructor**: Computed as the pseudo-inverse of the interaction matrix
5. **Quality check**: Condition number and reconstruction residual are computed

The calibration takes about 5 seconds (2.5ms per mode × 1240 modes × 2 pokes).

Update Main Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Now update the main configuration to use the calibrated files. Modify ``config/scao_tutorial.yaml``:

.. code-block:: yaml

   # Update these sections in your main config:
   
   slopec:
     class:             'ShSlopec'
     thr_value:         0.1                   
     subapdata_object:  'tutorial_subaps'     # ← Now available from calibration
     sn_object:         null                  
     inputs:
       in_pixels:       'detector.out_pixels'
     outputs:           ['out_slopes']
   
   modalrec:
     class:             'Modalrec'
     recmat_object:     'tutorial_rec'        # ← Now available from calibration
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs:           ['out_modes']

The system is now fully calibrated and ready for closed-loop operation!

Closed-Loop Simulation
~~~~~~~~~~~~~~~~~~~~~~

Now run the full closed-loop simulation:

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml

TODO write that SR is printed during the simulation.

Part 3: Results Analysis
------------------------

Analyze the results.
TODO

Part 4: Parameter Optimization
------------------------------

Now that you have a working baseline, let's optimize the system performance.

Loop Gain Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Test different control gains to find the optimum:

.. code-block:: yaml

   # Create


WFS Resolution Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~

Compare different subaperture numbers:

.. code-block:: yaml

   # yaml

Part 5: Advanced Topics
-----------------------
      
Guide Star Magnitude Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Study performance vs. star brightness:

.. code-block:: yaml

   sweep


Troubleshooting Common Issues
-----------------------------

TODO

Computational Issues
~~~~~~~~~~~~~~~~~~~

TODO

Summary and Next Steps
----------------------

Congratulations! You've successfully:

✅ **Configured** a complete SCAO system
✅ **Calibrated** the interaction and reconstruction matrices  
✅ **Executed** a closed-loop simulation
✅ **Analyzed** performance results
✅ **Optimized** system parameters

**Next Steps:**

1. **Experiment** with different atmospheric conditions
2. **Try** pyramid wavefront sensors
3. **Explore** laser guide star systems  
4. **Scale up** to MCAO configurations
5. **Compute** off-axis PSFs

**Additional Resources:**

* :doc:`../processing_objects` - Create custom components
* :doc:`../analysis` - Advanced performance analysis
* :doc:`mcao_tutorial` - Multi-conjugate AO systems
* :doc:`../troubleshooting` - Detailed problem solving

The complete tutorial files are available in the SPECULA examples directory.

.. seealso::
   
   TODO: Add links to relevant documentation sections for further reading