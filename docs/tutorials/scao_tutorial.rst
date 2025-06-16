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
   from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
   from astropy.io import fits
   import os
   
   def compute_and_save_influence_functions():
       """
       Compute zonal influence functions for the SCAO tutorial
       """
       # DM and pupil parameters for VLT-like telescope
       pupil_pixels = 160           # Pupil sampling resolution
       n_actuators = 41             # 41x41 = 1681 total actuators
       telescope_diameter = 8.2     # meters (VLT Unit Telescope)
       
       # Pupil geometry
       obsratio = 0.14              # 14% central obstruction
       diaratio = 1.0               # Full pupil diameter
       
       # Actuator geometry
       circGeom = True              # Circular geometry (better for round pupils)
       angleOffset = 0              # No rotation
       
       # Mechanical coupling between actuators
       doMechCoupling = True        # Enable realistic coupling
       couplingCoeffs = [0.31, 0.05]  # Nearest and next-nearest neighbor coupling
       
       # Actuator slaving (disable edge actuators outside pupil)
       doSlaving = True             # Enable slaving
       slavingThr = 0.1             # Threshold for valid actuators
       
       # Computation parameters
       dtype = np.float32           # Use single precision for speed
       
       print("Computing zonal influence functions...")
       print(f"Pupil pixels: {pupil_pixels}")
       print(f"Actuators: {n_actuators}x{n_actuators} = {n_actuators**2}")
       print(f"Telescope diameter: {telescope_diameter}m")
       print(f"Central obstruction: {obsratio*100:.1f}%")
       
       # Generate zonal influence functions
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
           xp=specula.xp,  # Use current device (GPU or CPU)
           dtype=dtype,
           return_coordinates=False
       )
       
       # Print statistics
       n_valid_actuators = influence_functions.shape[0]
       n_pupil_pixels = np.sum(pupil_mask)
       
       print(f"\nResults:")
       print(f"Valid actuators: {n_valid_actuators}/{n_actuators**2} ({n_valid_actuators/(n_actuators**2)*100:.1f}%)")
       print(f"Pupil pixels: {n_pupil_pixels}/{pupil_pixels**2} ({n_pupil_pixels/(pupil_pixels**2)*100:.1f}%)")
       print(f"Influence functions shape: {influence_functions.shape}")
       print(f"Memory usage: {influence_functions.nbytes / 1024**2:.1f} MB")
       
       # Create output directory
       os.makedirs('calibration', exist_ok=True)
       
       # Save influence functions and pupil mask
       # Convert to CPU arrays if needed for saving
       if hasattr(influence_functions, 'get'):  # CuPy array
           influence_functions_cpu = influence_functions.get()
           pupil_mask_cpu = pupil_mask.get()
       else:  # NumPy array
           influence_functions_cpu = influence_functions
           pupil_mask_cpu = pupil_mask
       
       # Save as FITS files
       print(f"\nSaving to calibration/ directory...")
       
       # Influence functions: shape (n_actuators, n_pupil_pixels)
       hdu_if = fits.PrimaryHDU(influence_functions_cpu)
       hdu_if.header['COMMENT'] = 'Zonal influence functions'
       hdu_if.header['NACT'] = n_valid_actuators
       hdu_if.header['NPIX'] = n_pupil_pixels
       hdu_if.header['PUPRES'] = pupil_pixels
       hdu_if.header['TELDIA'] = telescope_diameter
       hdu_if.header['OBSRAT'] = obsratio
       hdu_if.writeto('calibration/influence_functions.fits', overwrite=True)
       
       # Pupil mask: shape (pupil_pixels, pupil_pixels) 
       hdu_mask = fits.PrimaryHDU(pupil_mask_cpu.astype(np.uint8))
       hdu_mask.header['COMMENT'] = 'Telescope pupil mask'
       hdu_mask.header['PUPRES'] = pupil_pixels
       hdu_mask.header['TELDIA'] = telescope_diameter
       hdu_mask.header['OBSRAT'] = obsratio
       hdu_mask.writeto('calibration/pupil_mask.fits', overwrite=True)
       
       print("✓ influence_functions.fits")
       print("✓ pupil_mask.fits")
       
       # Optional: Visualize some influence functions
       try:
           import matplotlib.pyplot as plt
           
           print("\nGenerating visualization...")
           
           # Reconstruct 2D influence functions for plotting
           def reconstruct_2d_ifunc(ifunc_1d, mask):
               ifunc_2d = np.zeros(mask.shape)
               ifunc_2d[mask] = ifunc_1d
               return ifunc_2d
           
           # Plot a few example influence functions
           fig, axes = plt.subplots(2, 3, figsize=(12, 8))
           axes = axes.flatten()
           
           # Select representative actuators (center, edge, corner)
           example_indices = [
               n_valid_actuators // 2,        # Center actuator
               n_valid_actuators // 4,        # Quarter point
               n_valid_actuators // 8,        # Eighth point
               3 * n_valid_actuators // 4,    # Three quarters
               n_valid_actuators - 100,       # Near edge
               n_valid_actuators - 1,         # Last actuator
           ]
           
           for i, act_idx in enumerate(example_indices):
               if act_idx < n_valid_actuators:
                   ifunc_2d = reconstruct_2d_ifunc(influence_functions_cpu[act_idx], pupil_mask_cpu)
                   
                   im = axes[i].imshow(ifunc_2d, origin='lower', cmap='RdBu_r')
                   axes[i].set_title(f'Actuator {act_idx}')
                   axes[i].set_xticks([])
                   axes[i].set_yticks([])
                   plt.colorbar(im, ax=axes[i], shrink=0.8)
           
           plt.tight_layout()
           plt.savefig('calibration/influence_functions_examples.png', dpi=150, bbox_inches='tight')
           plt.show()
           
           print("✓ influence_functions_examples.png")
           
       except ImportError:
           print("Matplotlib not available - skipping visualization")
       
       print(f"\nInfluence functions computation completed!")
       print(f"Files saved in: {os.path.abspath('calibration/')}")
       
       return influence_functions, pupil_mask

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
     wavelengthInNm:    650                   # [nm] R-band for WFS
   
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
     n_modes:           [1000]                # Number of modes to control
     inputs:
       delta_comm:      'modalrec.out_modes'
     outputs:           ['out_comm']
   
   # Deformable mirror
   dm:
     class:             'DM'
     simul_params_ref:  'main'
     ifunc_object:      'tutorial_ifunc'      # Our computed influence functions
     nmodes:            1000                  # Number of controlled modes
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

Run the calibration to generate:

1. List of valid sub-aperture Indices.

You need to calibrate the subaperture geometry, create ``calib_subaps.yml``:

.. code-block:: yaml

   # Subaperture Geometry Calibration (Optional)
   # ===========================================
   
   # Subaperture calibrator
   sh_subaps:
     class: 'ShSubapCalibrator'
     subap_on_diameter: 40                   # 40×40 subapertures
     output_tag:        'tutorial_subaps_measured'
     energy_th:         0.25                 # 25% energy threshold
     inputs:
       in_i: 'sh.out_i'                     # WFS intensity input
   
   # Short calibration run
   main_override:
     total_time: 0.002                       # Very short (just measure pupil)
   
   # No atmosphere for clean pupil measurement
   prop_override:
     inputs:
       common_layer_list: ['pupilstop']      # Only telescope pupil
   
   # Remove unnecessary objects
   remove: ['atmo',
            'dm', 
            'slopec',
            'modalrec',
            'integrator',
            'psf']

Run the calibration

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml calib_subaps.yml

2. Interaction and reconstruction matrices.

Create ``calib_rec.yml`` for interaction matrix and reconstructor calibration:

TODO explain how to compute the push-pull commands file.

.. code-block:: yaml

   # SCAO Interaction Matrix Calibration
   # ===================================
   # Override file for main configuration
   
   # Push-pull command generator
   pushpull:
     class:     'FuncGenerator'
     func_type: 'PUSHPULL'
     nmodes:    1240                         # Number of DM actuators
     vect_amplitude_data: 'pushpull_1240modes_amp50'  # Amplitude vector tag
     outputs:   ['output']
   
   # Override main simulation parameters
   main_override:
     total_time: 4.96                        # Time for all modes (1240 × 2 × 0.002s)
   
   # Override atmospheric propagation (disable atmosphere)
   prop_override:
     inputs:
       common_layer_list: ['pupilstop', 'dm.out_layer']  # Only pupil + DM
   
   # Override DM to use push-pull commands
   dm_override:
     sign: 1                                 # Positive sign convention
     inputs:
       in_command: 'pushpull.output'         # Connect to push-pull generator
   
   # Disable noise for clean calibration
   detector_override:
     photon_noise:   false                   # No photon noise
     readout_noise:  false                   # No read noise
   
   # Interaction matrix calibrator
   calibrator:
     class:     'ImRecCalibrator'
     nmodes:    1240                         # Number of modes to calibrate
     im_tag:    'tutorial_scao_im'           # Interaction matrix filename
     rec_tag:   'tutorial_scao_rec'          # Reconstructor filename  
     data_dir:  './calibration'              # Output directory
     overwrite: true                         # Overwrite existing files
     pupdata_tag: 'tutorial_subaps'          # Subaperture data reference
     inputs:
       in_slopes:   'slopec.out_slopes'      # WFS slopes input
       in_commands: 'pushpull.output'        # Push-pull commands
   
   # Optional: Display pixels during calibration
   pixels_disp:
     class:            'PixelsDisplay'
     inputs:
       pixels:         'detector.out_pixels'
     window:           15                     # Display every 15 iterations
     title:            'Calibration Progress'
     disp_factor:      2                     # 2× magnification
     sh_as_pyr:        false                 # Shack-Hartmann display
     subapdata_object: 'tutorial_subaps'     # Subaperture layout
   
   # Remove unnecessary objects during calibration
   remove: ['atmo',                          # No atmosphere
            'source_science',                # No science target
            'psf',                           # No PSF computation
            'integrator']                    # No closed-loop control

Run the calibration

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml calib_rec.yml


Closed-Loop Simulation
~~~~~~~~~~~~~~~~~~~~~~

Now run the full closed-loop simulation:

.. code-block:: bash

   python main_simul.py config/scao_tutorial.yaml

TODO write that SR is printed during the simulation.

Part 3: Results Analysis
------------------------

Analyze the results.

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
   
   :doc:`pyramid_calibration` for Pyramid WFS systems
   :doc:`mcao_tutorial` for multi-conjugate AO
   :doc:`../configuration` for complete YAML reference
