.. _field_analyser_tutorial:

Field Analyser Tutorial: Post-Processing PSF, Modal Analysis, and Phase Cubes
=============================================================================

This tutorial demonstrates how to use SPECULA's `FieldAnalyser` to compute the Point Spread Function (PSF), modal coefficients, and phase cubes **after** running a simulation.  
Unlike the main simulation tutorials, here we focus on post-processing: extracting and analyzing results from previously generated simulation data.

**Goals:**

- Learn how to use the `FieldAnalyser` class for post-processing

- Understand what data to save during simulation for efficient replay

- Compute the PSF, modal coefficients, and phase cubes (units: nm) from simulation outputs

- Compare results with those generated during the simulation

**Prerequisites:**

- You have already run a simulation and have a data directory with results (see :ref:`scao_basic_tutorial` for running a simulation)

- The output directory contains the necessary replay data (see below)

Overview
--------

The `FieldAnalyser` is a powerful tool for post-processing SPECULA simulation results.  
It allows you to:

- Recompute the PSF for arbitrary field points and wavelengths

- Perform modal analysis on the residual phase

- Extract phase cubes for further analysis

This is especially useful for:

- Exploring the PSF at different field positions or wavelengths without rerunning the simulation

- Comparing different analysis methods

- Generating additional outputs for publications or diagnostics

**Key Concept: Efficient Replay with DM Commands**

The major computational advantage of `FieldAnalyser` is that it can replay the simulation using **only the saved DM commands**, without re-running the WFS and all related computationally expensive objects (detectors, slope computation, reconstructors, controllers, etc.). 

By saving the DM input commands (i.e., the control signals applied to the DM), the replay process can:

1. Skip the entire WFS processing chain (detector readout, slope computation, reconstruction, control law)
2. Directly apply the saved commands to the DM
3. Propagate through atmosphere and compute PSF/phases

This provides a significant **speedup** while maintaining full accuracy for wavefront propagation analysis.

Step 1: Configuring Your Simulation for Field Analysis
------------------------------------------------------

To enable efficient field analysis, you need to configure your simulation's `DataStore` to save the **DM input commands** (i.e., the output of the controller). Here's an example configuration for a Single Conjugate AO (SCAO) simulation:

.. code-block:: yaml

    # Example: Saving DM commands for later replay
    data_store:
      class: 'DataStore'
      store_dir: './output/'
      data_format: 'fits'  # or 'pickle'
      inputs:
        input_list: 
          - 'comm-control.out_comm'                   # Save DM commands (essential!)
          - 'res_modes-modal_analysis.out_modes'      # Optional: save modal coefficients
          - 'sr-psf.out_sr'                           # Optional: save original SR
          - 'psf-psf.out_int_psf'                     # Optional: save original PSF

**Critical Point:** You must save the **DM input commands** (the controller output) for `FieldAnalyser` to work efficiently. 

- In the example above, this is `control.out_comm` (from the Integrator controller)
- **The exact name depends on your simulation configuration** (e.g., `control.out_comm`, `my_controller.output`, etc.)
- This is the signal that **enters** the DM, not the DM surface output
- Without saving these commands, the entire simulation (including WFS processing) would need to be re-run

**Important:** The object name and output name may vary depending on your configuration:

- If your controller object is named `ctrl`, save `ctrl.out_comm`
- If you use a different controller class, check its output name in the documentation
- The key is to save the **commands sent to the DM**, not the DM surface itself

**What Gets Saved:**

- **DM commands** (`control.out_comm` or similar): Time series of control signals applied to DM - **required**
- **Modal coefficients** (optional): For comparison with recomputed values
- **PSF/SR** (optional): For validation of recomputed PSFs

**Storage Considerations:**

- DM command vectors are typically small compared to phase cubes (3D arrays)
- Storage size: ~(n_modes × n_frames × 8 bytes) for modal control
- Example: 100 modes, 1000 frames = ~0.8 MB per DM
- For influence function control: ~(n_actuators × n_frames × 8 bytes)
- Example: 1000 actuators, 1000 frames = ~8 MB per DM

**Comparison with Full Simulation Storage:**

- Phase cubes: (npixels × npixels × n_frames × 8 bytes) for single precision floating point (units: nm)
- Example: 160×160 pixels, 1000 frames = ~200 MB
- **Saving only DM commands typically reduces storage compared to phase cubes**

Step 2: Locate Your Simulation Output
-------------------------------------

After running a simulation, SPECULA saves results in a timestamped directory (e.g., `data/20240703_153000/`).  
This directory should contain:

**Required files:**

- `params.yml` - Original simulation parameters
- `replay_params.yml` - Automatically generated replay configuration
- `comm.fits` - Saved DM commands (or different name based on your prefix)

**Optional files (for comparison):**

- `res_modes.fits` - Original modal coefficients
- `sr.fits` - Original Strehl ratio
- `psf.fits` - Original PSF

**Note:** The exact filenames depend on the `input_list` in your `DataStore` configuration. The naming pattern is:

.. code-block:: none

   {prefix}.{extension}

where:

- `{prefix}` is the part **before** the dash in your input_list entry (e.g., `comm`, `res_modes`, `psf`)
- `{extension}` is the data format (e.g., `.fits`, `.pkl`)

**The part after the dash** (e.g., `control.out_comm`) is used **only** to identify which data to save from the simulation, not for the filename.

**Examples:**

.. code-block:: yaml

    data_store:
      inputs:
        input_list: 
          - 'comm-control.out_comm'        # Saves as: comm.fits
          - 'res_modes-modal_analysis.out_modes'  # Saves as: res_modes.fits
          - 'sr-psf.out_sr'                # Saves as: sr.fits
          - 'psf-psf.out_int_psf'          # Saves as: psf.fits

**Best practice:** Always use a descriptive prefix before the dash to create clean, meaningful filenames.

Step 3: Using FieldAnalyser in Python
-------------------------------------

You can use the `FieldAnalyser` class interactively or in a script.  
Below is an example script that loads the latest simulation output and computes the PSF, modal coefficients, and phase cubes for the on-axis source.

.. code-block:: python

    import os
    import glob
    import numpy as np
    import specula
    specula.init(0)
    
    from specula.field_analyser import FieldAnalyser

    # Find the latest data directory (assuming output is in ./data)
    data_dirs = sorted(glob.glob("data/2*"))
    if not data_dirs:
        raise RuntimeError("No data directory found.")
    latest_data_dir = data_dirs[-1]
    print(f"Using data directory: {latest_data_dir}")

    # Set up FieldAnalyser for on-axis source at 1650 nm
    polar_coords = np.array([[0.0, 0.0]])  # on-axis
    analyser = FieldAnalyser(
        data_dir="data",
        tracking_number=os.path.basename(latest_data_dir),
        polar_coordinates=polar_coords,
        wavelength_nm=1650,  # Science wavelength
        start_time=0.0,
        end_time=None,
        verbose=True
    )

    # Compute PSF
    psf_results = analyser.compute_field_psf(
        psf_sampling=7,         # Padding factor, should match your simulation
        force_recompute=True    # Recompute even if files exist
    )
    field_psf = psf_results['psf_list'][0]

    # Compute modal analysis
    modal_results = analyser.compute_modal_analysis()
    modes = modal_results['modal_coeffs'][0]

    # Compute phase cube (units: nm)
    cube_results = analyser.compute_phase_cube()
    phase_cube = cube_results['phase_cubes'][0]

    print("PSF shape:", field_psf.shape)
    print("Modal coefficients shape:", modes.shape)
    print("Phase cube shape:", phase_cube.shape)

**Behind the Scenes:**

When you call `compute_field_psf()`, `FieldAnalyser`:

1. Reads the saved DM commands from the DataStore files (e.g., `comm-control.out_comm.fits`)
2. Uses `build_targeted_replay` to create a minimal replay configuration
3. Creates a replay chain that includes:
   
   - `DataSource` object to read saved DM commands
   - `DM` object to apply commands and generate wavefront
   - `AtmoPropagation` to propagate through atmosphere
   - `PSF` object to compute PSF at specified wavelength

4. **Skips entirely:**
   
   - All WFS objects (Pyramid, Shack-Hartmann, detectors)
   - Slope computation and reconstruction
   - Controllers and all feedback loop components
   - All calibration objects

5. Re-runs **only** the forward propagation path: Saved DM commands → DM surface → atmosphere → PSF
6. Computes PSF for your specified field positions and wavelength

**Computational Savings:**

- **Full simulation:** Atmosphere + WFS detector + slopes + reconstruction + control + DM + propagation + PSF
- **Replay with FieldAnalyser:** Saved commands → DM + propagation + PSF
- **Speedup:** Typically 1-10× faster, depending on WFS configuration

This speedup allows you to explore different wavelengths, field positions, and analysis parameters interactively!

Step 4: Visualizing the Results
-------------------------------

You can use matplotlib to visualize the PSF, modal coefficients, or phase slices:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Display the PSF (log scale)
    plt.figure()
    plt.imshow(field_psf[0], origin='lower', cmap='hot', norm='log')
    plt.title('FieldAnalyser PSF (Log Scale)')
    plt.colorbar()
    plt.show()

    # Plot modal coefficients (first 10 modes)
    plt.figure()
    plt.plot(modes[:10])
    plt.title('First 10 Modal Coefficients')
    plt.xlabel('Mode')
    plt.ylabel('Coefficient')
    plt.show()

    # Show the last phase slice
    plt.figure()
    plt.imshow(phase_cube[-1, 1, :, :], origin='lower', cmap='hot')
    plt.title('Last Phase Slice (units: nm)')
    plt.colorbar()
    plt.show()

Step 5: Comparing with Simulation Outputs
-----------------------------------------

You can compare the results from `FieldAnalyser` with those saved during the simulation (e.g., `psf-psf.out_int_psf.fits`, `res_modes-modal_analysis.out_modes.fits`) to verify consistency.

.. code-block:: python

    from astropy.io import fits

    # Load original PSF from simulation
    with fits.open(os.path.join(latest_data_dir, 'psf-psf.out_int_psf.fits')) as hdul:
        original_psf = hdul[0].data
    
    # Normalize for fair comparison
    field_psf_norm = field_psf[0] / field_psf[0].sum()
    original_psf_norm = original_psf / original_psf.sum()

    # Compare visually
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_psf_norm, origin='lower', cmap='hot', norm='log')
    plt.title('Original PSF')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(field_psf_norm, origin='lower', cmap='hot', norm='log')
    plt.title('FieldAnalyser PSF')
    plt.colorbar()
    plt.show()

Advanced Usage: Multiple Field Points
-------------------------------------

One of the key advantages of `FieldAnalyser` is computing PSFs at multiple field positions efficiently:

.. code-block:: python

    # Define multiple field points
    polar_coords = np.array([
        [0.0, 0.0],      # on-axis
        [15.0, 0.0],     # 15 arcsec East
        [15.0, 90.0],    # 15 arcsec North
        [30.0, 45.0],    # 30 arcsec NE
    ])

    analyser = FieldAnalyser(
        data_dir="data",
        tracking_number=os.path.basename(latest_data_dir),
        polar_coordinates=polar_coords,
        wavelength_nm=1650,
        start_time=0.0,
        verbose=True
    )

    # Compute PSFs for all field points at once
    psf_results = analyser.compute_field_psf(psf_sampling=7)

    # Access individual PSFs
    for i, (r, theta) in enumerate(polar_coords):
        psf = psf_results['psf_list'][i]
        sr = psf_results['sr_list'][i]
        print(f"Field point ({r:.1f}\", {theta:.1f}°): SR = {sr:.3f}")

Tips and Customizations
-----------------------

- **Storage optimization:** Save DM commands in FITS format (more compact than pickle)
- **Time range:** Use `start_time` and `end_time` to analyze specific portions of the simulation
- **Wavelength scanning:** Recompute PSFs at different wavelengths without re-running atmosphere/WFS
- **Field mapping:** Generate PSF maps across the field of view efficiently
- **Caching:** Set `force_recompute=False` to reuse previously computed results
- **Controller names:** Always check your controller object name in `params.yml` before configuring DataStore
- **Multiple DMs:** For multi-DM systems, save all DM commands: `['dm1.out_comm', 'dm2.out_comm']`

**What to Save for Different Configurations:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - AO Configuration
     - Controller Output
     - DataStore Entry
   * - SCAO with Integrator
     - `control.out_comm`
     - `'comm-control.out_comm'`
   * - MCAO with 2 DMs
     - `control0.out_comm`, `control1.out_comm`
     - `['comm0-control0.out_comm', 'comm1-control1.out_comm']`
   * - Open loop control
     - `rec.out_modes`
     - `'ol_comm-rec.out_modes'`

**Conclusion**

With `FieldAnalyser`, you can efficiently post-process SPECULA simulation results by:

1. Saving **DM input commands** (controller outputs) during the original simulation
2. Replaying only the propagation path (DM → atmosphere → PSF)
3. Computing PSFs, modal coefficients, and phase cubes (units: nm) for arbitrary field positions and wavelengths

This provides significant computational savings while maintaining full accuracy for wavefront analysis, by eliminating the need to re-run the computationally expensive WFS processing chain.

.. seealso::

   - :ref:`scao_basic_tutorial` for running a full simulation with proper DataStore configuration
   - :ref:`scao_tutorial` for a complete SCAO workflow with calibration
   - `DataStore <../specula/processing_objects/data_store.py>`_ for data saving configuration
   - `build_targeted_replay <../specula/simul.py>`_ for understanding the replay mechanism
   - SPECULA API documentation for details on `FieldAnalyser` and controller classes