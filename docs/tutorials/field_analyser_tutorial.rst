.. _field_analyser_tutorial:

Field Analyser Tutorial: Post-Processing PSF, Modal Analysis, and Phase Cubes
=============================================================================

This tutorial demonstrates how to use SPECULA's `FieldAnalyser` to compute the Point Spread Function (PSF), modal coefficients, and phase cubes **after** running a simulation.  
Unlike the main simulation tutorials, here we focus on post-processing: extracting and analyzing results from previously generated simulation data.

**Goals:**
- Learn how to use the `FieldAnalyser` class for post-processing
- Compute the PSF, modal coefficients, and phase cubes from simulation outputs
- Compare results with those generated during the simulation

**Prerequisites:**
- You have already run a simulation and have a data directory with results (see :ref:`scao_tutorial` for running a simulation)
- The output directory contains files such as `res_psf.fits`, `res.fits`, and `phase.fits`

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

Step 1: Locate Your Simulation Output
-------------------------------------

After running a simulation, SPECULA saves results in a timestamped directory (e.g., `data/20240703_153000/`).  
This directory should contain files such as:
- `res_psf.fits` (PSF data)
- `res.fits` (modal coefficients)
- `phase.fits` (phase cubes)

Step 2: Using FieldAnalyser in Python
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

    # Compute phase cube
    cube_results = analyser.compute_phase_cube()
    phase_cube = cube_results['phase_cubes'][0]

    print("PSF shape:", field_psf.shape)
    print("Modal coefficients shape:", modes.shape)
    print("Phase cube shape:", phase_cube.shape)

Step 3: Visualizing the Results
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
    plt.title('Last Phase Slice')
    plt.colorbar()
    plt.show()

Step 4: Comparing with Simulation Outputs
-----------------------------------------

You can compare the results from `FieldAnalyser` with those saved during the simulation (e.g., `res_psf.fits`, `res.fits`, `phase.fits`) to verify consistency.

.. code-block:: python

    from astropy.io import fits

    # Load original PSF from simulation
    with fits.open(os.path.join(latest_data_dir, 'res_psf.fits')) as hdul:
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

Tips and Customizations
-----------------------

- You can specify multiple field points by passing a list of coordinates to `polar_coordinates`.
- Change `wavelength_nm` to compute the PSF at different wavelengths.
- Use `force_recompute=False` to avoid recomputing if output files already exist.
- The `FieldAnalyser` can also compute off-axis PSFs and analyze multi-source simulations.

**Conclusion**

With `FieldAnalyser`, you can flexibly post-process SPECULA simulation results, recompute PSFs, modal coefficients, and phase cubes for any field point or wavelength, and compare them with the original simulation outputs.

.. seealso::

   - :ref:`scao_tutorial` for running a full simulation
   - SPECULA API documentation for details on `FieldAnalyser`