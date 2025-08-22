.. _calibration_manager:

Calibration Manager
===================

The ``CalibrationManager`` class provides a structured way to organize and manage calibration files in SPECULA simulations. It maintains a hierarchical directory structure and handles file paths automatically.

Overview
--------

The calibration manager organizes files into predefined subdirectories based on their type:

- **phasescreens/**: Atmospheric phase screens
- **pupils/**: Pupil data and masks  
- **pupilstop/**: Pupil stop masks and configurations
- **subapdata/**: Sub-aperture data for Shack-Hartmann sensors
- **im/**: Interaction matrices
- **rec/**: Reconstruction matrices
- **data/**: General data files
- **filter/**: Control filters
- **m2c/**: Mirror-to-commands matrices
- And many more...

Basic Usage
-----------

Initialize the calibration manager with a root directory:

.. code-block:: python

    from specula.calib_manager import CalibManager
    
    # Initialize with root calibration directory
    calib = CalibManager('/path/to/calibration/root')

The manager will automatically create subdirectories under this root path as needed.

Automatic data_dir Handling in Simul
------------------------------------

When using the `Simul` class to build your simulation from a YAML file, any object parameter named `data_dir` is automatically replaced with a path managed by the `CalibrationManager`. This ensures that all calibration and data files are stored in the correct subdirectory of your calibration root.

**Example: Using ImCalibrator with automatic data_dir**

.. code-block:: yaml

    im_calibrator:
      class: ImCalibrator
      nmodes: 100
      # data_dir: ""   # This will be replaced automatically
      im_tag: auto
      # ... other parameters ...

When the simulation is built, `data_dir` will be set to something like:

.. code-block:: text

    /your/calibration/root/im/

This is handled transparently by Simul and the CalibrationManager.

Saving and Loading Pupil Stops
-------------------------------

One common use case is saving pupil stop configurations and masks:

**Example 1: Creating and Saving a Pupil Stop**

.. code-block:: python

    import numpy as np
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.calib_manager import CalibManager
    
    # Initialize calibration manager with root_dir
    # (same as root_dir in main section of the yml file)
    calib = CalibManager('/data/specula_calibrations')
    
    # Create simulation parameters
    simul_params = SimulParams(
        pixel_pupil=512,
        pixel_pitch=0.1  # meters
    )
    
    # Create a pupil stop with circular aperture
    pupilstop = Pupilstop(
        simul_params=simul_params,
        mask_diam=1.0,        # Normalized diameter
        obs_diam=0.15,        # Central obstruction (normalized)
        shiftXYinPixel=(0.0, 0.0),
        rotInDeg=0.0,
        magnification=1.0
    )
    
    # Save using calibration manager - automatically goes to pupilstop/ subdirectory
    pupil_filename = calib.filename('Pupilstop', 'main_telescope_pupil')
    pupilstop.save(pupil_filename)
    
    print(f"Pupil stop saved to: {pupil_filename}")

**Example 2: Loading a Pupil Stop**

.. code-block:: python

    # Load the pupil stop back
    pupil_filename = calib.filename('Pupilstop', 'main_telescope_pupil')
    loaded_pupilstop = Pupilstop.restore(pupil_filename)
    
    print(f"Loaded pupil stop: {loaded_pupilstop.pixel_pupil}x{loaded_pupilstop.pixel_pupil} pixels")
    print(f"Pixel pitch: {loaded_pupilstop.pixel_pitch} m")
    print(f"Shift: {loaded_pupilstop.shiftXYinPixel} pixels")

**Example 3: Creating Custom Pupil Masks**

.. code-block:: python

    from specula.lib.make_mask import make_mask
    
    # Create custom pupil mask with spiders
    pixel_pupil = 256
    custom_mask = make_mask(pixel_pupil, obs_diam=0.14, mask_diam=1.0)
    
    # Add spider vanes (simplified example)
    center = pixel_pupil // 2
    spider_width = 3
    custom_mask[center-spider_width//2:center+spider_width//2, :] = 0  # Horizontal spider
    custom_mask[:, center-spider_width//2:center+spider_width//2] = 0  # Vertical spider
    
    # Create pupil stop with custom mask
    simul_params = SimulParams(pixel_pupil, 0.05)
    pupilstop = Pupilstop(
        simul_params=simul_params,
        input_mask=custom_mask,  # Use custom mask
        shiftXYinPixel=(2.5, -1.0),  # Slight offset
        rotInDeg=15.0  # Rotate spiders
    )
    
    # Save custom pupil
    custom_pupil_filename = calib.filename('Pupilstop', 'telescope_with_spiders')
    pupilstop.save(custom_pupil_filename)

**Example 4: Batch Processing Multiple Pupil Configurations**

.. code-block:: python

    # Create multiple pupil configurations for different conditions
    configurations = [
        {'name': 'nominal', 'shift': (0.0, 0.0), 'rot': 0.0, 'obs': 0.14},
        {'name': 'misaligned', 'shift': (2.0, 1.5), 'rot': 0.0, 'obs': 0.14},
        {'name': 'rotated', 'shift': (0.0, 0.0), 'rot': 45.0, 'obs': 0.14},
        {'name': 'large_obstruction', 'shift': (0.0, 0.0), 'rot': 0.0, 'obs': 0.20},
    ]
    
    simul_params = SimulParams(512, 0.1)
    
    for config in configurations:
        pupilstop = Pupilstop(
            simul_params=simul_params,
            mask_diam=1.0,
            obs_diam=config['obs'],
            shiftXYinPixel=config['shift'],
            rotInDeg=config['rot'],
            magnification=1.0
        )
        
        # Save with descriptive name
        pupil_name = f"pupil_{config['name']}"
        pupil_filename = calib.filename('Pupilstop', pupil_name)
        pupilstop.save(pupil_filename)
        
        print(f"Saved pupil configuration: {config['name']}")

Working with Different Data Types
---------------------------------

The calibration manager supports many different data types:

**General Pupil Data:**

.. code-block:: python

    # Save general pupil mask data
    pupil_mask = create_pupil_mask()  # Your function
    calib.writefits('pupils', 'main_pupil_mask', pupil_mask)
    
    # Load pupil mask
    pupil_mask = calib.readfits('pupils', 'main_pupil_mask')

**Interaction Matrices:**

.. code-block:: python

    # Save interaction matrix
    interaction_matrix = calibrate_interaction_matrix()  # Your function
    calib.writefits('im', 'pyramid_interaction_matrix', interaction_matrix)

**Phase Screens:**

.. code-block:: python

    # Save atmospheric phase screen
    phase_screen = generate_phase_screen()  # Your function
    calib.writefits('phasescreen', 'kolmogorov_screen_001', phase_screen)

File Path Management
--------------------

The manager automatically handles file extensions and paths:

.. code-block:: python

    # These are equivalent:
    filename1 = calib.filename('Pupilstop', 'my_pupil')
    filename2 = calib.filename('Pupilstop', 'my_pupil.fits')
    
    # Both return: '/path/to/calibration/root/pupilstop/my_pupil.fits'

**Getting just the filename without reading:**

.. code-block:: python

    # Get filename for external use
    filename = calib.readfits('Pupilstop', 'my_pupil', get_filename=True)
    
    # Use with other libraries
    with fits.open(filename) as hdul:
        # Process FITS file manually
        pass

Directory Structure
-------------------

A typical calibration directory structure looks like:

.. code-block:: text

    /data/specula_calibrations/
    ├── data/
    │   ├── my_custom_data.fits
    │   └── measurement_data.fits
    ├── pupilstop/
    │   ├── main_telescope_pupil.fits
    │   ├── telescope_with_spiders.fits
    │   └── pupil_misaligned.fits
    ├── pupils/
    │   ├── main_pupil_mask.fits
    │   └── secondary_mask.fits
    ├── phasescreens/
    │   ├── kolmogorov_screen_001.fits
    │   └── von_karman_screen_001.fits
    ├── im/
    │   ├── pyramid_interaction_matrix.fits
    │   └── sh_interaction_matrix.fits
    └── rec/
        ├── mmse_reconstructor.fits
        └── least_squares_reconstructor.fits

API Reference
-------------

.. autoclass:: specula.calib_manager.CalibManager
   :members:
   :undoc-members:
   :show-inheritance: