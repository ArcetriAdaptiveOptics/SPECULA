.. _simulation_parameters:

Simulation Parameters Reference
================================

This page provides a comprehensive reference for simulation parameters, units, and conventions used throughout SPECULA.

Units and Conventions
---------------------

Unless otherwise specified, the following units are used throughout SPECULA:

- **Wavelengths**: nanometers (nm)
- **Wavefront/Phase**: nanometers (nm) of optical path difference
- **Lengths**: meters (m)
- **Angles**: arcseconds (arcsec) or degrees (deg)
- **Time**: seconds (s)
- **Wind speed**: meters per second (m/s)
- **Magnitude**: astronomical magnitude (mag)
- **Flux**: photons per second (ph/s)

Atmospheric Parameters
----------------------

Zenith Convention
~~~~~~~~~~~~~~~~~

In SPECULA, all atmospheric parameters such as **seeing**, **layer heights**, and **source heights** are defined **at zenith** (i.e., for a zenith angle of 0°).
The **zenith angle** specified in the ``main`` section (using the ``zenithAngleInDeg`` parameter) is used to compute the **airmass** (sec(zenith angle)).

**Important distinctions:**

- **Seeing**: The value (in arcsec) you provide is assumed at zenith and will be increased for off-zenith observations according to the airmass factor.
- **Layer heights**: The atmospheric layer heights (in m) are projected according to the zenith angle to account for the slant path through the atmosphere. These projected heights represent the **distance from the entrance pupil** along the line of sight.
- **Source heights**: If you use sources at finite distance (e.g., LGS), their heights are interpreted as zenith heights and projected according to the zenith angle. For example, an LGS at 90 km zenith height observed at 30° zenith angle will have an actual slant distance of 90/cos(30°) ≈ 104 km from the entrance pupil.
- **Source positions**: Source angular coordinates are always **relative to the telescope pointing direction** (on-axis), regardless of the zenith angle. The zenith angle does not change where sources appear in the field of view.

For example, if you set ``zenithAngleInDeg: 30``:
- Atmospheric turbulence is scaled by airmass (sec(30°) ≈ 1.15)
- An LGS at 90 km zenith height has a slant path of ~104 km from the entrance pupil
- A source at position ``[0, 0]`` remains on-axis
- A source at ``[10, 45]`` remains 10 arcsec away at 45° from the pointing direction

This convention allows you to simulate observations at different zenith angles while keeping the same field configuration, with automatic scaling of atmospheric parameters and source heights.

.. note::

   The **seeing** parameter is always defined at a wavelength of **500 nm** (standard astronomical convention).
   Similarly, **atmospheric phase screens** are always generated at 500 nm as a reference wavelength.
   If you want to simulate seeing at a different wavelength, you must convert it to the equivalent value at 500 nm before inserting it in the YAML file.

Atmospheric Layer Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **L0** (outer scale): Outer scale of turbulence for each layer [m]. Can be a scalar (same for all layers) or a list.
- **heights**: Heights of atmospheric layers at zenith [m].
- **Cn2**: Fractional Cn² values for each layer. Must sum to 1.0.
- **wind_speed**: Wind speed for each layer [m/s].
- **wind_direction**: Wind direction for each layer [degrees]. 0° is along +x axis, 90° is along +y axis.

Time Parameters
---------------

- **total_time**: Total simulation duration [s].
- **time_step**: Simulation time step [s]. This is the fundamental time resolution of the simulation.
- **dt** (in detectors): Detector integration time [s]. Can be a multiple of ``time_step`` to simulate slower detectors.
- **start_time**: Time after which to start recording statistics (e.g., PSF integration) [s]. Default is 0.0.

Coordinate Conventions
----------------------

- **Source positions**: Given in polar coordinates as ``[radius, angle]`` where radius is in arcseconds and angle in degrees.
- **Cartesian coordinates**: When used, x is horizontal (positive to the right) and y is vertical (positive upward).
- **Angular coordinates**: 0° is along +x axis, 90° is along +y axis (counter-clockwise).
- **Pupil position**: Can be specified in meters as ``[x, y]`` offset from the optical axis.

Precision and Device Settings
------------------------------

Numerical Precision
~~~~~~~~~~~~~~~~~~~

SPECULA supports both single and double precision floating-point arithmetic:

- **precision = 0**: Double precision (64-bit float, 128-bit complex)
- **precision = 1**: Single precision (32-bit float, 64-bit complex) - **default**

Single precision is faster and uses less memory, while double precision provides higher numerical accuracy. For most AO simulations, single precision is sufficient.

Device Selection
~~~~~~~~~~~~~~~~

- **target_device_idx = -1**: CPU execution (default)
- **target_device_idx = 0, 1, 2, ...**: GPU execution on the specified device

If CuPy is not installed, GPU execution is not available and all computations run on CPU.

Default Values and Behavior
----------------------------

- **zenithAngleInDeg**: If not specified in the ``main`` section, assumed to be 0° (on-axis).
- **pupil geometry**: If not specified, a circular pupil is assumed with diameter defined by ``pixel_pupil``.
- **wavelength**: When not specified for a component, the wavelength from the source is used.
- **fov**: Field of view in arcseconds. If 0 or not specified, a minimal FOV covering the pupil is used.

YAML File Structure
-------------------

Configuration File Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each SPECULA simulation is defined by a YAML configuration file where each top-level section corresponds to a simulation object (atmosphere, source, DM, WFS, etc.).

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Main simulation parameters
   main:
     total_time: 1.0
     time_step: 0.001
     pixel_pupil: 240
     diameter: 8.0

   # Individual components
   source_name:
     class: 'Source'
     parameter1: value1
     parameter2: value2

   component_name:
     class: 'ComponentClass'
     inputs:
       input_name: 'source_name.output'
     parameter: value

Parameter References
~~~~~~~~~~~~~~~~~~~~

Parameters can be referenced between blocks using dot notation:

- ``block_name.parameter_name``: Reference a parameter from another block
- ``block_name.output_name``: Reference an output from another block

Example:

.. code-block:: yaml

   dm:
     class: 'Dm'
     inputs:
       modes: 'zernike.out_modes'    # Reference output from zernike block
       commands: 'controller.out'    # Reference output from controller
     n_modes: 'main.n_modes'          # Reference parameter from main block

Component Inputs and Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most processing objects have:

- **Inputs**: Data consumed by the object (e.g., electric field, slopes, commands). Specified under the ``inputs:`` section.
- **Outputs**: Data produced by the object (e.g., wavefront, PSF, signals)
- **Parameters**: Configuration values that define the object's behavior

Inputs are specified using the ``inputs:`` dictionary with key-value pairs, while parameters are set directly at the component level.

Common Pitfalls
---------------

1. **Seeing wavelength**: Remember that seeing is always at 500 nm. Don't use seeing values measured at other wavelengths without conversion.

2. **Layer heights at zenith**: Don't pre-scale layer heights for the observation zenith angle - SPECULA does this automatically.

3. **Cn² normalization**: The sum of all Cn² fractions must equal exactly 1.0.

4. **Time step vs integration time**: The simulation ``time_step`` sets the temporal resolution, while detector ``dt`` can be larger for integration.

5. **GPU memory**: Large simulations may exceed GPU memory. Monitor memory usage or use CPU if needed.

See Also
--------

- :doc:`simulation_basics`: Introduction to simulation concepts
- :doc:`running_simulations`: How to run simulations
- :doc:`tutorials/scao_basic_tutorial`: Basic tutorial with example YAML file