Simulation Basics
=================

This section covers the fundamental concepts and architecture of SPECULA simulations.

What is SPECULA?
----------------

SPECULA is a comprehensive end-to-end adaptive optics simulator designed for:

* **Ground-based telescopes**: Any size, in particular from 8m class to ELTs (Extremely Large Telescopes)
* **Multiple AO modes**: SCAO, LTAO, MCAO, GLAO
* **Various wavefront sensors**: Shack-Hartmann, Pyramid, LGS systems
* **Realistic atmospheric modeling**: Kolmogorov turbulence, von Karman models, multi-layer atmospheric profiles
* **Performance**: GPU-accelerated computations
* **Calibration procedures**: Interaction matrix generation

SPECULA Architecture
--------------------

SPECULA follows a modular, object-oriented architecture based on three main components:

Processing Objects
~~~~~~~~~~~~~~~~~~

Processing objects perform the main computational tasks:

**Example Processing Objects:**

* ``AtmoPropagation`` - Turbulence propagation
* ``Slopesc`` - Wavefront sensor data processing
* ``ModalRec`` - Slope-to-modes conversion
* ``DM`` - Mirror command application

More information on processing objects can be found in the :doc:`processing_objects` documentation.

Data Objects
~~~~~~~~~~~~

Data objects encapsulate physical quantities and measurements:

**Example Data Objects:**

* ``ElectricField`` - Phase and amplitude information
    * Phase units: nanometers (nm) of optical path difference (wavefront)
* ``Slopes`` - WFS measurements
* ``Intensity`` - Detector images
* ``Intmat`` - Interaction matrices

More information on data objects can be found in the :doc:`data_objects` documentation.

Housekeeping Objects
~~~~~~~~~~~~~~~~~~~~

Housekeeping objects manage simulation state and configuration:

**Example Housekeeping Objects:**
* ``Simul`` - Main simulation controller
* ``LoopControl`` - Controls simulation iterations and time steps
* ``CalibManager`` - Handles data calibration structure
* ``Connections`` - Manages connections between objects

Configuration System
~~~~~~~~~~~~~~~~~~~~~

Simulations are defined through hierarchical YAML configuration files.
See `tutorials/scao_tutorial` for a SCAO system example and the files in the ``config/scao`` directory.

Special YAML Options: ``_data``, ``_object``, and ``_ref``
----------------------------------------------------------

SPECULA supports special configuration options in YAML files to load data from external sources, restore objects from disk, or reference other simulation objects. These options allow flexible initialization of simulation objects.

``<name>_data`` option
~~~~~~~~~~~~~~~~~~~~~~

- Loads a physical quantity (e.g., array, image, mask) from a FITS file.
- The value should be the path to a FITS file (relative to ``root_dir/data``).
- The loaded data is assigned to the parameter ``<name>`` in the object constructor.

.. code-block:: yaml

   pupil_data: "pupil_mask.fits"      # Loads the pupil mask from a FITS file
   atmo_data: "atmo_layers.fits"      # Loads atmospheric layers

``<name>_object`` option
~~~~~~~~~~~~~~~~~~~~~~~~

- Restores a full data object from disk (typically from a FITS file).
- The value should be a tag or filename identifying the object to restore.
- The object class is automatically inferred from the type hint of the corresponding initialization parameter in the Python class definition.
- When restoring an object, SPECULA calls the class's ``restore()`` method, passing the specified tag or filename as an argument.
- The restored object is assigned to the parameter ``<name>`` in the object constructor.

.. code-block:: yaml

   intmat_object: "intmat_tag"        # Restores the interaction matrix object
   slopes_object: "slopes_tag"        # Restores a Slopes data object
   ifunc_object:  "tutorial_ifunc"    # Restores influence functions

``<name>_ref`` option
~~~~~~~~~~~~~~~~~~~~~

- Creates a reference to another object defined in the same YAML file.
- The value should be the name of the target object (without quotes).
- The referenced object is passed directly to the parameter ``<name>`` in the object constructor.
- This is commonly used for:
  
  - Referencing simulation parameters (``simul_params_ref: 'main'``)
  - Sharing configuration objects between multiple components
  - Establishing dependencies between objects

.. code-block:: yaml

   # Common usage: reference to main simulation parameters
   pyramid:
     class: 'ModulatedPyramid'
     simul_params_ref: 'main'         # References the 'main' SimulParams object
     # ... other parameters ...

   # Another example: sharing a calibration manager
   dm:
     class: 'DM'
     calib_manager_ref: 'calib'       # References a CalibManager object
     # ... other parameters ...

``<name>_dict_ref`` option
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Creates a reference to multiple objects defined in the same YAML file.
- The value should be a list of object names.
- A dictionary mapping object names to object references is passed to the parameter ``<name>``.
- Useful when an object needs to access multiple related objects (e.g., multiple sources, multiple DMs).

.. code-block:: yaml

   # Example: propagation with multiple sources
   prop:
     class: 'AtmoPropagation'
     source_dict_ref: ['source_science', 'source_ngs']  # References multiple sources
     # ... other parameters ...

**How** ``_ref`` **Works:**

When SPECULA encounters a ``<name>_ref`` parameter:

1. It strips the ``_ref`` suffix to get the actual parameter name
2. It looks up the referenced object(s) in the current YAML configuration
3. It passes the object reference(s) directly to the constructor

This mechanism ensures proper initialization order: referenced objects are always created before objects that reference them.

Usage Notes
~~~~~~~~~~~

- These options are parsed automatically by the simulation loader (``simul.py``).
- If any ``_ref``, ``_dict_ref``, ``_object``, or ``_data`` value is ``None``, the parameter is set to ``None``.
- The type of restored objects (``_object``) is inferred from the class constructor type hints.
- References (``_ref``) establish a dependency graph that determines the object creation order.
- You can mix ``_data``, ``_object``, ``_ref``, and ``_dict_ref`` options with standard YAML parameters in your configuration files.

.. note::
   - Use ``_data`` for simple arrays/matrices from FITS files
   - Use ``_object`` for complex data objects with methods (e.g., IFunc, M2C, Intmat)
   - Use ``_ref`` for sharing configuration objects between components
   - Use ``_dict_ref`` when an object needs access to multiple related objects

Connection Graph
~~~~~~~~~~~~~~~~

Objects are connected through a directed graph where data flows from outputs to inputs:

.. code-block:: text

   Telescope → AtmosphericLayer → WFS → SlopesComputer → Reconstructor → DM
       ↑                                                                  |
       └─────────────────── Closed Loop ←─────────────────────────────────↓

This creates a flexible, modular system where components can be easily:

* **Replaced** - Swap WFS types without changing other components
* **Reused** - Same atmospheric model for different AO systems  
* **Extended** - Add new processing algorithms seamlessly

Time Management
---------------

SPECULA uses a discrete-time simulation model:

**Synchronous Execution**
   All objects execute in lockstep at each time iteration

**Configurable Time Steps**
   Any range is possible up to 1e-9s

**Temporal Delays**
   Realistic modeling of sensor readout and processing delays

**Frame Rates**
   Support for different subsystem frame rates (e.g., WFS vs NGS)

**Web-based Monitoring:**

SPECULA includes a real-time web-based monitoring system that runs during simulations:

.. code-block:: yaml

   # Enable in your configuration file
   main:
     class:             'SimulParams'
     ...
     display_server:    True                   # Display server on auto-selected port

**Architecture:**
   * **Display Server**: Runs within the simulation process, serves data via websockets
   * **Frontend**: Separate web application (if available) for visualization
   * **Real-time Updates**: Live plotting of data objects during simulation

**Access:**
   * The display server will print its URL when started: ``Display server running at http://localhost:[auto-selected-port]``
   * Frontend connection (if running): ``http://localhost:8080``

**Features:**
   * Real-time plotting of any data object
   * Simulation speed monitoring
   * Interactive data exploration
   * Multi-client support

.. note::
   The web interface is optional. Simulations run normally without it. Enable by adding a ``display_server: True`` object to your main configuration.