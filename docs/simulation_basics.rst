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