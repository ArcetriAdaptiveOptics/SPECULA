Simulation Basics
=================

This section covers the fundamental concepts and architecture of SPECULA simulations.

What is SPECULA?
----------------

SPECULA is a comprehensive end-to-end adaptive optics simulator designed for:

* **Ground-based telescopes**: From 8m class to ELTs (Extremely Large Telescopes)
* **Multiple AO modes**: SCAO, LTAO, MCAO, GLAO
* **Various wavefront sensors**: Shack-Hartmann, Pyramid, LGS systems
* **Performance**: GPU-accelerated computations
* **Calibration procedures**: Automated interaction matrix generation

Key Features
~~~~~~~~~~~~

**Physical Modeling**
   * Kolmogorov and von Karman turbulence models
   * Multi-layer atmospheric profiles
   * Realistic telescope pupil geometry
   * Optical aberrations and misalignments

**Sensor Models**
   * Photon and read noise
   * Pixel response variations
   * Optical features
   * Temporal behavior

**Control Systems**
   * Multiple control algorithms (integrator, IIR filters, ...)
   * Temporal delays and bandwidth limitations
   * Multi-conjugate control strategies

SPECULA Architecture
--------------------

SPECULA follows a modular, object-oriented architecture based on three main components:

Processing Objects
~~~~~~~~~~~~~~~~~~

Processing objects perform the main computational tasks:

**Common Processing Objects:**

* ``AtmoPropagation`` - Turbulence propagation
* ``Slopesc`` - Wavefront sensor data processing
* ``ModalRec`` - Slope-to-modes conversion
* ``DM`` - Mirror command application

Data Objects
~~~~~~~~~~~~~

Data objects encapsulate physical quantities and measurements:

**Key Data Objects:**

* ``ElectricField`` - Phase and amplitude information
* ``Slopes`` - WFS measurements
* ``Intensity`` - Detector images
* ``Intmat`` - Interaction matrices

Configuration System
~~~~~~~~~~~~~~~~~~~~~

Simulations are defined through hierarchical YAML configuration files:

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

.. seealso::
   
   :doc:`running_simulations` for execution details
   :doc:`configuration` for YAML syntax reference
   :doc:`processing_objects` for object development guidelines
