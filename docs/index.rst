.. SPECULA documentation master file, created by
   sphinx-quickstart on Tue Jan  7 15:26:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPECULA documentation
=====================

:Release: |release|
:Date: |today|

SPECULA is an adaptive optics end-to-end simulator.

SPECULA documentation
=====================

Getting Started
---------------
.. toctree::
   :maxdepth: 2

   Installation <installation>
   Simulation basics <simulation_basics>
   Simulation parameters <simulation_parameters>
   Running simulations <running_simulations>
   Profiling and timing <profiling>
   Displays <simul_displays>
   Diagrams <simul_diagrams>
   Calibration manager <calibration_manager>

Tutorials
---------
.. toctree::
   :maxdepth: 1

   tutorials/index

Developer Guide
---------------
.. toctree::
   :maxdepth: 1

   Development conventions <development>
   Processing objects <processing_objects>
   Data objects <data_objects>
   Base classes <base_classes>

Reference
---------

Detailed API documentation of SPECULA:

.. toctree::
   :maxdepth: 1

   Core Modules API <api/core>
   Processing Objects Summary <api/processing_objects_summary>
   Processing Objects API <api/processing_objects>
   Data Objects Summary <api/data_objects_summary>
   Data Objects API <api/data_objects>
   Utility Functions API <api/lib>

The iteras framework
--------------------

SPECULA is built on iteras, a generic simulation framework: base classes,
simulation loop, generic objects (filters, generators, data storage...),
accelerator, allocator, displays and plugins.

.. toctree::
   :maxdepth: 1

   Overview <iteras>
   Accelerating objects (CUDA graphs) <accelerator>
   Automatic allocation <allocator>
   Plugins <plugins>
   Extension API <extension_api>

.. toctree::
   :maxdepth: 1
   :caption: iteras API

   Core API <api/iteras_core>
   Processing Objects Summary <api/iteras_processing_objects_summary>
   Processing Objects API <api/iteras_processing_objects>
   Data Objects Summary <api/iteras_data_objects_summary>
   Data Objects API <api/iteras_data_objects>
   Utility Functions API <api/iteras_lib>
   Allocator API <api/iteras_allocator>
   Displays API <api/iteras_display>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
