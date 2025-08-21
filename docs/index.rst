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
   Running simulations <running_simulations>
   Diagrams <simul_diagrams>
   Calibration manager <calibration_manager>

Tutorials
---------
.. toctree::
   :maxdepth: 2

   Basic SCAO simulation tutorial <tutorials/scao_basic_tutorial>
   SCAO simulation tutorial <tutorials/scao_tutorial>
   Deformable Mirror actuator step response tutorial <tutorials/step_response_tutorial>
   Field Analyser (deferred computation of PSF, modal analysis and phase cube) tutorial <tutorials/field_analyser_tutorial>

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

Detailed API documentation:

.. toctree::
   :maxdepth: 1

   Base Classes API <api/base_classes>
   Processing Objects API <api/processing_objects>
   Data Objects API <api/data_objects>
   Utility Functions API <api/lib>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
