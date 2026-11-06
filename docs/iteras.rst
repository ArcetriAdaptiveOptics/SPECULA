.. _iteras:

The iteras framework
====================

SPECULA is built on ``iteras``, a simulation framework that knows nothing
about adaptive optics. It is in the same repository and is installed with
SPECULA. iteras provides:

- the base classes of the objects (``BaseDataObj``, ``BaseProcessingObj``,
  ``BaseValue``, ``BaseDisplay``...) and their connections (inputs and outputs);
- the simulation (``Simul``): parameter files, building and connecting the
  objects, the loop (``LoopControl``), MPI support;
- generic objects: filters and integrators, generators, operations on values,
  data storage (``DataStore``, ``DataSource``), displays;
- the :doc:`accelerator <accelerator>` (CUDA graphs) and the
  :doc:`allocator <allocator>` (placement of the objects on nodes, processes
  and GPUs);
- the :doc:`plugin <plugins>` registry, which SPECULA uses to add its AO
  objects.

A simulation without AO objects (for example the step response and control
stability tutorials) runs with the ``iteras`` command as well as with
``specula``.

The modules moved from SPECULA to iteras are still importable with their
SPECULA names (for example ``specula.lib.interp2d`` is ``iteras.lib.interp2d``),
and the classes of the generic objects can be named in the parameter files as
before. Their API is documented only once, in the iteras pages:

- :doc:`api/iteras_core`: base classes, simulation, connections, accelerator,
  tracing, registry;
- :doc:`api/iteras_processing_objects_summary` and
  :doc:`api/iteras_processing_objects`: generic processing objects;
- :doc:`api/iteras_data_objects_summary` and
  :doc:`api/iteras_data_objects`: generic data objects;
- :doc:`api/iteras_lib`: generic utility functions;
- :doc:`api/iteras_allocator` and :doc:`api/iteras_display`.

How to write objects for iteras and SPECULA is described in the
:doc:`extension_api` and in the Developer Guide (:doc:`processing_objects`,
:doc:`data_objects`, :doc:`base_classes`).
