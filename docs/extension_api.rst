Extension API
=============

Data objects, processing objects and displays, whether in SPECULA or in another
:doc:`plugin <plugins>`, derive from the iteras base classes. This page lists
what those classes offer to derived classes and what they expect from them:
the methods to override, the methods called by the framework, and the
attributes that cross the boundary between the framework and derived classes.

This is the stable interface between iteras and its plugins. Methods and
attributes not listed here are internal to the framework: derived classes
should not use or override them, since they may change.

The life cycle of processing objects, and how to write them, are described in
detail in :doc:`processing_objects`. Data objects are described in
:doc:`data_objects`.

Common to all objects
---------------------

All objects derive from ``BaseTimeObj``. Their constructors must accept the
``target_device_idx`` and ``precision`` arguments (default ``None``) and pass
them to the base class constructor, which sets the following attributes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Meaning
   * - ``target_device_idx``
     - Device of the object: -1 for the CPU, 0, 1, ... for the GPUs
   * - ``precision``
     - 0 for double precision, 1 for single precision
   * - ``xp``
     - Array module of the device: ``numpy`` or ``cupy``
   * - ``dtype``, ``complex_dtype``
     - Floating point and complex types for the precision and device
   * - ``logger``
     - Logger of the object, named ``<package>.<ClassName>``

These attributes are set by the framework and are **read-only** for derived
classes. Use them, instead of importing ``cp`` or ``xp`` at module level, to
allocate arrays and choose types: ``self.xp.zeros(n, dtype=self.dtype)``.
When ``target_device_idx`` or ``precision`` is ``None``, the values set by the
last call to ``init()`` are used.

Helper methods: ``to_xp(v, dtype=None, force_copy=False)`` converts an array to
the device of the object, ``seconds_to_t()`` and ``t_to_seconds()`` convert
between seconds and the integer time used by the framework.

Some scipy functions are available for the device of the object, from
``scipy`` on the CPU and from ``cupyx.scipy`` on the GPU: ``ndimage_shift``,
``ndimage_rotate``, ``ndimage_center_of_mass``, ``dct`` and ``idct``. They are
read-only attributes, imported the first time they are used, and they follow
the device of the object (for example after ``copyTo()``).

Processing objects
------------------

Processing objects derive from ``BaseProcessingObj``.

Attributes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Written by
     - Meaning and rules
   * - ``inputs``
     - the object, in ``__init__``
     - Dictionary of the inputs, created empty by the base class. Each value is
       an ``InputValue`` or ``InputList`` with the expected data type, and each
       name must match ``input_names()``.
   * - ``outputs``
     - the object, in ``__init__``
     - Dictionary of the outputs, created empty by the base class. Each value is
       a data object, and each name must match ``output_names()``. Outputs are
       created once and then updated in place: other objects keep references to
       them, and CUDA graphs keep the addresses of their arrays.
   * - ``local_inputs``
     - the framework
     - Read-only. The current input data objects, by input name, on the device
       of the object, set before ``prepare_trigger()``. ``None`` for optional
       inputs that are not connected.
   * - ``current_time``, ``current_time_seconds``
     - the framework
     - Read-only. Time of the current step, as an integer and in seconds.
   * - ``name``
     - the framework
     - Read-only. Name of the object in the parameter file.

Methods to override
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``input_names()``, ``output_names()``
     - Class methods returning the declared inputs and outputs, as dictionaries
       of ``InputDesc``/``OutputDesc`` (type and description). Checked against
       ``inputs`` and ``outputs`` after ``setup()``.
   * - ``setup()``
     - Initialization that needs the connected inputs. Must call the base class
       method first. Call ``build_stream()`` here to use a CUDA graph.
   * - ``prepare_trigger(t)``
     - Work that cannot be captured in a CUDA graph (host-side code,
       reallocations), run when the object is ready. Must call the base class
       method first.
   * - ``trigger_code()``
     - The computation of the step, on arrays of ``xp``. When a CUDA graph is
       used, it must only run device code on arrays whose addresses do not
       change, and must not keep state in Python variables.
   * - ``post_trigger()``
     - Work after the computation, in particular setting ``generation_time`` of
       the refreshed outputs. Must call the base class method first.
   * - ``finalize()``
     - Work at the end of the simulation (saving data, closing files...).
   * - ``check_input_names()``, ``check_output_names()``
     - Override to skip or replace the checks against ``input_names()`` and
       ``output_names()``, for objects whose inputs or outputs are created
       dynamically.
   * - ``get_resolved_seed()``
     - Return the random seed used, if chosen at construction, so that a replay
       reproduces the same values. Default: ``None``.
   * - ``graph_input_ptrs()``
     - Override to exclude, from the check made at each launch of the CUDA
       graph, inputs that the graph does not read.
   * - ``can_accelerate()``
     - Called by the :doc:`accelerator <accelerator>` just before it captures
       ``trigger_code()`` in a CUDA graph launched at each step instead. Return
       ``True`` if that graph does the same as ``trigger_code()`` at all the
       following steps: ``trigger_code()`` only launches GPU work, on arrays
       updated in place, with no Python value that changes from one step to the
       next (values that change are written to GPU arrays in
       ``prepare_trigger()``, which runs at each step); matrix products use
       ``specula.lib.graph_matmul.matmul()``. Otherwise return ``False``, or a
       string saying why (it is called again at the next trigger). Default:
       ``True`` for objects that build their own CUDA graph, ``False`` otherwise.
       A derived class that redefines ``trigger_code()`` is not accelerated
       unless it declares ``can_accelerate()`` itself.

Hooks called by Simul
~~~~~~~~~~~~~~~~~~~~~

The default implementations do nothing; override them to change how
``Simul`` treats the objects of a class.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``accepts_input_list()``
     - Class method. Return ``True`` to accept the ``input_list`` entry in the
       inputs of the parameter file: Simul creates one input per item, named
       after its alias (``'alias-obj.output'``). Used by ``DataStore`` and
       ``DataBuffer``.
   * - ``runs_on_cpu_only()``
     - Class method. Return ``True`` for classes that must run on the CPU
       (writing files, interacting with the user): the :doc:`allocator <allocator>`
       never places them on a GPU. Used by ``DataPrint``,
       ``DisplayServer`` and ``InteractiveInput``.
   * - ``runs_on_main_process_only()``
     - Class method. Return ``True`` for classes that must run in the main
       process (MPI rank 0), like ``DisplayServer`` and ``InteractiveInput``.
   * - ``devices_used(params)``
     - Class method. Number of devices used by an object built with *params*
       (1 by default): the device ``target_device_idx`` and the following ones,
       for objects that split their work, like ``DistributedSH``. The allocator
       divides their memory and GPU time among those devices.
   * - ``single_device_params(params)``
     - Class method. Parameters of an equivalent object on a single device, used
       by the allocator to measure the objects whose ``devices_used()`` is more
       than 1.
   * - ``is_schedule_barrier()``
     - Class method. Return ``True`` for classes whose position in the trigger
       order must never change, because they interact with the outside world
       or modify shared state (``DisplayServer``, ``InteractiveInput``).
   * - ``input_list_created()``
     - Called after the inputs of ``input_list`` have been created.
   * - ``set_simulation_params(params)``
     - Called after the object is built, with the parameters of the whole
       simulation.
   * - ``set_replay_params(replay_params)``
     - Called before the simulation starts, with the parameters of a
       simulation that would replay it from the stored data.

Methods called by the framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``check_ready()``, ``trigger()``, ``send_outputs()`` and the other methods
used by the loop are called by the framework and should not be overridden.
Two methods can be called by derived classes to use CUDA graphs:
``build_stream()``, normally in ``setup()``, and ``invalidate_graph()``, when
something frozen in the graph at capture time changes, so that the graph is
captured again at the next step.

Data objects
------------

Data objects derive from ``BaseDataObj``.

Attributes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Written by
     - Meaning and rules
   * - ``generation_time``
     - the processing object producing it
     - Time at which the data was last refreshed (-1 at creation). The producer
       sets it to ``current_time`` whenever it refreshes the output, normally in
       ``post_trigger()``: the objects reading the output are triggered only in
       the steps in which it is refreshed.
   * - ``tag``
     - the framework
     - Read-only. Calibration tag the object was restored from, if any.

Methods to implement
~~~~~~~~~~~~~~~~~~~~

These methods are used by the framework to move, store, restore and show the
data. A data type that does not implement one of them cannot be used with the
corresponding feature.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Used by
   * - ``get_value()``, ``set_value(v)``
     - The main array of the data. Used by ``DataStore`` (storage),
       ``DataSource`` (replay) and by MPI transfers after the first one.
       ``set_value()`` updates the existing array in place.
   * - ``get_fits_header()``, ``from_header(hdr, target_device_idx=None)``
     - Description of the data, stored with it by ``DataStore`` (``OBJ_TYPE``
       must be the class name), and creation of an empty object from it by
       ``DataSource``.
   * - ``save(filename)``, ``restore(filename, target_device_idx=None)``
     - Saving and loading: ``restore()`` is used for the ``_object`` and
       ``tag`` parameters, through the calibration manager. Implement them
       with ``iteras.lib.storage.write_object()`` and ``read_object()``, which
       store the header keywords and named arrays in any of the supported
       formats (FITS, npz, npy, HDF5), chosen by the extension of
       ``filename``. ``read_object().load()`` puts an array on the device of
       the object, streaming large FITS and npy arrays to the GPU.
   * - ``array_for_display()``
     - The array shown by the generic displays and the display server.
   * - ``debug_summary()``
     - What to show in the MPI debug log of the inputs. Default: the object.

All the arrays of a data object are copied by ``transferDataTo()`` and
``copyTo()``, which the framework uses to move data between devices: derived
classes do not need to implement them.

Class hooks
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``track_consumers_in_replay()``
     - Class method. Return ``True`` for data types whose producers must not be
       silently left out of a targeted replay (``ElectricField`` in SPECULA):
       ``Simul.build_targeted_replay()`` reports them. Default: ``False``.

Displays
--------

Displays derive from ``BaseDisplay``, a processing object.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``can_skip_updates()``
     - With ``--async-displays``, return ``True`` (the default) if updates can
       be dropped when the display process is busy, ``False`` for displays that
       accumulate a history and must receive every update (see
       :doc:`simul_displays`).

Main parameters
---------------

The main parameters of a simulation are the section whose class is
``SimParams`` or a subclass of it. A plugin can define a subclass to add
domain-specific parameters, as SPECULA does with ``SimulParams``.
