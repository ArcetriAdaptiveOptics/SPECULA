

Guidelines for processing objects
=================================


Class derivation
----------------

All processing objects derive from the :py:class:`~specula.base_processing_obj.BaseProcessingObj` class.

Life cycle
----------

A processing object life is divided into several discrete steps

* Initialization
* Input/output connection
* Setup
* Loop (repeated N times):

  * check_ready(), which calls prepare_trigger() if the object is ready
  * trigger(), which runs trigger_code() directly or as a captured CUDA graph
  * post_trigger()

* Finalize
 

Initialization
--------------

The *__init__* method has several tasks to do, each explained in more detail below. They are:

* receive the standard parameters *target_device_idx* and *precision* in addition to the object-specific ones
* call the base class *__init__* method with the standard parameters
* initialize the object as far as possible, allocating arrays with *self.xp*
* define the class inputs and outputs

Init parameters
***************

The *__init__* parameters should configure the object as much as possible. Default values are allowed. Use type hints whenever possible, either Python built-in types or SPECULA data objects. Parameters (of any type) that can be omitted completely should be initialized with None.

All processing objects must accept the *target_device_idx* and *precision* arguments, initialized to default values of None, and call the superclass *__init__* method with those two arguments::

    from specula.base_processing_obj import BaseProcessingObj

    class ExampleProcessingObj(BaseProcessingObj):
        '''Example processing object'''

        def __init__(self,
                     foo: int,
                     bar: str=None,
                     target_device_idx=None,
                     precision=None
                ):
            super().__init__(target_device_idx=target_device_idx, precision=precision)


*target_device_idx* has the following standard values:

* -1: CPU
* 0: first GPU
* 1: second GPU
* ...

Array allocation
****************

Any numpy-like array should be allocated using *self.xp* instead of *np*. In this way, they will be automatically allocated on the correct GPU or on CPU memory depending on the current target. The dtypes to be used are *self.dtype* for float, and *self.complex_dtype* for complex numbers. These dtypes are automatically set based on the *precision* parameter. For example::

        self.buffer = self.xp.zeros((100, 100), dtype=self.dtype)

If arrays are very small and are not intended to be loaded on a GPU, even if present, they can be allocated with the *np* module as usual.


Inputs
******

Each input is an entry on the *self.inputs* dictionary, and is an instance of the :py:class:`~specula.connections.InputValue` class, initialized with the expected input type. It is possible for an input to be a list of values of the same type, in this case use an instance of :py:class:`~specula.connections.InputList`::

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['in_slopes_list'] = InputList(type=Slopes)

If the input is optional, add *optional=True* (by default, all inputs are mandatory)::

        self.inputs['in_slopes'] = InputValue(type=Slopes, optional=True)

An optional input that has not been connected in the configuration file has a value of None
in *self.local_inputs* (see :ref:`prepare_trigger() <prepare_trigger>` below).

Inputs must be created in the constructor, because connections are made before *setup()* is called.
An input can be created conditionally, depending on the constructor parameters
(for example, the sodium profile inputs of :py:class:`~specula.processing_objects.sh.SH`).
In that case its declaration must be marked as optional, see :ref:`io_declaration` below.

Outputs
*******

Outputs are instances of SPECULA data objects that must be allocated exactly once in the object constructor, and then never reassigned: rather, change their contents to refresh the output. The *self.outputs* dictionary must contain a reference to each output::

        self.modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self.outputs['out_modes'] = self.modes

The output type is the type of the data object itself, so no type needs to be specified
when adding an entry to *self.outputs*.


.. _io_declaration:

Declaring inputs and outputs
****************************

In addition to filling *self.inputs* and *self.outputs*, a processing object declares its interface
with the two class methods ``input_names()`` and ``output_names()``. Each one returns a dictionary
that maps a name to an ``InputDesc(type, description)`` or ``OutputDesc(type, description)``
named tuple, both defined in :py:mod:`specula.base_processing_obj`::

    from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc

    class ExampleProcessingObj(BaseProcessingObj):

        @classmethod
        def input_names(cls):
            return {
                'in_slopes': InputDesc(Slopes, 'Input slope vector'),
                'gain_mod': InputDesc(BaseValue, 'Gain modulation vector (optional)'),
            }

        @classmethod
        def output_names(cls):
            return {'out_modes': OutputDesc(BaseValue, 'Reconstructed modal vector')}

The type is the data object class (for an :py:class:`~specula.connections.InputList`,
the type of the list elements). The description is a short human-readable text.

A derived class extends the declarations of its base class by updating the dictionary
returned by the base class method::

        @classmethod
        def input_names(cls):
            inputs = super().input_names()
            inputs.update({'in_commands': InputDesc(BaseValue, 'Command vector (optional)')})
            return inputs

The declarations are also used when building this documentation: the names of the declared inputs and outputs
appear in the :doc:`api/processing_objects_summary` table, with optional inputs marked as *(opt)*.
Types and descriptions are not shown there.

Validation
++++++++++

The declarations are checked against the actual *self.inputs* and *self.outputs* dictionaries by
``sanity_check()``, which the framework calls for each object right after its *setup()* method.
The check raises an exception if:

* an input or output present in the dictionaries is not declared;
* a declared output is not present in *self.outputs*;
* a declared input is not present in *self.inputs*, unless its declaration is optional (see below);
* the input type is not exactly the declared type (a subclass is rejected),
  or the input is not an :py:class:`~specula.connections.InputValue`
  or an :py:class:`~specula.connections.InputList`;
* the output is not an instance of the declared type (a subclass is accepted).

If a class does not define ``input_names()`` or ``output_names()``, the corresponding check is skipped.
Every new processing object should define both.

Optional declarations
+++++++++++++++++++++

An input declaration is optional if its description ends with ``(optional)``.
This is a different concept from the *optional=True* flag of an input:

* *optional=True* in :py:class:`~specula.connections.InputValue`: the input always exists,
  but it may be left unconnected in the configuration file. This is checked by *setup()*.
* ``(optional)`` at the end of the description: the input may be absent from *self.inputs*,
  because it is created only for some constructor parameters.
  This is checked by ``sanity_check()``.

An input that is created conditionally must therefore have an ``(optional)`` declaration.
Output declarations have no optional form: each declared output must be present.

Name patterns
+++++++++++++

Declared names can be:

* exact names, for example ``out_ef``;
* placeholder names, for example:

  * ``out_modes_{sensor_idx}`` in ``ModalrecMultirate.output_names()``
  * ``out_{source_name_}layer`` and ``out_{source_name_}ef`` in ``AtmoRandomPhase.output_names()``

Each segment delimited by ``{...}`` is replaced by a ``*`` wildcard and the actual name is matched
with :py:func:`fnmatch.fnmatchcase`. The wildcard also matches an empty string, so
``out_{source_name_}ef`` matches both ``out_on_axis_ef`` and ``out_ef``.
When an actual name matches both an exact declaration and a pattern, the exact declaration is used.

Raw ``*`` patterns are still accepted by the matcher for backward compatibility,
but new declarations must use the ``{...}`` placeholder form.

Custom checks
+++++++++++++

When the inputs or outputs are truly dynamic and cannot be described by a stable name pattern and type,
a class can override ``check_input_names()`` or ``check_output_names()``, as done for example by
:py:class:`~specula.processing_objects.data_store.DataStore`
and :py:class:`~specula.processing_objects.data_source.DataSource`.


Input/Output connections
------------------------

This step is performed automatically by the framework after all objects have been initialized.
Inputs and outputs are connected based on the configuration file directives.
For each connection, the output must be an instance of the type specified in the input definition
(a subclass is accepted). If it is not, an exception will be raised.
For connections between MPI processes this check is not performed.


Setup
-----

The *setup()* method is called after all connections have been completed but before starting the simulation,
and it is intended for later initialization that needs information from the connected inputs, or from some
other global simulation parameter. The method signature is::

    def setup(self):

The default implementation checks that all non-optional inputs have been set, and selects the correct GPU if needed,
so that the derived class' code runs on the correct target. A class that reimplements this method *must* call the base class one::

    def setup(self):
        super().setup()
        [... additional setup as needed ...]

An important task of the *setup()* method is to call the *build_stream()* method to enable CUDA graph capturing
of the trigger code described below::

    def setup(self):
        super().setup()
        self.build_stream()


Trigger
-------

Trigger order
*************

The order in which instances are triggered is computed once, before the simulation starts,
from the input/output connections in the configuration file. The algorithm is:

#. Find all objects without inputs (leaves) and assign them to the current group
#. Remove these objects, and all the connections to their outputs, from the dependency graph.
   Objects whose inputs were all connected to the removed objects become leaves
#. Repeat from step 1 with the next group, until no objects are left

A closed loop is a cycle in the connections, and a cycle has no valid trigger order.
The cycle is broken by a delayed connection (for example ``'dm.out_layer:-1'``), which refers to the output
of the previous iteration and is ignored when looking for leaves. With at least one delayed connection
in each loop, the dependency graph becomes acyclic and the trigger order can be computed.
If a cycle without delayed connections is present, its objects (and all the objects depending on them)
never become leaves: they are never triggered, and only a warning listing them is logged.
An exception is raised instead when the remaining leaves all produce outputs that are read with a delay
by each other (for example, two objects each reading the other's output with ``:-1``),
because no order satisfies all the delays.

The one-step delay is obtained from the trigger order, not from a copy of the data: the input refers to the same
data object as the output. The object producing a delayed output is therefore placed after all the objects that
read it, so that they read the value of the previous iteration before it is overwritten.
As a consequence, only one-step delays are supported: delays other than ``0`` and ``-1`` (e.g. ``:-2``) raise an error.
Longer delays must be implemented inside the processing objects (for example with the *delay* parameter of the filters).

Objects in the same group can be triggered at the same time.

Readiness check
***************

The trigger order is fixed, but at each iteration an object is actually triggered only if it is ready.
The readiness check is implemented in the *check_ready()* method of the base class, and there is usually
no need to override it. An object is ready if:

* it has no inputs, or
* all its inputs are optional and none of them has been connected, or
* at least one input has been refreshed in the current iteration,
  that is its *generation_time* is equal to the current time.

This allows part of the simulation to run at a slower rate than the rest. For example, an object
simulating a CCD might integrate data for many loop iterations before producing its output, in order
to simulate a long integration time. All objects depending on this output will automatically
be triggered at the slower rate.

If the object is ready, *check_ready()* also calls *prepare_trigger()*.

Trigger process
***************

For each group identified by the trigger order algorithm:

#. Call *check_ready()* for all objects, which in turn calls *prepare_trigger()* for the ready ones
#. Call *trigger()* for all ready objects, which executes *trigger_code()* directly, or launches its CUDA graph if one was captured
#. Call *post_trigger()* for all ready objects

The general idea is to have a GPU-friendly algorithm in *trigger_code()*, that operates on statically-allocated
arrays. This algorithm can be captured in a CUDA graph and executed on a private CUDA stream, which is both
be more efficient than a series of Python/CuPY operations and also allows multiple objects to be run in parallel
on the same or different GPUs. *prepare_trigger()* and *post_trigger()* take care of operations
that cannot be captured in CUDA graph, in particular:

* *prepare_trigger()*: perform any needed setup, for example CPU-only numpy calculations
* *post_trigger()*: as a minimum, set the *generation_time* attribute of any output arrays.

The three methods have a very simple signature: only *prepare_trigger()* takes a single argument,
the current simulated time *t*. The base class method must be called as well, except for *trigger_code()*::

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

    def trigger_code(self):
        pass

    def post_trigger(self):
        super().post_trigger()

By default, all streams in an object group are executed in parallel. If an object wishes to turn off parallelization,
it can call *build_stream()* setting the optional *allow_parallel* parameter to False::

    def setup(self):
        super().setup()
        self.build_stream(allow_parallel=False)

In this case, the trigger graph will be run on a default stream that serializes all such graphs. It is still
possible to parallelize object instances across multiple GPUs, by explicitly setting their *target_device_idx*
to a specific GPU in their initialization.

.. _prepare_trigger:

prepare_trigger():
++++++++++++++++++

The base class implementation takes care of transferring any needed data to and from CPU and GPUs,
in case objects have been allocated to different targets. After being transferred for the first
time, data objects are not reallocated: their contents are refilled each time.

The transferred inputs are available in the *self.local_inputs* dictionary, from which
they can be copied into pre-allocated static arrays::

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.data[:] = self.local_inputs['data'].value


trigger_code()
++++++++++++++

The base class implementation does nothing, and there is no need to call it.

Any code implemented by derived classes must:

#. only perform GPU operations using the xp module on arrays allocated with self.xp
#. avoid any explicity numpy or normal python operation.
#. NOT use any value in variables that are reallocated by prepare_trigger() or post_trigger(), and in general avoid any value defined outside this class (like object inputs)

because if stream capture is used, a CUDA graph will be generated that will skip
over any non-GPU operation and re-use GPU memory addresses of its first run.

Defining local variables inside this function is OK, they will persist in GPU memory.

post_trigger()
++++++++++++++

The base class implementation synchronizes any previous CUDA stream, if active.

Derived classes will use this method to set the *generation_time* attributes of any output
(because it is a task that cannot be capture by the CUDA graph in *trigger_code()*) and
any other custom cleanup tasks.

Non-parallelizable code
+++++++++++++++++++++++

If the trigger code is known to be non-parallelizable, or numpy-only code is used,
it is possible to avoid the previous complexity and:

* Put all code into *trigger_code()*
* Avoid calling *build_stream()* during setup

This way, no CUDA graph will be generated and *trigger_code()* will be executed
as ordinary Python code.

