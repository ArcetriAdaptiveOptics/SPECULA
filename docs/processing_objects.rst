

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
  * check_ready()
  * prepare_trigger()
  * trigger_code()
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

Calling the .get() method of an optional input may return None if it has not been set.

Outputs
*******

Outputs are instances of SPECULA data objects that must be allocated exactly once in the object constructor, and then never reassigned: rather, change their contents to refresh the output. The *self.outputs* dictionary must contain a reference to each output::

        self.modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self.outputs['out_modes'] = self.modes

The output type is inferred automatically.


Input/Output connections
------------------------

This step is performed automatically by the framework after all objects have been initialized.
Inputs and outputs are connected based on the configuration file directives.
For each connection, the output type must match what is specified in the input type definition as shown above.
If the types do not match, an exception will be raised.


Setup
-----

The *setup()* method is called after all connections have been completed but before starting the simulation,
and it is intended for later initialization that needs information from the connected inputs, or from some
other global simulation parameter. The method signature is::

    def setup(self, loop_dt, loop_niters):

where *loop_dt* is the simulation time step (in units of *self._time_resolution*), and *loop_niters* is the foreseen
total number of simulation iterations.

The default implementation checks that all non-optional inputs have been set, and selects the correct GPU if needed,
so that the derived class' code runs on the correct target. A class that reimplements this method *must* call the base class one::

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        [... additional setup as needed ...]

An important task of the *setup()* method is to call the *build_stream()* method to enable CUDA graph capturing
of the trigger code described below::

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        self.build_stream()


Trigger
-------

Trigger order
*************

The order in which instances will be triggered is automatically inferred from the input/output connections. The algorithm is:

#. First, all instances without inputs are triggered
#. Then, all instances for which the objects in the previous step were the sole input
#. Then, all instances for which all inputs have been fullfilled in the previous step

The last step is repeated until no new inputs have been set. Therefore, it is possible for some instances not to be triggered
if an object in a previous step has not produced an output. This is expected behaviour: it allows to have part of the simulation
to run at a slower rate than the rest. For example, a object simulating a CCD might integrate data for many loop iterations
before producing its output, in order to simulate a long integration time. All objects depending on this output will automatically
be triggered at the slower rate.

Readyness check
***************

Before triggering each object, its inputs are checked. The object is triggered only if at least one input
has been refreshed since the last trigger, or if the object has no inputs.

The readiness check is implemented in the *check_ready()* of the base class, and there is usually
no need to override it.

Trigger process
***************

The trigger order algorithm identifies groups of object that can be triggered at the same time. For each group:

#. Call *prepare_trigger()* for all objects
#. Call *trigger_code()* for all objects
#. Call *post_trigger()* for all objects

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

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        self.build_stream(allow_parallel=False)

In this case, the trigger graph will be run on a default stream that serializes all such graphs. It is still
possible to parallelize object instances across multiple GPUs, by explicitly setting their *target_device_idx*
to a specific GPU in their initialization.

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

This method has no base class implementation.

Any code implemented by derived classes must:

#. only perform GPU operations using the xp module
   on arrays allocated with self.xp
#. avoid any explicity numpy or normal python operation.
#. NOT use any value in variables that are reallocated by prepare_trigger() or post_trigger(),
  and in general avoid any value defined outside this class (like object inputs)

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

