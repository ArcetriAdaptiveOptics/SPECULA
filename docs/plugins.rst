Plugins
=======

The simulation framework used by SPECULA (base classes, connections, loop
control, MPI support, data storage, displays) lives in a separate package,
``iteras``, which knows nothing about adaptive optics. SPECULA is an iteras
*plugin*: a package that adds data objects, processing objects and displays
for a specific domain. Other packages can be plugins too, of iteras directly
or of another plugin.

This page describes how plugins are registered, how the classes named in a
parameter file are found, and how this relates to the ``add_modules``
parameter.

What a plugin is
----------------

A plugin is an ordinary Python package whose classes derive from the iteras
base classes (``BaseDataObj``, ``BaseProcessingObj``, ``BaseDisplay``, ...),
directly or through another plugin. The only convention to follow is the one
already used in SPECULA: class ``FooBar`` is defined in a module called
``foo_bar.py``, inside one of the packages that the plugin registers. This is
how a ``class: FooBar`` entry in a parameter file is resolved.

Plugins can be stacked, since a plugin is just code that imports another
package. For example::

    iteras        framework: base classes, Simul, LoopControl, DataStore, displays, ...
      specula     AO plugin: ElectricField, Pixels, IFunc, DM, PyrSlopec, SimulParams, ...
        speculab  laboratory plugin: PlicoDM, PysilicoCamera, ... (uses Pixels, IFunc, ...)

The registry
------------

The module ``iteras.registry`` holds the information provided by the plugins.
It is filled when the plugins are imported, and is shared by the whole
process.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Entry
     - Filled by
     - Used by
   * - Search packages
     - the iteras packages first, then the packages of each plugin, in
       registration order
     - ``iteras.lib.utils.import_class()``, which resolves the class names of
       the parameter files, the object types stored by ``DataStore`` (read back
       by ``DataSource``) and the main parameters class
   * - Plugin roots
     - ``register_plugin(root, ...)``
     - the GPU memory accounting, which follows the objects of these packages
   * - Calibration subdirectories
     - ``register_plugin(calib_subdirs=...)``
     - ``CalibManager`` (e.g. ``IFunc`` objects are stored in ``ifunc/``),
       see :doc:`calibration_manager`
   * - Environment variable aliases
     - ``register_plugin(env_aliases=...)``
     - ``iteras.registry.getenv()``: for example ``SPECULA_DISABLE_GPU`` is an
       alias of ``ITERAS_DISABLE_GPU``, and ``SPECULA_PORT`` of ``ITERAS_PORT``

How a plugin is registered
--------------------------

A plugin registers itself when its top-level package is imported, calling
``iteras.registry.register_plugin()``. This is what SPECULA does in
``specula/__init__.py``:

.. code-block:: python

    from iteras import registry

    registry.register_plugin(
        'specula',
        packages=['specula.processing_objects',
                  'specula.data_objects',
                  'specula.display'],
        calib_subdirs=CALIB_SUBDIRS,
        env_aliases={'ITERAS_DISABLE_GPU': 'SPECULA_DISABLE_GPU',
                     'ITERAS_PORT': 'SPECULA_PORT'},
    )

Any program that imports the plugin, or any of its modules, therefore has it
registered before a parameter file is read. This is the case of the ``specula``
command, and of any script that calls ``specula.init()``.

A plugin can also declare an ``iteras.plugins`` entry point in its package
metadata. The plugins declared in this way are imported by the ``iteras``
command before building the simulation, and by any program the first time a
class is not found (see below), so that their parameter files can be run
without importing them explicitly. SPECULA declares it in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."iteras.plugins"]
    specula = "specula"

Entry points are part of the installed package metadata: after adding one,
the package must be installed again (``pip install -e .`` for a development
installation).

How classes are found
---------------------

When ``Simul`` builds a simulation, ``iteras.lib.utils.import_class()`` converts
each class name to a module name (``FooBar`` to ``foo_bar``) and looks for that
module in all the following packages:

1. the iteras packages: ``iteras.processing_objects``, ``iteras.data_objects``,
   ``iteras.display``;
2. the packages of the registered plugins;
3. the packages listed in the ``add_modules`` parameter of the main parameters.

If the module is found in a single package, or always leads to the same class
(as for the SPECULA modules that are aliases of iteras modules), that class is
used. If different classes with the same name are found in more than one
package, an ``AmbiguousClassError`` is raised, listing them: no class silently
hides another one, whatever the order in which the plugins are imported.

The class name can be qualified, to choose one of several classes with the same
name, or simply to be explicit:

- ``<prefix>.FooBar``, where ``<prefix>`` is a plugin or one of the searched
  packages: the class is looked for only in the packages equal to ``<prefix>``
  or contained in it. For example ``class: specula.IirFilter`` searches only
  the SPECULA packages;
- otherwise, the full path of the module defining the class, for example
  ``class: mypackage.mymodule.FooBar``.

.. code-block:: yaml

    rec:
      class: 'specula.Modalrec'
      ...

If a class is not found, the plugins declared with an ``iteras.plugins`` entry
point are imported (once per process, and only if needed), and the search is
repeated. Installed plugins are therefore found by any program, also when they
were not imported explicitly.

The main parameters are the section whose class is ``SimParams`` (defined by
iteras), or a subclass of it, like SPECULA's ``SimulParams``. ``Simul`` finds it
by resolving the class of each section, looking also in the ``add_modules``
packages listed in any section, so that the main parameters class itself can
come from one of them.

The add_modules parameter
-------------------------

``add_modules`` is a parameter of ``SimParams`` (and therefore of
``SimulParams``) listing packages to search, after the registered ones, when
building that simulation:

.. code-block:: yaml

    main:
      class:        'SimulParams'
      root_dir:     './calib'
      add_modules:  ['my_lab_objects']

``add_modules`` and plugin registration can be used together, and differ in
scope:

- ``add_modules`` only extends the class search of the simulation that lists
  it. It does not register anything, so it cannot add calibration
  subdirectories or environment variable aliases. It suits objects that are
  local to a project, or experimental.
- Registration is permanent for the process and applies to every simulation:
  it suits a package installed and used across many simulations, and it is the
  only way to add calibration subdirectories and environment variable aliases.

Listing a registered package in ``add_modules`` too is harmless. A package
listed in ``add_modules`` that cannot be found is an error.

Logger names
------------

Each object logs under the top-level package of the module defining its class,
followed by the class name: ``iteras.DataStore``, ``specula.DM``, or
``speculab.PlicoDM`` for a class found in the ``speculab`` package, whether
it is a registered plugin or listed in ``add_modules``. The framework modules
log under ``iteras`` (e.g. ``iteras.simul``).

Writing a plugin
----------------

As an example, SpecuLab (``speculab``) provides processing objects that drive
laboratory hardware, and loads them with ``add_modules: ['speculab']``. This
keeps working unchanged. Since its classes use SPECULA data objects (``Pixels``,
``ElectricField``, ``IFunc``), it would be a plugin of SPECULA rather than of
iteras, and would need two additions to become one.

In ``speculab/__init__.py``:

.. code-block:: python

    import specula                  # SPECULA is registered first
    from iteras import registry

    registry.register_plugin('speculab', packages=['speculab'])

In ``setup.py`` (or ``pyproject.toml``):

.. code-block:: python

    entry_points={'iteras.plugins': ['speculab = speculab']},

After this, ``add_modules: ['speculab']`` is no longer needed once SpecuLab is
installed: its classes are found through the entry point, whatever program
runs the simulation.

For a new plugin, the steps are:

1. create a package, with the classes in modules named after them (one class
   per module, ``FooBar`` in ``foo_bar.py``), possibly grouped in subpackages
   like ``processing_objects`` and ``data_objects``;
2. derive the classes from the iteras base classes, or from the classes of the
   plugin they extend, following the :doc:`extension_api`. Use the ``xp``, ``dtype`` and ``target_device_idx``
   attributes of the objects rather than importing ``cp`` or ``xp`` from iteras
   (or from SPECULA) at module level: the iteras modules read these settings
   when objects are created, so ``init()`` may be called after importing them,
   but a module-level import copies the value present at import time;
3. call ``iteras.registry.register_plugin()`` in the top-level ``__init__.py``,
   listing the packages that contain the classes;
4. optionally, declare an ``iteras.plugins`` entry point;
5. optionally, define a subclass of ``SimParams`` to add domain-specific main
   parameters.

Compatibility with older SPECULA code
-------------------------------------

The framework modules moved from SPECULA to iteras are still available from
their old paths: for example ``specula.base_processing_obj`` is the same module
object as ``iteras.base_processing_obj``, so its classes are the same objects
and ``isinstance()`` checks, ``unittest.mock.patch()`` targets and pickled
objects referring to the old paths keep working. The values set by
``specula.init()`` (``cp``, ``xp``, ``global_precision``, ...) are read from
iteras when accessed through ``specula``. Plugins written for SPECULA can keep
importing from ``specula`` and do not need any change.
