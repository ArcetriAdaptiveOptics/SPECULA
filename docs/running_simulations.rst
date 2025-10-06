.. _running_simulations:

Running Simulations
===================

In ``config`` directory there are example configuration / parameters files for several systems.

The main way to run a simulation is to use the ``specula`` command line tool, installed together
with the SPECULA package, giving the configuration file as an argument, in addition to several
optional arguments (visible with the ``specula -h`` command).

When embedding in another Python program, it is possible to use the :class:`specula.simul.Simul` class directly:

.. code-block:: python

    import specula
    specula.init(target_device_idx, precision=1)

    print(args)    
    from specula.simul import Simul
    simul = Simul(yml_file,
                  overrides=overrides,
                  diagram=diagram,
                  diagram_filename=diagram_filename,
                  diagram_title=diagram_title,
                  diagram_colors_on=diagram_colors_on
    )
    simul.run()

where ``target_device_idx`` is the GPU device number (or ``-1`` for CPU), and ``yml_file`` is the path to your configuration / parameters file.
The ``overrides`` parameter allows you to combine the parameter of the configuration file with the one of an additional file (or additional files).
This is useful when we need to override, add and/or remove some parameters of the main simulation.
The other parameters, ``diagram``, ``diagram_filename``, ``diagram_title``, and ``diagram_colors_on``, are optional and can be used to generate a diagram of the simulation, which is useful for understanding and debugging the flow of data.
The diagram is the graphical representation of the simulation, showing the objects and their connections.

These arguments are similar to the ones used by ``specula`` itself, whose implementation can be find in the :py:func:`specula.__init__.main` function in file :file:`specula.__init__.py`.

Examples of the diagram can be found in :doc:`simul_diagrams` page.
A tutorial for running SCAO simulations is available in the :doc:`tutorials/scao_tutorial` page.

Multiple Simulations and Override System
========================================

SPECULA provides a powerful system for running multiple simulations with different parameters using override files and simulation indices. This is particularly useful for calibration procedures, parametric studies, and multi-configuration analysis.

Override System
---------------

The override system allows you to modify parameters from a base configuration file using suffixed parameter names:

Global Overrides
~~~~~~~~~~~~~~~~

Parameters with the ``_override`` suffix are applied to **all** simulations:

.. code-block:: yaml

    # Applied to ALL simulations
    detector_override:
      photon_noise: false
      readout_noise: false
    
    dm_override:
      inputs:
        in_command: 'default_command'

Global Object Removal
^^^^^^^^^^^^^^^^^^^^^^

The ``remove`` keyword removes objects from **all** simulations:

.. code-block:: yaml

    # Remove these objects from ALL simulations
    remove: ['atmo', 'tomo_polc_lgs', 'iir_lgs', 'psf']

This is particularly useful for calibration procedures where certain objects (like atmosphere, controllers, or analysis tools) are not needed in any simulation variant.

Simulation-Specific Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters with the ``_override_N`` suffix are applied **only** to simulation with ``simul_idx=N``:

.. code-block:: yaml

    # Applied ONLY to simulation with simul_idx=0
    main_override_0:
      total_time: 1
    
    dm_override_0:
      inputs:
        in_command: 'pushpull1_dm.output'
    
    # Applied ONLY to simulation with simul_idx=2
    main_override_1:
      total_time: 2

    dm_override_1:
      inputs:
        in_command: 'pushpull2_dm.output'

Simulation-Specific Object Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``remove_N`` keyword removes objects **only** from simulation with ``simul_idx=N``:

.. code-block:: yaml

    # Remove only from simulation 0
    remove_0: ['dm2', 'dm3']
    
    # Remove only from simulation 1
    remove_1: ['dm1', 'dm3']
    
    # Remove only from simulation 2
    remove_2: ['dm1', 'dm2']

This allows you to selectively disable different components for each simulation, such as individual deformable mirrors in a multi-DM calibration.

Parameter Application Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters are applied in the following order:

1. **Base parameters**: From the main YAML file
2. **Global overrides**: ``_override`` (applied to all simulations)
3. **Simulation-specific overrides**: ``_override_N`` (only if ``simul_idx == N``)
4. **Simulation-specific objects**: Objects with ``_N`` suffix (only if ``simul_idx == N``)

If the same parameter is defined in multiple places, simulation-specific overrides take precedence over global ones.

Running Multiple Simulations
-----------------------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Use the ``--nsimul`` option to run multiple simulations automatically:

.. code-block:: bash

    # Run 3 simulations automatically (simul_idx = 0, 1, 2)
    specula config/base.yml config/override.yml --nsimul 3
    
    # Run single simulation (default)
    specula config/base.yml config/override.yml

The ``--nsimul`` parameter automatically runs multiple simulations in sequence, with ``simul_idx`` ranging from 0 to ``nsimul-1``.

Python API
~~~~~~~~~~

Using the :class:`specula.simul.Simul` class directly with explicit ``simul_idx``:

.. code-block:: python

    from specula.simul import Simul
    
    # Single simulation with specific simul_idx
    simul = Simul('base.yml', 'override.yml', simul_idx=1)
    simul.run()
    
    # Multiple simulations loop
    for simul_idx in range(3):
        simul = Simul('base.yml', 'override.yml', simul_idx=simul_idx)
        simul.run()

Using ``main_simul()`` for automatic multiple runs:

.. code-block:: python

    import specula
    
    # Automatically runs 3 simulations (simul_idx = 0, 1, 2)
    specula.main_simul(['base.yml', 'override.yml'], nsimul=3)
    
    # Single simulation (default)
    specula.main_simul(['base.yml', 'override.yml'])