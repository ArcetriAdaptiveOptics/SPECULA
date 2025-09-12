.. _running_simulations:

Running Simulations
===================

In the directory ``main/scao`` there are several example configuration / parameters files for SCAO systems.

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