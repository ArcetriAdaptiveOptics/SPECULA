.. _running_simulations:

Running Simulations
===================

In the directory ``main/scao`` there are several example configuration / parameters files for SCAO systems.

The basic way to run the simulation is to use the :class:`specula.simul.Simul` class directly:

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
    )
    simul.run()

where ``target_device_idx`` is the GPU device number (or ``-1`` for CPU), and ``yml_file`` is the path to your configuration / parameters file.
The ``overrides`` parameter allows you to combine the parameter of the configuration file with the one of an additional file (or additional files).
This is useful when we need to override, add and/or remove some parameters of the main simulation.
The other parameters, ``diagram``, ``diagram_filename``, and ``diagram_title``, are optional and can be used to generate a diagram of the simulation, which is useful for understanding and debugging the flow of data.
The diagram is the graphical representation of the simulation, showing the objects and their connections.

This is embedded in the main simulation script ``main_simul.py`` that can be found in the ``main/scao`` directory.

Examples of the diagram can be found in :doc:`simul_diagrams` page.
A tutorial for running SCAO simulations is available in the :doc:`tutorials/scao_tutorial` page.