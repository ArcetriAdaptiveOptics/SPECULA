Running Simulations
===================

In the directory ``main/scao`` there are several example configuration / parameters files for SCAO systems.

The basic way to run the simulation is to use the Simul class directly:

.. code-block:: python

    import specula
    specula.init(target_device_idx, precision=1)

    print(args)    
    from specula.simul import Simul
    simul = Simul(yml_file,
                  overrides=args.overrides,
                  diagram=args.diagram,
                  diagram_filename=args.diagram_filename,
                  diagram_title=args.diagram_title,
    )
    simul.run()

where ``target_device_idx`` is the GPU device number (or ``-1`` for CPU), and ``yml_file`` is the path to your configuration / parameters file.

This is embedded in the main simulation script ``main_simul.py`` that can be found in the ``main/scao`` directory.

A tutorial for running SCAO simulations is available in the :doc:`tutorials/scao_tutorial` page.