Running Simulations
===================

This comprehensive guide covers all aspects of executing SPECULA simulations, from basic usage to advanced optimization techniques.

Quick Start
-----------

For impatient users, here's the fastest way to run a simulation:
TODO add caibration info

.. code-block:: bash

   cd main/scao
   python main_simul.py params_scao_sh.yml
   
This runs a complete SCAO simulation with default parameters.

Command Line Interface
----------------------

SPECULA provides flexible command-line options:

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   python main_simul.py [options] [yaml file]

**Common Options:**

TODO

.. code-block:: bash

   # Use custom configuration
   python main_simul.py --config my_setup.yaml


Advanced Options
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile performance
   python main_simul.py --profile --profile-output timing.txt


Configuration Management
------------------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ???

Performance Optimization
------------------------

Hardware Acceleration
~~~~~~~~~~~~~~~~~~~~~

**GPU Usage:**

.. code-block:: yaml

   main:
     target_device_idx: 0        # GPU device number
     precision: 32               # 32-bit float (faster)
     memory_pool: true           # Enable memory pooling

**Multi-GPU:**

   TODO target_device_idx

**CPU:**

.. code-block:: yaml

   main:
     target_device_idx: -1       # Force CPU execution

Results Management
------------------

Output Organization
~~~~~~~~~~~~~~~~~~~

SPECULA automatically organizes simulation outputs:

TODO

Data Formats
~~~~~~~~~~~~

**FITS for astronomical data:**

.. code-block:: python

   from astropy.io import fits
   
   # Load PSF evolution
   psf_data = fits.getdata('results/psf.fits')
   
Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Memory Errors:**

TODO

.. seealso::
   
   :doc:`configuration` for complete YAML reference
   :doc:`tutorials/scao_tutorial` for step-by-step examples  
   :doc:`troubleshooting` for detailed problem resolution
