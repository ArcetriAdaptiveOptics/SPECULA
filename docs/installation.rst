.. _installation:

Installation
============

SPECULA requires Python 3.8 or higher and we strongly recommend using conda for package management.

Prerequisites
-------------

**System Requirements:**
   * Python 3.8 or higher
   * Git (for repository cloning)
   * CUDA-compatible GPU (optional, for acceleration)

**Recommended Setup:**
   * Anaconda or Miniconda
   * 16GB+ RAM
   * 12GB+ GPU memory (if using GPU acceleration)

Step 1: Create Conda Environment
--------------------------------

Create a dedicated conda environment for SPECULA (here with python 3.11):

.. code-block:: bash

   # Create environment with Python 3.11
   conda create --name specula python=3.11
   
   # Activate the environment
   conda activate specula

Step 2: GPU Support (Optional but Recommended)
----------------------------------------------

If you have a CUDA-compatible GPU and want to benefit from GPU acceleration, install CuPy:

.. code-block:: bash

   # Install CuPy for GPU acceleration
   conda install -c conda-forge cupy

**GPU Benefits:**
   * 10-100× faster simulations

**Without GPU:**
   SPECULA will automatically fall back to CPU computation using NumPy. Performance will be slower but all functionality remains available.

Step 3: Install SPECULA
-----------------------

You can install SPECULA in two ways:

**A) Install from PyPI (recommended for most users):**

.. code-block:: bash

   # Install SPECULA from PyPI
   pip install specula

**B) Install from source (for development or latest features):**

Clone the SPECULA repository from GitHub:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ArcetriAdaptiveOptics/SPECULA.git

   # Navigate to the directory
   cd SPECULA

Then install in development mode:

.. code-block:: bash

   pip install -e .

This installs SPECULA in "editable" mode, allowing you to modify the code and see changes immediately.

**Required Dependencies:**
All required dependencies will be installed automatically, including:

* **numpy**: Numerical computing foundation
* **scipy**: Scientific computing algorithms
* **astropy**: Astronomical data handling and FITS I/O
* **matplotlib**: Plotting and visualization
* **flask**: Web framework for display server
* **flask-socketio**: Real-time web communication
* **python-socketio**: WebSocket client support

Optional Libraries
^^^^^^^^^^^^^^^^^^

Block Diagram Generation
""""""""""""""""""""""""

* **pycairo**: Graphics library for rendering diagrams
* **orthogram**: Automatic block diagram creation from SPECULA configurations

To install these libraries, first install pkg-config and cairo with conda:

.. code-block:: bash

   # Install dependencies for optional diagram tools
   conda install -c conda-forge pkg-config cairo

To ensure that the correct version of pycairo is installed, use conda:

.. code-block:: bash

   # Install pycairo version 1.21.0
   conda install -c conda-forge pycairo=1.21.0

and then install orthogram with pip.

.. code-block:: bash

   # Install optional diagram tools
   pip install orthogram

Transfer function system management
"""""""""""""""""""""""""""""""""""

* **control**: Library for control system analysis and design

.. code-block:: bash

   # Install control library
   pip install control

Environment Management
----------------------

**Useful conda commands:**

.. code-block:: bash

   # List environments
   conda env list
   
   # Activate SPECULA environment
   conda activate specula
   
   # Deactivate environment
   conda deactivate
   
   # Update all packages
   conda update --all
   
   # Remove environment (if needed)
   conda env remove --name specula

**Updating SPECULA:**

.. code-block:: bash

   # Navigate to SPECULA directory
   cd SPECULA
   
   # Pull latest changes
   git pull origin main
