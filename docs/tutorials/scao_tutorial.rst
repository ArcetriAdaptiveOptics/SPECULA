SCAO Tutorial: Complete Walkthrough
====================================

This comprehensive tutorial guides you through creating, running, and analyzing a complete Single Conjugate Adaptive Optics (SCAO) simulation using SPECULA.

**What you'll learn:**

* Setting up a realistic SCAO system configuration
* Running calibration and closed-loop phases
* Analyzing performance results
* Optimizing system parameters
* Troubleshooting common issues

**Prerequisites:**

* SPECULA installed and working
* Basic understanding of adaptive optics concepts
* Python and YAML familiarity

Tutorial Overview
-----------------

We'll simulate a modern SCAO system similar to those used on 8-10m class telescopes:

**System Specifications:**
* 8.2m telescope (VLT-like) with 14% central obstruction
* Kolmogorov turbulence, r₀ = 15cm at 500nm
* 40×40 Shack-Hartmann WFS (1600 subapertures)
* 1240 actuator deformable mirror
* 1 kHz control loop with integrator controller
* R-band natural guide star (magnitude 8)

**Performance Goals:**
* Strehl ratio > 60% in H-band
* RMS wavefront error < 150nm
* Stable closed-loop operation

Part 1: System Configuration
----------------------------

Setting Up the Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Main Configuration File
~~~~~~~~~~~~~~~~~~~~~~~

Create ``config/main.yaml``:

.. code-block:: yaml

   # SCAO Tutorial - Main Configuration

Part 2: Running the Simulation
------------------------------

Calibration Phase
~~~~~~~~~~~~~~~~~

Run the calibration to generate interaction matrices:

.. code-block:: bash

   python main_simul.py TODO

This will:

1. **Generate pupil data** from telescope geometry
3. **Measure WFS response** for each actuator poke
5. **Compute interaction matrix** and save to ``calibration/``
6. **Generate reconstructor** using SVD inversion

Expected output:

.. code-block:: text

   SPECULA Calibration Phase
   =========================
   
   [10:30:15] Initializing 12 simulation objects...
   [10:30:16] Computing pupil geometry...
   [10:30:16] Found 1634 valid subapertures (40×40 grid)
   [10:30:16] Starting interaction matrix calibration...
   [10:30:16] Mode 1/1240: Actuator (15,20) - Response: 0.34 arcsec RMS
   [10:30:17] Mode 2/1240: Actuator (15,21) - Response: 0.31 arcsec RMS
   ...
   [10:45:22] Mode 1240/1240: Actuator (35,12) - Response: 0.29 arcsec RMS
   [10:45:23] Calibration completed in 15.1 minutes
   [10:45:23] Interaction matrix: 3268 × 1240 (slopes × modes)
   [10:45:24] Computing reconstructor with SVD...
   [10:45:26] Condition number: 1.2e4 (good)
   [10:45:26] Effective modes: 1186 / 1240 (95.6%)
   [10:45:27] Reconstructor saved: calibration/reconstructor.fits
   
   Calibration Summary:
   - Valid subapertures: 1634
   - Reconstructed modes: 1186  
   - Matrix condition: Good
   - Ready for closed-loop!

Closed-Loop Simulation
~~~~~~~~~~~~~~~~~~~~~~

Now run the full closed-loop simulation:

.. code-block:: bash

   python main_simul.py --verbose

Watch the real-time performance:

.. code-block:: text

   SPECULA Closed-Loop Simulation
   ==============================
   Configuration: config/main.yaml
   Duration: 2.0s (2000 iterations at 1000 Hz)
   
   [10:50:15] Loading calibration data...
   [10:50:16] Starting closed-loop at t=0.000s
   
   Iteration    Time     Strehl    RMS WFE   Loop Gain   Status
   ---------    ----     ------    -------   ---------   ------
          10   0.010s     0.12      287nm       0.30     Converging
         100   0.100s     0.45      198nm       0.30     Stable  
         200   0.200s     0.58      156nm       0.30     Stable
         500   0.500s     0.64      142nm       0.30     Stable
        1000   1.000s     0.67      136nm       0.30     Stable
        1500   1.500s     0.68      134nm       0.30     Stable
        2000   2.000s     0.69      132nm       0.30     Excellent
   
   [10:51:28] Simulation completed in 1.2 minutes
   [10:51:28] Average performance: Strehl = 0.65 ± 0.03
   [10:51:28] Final RMS WFE: 135 ± 8 nm
   [10:51:29] Results saved to: results/scao_20250115_105015/

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

TODO

Part 3: Results Analysis
------------------------

Data Organization
~~~~~~~~~~~~~~~~~

After simulation, your results directory contains:

.. code-block:: text

   TOdO

Quick Performance Check
~~~~~~~~~~~~~~~~~~~~~~~

Load and examine the key results:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt


PSF Analysis
~~~~~~~~~~~~

Examine the point spread function quality:

.. code-block:: python

   from astropy.io import fits
   import numpy as np


Part 4: Parameter Optimization
------------------------------

Now that you have a working baseline, let's optimize the system performance.

Loop Gain Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Test different control gains to find the optimum:

.. code-block:: yaml

   # Create


WFS Resolution Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~

Compare different subaperture numbers:

.. code-block:: yaml

   # yaml

Part 5: Advanced Topics
-----------------------
      
Guide Star Magnitude Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Study performance vs. star brightness:

.. code-block:: yaml

   sweep


Troubleshooting Common Issues
-----------------------------

Calibration Problems
~~~~~~~~~~~~~~~~~~~

**Issue**: "Interaction matrix rank deficient"

**Solution**: 

Poor Performance
~~~~~~~~~~~~~~~

**Issue**: Strehl ratio much lower than expected

**Checklist**:
1. TODO

**Issue**: Loop instability or oscillations

**Solutions**:
.. code-block:: yaml

   integrator:

Computational Issues
~~~~~~~~~~~~~~~~~~~

**Issue**: Out of memory errors

**Solutions**:
.. code-block:: yaml

**Issue**: Slow execution

**Solutions**:
.. code-block:: bash

   # Check GPU usage

Summary and Next Steps
----------------------

Congratulations! You've successfully:

✅ **Configured** a complete SCAO system
✅ **Calibrated** the wavefront sensor and control system  
✅ **Executed** a closed-loop simulation
✅ **Analyzed** performance results
✅ **Optimized** system parameters

**Your SCAO system achieved:**
* Strehl ratio: ~69% (excellent for ground-based AO)
* RMS wavefront error: ~134 nm  
* Correction bandwidth: ~39 Hz
* Stable closed-loop operation

**Next Steps:**

1. **Experiment** with different atmospheric conditions
2. **Try** pyramid wavefront sensors (see :doc:`pyramid_calibration`)
3. **Explore** laser guide star systems  
4. **Scale up** to MCAO configurations
5. **Integrate** with science instruments

**Additional Resources:**

* :doc:`../processing_objects` - Create custom components
* :doc:`../analysis` - Advanced performance analysis
* :doc:`mcao_tutorial` - Multi-conjugate AO systems
* :doc:`../troubleshooting` - Detailed problem solving

The complete tutorial files are available in the SPECULA examples directory.

.. seealso::
   
   :doc:`pyramid_calibration` for Pyramid WFS systems
   :doc:`mcao_tutorial` for multi-conjugate AO
   :doc:`../configuration` for complete YAML reference
