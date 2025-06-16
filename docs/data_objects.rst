Data Objects
============

Data objects in SPECULA serve as **intelligent containers** that connect processing objects and carry temporal information about when data was generated. They extend the [`BaseDataObj`](specula/base_data_obj.py) class and provide the essential data flow between computational components.

Core Concepts
-------------

**Temporal Awareness**
   Every data object tracks its [`generation_time`](specula/base_data_obj.py), allowing the simulation to maintain temporal consistency and detect when data needs to be refreshed.

**Device Management**
   Data objects automatically handle GPU/CPU transfers through the [`copyTo()`](specula/base_data_obj.py) and [`transferDataTo()`](specula/base_data_obj.py) methods, enabling seamless computation across different devices.

**Persistent Storage**
   All data objects implement [`save()`](specula/data_objects/pupdata.py) and [`read()`](specula/data_objects/pupdata.py) methods using FITS format, ensuring simulation data can be stored and reloaded.

**Connection Framework**
   Data objects flow through the simulation graph as outputs from one processing object become inputs to another, creating a directed acyclic graph of computation.

Available Data Objects
----------------------

**Optical Wavefronts and Atmosphere**
   * [`ElectricField`](specula/data_objects/electric_field.py) - Complex amplitude and phase information
   * [`Intensity`](specula/data_objects/intensity.py) - Detected intensity maps
   * [`Layer`](specula/data_objects/layer.py) - Atmospheric or optical layers
   * [`InfinitePhaseScreen`](specula/data_objects/infinite_phase_screen.py) - Atmospheric turbulence screens

**Wavefront Sensing**
   * [`Slopes`](specula/data_objects/slopes.py) - Wavefront sensor measurements (x,y slopes)
   * [`Lenslet`](specula/data_objects/lenslet.py) - Shack-Hartmann lenslet arrays
   * [`SubapData`](specula/data_objects/subap_data.py) - Subaperture geometry and validity maps
   * [`PupData`](specula/data_objects/pupdata.py) - Telescope pupil geometry and indexing

**System Geometry**
   * [`Pupilstop`](specula/data_objects/pupilstop.py) - Pupil masks and obstruction patterns
   * [`Source`](specula/data_objects/source.py) - Guide star and science target definitions

**Calibration Data**
   * [`Intmat`](specula/data_objects/intmat.py) - Interaction matrices (slopes→commands)
   * [`Recmat`](specula/data_objects/recmat.py) - Reconstruction matrices (commands→slopes)
   * [`IFunc`](specula/data_objects/ifunc.py) - Deformable mirror influence functions
   * [`M2C`](specula/data_objects/m2c.py) - Mode-to-command transformation matrices

**Signal Processing**
   * [`IirFilterData`](specula/data_objects/iir_filter_data.py) - Digital filter coefficients
   * [`TimeHistory`](specula/data_objects/time_history.py) - Temporal data sequences
   * [`Pixels`](specula/data_objects/pixels.py) - Digitized detector readouts

**Specialized Components**
   * [`LaserLaunchTelescope`](specula/data_objects/laser_launch_telescope.py) - Laser guide star launcher geometry
   * [`ConvolutionKernel`](specula/data_objects/convolution_kernel.py) - Generic convolution kernels
   * [`GaussianConvolutionKernel`](specula/data_objects/gaussian_convolution_kernel.py) - Gaussian PSF kernels

Usage Example
-------------

Data objects automatically manage temporal consistency:

.. code-block:: python

   class MyProcessor(BaseProcessingObj):
       def trigger_code(self):
           # Check if input data is current
           if self.local_inputs['wavefront'].generation_time != self.current_time:
               return  # Skip processing with stale data
           
           # Process current data
           input_wf = self.local_inputs['wavefront']
           result = self.process(input_wf.phase)
           
           # Update output with current timestamp
           self.outputs['processed'].value = result
           self.outputs['processed'].generation_time = self.current_time

Device Transfer Example
-----------------------

Moving data between GPU and CPU:

.. code-block:: python

   # Original data on GPU
   gpu_slopes = Slopes(target_device_idx=0)  # GPU device 0
   
   # Transfer to CPU for analysis
   cpu_slopes = gpu_slopes.copyTo(target_device_idx=-1)  # CPU
   
   # Data is automatically converted between CuPy and NumPy arrays

Persistence Example
-------------------

Saving and loading calibration data:

.. code-block:: python

   # Save interaction matrix
   intmat = Intmat(matrix_data, pupdata_tag='telescope_pupil')
   intmat.save('calibration/interaction_matrix.fits')
   
   # Load in another simulation
   loaded_intmat = Intmat.restore('calibration/interaction_matrix.fits')

**Key Design Principles:**

1. **Temporal Consistency**: Every data object knows when it was created
2. **Device Agnostic**: Automatic GPU/CPU memory management  
3. **Persistent**: All data can be saved and restored
4. **Type Safety**: Each data type has specific validation and methods
5. **Modular**: Data objects can be combined and reused across simulations

Data objects form the **connective tissue** of SPECULA simulations, ensuring that information flows correctly through the processing pipeline while maintaining temporal and spatial consistency.