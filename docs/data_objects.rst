Data Objects
============

Data objects in SPECULA serve as **intelligent containers** that connect processing objects and carry temporal information about when data was generated. They extend the :class:`~specula.base_data_obj.BaseDataObj` class and provide the essential data flow between computational components.

Core Concepts
-------------

**Temporal Awareness**
   Every data object tracks its :attr:`~specula.base_data_obj.BaseDataObj.generation_time`, allowing the simulation to maintain temporal consistency and detect when data needs to be refreshed.

**Device Management**
   Data objects automatically handle GPU/CPU transfers through the :meth:`~specula.base_data_obj.BaseDataObj.copyTo` and :meth:`~specula.base_data_obj.BaseDataObj.transferDataTo` methods, enabling seamless computation across different devices.

**Persistent Storage**
   All data objects implement :meth:`~specula.base_data_obj.BaseDataObj.save` and :meth:`~specula.base_data_obj.BaseDataObj.read` methods using FITS format, ensuring simulation data can be stored and reloaded.

**Connection Framework**
   Data objects flow through the simulation graph as outputs from one processing object become inputs to another, creating a directed acyclic graph of computation.

Available Data Objects
----------------------

**Optical Wavefronts**
   * :class:`~specula.data_objects.electric_field.ElectricField` - Complex amplitude and phase information
   * :class:`~specula.data_objects.intensity.Intensity` - Detected intensity maps
   * :class:`~specula.data_objects.pixels.Pixels` - Digitized detector readouts

**Wavefront Sensing**
   * :class:`~specula.data_objects.slopes.Slopes` - Wavefront sensor measurements (x,y slopes)
   * :class:`~specula.data_objects.subap_data.SubapData` - Subaperture geometry and validity maps

**System Geometry**
   * :class:`~specula.data_objects.pupdata.PupData` - Telescope pupil geometry and indexing
   * :class:`~specula.data_objects.pupilstop.Pupilstop` - Pupil masks and obstruction patterns
   * :class:`~specula.data_objects.layer.Layer` - Atmospheric or optical layers
   * :class:`~specula.data_objects.source.Source` - Guide star and science target definitions

**Calibration Data**
   * :class:`~specula.data_objects.intmat.Intmat` - Interaction matrices (slopes→commands)
   * :class:`~specula.data_objects.recmat.Recmat` - Reconstruction matrices (commands→slopes)
   * :class:`~specula.data_objects.ifunc.IFunc` - Deformable mirror influence functions
   * :class:`~specula.data_objects.m2c.M2C` - Mode-to-command transformation matrices

**Signal Processing**
   * :class:`~specula.data_objects.convolution_kernel.ConvolutionKernel` - Generic convolution kernels
   * :class:`~specula.data_objects.gaussian_convolution_kernel.GaussianConvolutionKernel` - Gaussian PSF kernels
   * :class:`~specula.data_objects.iir_filter_data.IirFilterData` - Digital filter coefficients
   * :class:`~specula.data_objects.time_history.TimeHistory` - Temporal data sequences

**Specialized Components**
   * :class:`~specula.data_objects.laser_launch_telescope.LaserLaunchTelescope` - Laser guide star launcher geometry
   * :class:`~specula.data_objects.lenslet.Lenslet` - Shack-Hartmann lenslet arrays
   * :class:`~specula.data_objects.infinite_phase_screen.InfinitePhaseScreen` - Atmospheric turbulence screens

Persistence and Data Flow
-------------------------

SPECULA data objects implement specialized methods for seamless integration with the data storage and replay system:

**Storage and Replay Integration**
   Data objects work automatically with :class:`~specula.processing_objects.data_store.DataStore` and :class:`~specula.processing_objects.data_source.DataSource` through standardized methods:

   * :meth:`get_value` - Extracts the core numerical data for storage
   * :meth:`from_header` - Reconstructs objects from stored headers
   * :meth:`set_value` - Restores numerical data from storage

**Supported Data Formats:**

- **FITS**: Default format, preserves metadata and ensures portability
- **Pickle**: Python-specific format for complex objects

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
3. **Persistent**: All data can be saved and restored with full metadata
4. **Type Safety**: Each data type has specific validation and methods
5. **Modular**: Data objects can be combined and reused across simulations
6. **Automatic Storage and Replay**: Seamless integration with DataStore/DataSource pipeliner analysis

Data objects form the **connective tissue** of SPECULA simulations, ensuring that information flows correctly through the processing pipeline while maintaining temporal and spatial consistency. The integrated storage system enables **complete reproducibility** of simulation results and facilitates post-processing analysis workflows.

.. seealso::
   
   :doc:`processing_objects` for how data objects connect to processing components
   :doc:`base_classes` for the underlying BaseDataObj implementation
   :doc:`tutorials/scao_tutorial` for practical examples of data object usage
