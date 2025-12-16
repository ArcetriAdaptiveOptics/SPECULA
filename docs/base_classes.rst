.. _base_classes:

Base Classes
============

SPECULA uses several base classes that provide the foundation for all processing objects and data structures. These classes are designed to be inherited and should not be instantiated directly.

Overview
--------

The base classes in SPECULA provide:

- **Common interfaces** for all processing objects
- **Standardized data handling** patterns
- **Connection management** between simulation components
- **Time synchronization** mechanisms
- **Device management** (CPU/GPU) capabilities

Core Base Classes
-----------------

**BaseProcessingObj**
    The foundation for all processing objects in SPECULA. Provides:
    
    - Input/output connection management
    - Setup and trigger mechanisms
    - Stream processing support
    - Device allocation (CPU/GPU)

**BaseValue**
    Base class for all data containers. Handles:
    
    - Data storage and access
    - Type validation
    - Memory management
    - Device-specific operations

**BaseDataObj**
    Extended data objects with additional metadata:
    
    - Generation time tracking
    - Data provenance
    - Serialization support

**BaseTimeObj**
    Time-aware objects for simulation synchronization:
    
    - High-resolution timestamp management (nanosecond precision)
    - Time conversion utilities (seconds ↔ internal format)
    - Device allocation and memory monitoring
    - Mathematical library integration (scipy/cupyx)

**TemplateProcessingObj**
    Template and example for creating new processing objects:
    
    - Standard structure patterns
    - Input/output setup examples
    - Best practice demonstrations

Connection System
-----------------

The connection system (``specula.connections``) provides:

- **InputValue/OutputValue**: Typed connections between objects
- **Connection validation**: Ensures type compatibility
- **Data flow management**: Handles data transfer between components
- **Lazy evaluation**: Data is computed only when needed

Usage Guidelines
----------------

When creating new SPECULA components:

1. **Always inherit** from the appropriate base class
2. **Don't instantiate** base classes directly
3. **Override required methods** (``setup()``, ``trigger_code()``, etc.)
4. **Use the connection system** for inter-object communication
5. **Follow naming conventions** established by base classes

Example Structure
-----------------

.. code-block:: python

    from specula.base_processing_obj import BaseProcessingObj
    from specula.connections import InputValue, OutputValue
    
    class MyProcessor(BaseProcessingObj):
        def __init__(self, ...):
            super().__init__(...)
            
            # Define inputs and outputs
            self.inputs['data_in'] = InputValue(type=MyDataType)
            self.outputs['result'] = OutputValue(type=MyResultType)
        
        def setup(self):
            # Initialize processing
            pass
            
        def trigger_code(self):
            # Main processing logic
            input_data = self.inputs['data_in'].get()
            result = self.process(input_data)
            self.outputs['result'].set(result)

For detailed API documentation of all base classes, see :doc:`api/base_classes`.

See Also
--------

- :doc:`api/base_classes` - Complete API reference
- :doc:`development` - Development conventions
- :doc:`processing_objects` - Guide to processing objects
- :doc:`data_objects` - Guide to data objects