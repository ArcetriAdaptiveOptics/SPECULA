
'''
How to write tests:

#. At the beginning::
    import specula
    specula.init(0)  # Default target device
    
#. In most cases, also import cpuArray and the `cpu_and_gpu` decorator::

    from specula import cpuArray
    from specula_testlib import cpu_and_gpu

#. Each test function must be decorated with `@cpu_and_gpu`, and take two
   keyword arguments: `target_device_idx` and `xp`.

#. Inside the test function, always use `xp.array` and similar to allocate data,
   and use `target_device_idx` in SPECULA object constructors if any

#. Remember to use `cpuArray()` on any data that you want to compare to verify
'''
