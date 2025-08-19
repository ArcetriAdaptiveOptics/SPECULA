.. _step_response_tutorial:

Step Response Tutorial: Simulating the Step Response of a DM Actuator
=====================================================================

This tutorial demonstrates how to set up and analyze the **step response** of a simplified deformable mirror (DM) actuator model using SPECULA.

**Goals:**
- Simulate the dynamic response of a DM actuator to a step input in closed-loop control
- Model the control chain with sample&hold, integrator, and low-pass filter
- Visualize the response in real time (no data is saved to disk)

System Overview
---------------

The simulated system consists of:
- **Input generator**: a step (or square wave) signal representing the desired actuator command
- **Sample&Hold**: limits the control framerate (e.g., 1 kHz → 1 ms)
- **Integrator**: accumulates the error (controller)
- **Low-pass filter**: models the actuator dynamics, with a configurable cutoff frequency and amplification factor (to simulate resonance)

Block diagram:

.. code-block:: text

    [Step Input] → [Sample&Hold] → [Integrator] → [LowPassFilter] → [Output]

YAML Configuration
------------------

Create a configuration file, for example ``params_control_lpf.yml``:

.. code-block:: yaml

   main:
     class:             'SimulParams'
     pixel_pupil:       160
     pixel_pitch:       0.05
     total_time:        1.0
     time_step:         0.0001

   disturbance:
     class:             'WaveGenerator'
     wave_type:         'SQUARE'
     amp:               [5.0]
     freq:              [50.0] 

   diff:
     class:             'BaseOperation'
     sub:               True
     inputs:
         in_value1: 'disturbance.output'
         in_value2: 'lowpass.out_comm:-1'

   sampHold:
     class:             'WindowedIntegration'
     simul_params_ref:  'main'
     n_elem:            1
     dt:                0.001
     inputs:
         input: 'diff.out_value'

   control:
     class:             'Integrator'
     simul_params_ref:  'main'
     delay:             1
     int_gain:          [0.3]
     inputs:
         delta_comm: 'sampHold.output'

   lowpass:
     class:             'LowPassFilter'
     simul_params_ref:  'main'
     delay:             0
     cutoff_freq:       [1000]
     amplif_fact:       [3]
     inputs:
         delta_comm: 'control.out_comm'

   all_disp:
     class:            'PlotDisplay'
     inputs:
       value_list: ['disturbance.output',
                    'diff.out_value',
                    'control.out_comm',
                    'lowpass.out_comm']
     yrange:           [-10,10]
     title:            'Step Response'

**Block description:**
- `disturbance`: generates the input signal (step or square wave)
- `sampHold`: sample&hold to limit the control framerate
- `control`: integrator (controller)
- `lowpass`: low-pass filter modeling the actuator dynamics
- `all_disp`: real-time plot of the main signals

Running the Simulation
----------------------

To run the simulation, use:

.. code-block:: bash

   specula params_control_lpf.yml

During the simulation, a real-time plot will show:
- Input (step)
- Error (diff)
- Integrator command
- Low-pass filter output (actuator response)

Response Analysis
-----------------

Observe the actuator response:
- **Rise time**: how quickly the actuator follows the command
- **Overshoot**: possible resonance due to the amplification factor
- **Steady-state error**: difference between input and output after the transition

You can modify the `cutoff_freq` and `amplif_fact` parameters in the low-pass filter to see how the actuator performance and resonance change.

**Note:**  
In this simulation, data is **not saved to disk**, only plotted on screen.

Customizations and Experiments
------------------------------

- Change the cutoff frequency (`cutoff_freq`) to simulate faster or slower actuators.
- Adjust the integrator gain (`int_gain`) to see its effect on response speed.
- Try different values of `amplif_fact` to simulate actuator resonance.
- Replace the input with other signals (`SIN`, etc.) to test different responses.

**Conclusion**

You have simulated and visualized the step response of a DM actuator with digital control and realistic dynamics.  
This setup is a foundation for testing and optimizing controllers and actuator models in adaptive optics systems.

.. seealso::

   - :ref:`scao_tutorial` for a complete SCAO simulation example
   - SPECULA block documentation
