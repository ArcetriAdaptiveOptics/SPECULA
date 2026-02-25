Control Stability and Performance Analysis
==========================================

This guide demonstrates how to use the :class:`~specula.data_objects.iir_filter_data.IirFilterData` class 
to perform stability and frequency response analysis of an Adaptive Optics (AO) control loop.

Transfer Function Convention
----------------------------

In SPECULA, the IIR filters are defined using the following convention:

.. math::
    H(z) = \frac{num[0] + num[1]z + num[2]z^2 + \dots}{den[0] + den[1]z + den[2]z^2 + \dots}

Where index ``0`` corresponds to the constant term ($z^0$).

Overview
--------

In an AO system, the closed-loop performance is determined by the interaction between the **Controller** (:math:`C`) 
and the **Plant** (:math:`P`). The plant typically includes the WFS integration time, the RTC latency (delay), 
and the Deformable Mirror dynamics (Low Pass Filter).

The fundamental transfer functions are:

* **Rejection Transfer Function (RTF)**: Describes how the system suppresses atmospheric turbulence.
  
  .. math::
     RTF(z) = \frac{1}{1 + C(z)P(z)}

* **Noise Transfer Function (NTF)**: Describes how measurement noise propagates to the actuators.
  
  .. math::
     NTF(z) = \frac{C(z)P(z)}{1 + C(z)P(z)}

Building the Control Loop
-------------------------

A realistic AO plant consists of a fractional delay and a low-pass filter representing the mirror dynamics.

1. **Controller**: An integrator with gain and forgetting factor.
2. **Plant Delay**: WFS integration and RTC latency using :meth:`discrete_delay_tf`.
3. **Plant LPF**: Mirror response using :meth:`lpf_from_fc`.

Analysis Script
---------------

The following script computes the RTF, NTF, and the Nyquist plot for stability verification.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import specula
    specula.init(-1)  # CPU execution
    from specula.data_objects.iir_filter_data import IirFilterData

    # --- Configuration ---
    fs = 1000.0         # Sampling frequency [Hz]
    gain = 0.35         # Loop gain
    ff = 0.999          # Forgetting factor (leaky integrator)
    delay_frames = 2.4  # Total loop delay [frames]
    fc_lpf = 400.0      # Mirror LPF cutoff [Hz]

    # 1. Define Controller (C)
    controller = IirFilterData.from_gain_and_ff(gain=[gain], ff=[ff])

    # 2. Define Plant Components (P)
    # Delay TF: interpolates between samples for fractional values
    nw_delay, dw_delay = controller.discrete_delay_tf(delay_frames)

    # Mirror dynamics (2nd order Butterworth)
    lpf_obj = IirFilterData.lpf_from_fc(fc=fc_lpf, fs=fs, n_ord=2)
    lpf_num, lpf_den = lpf_obj.num[0], lpf_obj.den[0]

    # 3. Assemble the Open Loop (CP)
    # Open Loop Numerator = C_num * P_delay_num * LPF_num
    nw_plant = np.convolve(nw_delay, lpf_num)
    dm_plant = np.convolve(dw_delay, lpf_den)

    # --- Performance Plotting ---
    freq = np.logspace(-1, np.log10(fs/2), 1000)
    rtf_mag = controller.RTF(mode=0, fs=fs, freq=freq, dm=dm_plant, nw=nw_plant, dw=1.0, plot=False)
    ntf_mag = controller.NTF(mode=0, fs=fs, freq=freq, dm=dm_plant, nw=nw_plant, dw=1.0, plot=False)

    plt.figure(figsize=(10, 5))
    plt.loglog(freq, rtf_mag, label='RTF (Rejection)', linewidth=2)
    plt.loglog(freq, ntf_mag, label='NTF (Noise)', linewidth=2, color='red')
    idx_cross = np.argmin(np.abs(rtf_mag - 1))
    plt.axvline(freq[idx_cross], color='green', linestyle='--', linewidth=1.0,
                label=f'Bandwidth (0 dB): {freq[idx_cross]:.1f} Hz')
    plt.title(f'Frequency Response (Gain={gain}, Delay={delay_frames}f)')
    plt.legend(loc='lower left')
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()

    # --- Stability Analysis (Nyquist) ---
    ol_num = np.convolve(controller.num[0], nw_plant)
    ol_den = np.convolve(controller.den[0], dm_plant)

    open_loop_obj = IirFilterData(ordnum=[len(ol_num)], ordden=[len(ol_den)], num=[ol_num], den=[ol_den])
    
    plt.figure(figsize=(6, 6))
    open_loop_obj.nyquist_plot(dt=1/fs, unit_circle=True)
    plt.show()

Interpreting the Results
------------------------

Frequency Response (Bode/Log-Log)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Bandwidth**: The frequency where the RTF crosses 1 (0 dB). It represents the limit of correction.
* **Resonance (Overshoot)**: A high peak in the NTF indicates that the system is close to instability and will significantly amplify WFS noise.

Nyquist Plot
^^^^^^^^^^^^
The Nyquist plot shows the stability margin by observing how close the open-loop response (:math:`CP`) gets to the critical point (:math:`(-1, 0)`).

* **Stability**: The system is stable if the point (:math:`(-1, 0)`) is not encircled.
* **Margins**: The distance from the curve to (:math:`(-1, 0)`) on the unit circle defines the **Phase Margin**, while the distance on the real axis defines the **Gain Margin**.

Note on Library Versions
------------------------

The ``control`` library has changed its API significantly across versions:

* **Version < 0.9**: Functions like ``nyquist_plot`` returned a tuple ``(real, imag, omega)``.
* **Version 0.9.x**: ``nyquist_plot`` returned the number of encirclements (int).
* **Version 0.10+**: Plotting functions return a ``ControlPlot`` object.

The :class:`IirFilterData` methods are designed to handle these variations internally, 
always returning the numerical data arrays for consistency.