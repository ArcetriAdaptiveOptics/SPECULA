.. _nonlinear_modalrec:

Nonlinear modal reconstructor
=============================

``NonlinearModalrec`` is the linear ``Modalrec`` plus a per-mode correction of
the sensor saturation, for sensors whose response saturates mode by mode.
``NonlinearResponseCalibrator`` measures the response of each mode,
injected alone at a grid of amplitudes: the mode's linear reconstruction,
``(recmat @ slopes)[mode]``. The curves are saved as a
``NonlinearCalibration``. At each step the reconstructor computes the linear
estimate with the same recmat and inverts the curves on it: the amplitude of
a mode injected alone is recovered exactly, also when the saturated slopes
change shape. The coupling between modes in the saturated regime is not
corrected.

The curves depend on the residual wavefront during the calibration, as the
optical gains do: a calibration without turbulence overestimates the response
in closed loop. Calibrate on a residual representative of the closed loop,
for example with a turbulence that is partially corrected.

The calibrator is driven one mode at a time, for example by a
``PushPullGenerator`` with the amplitude grid as ``pattern``:

.. code-block:: yaml

   pushpull:
     class:             'PushPullGenerator'
     nmodes:            54
     push_pull_type:    'PUSHPULL'
     amp:               1.0
     constant_amp:      true
     pattern:           [-200., -100., -50., 50., 100., 200.]   # [nm], no zero
     outputs: ['output']

   nonlinear_calibrator:
     class:             'NonlinearResponseCalibrator'
     nmodes:            54
     recmat_object:     'my_recmat'
     calib_tag:         'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
       in_commands:     'pushpull.output'

and the reconstructor replaces ``Modalrec``:

.. code-block:: yaml

   modalrec:
     class:             'NonlinearModalrec'
     recmat_object:     'my_recmat'
     nonlinear_calib_object: 'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs: ['out_modes']

The response curves must increase with the amplitude: the calibrator warns
about the modes whose curve rolls over, and the reconstructor refuses them.
Reduce the largest amplitudes, or calibrate fewer modes (``first_mode`` and
``nmodes`` select them). Modes outside the calibration are reconstructed
linearly. ``NonlinearModalrec`` can be accelerated.

Outside the calibrated range, the estimate stops at the ends of the amplitude
grid: a mode whose linear estimate is beyond the largest response of its curve
is reconstructed as the largest amplitude of the grid (and likewise for the
smallest). This is still larger than the linear estimate of a saturated
sensor, but it underestimates larger amplitudes: the grid should cover the
amplitudes expected in closed loop, as far as the curves stay increasing.
