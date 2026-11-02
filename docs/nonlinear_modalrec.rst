.. _nonlinear_modalrec:

Nonlinear modal reconstructor
=============================

``NonlinearModalrec`` is the linear ``Modalrec`` plus a per-mode correction of
the sensor saturation, for sensors whose response saturates mode by mode.
``NonlinearResponseCalibrator`` measures the response of each mode,
injected alone at a grid of amplitudes: the slopes projected on the mode's
small-signal interaction vector. The curves are saved as a
``NonlinearCalibration``. At each step the reconstructor inverts the curves on
the linear estimate, then adds one linear correction of the signal the
curves do not explain (see the class documentation).

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
     intmat_object:     'my_intmat'
     calib_tag:         'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
       in_commands:     'pushpull.output'

and the reconstructor replaces ``Modalrec``:

.. code-block:: yaml

   modalrec:
     class:             'NonlinearModalrec'
     recmat_object:     'my_recmat'
     intmat_object:     'my_intmat'
     nonlinear_calib_object: 'my_nonlinear_calib'
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs: ['out_modes']

The response curves must increase with the amplitude: the calibrator warns
about the modes whose curve rolls over, and the reconstructor refuses them.
Reduce the largest amplitudes, or calibrate fewer modes (``first_mode`` and
``nmodes`` select them). Modes outside the calibration are reconstructed
linearly. ``NonlinearModalrec`` can be accelerated.
