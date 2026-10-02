
Displays
========

Display objects provide real-time visualization of simulation data and results. They are typically connected to data objects and processing objects to monitor the state of the adaptive optics system during execution.

All display objects derive from `BaseDisplay` and use Matplotlib for rendering. Displays are updated asynchronously and can also be recorded using the `DisplayRecorder` processing object.

Phase Displays
--------------

Displays based on phase data represented as optical path difference maps.

* ``PhaseDisplay`` - Single phase screen visualization
* ``DoublePhaseDisplay`` - Comparison of two phase maps

Typical applications include:

* Atmospheric phase screens
* Residual wavefront error monitoring
* Deformable mirror surface visualization
* Reconstructed wavefront inspection

Image Displays
--------------

Displays for 2D pixel-based data products.

* `PixelsDisplay` - Generic image display
* `PixelsPupDisplay` - Pupil-aware image display
* `PsfDisplay` - PSF visualization

Typical applications include:

* Detector images
* Wavefront sensor frames
* PSF monitoring
* Intensity distributions

Plot Displays
-------------

Displays for scalar values, vectors, and temporal evolution of quantities.

* ``PlotDisplay`` - Scalar or curve plotting
* ``PlotVectorDisplay`` - Vector and time-history visualization
* ``ModesDisplay`` - Modal coefficient plotting

Typical applications include:

* Modal evolution
* Residual error tracking
* Performance metrics
* Time-series analysis

Wavefront Sensor Displays
-------------------------

Specialized displays for wavefront sensor measurements.

* `SlopescDisplay` - Slope visualization and diagnostics

Typical applications include:

* Slope inspection
* Centroid diagnostics
* WFS monitoring
* Reconstruction debugging

Display Recording
-----------------

The ``DisplayRecorder`` processing object allows one or more display windows to be recorded to an MP4 video file during execution.

The recorder can capture multiple display windows simultaneously, combining them into a single video stream by stacking horizontally or vertically.
It cannot be used together with :ref:`asynchronous displays <async_displays>`.

Display Updates
---------------

Displays use Matplotlib's interactive rendering system and are refreshed only when their underlying data changes. Multiple display updates are aggregated and a single redraw is performed for each simulation iteration, minimizing rendering overhead.

For high-frequency simulations, displays should be considered diagnostic tools since they may be updated at a lower rate than the simulation itself to reduce visualization costs.

.. _async_displays:

Asynchronous displays
---------------------

By default, displays are drawn in the simulation process, and the simulation waits for each redraw.
Drawing is often much slower than the simulation step, so a few displays can dominate the
simulation time. The ``--async-displays`` command line flag moves all displays to a separate process:

.. code-block:: bash

    specula params.yml --async-displays

When embedding SPECULA in a Python program, the same behavior is selected with the
``async_displays=True`` argument of :class:`specula.simul.Simul` or :func:`specula.main_simul`.
The display process is started with the multiprocessing *spawn* method, which imports the main
script again in the new process, so the script must protect its top-level code with the usual guard:

.. code-block:: python

    import specula

    if __name__ == '__main__':
        specula.main_simul(['params.yml'], async_displays=True)

Without the guard, the display process fails at startup. The simulation still runs, and an error
is logged saying that the displays will not be updated.

With this flag:

* the simulation does not draw anything. At each trigger, a display copies its inputs to the CPU
  and sends them to the display process through a queue. The simulation never waits for the displays.
* a single display process, started together with the simulation, holds a copy of each display object,
  built with the same parameters, and does all the drawing, using the same window numbers.
  If the simulation has no displays, no process is started.
* the display process runs on the CPU only, and does not use GPU memory.
* when the display process is busy, image displays (``PhaseDisplay``, ``PixelsDisplay``, ``PsfDisplay``, etc.)
  skip updates: a new frame is dropped, before copying it, while at most two frames per display are waiting
  to be drawn. This keeps the memory used by the queue bounded even with large arrays.
  The number of skipped updates is logged at the end of the simulation.
* displays that build a time history (``PlotDisplay``, ``PlotVectorDisplay``) never skip updates, so that no
  point is lost. There is no limit to their pending updates: their data is usually small
  (a few values per step), and the display process catches up by applying all pending points before redrawing.
* at the end of the simulation, the display process draws the pending data and exits.

This mode has some limitations:

* a slow display still delays the other displays, since they share the same process (but not the simulation).
* the displays show data with some delay with respect to the simulation.
* image displays that average over time, like the PSD average of ``DoublePhaseDisplay``, average only
  the frames that they receive. Their data is too large to be queued without limits.
* ``DisplayRecorder`` is not supported, since the windows it records live in another process:
  an error is raised if it is used together with ``--async-displays``.

Custom displays that accumulate data over time, and so must receive every update, should set the
``skip_updates`` class attribute to ``False``:

.. code-block:: python

    class MyHistoryDisplay(BaseDisplay):

        skip_updates = False   # keep every point of the history

All the display code runs in the display process, where the figure is. There, the inputs of each
display are set to CPU copies of the simulation data. ``setup()`` is called before the first update,
when the inputs are available, then ``trigger_code()`` is called at each update as usual,
and ``finalize()`` at the end of the simulation. In the simulation process, the display object only
checks its inputs in ``setup()``, and ``setup()`` and ``finalize()`` of derived classes are not called.

Display grouping
----------------

Multiple displays can be grouped into a single window and updated together. Each display accepts optional *window* and *subplots* parameters. All displays with the same value for *window* will be grouped together, arranged according to the *subplot* parameter which has the same syntax as Matplotlib::

  modes_disp:
    class: 'ModesDisplay'
    inputs:
      modes: ['modalrec.out_modes']
    title: 'Modes'
    window: 1
    subplot: 221
    outputs: ['out_window_id']

  phase_disp:
    class: 'PhaseDisplay'
    inputs:
      phase: ['atmo.out_phase']
    title: 'Phase'
    window: 1
    subplot: 222
    outputs: ['out_window_id']

  psf_disp:
    class: 'PsfDisplay'
    inputs:
      psf: ['prop.out_psf']
    title: 'PSF'
    window: 1
    subplot: 223
    outputs: ['out_window_id']

  slopes_disp:
    class: 'SlopescDisplay'
    inputs:
      slopes: ['wfs.out_slopes']
    title: 'Slopes'
    window: 1
    subplot: 224
    outputs: ['out_window_id']
 

This produces a single Matplotlib window arranged as::

    +-----------+-----------+
    | Modes     | Phase     |
    | (221)     | (222)     |
    +-----------+-----------+
    | PSF       | Slopes    |
    | (223)     | (224)     |
    +-----------+-----------+



