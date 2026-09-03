
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

Display Updates
---------------

Displays use Matplotlib's interactive rendering system and are refreshed only when their underlying data changes. Multiple display updates are aggregated and a single redraw is performed for each simulation iteration, minimizing rendering overhead.

For high-frequency simulations, displays should be considered diagnostic tools since they may be updated at a lower rate than the simulation itself to reduce visualization costs.

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



