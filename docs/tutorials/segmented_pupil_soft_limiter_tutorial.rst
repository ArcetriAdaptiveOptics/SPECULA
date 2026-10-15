.. _segmented_pupil_soft_limiter_tutorial:

Segmented Pupil and Soft-Limiter Tutorial
==========================================

This tutorial shows how to control **petal (piston) modes** -- the
differential piston between the independent support structures of a
segmented, ELT-class primary mirror -- with the *Soft-Limiter* leak-gain
mechanism, in closed loop with a pyramid WFS.

The problem
-------------

A pyramid WFS cannot distinguish a petal piston offset from the same
offset plus an integer number of sensing wavelengths: the two produce an
identical wrapped signal. A standard integrator therefore has no
information to correct that integer part, and a naive attempt to
compensate it will drift arbitrarily. The Soft-Limiter addresses this with
a leak term on the petal subspace, chosen so that the loop retains any
*fractional* part of a physical step -- see the paper this tutorial
series accompanies for the full derivation:

    Agapito, G. et al., *Temporal regularization of petal modes in SCAO
    systems: the Soft-Limiter approach*, submitted to Astronomy &
    Astrophysics.

This tutorial is split into three parts:

* **Part 1 -- Building the segmented pupil and modal basis**, a
  self-contained, reusable calibration-generation tutorial:
  :ref:`elt_segmented_dm_tutorial`. It builds an ELT-class pupil (6
  petals, 39 m) and a 4000-mode KL modal basis from scratch, without
  depending on any project-specific calibration file.
* **Part 2 -- Building an MMSE petal reconstructor**:
  :ref:`elt_petal_mmse_reconstructor_tutorial`. It uses the shared pupil
  mask and KL basis from Part 1 to build a self-consistent reconstructor
  from KL-mode commands to petal-piston estimates, and validates it
  statistically against atmosphere-like turbulence before it is ever used
  in closed loop.
* **Part 3 -- Closed-loop simulation with the Soft-Limiter**:
  :ref:`elt_petal_soft_limiter_closed_loop_tutorial`. It calibrates the
  pyramid WFS (pupil geometry, interaction matrix, reconstruction
  matrix), wires everything into a closed-loop simulation YAML file, and
  compares a Classical loop against the Soft-Limiter on the same
  atmosphere-only turbulence -- no injected petal disturbance is needed --
  showing the Soft-Limiter confining the fringe-jump ambiguity to a
  smaller range with a correspondingly higher and steadier Strehl ratio.

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* :ref:`elt_segmented_dm_tutorial` and
  :ref:`elt_petal_mmse_reconstructor_tutorial` completed -- the full,
  ELT-class run, not a shrunk-down one, so that the pupil,
  influence-function, modal-basis and reconstructor products they produce
  are available
* Basic understanding of adaptive optics concepts (influence functions,
  modal bases, wavefront sensing, closed-loop control)
