.. _elt_petal_mmse_reconstructor_tutorial:

Building an MMSE Petal Reconstructor from a KL Modal Basis
==============================================================

This is Part 2 of the segmented-pupil tutorial series. It uses the pupil
mask, petal influence functions, and KL modal basis built in
:ref:`elt_segmented_dm_tutorial` (Part 1) to build a **reconstructor from
KL-mode commands to petal-piston estimates**: the piece a closed-loop
simulation needs to know the current petal state without depending on any
project-specific calibration file.

**What you'll learn:**

* Why the petal-to-mode transform must be built *consistently* with the
  modal basis it will operate on, and how to guarantee that starting from
  the shared-mask design of Part 1
* Building a petals-to-modes interaction matrix from two influence-function
  sets that live on the same pixel grid
* Why a plain geometric projection is not enough, and what the *M* in MMSE
  buys you
* Computing a turbulence covariance matrix in modal space with
  :func:`compute_ifs_covmat`, and a reconstructor with
  :func:`compute_mmse_reconstructor`
* Validating the reconstructor statistically, against atmosphere-like
  turbulence plus known injected petal offsets, before ever wiring it into
  a closed loop

**Prerequisites:**

* :ref:`elt_segmented_dm_tutorial` completed: the full run, not a
  shrunk-down version. This tutorial restores its products by tag and does
  not regenerate them, and (see the
  :ref:`note on modal bandwidth <elt_modal_bandwidth_note>` in Part 1) a
  modal basis with too little spatial bandwidth cannot represent the
  petal signal it needs to reconstruct, regardless of how correct the
  reconstructor code is
* Basic familiarity with modal wavefront reconstruction (interaction
  matrices, reconstruction matrices)

The problem
--------------

In closed loop, a pyramid WFS gives you a residual expressed in whatever
modal basis the loop controls (here, KL-mode coefficients). But the
Soft-Limiter needs to know something the WFS does not report directly: the
current *petal-piston* state, i.e. how the six independent primary-mirror
support structures are offset from each other. That state is buried inside
the KL-mode residual, mixed in with ordinary atmospheric turbulence.

We need a matrix, call it :math:`W`, that turns a KL-mode vector into a
petal-piston estimate:

.. math::

   \hat{p} = W \cdot m

and (for the reverse direction, used later to inject a petal correction
back into the loop as a DM command) an interaction matrix :math:`A` that
turns a petal offset into the KL-mode vector it would produce:

.. math::

   m = A \cdot p

Why a plain projection is not enough
-----------------------------------------

The geometrically obvious way to get :math:`A` is to project the piston
pattern of each petal onto the KL basis, and to get :math:`W` by inverting
:math:`A`. That works in the noise-free case, but a real KL-mode vector is
never *just* the petal signal: it is the petal signal plus whatever
atmospheric turbulence happens to be present that frame, and the two are
not separable mode-by-mode. A plain pseudo-inverse of :math:`A` would treat
every KL mode as equally informative about the petal state, when in
reality the high-order modes carry little turbulent power and are mostly
noise for this estimation problem.

The **MMSE** (Minimum Mean Square Error) reconstructor instead down-weights
each mode according to how much genuine turbulence power it is expected to
carry, using the actual mode covariance as a statistical prior. Concretely:

.. math::

   W_{\rm mmse} = \left( A^T C_m^{-1} A + C_p^{-1} \right)^{-1} A^T C_m^{-1}

where :math:`C_p` is the prior covariance of the petal state we are trying
to estimate, and :math:`C_m` is the covariance of the KL-mode turbulence
that competes with the petal signal. SPECULA implements this directly as
:func:`compute_mmse_reconstructor`; the work in this tutorial is building
:math:`A`, :math:`C_p`, and :math:`C_m` correctly, not the estimator
formula itself.

.. note::

   **A cautionary tale.** Building :math:`A` and :math:`W` from two
   *independently generated* petal/modal bases, rather than from the single
   shared pupil mask Part 1 sets up, is a real trap: a subtly different
   pixel rasterization between the two bases leaves them not quite
   consistent inverses of each other, and the resulting cross-talk is hard
   to diagnose after the fact. It only shows up as a puzzling bias deep
   inside a closed-loop run. This tutorial avoids that by building both
   :math:`A` and :math:`W` from the petal and KL bases generated together
   in Part 1, on the same shared mask.

Step 1: Restore the products of Part 1
-------------------------------------------

.. code-block:: python

    import specula
    specula.init(-1)

    from specula import xp, np, cpuArray
    from specula.calib_manager import CalibManager
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.ifunc_inv import IFuncInv

    calib = CalibManager('./calib_elt_segmented_dm_tutorial')

    petal_ifunc_obj = IFunc.restore(calib.filename('ifunc', 'ELT39_6petals'))
    kl_ifunc_obj = IFunc.restore(calib.filename('ifunc', 'ELT39_KL4000'))
    kl_inv_obj = IFuncInv.restore(calib.filename('ifunc', 'ELT39_KL4000_inv'))

    petal_ifunc = petal_ifunc_obj.influence_function       # (6, n_valid)
    kl_basis = kl_ifunc_obj.influence_function              # (n_modes, n_valid)
    influence_function_inv = kl_inv_obj.ifunc_inv            # (n_valid, n_modes)
    pupil_mask = kl_ifunc_obj.mask_inf_func

    n_petals = petal_ifunc.shape[0]
    n_modes, n_valid = kl_basis.shape

    assert petal_ifunc.shape[1] == n_valid, \
        "petal and KL bases are on different pixel grids!"

    print(f'petal_ifunc {petal_ifunc.shape}, kl_basis {kl_basis.shape}, '
          f'influence_function_inv {influence_function_inv.shape}')

``petal_ifunc`` and ``kl_basis`` should agree on the second dimension, the
number of valid pixels in the shared mask from Part 1, with the first
dimension of ``kl_basis`` at 4000 (the modes kept) and
``influence_function_inv`` simply that shape transposed.

The ``assert`` is the same shared-mask safety check from Part 1. Restoring
by tag does not remove the risk of a mismatch, it just moves it later in
time, so it pays to check again here.

Step 2: The petals-to-modes interaction matrix
----------------------------------------------------

We use the six raw petal influence functions from Part 1 directly (piston
= 1 inside a petal, 0 elsewhere); no separate "relative petal" basis is
needed. The pyramid cannot see an absolute piston offset applied
identically to all six petals, so that global mode is invisible to any
reconstructor built this way; the standard fix is to treat one petal as a
reference and describe the other five *relative to it*, which we do simply
by dropping the last petal's row before projecting (checked explicitly
below):

.. code-block:: python

    petal_ifunc_5 = petal_ifunc[:n_petals - 1, :]     # (5, n_valid), petal 6 is the reference

    # Project the raw piston pattern of each petal onto the KL basis: this
    # is the same pseudo-inverse trick as the IFuncInv from Part 1, just
    # applied here to a different signal (a petal shape) instead of to a
    # phase screen.
    intmat = petal_ifunc_5 @ influence_function_inv    # (5, n_modes)
    intmat = intmat.T                                     # (n_modes, 5), the A matrix

    print('intmat (n_modes, 5):', intmat.shape)

.. note::

   **Is dropping one petal equivalent to a proper differential basis?**
   Yes. The six raw petal masks sum, pixel for pixel, to the uniform
   (global piston) pattern, and that pattern projects through
   ``influence_function_inv`` to KL-mode RMS ~3e-17 -- machine-precision
   zero, since the KL basis cannot represent piston. Projected into
   KL-mode space this way, the 6-row petal matrix is therefore rank 5, not
   rank 6 (the raw, un-projected petal masks themselves are full rank 6:
   it is specifically the projection through a piston-free basis that
   creates the redundancy). Dropping petal 6 discards exactly that
   redundant direction and nothing else.

Step 3: Turbulence covariance in KL-mode space
----------------------------------------------------

This is :math:`C_m`: how much power ordinary atmospheric turbulence puts
into each KL mode (and how the modes correlate), independent of any petal
signal. :func:`compute_ifs_covmat`, the same routine Part 1 uses internally
to rank the KL basis, gives us this directly from the forward KL basis and
an ``r0``/``L0`` pair:

.. code-block:: python

    from specula.lib.modal_base_generator import compute_ifs_covmat

    telescope_diameter = 39.0
    r0 = 0.15
    L0 = 25.0

    c_modes = compute_ifs_covmat(pupil_mask, telescope_diameter, kl_basis,
                                  r0, L0, oversampling=2, xp=xp, dtype=xp.float32)
    print('c_modes:', c_modes.shape)

::

    c_modes: (4000, 4000)

Step 4: Petal prior covariance
------------------------------------

This is :math:`C_p`: the prior covariance of the petal state itself, as
opposed to :math:`C_m`, the covariance of what competes with it. Following
the paper this tutorial accompanies (Sect. 3, Eq. 2), we use a plain
isotropic, deliberately *uninformative* prior, :math:`C_p = \sigma_p^2 I`:

.. code-block:: python

    petal_sigma_nm = 800.0   # see note below: the exact value barely matters

    c_petals = (petal_sigma_nm ** 2) * xp.eye(n_petals - 1, dtype=xp.float32)
    print('c_petals:', c_petals.shape)

::

    c_petals: (5, 5)

.. note::

   The paper shows the output of this static estimator is essentially
   invariant to :math:`\sigma_p` across more than six orders of magnitude (8 nm to
   8e6 nm): for any sufficiently uninformative prior, the estimator
   converges to the Gauss-Markov (minimum-variance-unbiased) limit, so
   ``petal_sigma_nm`` is not really a tunable parameter of the method; any
   plausible order of magnitude works.

   The same reasoning covers the choice of prior structure. If the six
   absolute petals were independent with variance :math:`\sigma_p^2`, the
   exact covariance of the five petals relative to the reference would be
   :math:`\sigma_p^2 (I + \mathbf{1}\mathbf{1}^T)`, not :math:`\sigma_p^2 I`.
   For :math:`\sigma_p` of a few hundred nm or more the two priors give
   the same reconstructor to within numerical precision, so we use the
   simpler diagonal form.

Step 5: Computing and checking the reconstructor
------------------------------------------------------

.. code-block:: python

    from specula.lib.mmse_reconstructor import compute_mmse_reconstructor

    mmse_recmat = compute_mmse_reconstructor(
        intmat, c_petals, xp, xp.float32,
        noise_variance=None, c_noise=c_modes, c_inverse=False)
    print('mmse_recmat:', mmse_recmat.shape)

    # Sanity check: applied to the noise-free interaction matrix, the
    # reconstructor should recover a (5, 5) identity. This is the check
    # that would catch a shared-mask mismatch of the kind described above,
    # and it also confirms that dropping the sixth (reference) petal,
    # rather than building a separate 5-mode "relative petal" basis, was a
    # safe simplification: there is no leftover cross-talk.
    print(np.round(cpuArray(mmse_recmat @ intmat), 3))

::

    mmse_recmat: (5, 4000)
    [[ 1. -0. -0. -0. -0.]
     [-0.  1. -0. -0. -0.]
     [-0. -0.  1. -0. -0.]
     [-0. -0. -0.  1. -0.]
     [-0. -0. -0. -0.  1.]]

(Computing ``c_modes`` and ``mmse_recmat`` at this scale takes a few
minutes, comparatively cheap next to Part 1.)

Saving the reconstructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from specula.data_objects.recmat import Recmat

    rec_obj = Recmat(mmse_recmat)
    rec_filename = calib.filename('rec', 'ELT39_modes_to_petals_mmse')
    rec_obj.save(rec_filename, overwrite=True)

    im_obj = Recmat(intmat)
    im_filename = calib.filename('rec', 'ELT39_petals_to_modes')
    im_obj.save(im_filename, overwrite=True)

Step 6: Statistical validation with atmosphere-like commands
--------------------------------------------------------------------

The identity check above only proves the reconstructor is self-consistent
in the noise-free case. What matters in practice is how well it recovers a
*known* petal offset once realistic atmospheric turbulence is mixed in --
the static estimation test described in the paper, Sect. 4.3 (Fig. 4 and
Fig. 5): known petal offsets and turbulence are injected together, and the
estimator error is compared against the injected truth.

Code, using :class:`AtmoRandomPhase`-generated turbulence plus injected,
known-truth random petal offsets:

.. code-block:: python

    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.base_value import BaseValue
    from specula.processing_objects.atmo_random_phase import AtmoRandomPhase

    n_realizations = 300
    seeing_arcsec = 0.8
    np.random.seed(1234)

    dim = pupil_mask.shape[0]
    mask_bool = cpuArray(pupil_mask) > 0
    simul_params = SimulParams(time_step=1.0, pixel_pupil=dim,
                                pixel_pitch=telescope_diameter / dim)
    pupilstop = Pupilstop(simul_params, input_mask=mask_bool.astype(np.float32))

    atmo = AtmoRandomPhase(simul_params, L0=L0, data_dir='./atmo_phasescreens',
                            wavelengthInNm=500.0, pixel_phasescreens=2048, seed=1,
                            update_interval=1)
    seeing = BaseValue(value=xp.array([seeing_arcsec], dtype=xp.float32))
    atmo.inputs['pupilstop'].set(pupilstop)
    atmo.inputs['seeing'].set(seeing)
    atmo.setup()

    # Known-truth petal offsets, re-referenced to petal 6 exactly as in Step 2.
    true_petals = np.random.randn(n_realizations, n_petals).astype(np.float32) * petal_sigma_nm
    true_petals -= true_petals[:, [n_petals - 1]]
    true_petals = xp.asarray(true_petals)
    true_rel = true_petals[:, :n_petals - 1]

    # Project the injected petal offsets into KL-mode space the same way a
    # real petal step would appear in the modal residual of the loop.
    petal_modes = (true_petals @ petal_ifunc) @ influence_function_inv

    turb_modes = xp.zeros((n_realizations, n_modes), dtype=xp.float32)
    for i in range(n_realizations):
        t = atmo.seconds_to_t(float(i))
        seeing.generation_time = t
        pupilstop.generation_time = t
        atmo.check_ready(t)
        atmo.trigger()
        atmo.post_trigger()
        phase_vec = atmo.outputs['out_layer'].phaseInNm[mask_bool]
        turb_modes[i] = phase_vec @ influence_function_inv

    est_rel = (turb_modes + petal_modes) @ mmse_recmat.T
    err = cpuArray(est_rel - true_rel)

    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    print(f'RMSE = {rmse:.2f} nm, bias = {bias:.2f} nm')

Compare the resulting bias and RMS error against Fig. 4 and Sect. 4.3 of
the paper, evaluated there on the real ELT-class basis rather than a
stand-in.

Summary and what's next
---------------------------

Starting from the products of Part 1, this tutorial built:

* a petals-to-modes interaction matrix (:math:`A`), consistent by
  construction with the KL basis because both come from the same shared
  pupil mask
* a turbulence covariance matrix in KL-mode space
* a petal-piston prior covariance
* an MMSE reconstructor from KL-mode commands to relative petal-piston
  estimates (:math:`W_{\rm mmse}`), saved as a :class:`Recmat`
* a statistical validation against atmosphere-like turbulence plus known
  petal offsets, confirming the reconstructor is unbiased before it is ever
  used in closed loop

What remains for a full closed-loop demonstration, calibrating the
pyramid WFS itself (pupil geometry, interaction matrix, reconstruction
matrix) and running the loop with and without the Soft-Limiter, is
covered in :ref:`elt_petal_soft_limiter_closed_loop_tutorial` (Part 3).
