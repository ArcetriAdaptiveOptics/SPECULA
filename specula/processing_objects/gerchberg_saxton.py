import logging
import math

from specula import cpuArray, np
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.ifunc import IFunc
from specula.lib.toccd import toccd
from specula.processing_objects.lift import Lift


class GerchbergSaxton(Lift):
    """
    Gerchberg-Saxton (GS) phase retrieval processing object.

    Drop-in alternative to :class:`~specula.processing_objects.lift.Lift`:
    it takes the same parameters, the same ``in_pixels`` input and produces
    the same ``out_pistons`` / ``out_zern`` outputs (nm, wrapped to
    [-lambda/2, +lambda/2), reference ``ref_zern_amp`` removed), so that the
    two estimators can be swapped in a configuration file and compared
    directly.

    The forward model (pupil sampling, FFT padding, half-pixel tilt, binning
    to the detector pixel scale and cropping to ``npix_side``) is inherited
    from Lift, so that GS and LIFT see exactly the same image formation.

    Algorithm (error-reduction form of GS, Gerchberg & Saxton 1972,
    Fienup 1982), with the known reference aberration ``ref_zern_amp`` used
    as phase diversity and as starting point (this lifts the twin-image
    sign ambiguity, exactly as in LIFT):

    1. pupil field ``E = mask * exp(i * phi)``
    2. focal field ``F = FFT(E)``; model image binned to detector pixels
    3. focal constraint: inside the ROI (``cropped_size`` around the
       measured PSF centre) the modulus of ``F`` is rescaled by
       ``sqrt(I_meas / I_model)`` (computed on detector pixels and spread
       back to the FFT grid, i.e. a generalised projection that handles
       binning / sub-Nyquist pixels); outside the ROI ``F`` is left free
    4. back to the pupil, keep the phase, impose the known amplitude
       (optionally projecting the phase onto the modal basis)

    The phase is kept unwrapped by accumulating wrapped increments, then
    projected onto the modal basis (pistons + Zernike) with a weighted
    least-squares fit.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 nPistons: int,
                 nZern: int,
                 wavelengthInNm: float,
                 pix_scale: float,
                 cropped_size: int,
                 ifunc: IFunc,
                 ref_zern_amp,
                 npix_side: int = None,
                 ron: float = 0.0,
                 n_iter: int = 100,
                 fft_res: int = 2,
                 fix: bool = False,
                 modal_projection: bool = False,
                 rel_tol: float = 1e-4,
                 abs_tol: float = 1e-4,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters are identical to :class:`Lift`, plus:

        n_iter : int [1], optional
            Maximum number of GS iterations. Defaults to 100 (GS converges
            more slowly than LIFT's Gauss-Newton iterations).
        modal_projection : bool, optional
            If True, at each iteration the pupil phase is projected onto the
            modal basis (modal-constrained GS). If False (default), the
            classic zonal GS is run and the modal projection is done only
            at the end.
        rel_tol : float, optional
            Stop when the relative improvement of the focal-plane error
            metric is below this value. Defaults to 1e-4.
        abs_tol : float [rad], optional
            Stop when the mean absolute change of the modal coefficients in
            one iteration is below this value. Defaults to 1e-4.
        """
        super().__init__(simul_params=simul_params,
                         nPistons=nPistons,
                         nZern=nZern,
                         wavelengthInNm=wavelengthInNm,
                         pix_scale=pix_scale,
                         cropped_size=cropped_size,
                         ifunc=ifunc,
                         ref_zern_amp=ref_zern_amp,
                         npix_side=npix_side,
                         ron=ron,
                         n_iter=n_iter,
                         fft_res=fft_res,
                         fix=fix,
                         target_device_idx=target_device_idx,
                         precision=precision)
        self.modal_projection = bool(modal_projection)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.n_iter_done = 0
        self.error_history = []
        self.logger.info(f"Gerchberg-Saxton initialized with {self.nmodes} modes")

    # ------------------------------------------------------------------
    # Modal basis
    # ------------------------------------------------------------------
    def set_modalbase(self, modalbase, mask2d):
        super().set_modalbase(modalbase, mask2d)
        # Weighted least-squares projector on the (resampled) pupil
        valid = self.mask > 0.5
        self._valid_idx = self.xp.nonzero(valid)
        w = self.xp.sqrt(self.mask[self._valid_idx])
        M = self.modesCube[:, self._valid_idx[0], self._valid_idx[1]].T  # (nvalid, nmodes)
        Mw = M * w[:, None]
        self._fit_matrix = (self.xp.linalg.pinv(Mw) * w[None, :]).astype(self.dtype)

    def fitModes(self, phase):
        """Weighted least-squares projection of an (unwrapped) pupil phase on the modal basis [rad]."""
        return self._fit_matrix @ phase[self._valid_idx]

    @staticmethod
    def _wrap(x, xp):
        return (x + xp.pi) % (2 * xp.pi) - xp.pi

    # ------------------------------------------------------------------
    # Geometry helpers (adjoint of the Lift forward model)
    # ------------------------------------------------------------------
    def ift_ft2(self, F):
        """Inverse of :meth:`Lift.ft_ft2`: back to the pupil grid."""
        pad = (self.fftSize - self.gridSize) // 2
        E = self.xp.fft.ifft2(self.xp.fft.ifftshift(F)) * self.fftSize
        return E[pad:pad + self.gridSize, pad:pad + self.gridSize]

    def _binned_shape(self):
        return (round(self.fftSize / self.settings.bin_fact),
                round(self.fftSize / self.settings.bin_fact))

    def _detector_window(self):
        """
        Slices mapping the binned FFT image onto the npix_side detector frame,
        replicating Lift.crop_or_enlarge_around_peak with peak at shape//2.
        Returns (src_slices, dst_slices).
        """
        new_shape = self._binned_shape()
        side = int(self.npix_side)
        half = side // 2
        src, dst = [], []
        for n in new_shape:
            peak = n // 2
            start = max(0, peak - half)
            end = min(n, peak + half)
            length = end - start
            pad = (side - length) // 2 if length < side else 0
            src.append(slice(start, end))
            dst.append(slice(pad, pad + length))
        return tuple(src), tuple(dst)

    def _roi_slices(self, center, shape):
        """Same indices as Lift.crop, clipped to the frame."""
        side = self.cropped_size
        r0 = int(math.ceil(float(center[1]) - side // 2))
        r1 = int(math.ceil(float(center[1]) + side // 2))
        c0 = int(math.ceil(float(center[0]) - side // 2))
        c1 = int(math.ceil(float(center[0]) + side // 2))
        return (slice(max(r0, 0), min(r1, shape[0])),
                slice(max(c0, 0), min(c1, shape[1])))

    def _ratio_to_fft_grid(self, ratio_det):
        """Spread a detector-plane modulus ratio back onto the FFT grid."""
        src, dst = self._detector_window()
        ratio_bin = self.xp.ones(self._binned_shape(), dtype=self.dtype)
        ratio_bin[src] = ratio_det[dst]
        ratio_fft = toccd(ratio_bin, (self.fftSize, self.fftSize), xp=self.xp, set_total=0)
        return ratio_fft.astype(self.dtype, copy=False)

    # ------------------------------------------------------------------
    # Phase retrieval
    # ------------------------------------------------------------------
    def phaseEstimation(self, psf_orig, relTol=None, absTol=None):
        if not self.modes:
            raise Exception('Modal base has not been set yet')
        relTol = self.rel_tol if relTol is None else relTol
        absTol = self.abs_tol if absTol is None else absTol
        xp = self.xp

        psf = self.setPsf(psf_orig).astype(self.dtype, copy=False)
        center = self.calcCenter(psf)
        self.setRefTT(float(center[0]), float(center[1]), float(psf.shape[0]))
        roi = self._roi_slices(center, psf.shape)

        meas_roi = xp.maximum(psf[roi], 0)
        meas_flux = meas_roi.sum()
        sqrt_meas = xp.sqrt(meas_roi)

        # Phase estimate relative to the LIFT-like reference phase
        # (ref_zern_amp + CoG tip/tilt + half-pixel tilt), kept unwrapped
        phase_base = self.phaseLIFT(xp.zeros_like(self.mask)) + \
            self._get_tlt_f(self.gridSize, self.settings.fft_size)
        estimate = xp.zeros_like(self.mask)
        coeffs = xp.zeros(self.nmodes, dtype=self.dtype)

        olderror = float(1e18)
        self.error_history = []
        n_done = 0
        for i in range(self.n_iter):
            n_done = i + 1
            # 1-2. Pupil -> focal plane
            _, F = self.complexField(estimate)
            I_det = self.focalPlaneImageFromFFT(F, set_flux=1.0)
            model_roi = I_det[roi]
            model_roi *= meas_flux / xp.maximum(model_roi.sum(), xp.finfo(self.dtype).tiny)

            # Focal-plane error metric (before the projection)
            newerror = float(cpuArray(((sqrt_meas - xp.sqrt(model_roi))**2).sum() / meas_flux))
            self.error_history.append(newerror)

            # 3. Focal-plane modulus constraint (generalised projection)
            eps = max(float(self.ron)**2, 1e-6 * float(cpuArray(model_roi.max())))
            ratio_det = xp.ones(I_det.shape, dtype=self.dtype)
            ratio_det[roi] = xp.sqrt((meas_roi + eps) / (model_roi + eps))
            F = F * self._ratio_to_fft_grid(ratio_det)

            # 4. Back to the pupil plane, keep phase, impose known amplitude
            E = self.ift_ft2(F)
            delta = self._wrap(xp.angle(E) - phase_base - estimate, xp).astype(self.dtype)
            delta *= (self.mask > 0.5)
            estimate = estimate + delta

            new_coeffs = self.fitModes(estimate)
            if self.modal_projection:
                estimate = self.phaseFromCoeffs(new_coeffs)
            delta_coeffs = cpuArray(new_coeffs - coeffs)
            coeffs = new_coeffs

            improvement = (olderror - newerror) / olderror
            olderror = newerror
            logging.debug(f"GS iteration {i}: error={newerror}, coeffs={cpuArray(coeffs)}")
            if i > 0 and (abs(improvement) < relTol or np.mean(np.abs(delta_coeffs)) < absTol):
                logging.debug(f'GS criterion reached at iteration {i+1}')
                break

        self.n_iter_done = n_done

        out = self.to_xp(coeffs, dtype=self.dtype, force_copy=True)
        out[0 + self.nPistons] += self.ref_tip
        out[1 + self.nPistons] += self.ref_tilt
        final_coeffs = out * self.wavelengthInNm / (2 * np.pi)
        final_coeffs = self._wrap_coefficients(final_coeffs, self.wavelengthInNm)
        return estimate, final_coeffs, n_done
