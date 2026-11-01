from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat
from specula.data_objects.intmat import Intmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.lib.graph_matmul import matmul


class NonlinearModalrec(BaseModalrec):
    """
    Nonlinear modal reconstructor: the linear
    :class:`~specula.processing_objects.modalrec.Modalrec` plus a per-mode
    saturation correction ("parallel channel") and a linear correction of the
    signal it does not explain ("orthogonal channel"). It suits sensors whose
    response saturates mode by mode, such as a non-modulated pyramid.

    At each step:

    1. ``a_lin = recmat @ slopes``: the linear estimate, as Modalrec.
    2. ``a_parallel = calibration.invert(a_lin)``: each mode's linear estimate
       de-saturated with its calibrated response curve (see
       :class:`~specula.data_objects.nonlinear_calibration.NonlinearCalibration`).
    3. ``predicted_response = calibration.forward(a_parallel)``: the response
       that a_parallel produces, from the same curves. Using the curves here,
       not the interaction matrix alone, keeps step 4 from undoing step 2: a
       mode fully explained by its own saturation leaves no residual.
    4. ``a_orthogonal = recmat @ (slopes - intmat @ predicted_response)``: one
       linear correction step on the signal left over, which a per-mode curve
       cannot explain (e.g. coupling between modes).
    5. ``out_modes = a_parallel + a_orthogonal``.

    Modes outside the calibrated range have a linear (identity) curve. With
    identity curves everywhere, a_parallel = a_lin and a_orthogonal is zero
    when recmat is the pseudo-inverse of intmat: the result is Modalrec's.

    Parameters
    ----------
    recmat : Recmat
        Reconstruction matrix (nmodes x nslopes).
    intmat : Intmat
        Small-signal interaction matrix (nslopes x nmodes), the one the
        calibration projects on.
    nonlinear_calib : NonlinearCalibration
        Per-mode response curves. All of them must be strictly increasing.
    nmodes : int, optional
        Number of modes to reconstruct (default: all the modes of the recmat).
    apply_orthogonal_correction : bool
        Add the orthogonal channel (steps 3-4); if False, out_modes = a_parallel.
    """

    def __init__(self,
                 recmat: Recmat,
                 intmat: Intmat,
                 nonlinear_calib: NonlinearCalibration,
                 nmodes: int = None,
                 apply_orthogonal_correction: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        rec_nmodes = recmat.recmat.shape[0]
        self.nmodes = nmodes if nmodes is not None else rec_nmodes
        if self.nmodes > rec_nmodes:
            raise ValueError(f'Requested nmodes={self.nmodes} exceeds recmat nmodes={rec_nmodes}')
        if self.nmodes > intmat.nmodes:
            raise ValueError(f'Requested nmodes={self.nmodes} exceeds intmat nmodes={intmat.nmodes}')
        bad = nonlinear_calib.non_monotonic_modes()
        if bad:
            raise ValueError(f'the nonlinear calibration cannot be inverted: the response of modes {bad} '
                             f'is not strictly increasing with the amplitude')

        self.nonlinear_calib = nonlinear_calib
        self.apply_orthogonal_correction = apply_orthogonal_correction
        # Contiguous copies of the parts used, as graph_matmul needs
        self.rec = self.xp.ascontiguousarray(self.to_xp(recmat.recmat[:self.nmodes], dtype=self.dtype))
        self.im = self.xp.ascontiguousarray(self.to_xp(intmat.intmat[:, :self.nmodes], dtype=self.dtype))
        self.modes.value = self.xp.zeros(self.nmodes, dtype=self.dtype)

    def can_accelerate(self):
        # Only GPU work on arrays that are not reallocated, and graph_matmul products
        return True

    def trigger_code(self):
        a_lin = matmul(self.rec, self.slopes, self.xp)
        a_parallel = self.nonlinear_calib.invert(a_lin)

        if self.apply_orthogonal_correction:
            predicted_response = self.nonlinear_calib.forward(a_parallel)
            residual_slopes = self.slopes - matmul(self.im, predicted_response, self.xp)
            self.modes.value[:] = a_parallel + matmul(self.rec, residual_slopes, self.xp)
        else:
            self.modes.value[:] = a_parallel
        self.modes.generation_time = self.current_time
