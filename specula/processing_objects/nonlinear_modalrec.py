from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.lib.graph_matmul import matmul


class NonlinearModalrec(BaseModalrec):
    """
    Nonlinear modal reconstructor: the linear
    :class:`~specula.processing_objects.modalrec.Modalrec` plus a per-mode
    saturation correction. It suits sensors whose response saturates mode by
    mode, such as a non-modulated pyramid.

    At each step:

    1. ``a_lin = recmat @ slopes``: the linear estimate, as Modalrec.
    2. ``out_modes = calibration.invert(a_lin)``: each mode's linear estimate
       de-saturated with its calibrated response curve (see
       :class:`~specula.data_objects.nonlinear_calibration.NonlinearCalibration`).

    The curves must be measured with the same recmat, by
    :class:`~specula.processing_objects.nonlinear_response_calibrator.NonlinearResponseCalibrator`:
    then step 2 recovers exactly the amplitude of a mode injected alone. The
    coupling between modes in the saturated regime is not corrected.

    Modes outside the calibrated range have a linear (identity) curve: with
    identity curves everywhere, the result is Modalrec's.

    Parameters
    ----------
    recmat : Recmat
        Reconstruction matrix (nmodes x nslopes), the one the calibration
        was measured with.
    nonlinear_calib : NonlinearCalibration
        Per-mode response curves. All of them must be strictly increasing.
    nmodes : int, optional
        Number of modes to reconstruct (default: all the modes of the recmat).
    """

    def __init__(self,
                 recmat: Recmat,
                 nonlinear_calib: NonlinearCalibration,
                 nmodes: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        rec_nmodes = recmat.recmat.shape[0]
        self.nmodes = nmodes if nmodes is not None else rec_nmodes
        if self.nmodes > rec_nmodes:
            raise ValueError(f'Requested nmodes={self.nmodes} exceeds recmat nmodes={rec_nmodes}')
        bad = nonlinear_calib.non_monotonic_modes()
        if bad:
            raise ValueError(f'the nonlinear calibration cannot be inverted: the response of modes {bad} '
                             f'is not strictly increasing with the amplitude')

        self.nonlinear_calib = nonlinear_calib
        # Contiguous copy of the part used, as graph_matmul needs
        self.rec = self.xp.ascontiguousarray(self.to_xp(recmat.recmat[:self.nmodes], dtype=self.dtype))
        self.modes.value = self.xp.zeros(self.nmodes, dtype=self.dtype)

    def can_accelerate(self):
        # Only GPU work on arrays that are not reallocated, and graph_matmul products
        return True

    def trigger_code(self):
        a_lin = matmul(self.rec, self.slopes, self.xp)
        self.modes.value[:] = self.nonlinear_calib.invert(a_lin)
        self.modes.generation_time = self.current_time
