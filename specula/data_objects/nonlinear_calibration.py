import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from iteras.lib import storage


class NonlinearCalibration(BaseDataObj):
    """
    Per-mode nonlinear response of a wavefront sensor (e.g. a non-modulated
    pyramid), as measured by
    :class:`~specula.processing_objects.nonlinear_response_calibrator.NonlinearResponseCalibrator`.

    For each calibrated mode it holds a sampled curve of the "response" (the
    mode's linear reconstruction, ``(recmat @ slopes)[mode]``) versus the true
    amplitude of the mode, injected alone. In the linear regime the response
    equals the amplitude; at larger amplitudes it saturates.

    The calibrated modes are first_mode, ..., first_mode + nmodes - 1; all of
    them share the same amplitude grid, so the calibration is a 1D amplitude
    vector plus a (nmodes, nsamples) response matrix. :meth:`forward` and
    :meth:`invert` work on a whole modal vector: modes outside the calibrated
    range are left unchanged.

    This is a lookup-table calibration that assumes each mode's response
    depends only on its own amplitude (no cross-mode coupling).

    Parameters
    ----------
    amplitudes : array-like [nsamples]
        Amplitude grid shared by every mode, strictly increasing.
    responses : array-like [nmodes, nsamples]
        Response of each calibrated mode at each amplitude. It must
        be strictly increasing along the amplitudes to be inverted (see
        :meth:`non_monotonic_modes`).
    first_mode : int
        Index, in the modal vector, of the first calibrated mode.
    """

    def __init__(self,
                 amplitudes,
                 responses,
                 first_mode: int = 0,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        amplitudes = np.asarray(cpuArray(amplitudes), dtype=float)
        responses = np.asarray(cpuArray(responses), dtype=float)

        if amplitudes.ndim != 1 or amplitudes.shape[0] < 2:
            raise ValueError(f'amplitudes must be 1D with at least 2 samples, got shape {amplitudes.shape}')
        if responses.ndim != 2 or responses.shape[1] != amplitudes.shape[0]:
            raise ValueError(f'responses must have shape (nmodes, {amplitudes.shape[0]}), got {responses.shape}')
        if np.any(np.diff(amplitudes) <= 0):
            raise ValueError('amplitudes must be strictly increasing')
        if first_mode < 0:
            raise ValueError(f'first_mode must be >= 0, got {first_mode}')

        self.first_mode = int(first_mode)
        self.amplitudes = self.to_xp(amplitudes, dtype=self.dtype)
        self.responses = self.to_xp(responses, dtype=self.dtype)

    @property
    def nmodes(self):
        return self.responses.shape[0]

    @property
    def nsamples(self):
        return self.amplitudes.shape[0]

    def get_value(self):
        '''Get the per-mode responses as a numpy/cupy array'''
        return self.responses

    def set_value(self, v):
        '''Set new per-mode responses. Arrays are not reallocated.'''
        assert v.shape == self.responses.shape, \
            f"Error: input array shape {v.shape} does not match responses shape {self.responses.shape}"
        self.responses[:] = self.to_xp(v)

    def non_monotonic_modes(self):
        '''Indices, in the modal vector, of the modes whose response is not strictly
        increasing with the amplitude: their curve cannot be inverted'''
        bad = np.nonzero(np.any(np.diff(cpuArray(self.responses), axis=1) <= 0, axis=1))[0]
        return (bad + self.first_mode).tolist()

    def _calibrated(self, n):
        '''Slice of the calibrated modes within a modal vector of length n'''
        return slice(min(self.first_mode, n), min(self.first_mode + self.nmodes, n))

    def _interp(self, x, xs, ys):
        '''Row-wise linear interpolation, as np.interp on each row: x [k], xs and ys
        [k, nsamples] with xs increasing along each row. Values outside a row's
        range get its boundary value.'''
        xp = self.xp
        rows = xp.arange(x.shape[0])
        j = xp.clip(xp.sum(xs <= x[:, None], axis=1), 1, self.nsamples - 1)
        x0, x1 = xs[rows, j - 1], xs[rows, j]
        y0, y1 = ys[rows, j - 1], ys[rows, j]
        t = xp.clip((x - x0) / (x1 - x0), 0, 1)
        return y0 + t * (y1 - y0)

    def forward(self, amplitude):
        """
        The (saturated) response that each mode of the modal vector `amplitude`
        produces, from the calibrated curves (linear interpolation, boundary
        value outside the calibrated range). Modes outside the calibration are
        unchanged. This is the forward counterpart of :meth:`invert`.
        """
        s = self._calibrated(amplitude.shape[0])
        k = s.stop - s.start
        out = amplitude.copy()
        xs = self.xp.broadcast_to(self.amplitudes, (k, self.nsamples))
        out[s] = self._interp(amplitude[s], xs, self.responses[:k])
        return out

    def invert(self, response):
        """
        The true amplitude that produced each mode of the modal vector
        `response` (e.g. a linear reconstruction), inverting the calibrated
        curves (linear interpolation, boundary amplitude outside the calibrated
        range). Modes outside the calibration are unchanged.
        """
        s = self._calibrated(response.shape[0])
        k = s.stop - s.start
        out = response.copy()
        amps = self.xp.broadcast_to(self.amplitudes, (k, self.nsamples))
        out[s] = self._interp(response[s], self.responses[:k], amps)
        return out

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NMODES'] = self.nmodes
        hdr['NSAMPLES'] = self.nsamples
        hdr['FIRSTMOD'] = self.first_mode
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'amplitudes': self.amplitudes, 'responses': self.responses},
                             overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            version = int(f.header['VERSION'])
            if version != 1:
                raise ValueError(f'Error: unknown version {version} in file {filename}')
            amplitudes = f.read('amplitudes', 1)
            responses = f.read('responses', 2)
            first_mode = int(f.header.get('FIRSTMOD', 0))
        return NonlinearCalibration(amplitudes, responses, first_mode=first_mode,
                                    target_device_idx=target_device_idx)
