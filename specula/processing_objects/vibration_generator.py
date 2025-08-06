import numpy as np
from specula.processing_objects.base_generator import BaseGenerator
from specula.data_objects.simul_params import SimulParams
from specula.lib.utils import psd_to_signal


class Vibrations:
    """Helper class for PSD-based vibration generation"""
    def __init__(self, nmodes, psd=None, freq=None, seed=1987, samp_freq=1000,
                 niter=1000, start_from_zero=False, xp=np, dtype=np.float32,
                 complex_dtype=np.complex64):
        self.nmodes = nmodes
        self.seed = seed
        self.start_from_zero = start_from_zero
        self.niter = niter
        self.samp_freq = samp_freq
        self.xp = xp
        self.dtype = dtype
        self.complex_dtype = complex_dtype

        # Store PSD and freq as lists of arrays (one per mode)
        self._psd = []
        self._freq = []

        psd = self.xp.array(psd)
        for i in range(self.nmodes):
            self._psd.append(psd[i, :])

        freq = self.xp.array(freq)
        if freq.ndim == 1:
            freq = np.tile(freq, (self.nmodes, 1)).T
        for i in range(self.nmodes):
            self._freq.append(freq[:, i])

    def get_time_hist(self):
        n = int(np.floor((self.niter + 1) / 2.))
        time_hist = self.xp.zeros((2 * n, self.nmodes), dtype=self.dtype)

        for i in range(self.nmodes):
            freq_mode = self._freq[i]
            psd_mode = self._psd[i]
            freq_bins = self.xp.linspace(freq_mode[0], freq_mode[-1], n, dtype=self.dtype)
            psd_interp = self.xp.interp(freq_bins, freq_mode, psd_mode)

            temp, _ = psd_to_signal(psd_interp, self.samp_freq, self.xp, self.dtype,
                                  self.complex_dtype, seed=self.seed + i)
            if self.start_from_zero:
                temp -= temp[0]
            time_hist[:, i] = temp
  
        return time_hist


class VibrationGenerator(BaseGenerator):
    """
    Generates vibration signals from PSD specifications.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 nmodes: int,
                 psd,
                 freq,
                 seed: int = 1987,
                 start_from_zero: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Setup vibration generator
        samp_freq = 1 / simul_params.time_step
        niter = int(simul_params.total_time / simul_params.time_step)

        self.vib = Vibrations(
            nmodes, psd=psd, freq=freq, seed=seed,
            samp_freq=samp_freq, niter=niter, start_from_zero=start_from_zero,
            xp=self.xp, dtype=self.dtype, complex_dtype=self.complex_dtype
        )

        self.time_hist = self.vib.get_time_hist()

    def trigger_code(self):
        if self.iter_counter < self.time_hist.shape[0]:
            self.output.value[:] = self.time_hist[self.iter_counter, :]
        else:
            # Beyond available data, use last values
            self.output.value[:] = self.time_hist[-1, :]