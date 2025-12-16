import numpy as np
from specula.processing_objects.base_generator import BaseGenerator
from specula.data_objects.simul_params import SimulParams
from specula.lib.utils import psd_to_signal


def get_vibrations_time_hist(nmodes, psd, freq, seed=1987, samp_freq=1000,
                             niter=1000, start_from_zero=False, xp=np, dtype=np.float32,
                             complex_dtype=np.complex64):
        '''
        PSD-based vibration generation

        For PASSATA compatibility, freq is a 2d array-like with modes on the second index.

        Parameters
        ----------
        psd: 2d array-like
            psd for each mode, modes on first index: [mode, psd]
        freq: 1d or 2d array-like
            frequency vector for each mode. If 1d, the same frequency vector
            will be replicated for all modes. If 2d, modes must be on the second index: [freq, mode]
        seed: int, optional
            generation seed for first mode, will be increment by 1 for each additonal mode
        samp_freq: float, optional
            PSD sampling frequency in Hz, default 1000
        niter: int, optional
            number of data points per mode to generate, default 1000
        start_from_zero: bool, optional
            if True, first data point for each mode is zero. Defaults to False
        xp: module, optional
            either np or cp
        dtype: dtype, optional
            dtype for results
        complex_dtype: dtype, optional
            dtype for complex numbers in PSD generation

        Returns
        -------
        time_hist: 2d array
            time history as a [sample, mode] array

        '''
        # PSD is a 2d array-like with modes already on the first index
        psd = xp.array(psd[:nmodes])


        # If freq 1d only, replicate the data for each mode, make sure that modes are on the second index
        freq = xp.array(freq)
        if freq.ndim == 1:
            freq = np.tile(freq, (nmodes, 1)).T

        # Get modes on the first index
        freq = freq.T[:nmodes]

        if len(psd) < nmodes:
            raise ValueError(f'Requested {nmodes=} but PSD array only contains {psd.shape[0]} modes (shape={psd.shape})')

        if len(freq) < nmodes:
            raise ValueError(f'Requested {nmodes=} but frequency array only contains {freq.shape[0]} modes (shape={freq.shape})')

        n = int(np.floor((niter + 1) / 2.))
        time_hist = xp.zeros((2 * n, nmodes), dtype=dtype)

        for i in range(nmodes):
            freq_mode = freq[i]
            psd_mode = psd[i]
            freq_bins = xp.linspace(freq_mode[0], freq_mode[-1], n, dtype=dtype)
            psd_interp = xp.interp(freq_bins, freq_mode, psd_mode)

            temp, _ = psd_to_signal(psd_interp, samp_freq, xp, dtype,
                                  complex_dtype, seed=seed + i)
            if start_from_zero:
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
        '''
        PSD-based vibration generation

        For PASSATA compatibility, psd is a 2d array-like with modes on the first index,
        while freq is a 1d array-like or a 2d array-like with modes on the second index.

        Parameters
        ----------
        simul_params: SimulParams object
            main simulation parameters. Only the *total_time* and *time_step* members are accessed
        nmodes: int
            number of modes to generate
        psd: 2d array-like
            psd for each mode, modes on first index: [mode, psd]
        freq: 1d or 2d array-like
            frequency vector for each mode. If 1d, the same frequency vector
            will be replicated for all modes. If 2d, modes must be on the second index: [freq, mode]
        seed: int, optional
            generation seed for first mode, will be increment by 1 for each additonal mode
        start_from_zero: bool, optional
            if True, first data point for each mode is zero. Defaults to False
        '''
        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Setup vibration generator
        samp_freq = 1 / simul_params.time_step
        niter = int(simul_params.total_time / simul_params.time_step)

        self.time_hist = get_vibrations_time_hist(
            nmodes, psd=psd, freq=freq, seed=seed,
            samp_freq=samp_freq, niter=niter, start_from_zero=start_from_zero,
            xp=self.xp, dtype=self.dtype, complex_dtype=self.complex_dtype
        )

    def trigger_code(self):
        if self.iter_counter < self.time_hist.shape[0]:
            self.output.value[:] = self.time_hist[self.iter_counter, :]
        else:
            # Beyond available data, use last values
            self.output.value[:] = self.time_hist[-1, :]