import numpy as np
from scipy import signal
from functools import lru_cache

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams
from specula import cpuArray

import matplotlib.pyplot as plt

class GainOptimizer(BaseProcessingObj):
    """
    Gain optimizer for IIR filters based on modal gain optimization (GENDRON 1994).
    This class optimizes the gains of an IIR filter by minimizing the residual variance
    in the closed-loop system using pseudo open-loop measurements.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 opt_dt: float = 1.0,  # Optimization interval in seconds
                 delay: float = 2.0,   # Loop delay in frames
                 max_gain_factor: float = 0.95,  # Safety factor for maximum gain
                 safety_factor: float = 0.90,    # Additional safety margin
                 max_inc: float = 0.5,           # Maximum gain increment per step
                 limit_inc: bool = True,         # Limit gain increments
                 ngains: int = 20,               # Number of gain values to test
                 running_mean: bool = False,     # Use running mean for PSD
                 verbose: bool = True,          # Verbose output
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.iir_filter_data = iir_filter_data
        self.time_step = simul_params.time_step

        # Convert optimization parameters
        self.opt_dt = self.seconds_to_t(opt_dt)
        self.delay = delay
        self.max_gain_factor = max_gain_factor
        self.safety_factor = safety_factor
        self.max_inc = max_inc
        self.limit_inc = limit_inc
        self.ngains = ngains
        self.running_mean = running_mean

        # Get number of modes from filter
        self.nmodes = iir_filter_data.nfilter

        # History storage
        self.time_hist = []
        self.delta_comm_hist = []
        self.comm_hist = []
        self.optical_gain_hist = []
        self.psd_ol = None
        self.prev_optimized_gain = self.iir_filter_data.gain.copy()

        self.plot_debug = False  # Enable plotting for debugging

        # Outputs
        self.optimized_gain = BaseValue(
            value=self.xp.ones(self.nmodes, dtype=self.dtype),
            target_device_idx=target_device_idx
        )
        # Initialize optimal gain to ones
        self.optimized_gain.value = self.xp.ones(self.nmodes, dtype=self.dtype)

        self.optimization_done = False

        # Inputs
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['out_comm'] = InputValue(type=BaseValue)
        self.inputs['optical_gain'] = InputValue(type=BaseValue, optional=True)

        # Outputs
        self.outputs['optimized_gain'] = self.optimized_gain

        self.verbose = verbose

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Get current inputs
        self.current_delta_comm = self.local_inputs['delta_comm'].value
        self.current_out_comm = self.local_inputs['out_comm'].value

        # Store optical gain if available
        if self.local_inputs['optical_gain'] is not None:
            self.current_optical_gain = self.local_inputs['optical_gain'].value
            self.optical_gain_hist.append(float(self.current_optical_gain))
        else:
            self.current_optical_gain = 1.0

    def trigger_code(self):
        t = self.current_time

        # Store history
        self.time_hist.append(t)
        self.delta_comm_hist.append(self.current_delta_comm.copy())
        self.comm_hist.append(self.current_out_comm.copy())

        # Check if it's time to optimize
        if t >= self.opt_dt and (t % self.opt_dt) == 0:
            self._optimize_gains(t)
            self.optimization_done = True
        else:
            self.optimization_done = False

    def _optimize_gains(self, t):
        """
        Perform gain optimization based on accumulated history.
        """
        if len(self.delta_comm_hist) < 2:
            if self.verbose:
                print("Not enough history for optimization")
            return

        # Convert history to arrays
        delta_comm_hist = self.xp.array(self.delta_comm_hist)
        comm_hist = self.xp.array(self.comm_hist)

        # Calculate pseudo open-loop signal
        pseudo_ol = self._calculate_pseudo_open_loop(delta_comm_hist, comm_hist)

        # Calculate maximum stable gains
        gmax_vec = self._calculate_max_gains()

        # Optimize gains for each mode
        opt_gains = self.xp.zeros(self.nmodes, dtype=self.dtype)

        for mode in range(self.nmodes):
            opt_gains[mode] = self._optimize_single_mode(
                mode, pseudo_ol[:, mode], self.time_step, gmax_vec[mode]
            )

        if self.plot_debug:
            plt.figure()
            plt.plot(cpuArray(opt_gains), marker='o')
            plt.xlabel('Mode Index')
            plt.ylabel('Optimized Gain')
            plt.title('Optimized Gains for Each Mode')
            plt.grid()
            plt.show()

        # Apply increment limiting
        if self.limit_inc and self.prev_optimized_gain is not None:
            opt_gains = (self.prev_optimized_gain +
                        self.max_inc * (opt_gains - self.prev_optimized_gain))

        # Apply optical gain compensation
        if len(self.optical_gain_hist) >= 2:
            gain_ratio = self.optical_gain_hist[-1] / self.optical_gain_hist[-2]
            opt_gains *= gain_ratio

        # Apply safety factors
        opt_gains *= self.safety_factor
        opt_gains = self.xp.minimum(opt_gains, gmax_vec)

        # Store results
        self.prev_optimized_gain = opt_gains.copy()
        self.optimized_gain.value[:] = opt_gains

        if self.verbose:
            print(f"Optimized gains at t={self.t_to_seconds(t):.3f}s: "
                  f"mean={float(self.xp.mean(opt_gains)):.4f}")

    def _calculate_pseudo_open_loop(self, delta_comm_hist, comm_hist):
        """
        Calculate pseudo open-loop signal from delta commands and output commands.
        pseudo_ol[t] = comm[t-1] + delta_comm[t]
        """
        n_time, n_modes = delta_comm_hist.shape
        pseudo_ol = self.xp.zeros((n_time, n_modes), dtype=self.dtype)

        # First time step: use delta command only
        pseudo_ol[0, :] = delta_comm_hist[0, :]

        # Subsequent time steps: add previous command
        pseudo_ol[1:, :] = comm_hist[:-1, :] + delta_comm_hist[1:, :]

        if self.plot_debug:
            plt.figure()
            plt.plot(cpuArray(comm_hist[:,0]), label='Output Command')
            plt.plot(cpuArray(delta_comm_hist[:,0]), label='Delta Command')
            plt.plot(cpuArray(pseudo_ol[:,0]), label='Pseudo Open-Loop Signal')
            plt.legend()
            plt.xlabel('Time Step')
            plt.ylabel('Signal Value')
            plt.title('Pseudo Open-Loop Signal Calculation')
            plt.grid()
            plt.show()

        return pseudo_ol

    def _calculate_max_gains(self):
        """
        Calculate maximum stable gains for each mode using IirFilterData stability analysis.
        """
        # Use the new max_stable_gain method from IirFilterData
        gmax_vec = self.iir_filter_data.max_stable_gain(
            delay=self.delay,
            max_gain=20.0,  # Maximum gain to test
            n_gain=20000,   # Number of gain values to test for high precision
        )

        # Apply the maximum gain factor safety margin
        gmax_vec = self.to_xp(gmax_vec) * self.max_gain_factor

        if self.verbose:
            print(f"Maximum stable gains calculated:")
            print(f"  Raw max gains: mean={float(self.xp.mean(gmax_vec/self.max_gain_factor)):.4f}, "
                f"std={float(self.xp.std(gmax_vec/self.max_gain_factor)):.4f}")
            print(f"  With safety factor ({self.max_gain_factor}): mean={float(self.xp.mean(gmax_vec)):.4f}, "
                f"std={float(self.xp.std(gmax_vec)):.4f}")

        return gmax_vec

    def _optimize_single_mode(self, mode, pseudo_ol_mode, t_int, gmax):
        """
        Optimize gain for a single mode using PSD minimization.
        """

        # Get filter coefficients for this mode from iir_filter_data
        num = cpuArray(self.iir_filter_data.num.copy()[mode, :])
        den = cpuArray(self.iir_filter_data.den.copy()[mode, :])
        # normalize numerator by the gain
        # This is done because _calculate_rejection_tf wants a num with unitary gain
        num /= cpuArray(self.iir_filter_data.gain[mode])

        # Calculate PSD of pseudo open-loop signal
        psd_pseudo_ol, freq = self._calculate_psd(pseudo_ol_mode, t_int)

        # Apply running mean if enabled
        if self.running_mean and self.psd_ol is not None:
            if self.psd_ol.shape[1] > mode:
                psd_pseudo_ol = (psd_pseudo_ol + self.psd_ol[:, mode]) / 2.0

        # t_intst different gain values
        gains = self.xp.linspace(gmax/self.ngains, gmax, self.ngains)
        totals = self.xp.zeros(self.ngains, dtype=self.dtype)

        for i, gain in enumerate(gains):
            # Calculate rejection transfer function
            h_rej = self._calculate_rejection_tf(freq, t_int, gain, num, den)
            # Calculate total residual variance
            psd_res = self.xp.nan_to_num(self.xp.abs(h_rej)**2 * psd_pseudo_ol)
            totals[i] = self.xp.sum(psd_res)

        if self.plot_debug:
            plt.figure()
            plt.plot(cpuArray(freq), cpuArray(psd_pseudo_ol), label='Pseudo Open-Loop PSD')
            plt.plot(cpuArray(freq), cpuArray(psd_res), label='Residual PSD')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.title(f'PSDs for Mode {mode}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            plt.legend()
            plt.figure()
            plt.plot(cpuArray(freq), cpuArray(self.xp.nan_to_num(self.xp.abs(h_rej))))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'RTF for Mode {mode}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            plt.figure()
            plt.plot(cpuArray(gains), cpuArray(totals), marker='o')
            plt.xlabel('Gain')
            plt.ylabel('Total Residual Variance')
            plt.title(f'Mode {mode} Gain Optimization')
            plt.grid()
            plt.show()

        # Find optimal gain
        min_idx = self.xp.argmin(totals)
        optimal_gain = gains[min_idx]

        return optimal_gain

    def _calculate_psd(self, data, t_int):
        """
        Calculate Power Spectral Density using Welch's method.
        """
        # Convert to CPU for scipy operations
        data_cpu = self.xp.asnumpy(data) if hasattr(self.xp, 'asnumpy') else data

        # Use Welch's method with Hanning window
        fs = 1.0 / t_int
        freq, psd = signal.welch(data_cpu, fs=fs, window='hann',
                                nperseg=min(len(data_cpu), 256))

        # Convert back to target device
        freq = self.to_xp(freq, dtype=self.dtype)
        psd = self.to_xp(psd, dtype=self.dtype)

        return psd, freq

    def _round_values_for_cache(self, freq, t_int, gain, num, den):
        """Round values to reasonable precision for cache key consistency."""
        # Round frequency to avoid floating point precision issues
        freq_rounded = tuple(np.round(cpuArray(freq), 8))

        # Round other parameters
        t_int_rounded = round(float(t_int), 10)
        gain_rounded = round(float(gain), 8)
        num_rounded = tuple(np.round(cpuArray(num), 10))
        den_rounded = tuple(np.round(cpuArray(den), 10))
        delay_rounded = round(float(self.delay), 6)

        return freq_rounded, t_int_rounded, gain_rounded, num_rounded, den_rounded, delay_rounded

    @lru_cache(maxsize=16384)
    def _calculate_rejection_tf_cached(self, freq_tuple, t_int, gain, num_tuple, den_tuple, delay):
        """
        Calculate rejection transfer function using lru_cache.
        All parameters must be hashable (tuples, not arrays).
        """
        # Convert tuples back to arrays
        freq = self.to_xp(freq_tuple, dtype=self.dtype)
        num = self.to_xp(num_tuple, dtype=self.dtype)
        den = self.to_xp(den_tuple, dtype=self.dtype)

        # Calculate transfer function
        omega = 2 * np.pi * freq * t_int
        z = self.xp.exp(1j * omega)

        # Calculate controller transfer function
        num_val = self.xp.polyval(num[::-1], z)
        den_val = self.xp.polyval(den[::-1], z)

        # Avoid division by zero
        den_val = self.xp.where(self.xp.abs(den_val) < 1e-12, 1e-12, den_val)
        c_tf = num_val / den_val

        # Add delay
        delay_tf = z**(-delay)

        # Open-loop transfer function
        ol_tf = gain * c_tf * delay_tf

        # Closed-loop rejection transfer function: 1/(1 + L)
        denom = 1.0 + ol_tf
        denom = self.xp.where(self.xp.abs(denom) < 1e-12, 1e-12, denom)
        h_rej = 1.0 / denom

        # Clean up any NaN/Inf
        h_rej = self.xp.nan_to_num(h_rej, nan=0.0, posinf=0.0, neginf=0.0)

        return h_rej

    def _calculate_rejection_tf(self, freq, t_int, gain, num, den):
        """
        Calculate rejection transfer function with lru_cache.
        """
        # Round values for consistent cache keys
        freq_rounded, t_int_rounded, gain_rounded, num_rounded, den_rounded, delay_rounded = \
            self._round_values_for_cache(freq, t_int, gain, num, den)

        # Try to get from cache
        h_rej = self._calculate_rejection_tf_cached(
            freq_rounded, t_int_rounded, gain_rounded,
            num_rounded, den_rounded, delay_rounded
        )
        return h_rej

    def post_trigger(self):
        super().post_trigger()

        if self.optimization_done:
            self.optimized_gain.generation_time = self.current_time
            self.iir_filter_data.set_gain(self.optimized_gain.value)
