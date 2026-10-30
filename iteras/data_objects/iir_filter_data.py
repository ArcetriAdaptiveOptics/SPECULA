import numpy as np
from functools import lru_cache

from iteras import cpuArray
from iteras.base_data_obj import BaseDataObj

from astropy.io import fits
from iteras.lib import storage

# Try to import control library, but make it optional
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None


def _real_poly(roots, what):
    """
    Coefficients, lowest order first, of the monic polynomial with the given
    roots. The coefficients of a filter are real, so complex roots must come
    in conjugate pairs: a ValueError is raised otherwise.
    """
    coeffs = np.poly(roots)[::-1]     # np.poly: highest order first
    if np.iscomplexobj(coeffs):
        if np.max(np.abs(coeffs.imag)) > 1e-9 * np.max(np.abs(coeffs)):
            raise ValueError(f'IirFilterData: complex {what} must come in conjugate pairs, '
                             f'got {roots}')
        coeffs = coeffs.real
    return coeffs


def _sampling_time(dt):
    """Sampling time of a python-control discrete system: 1 if dt is True (unspecified)"""
    return 1.0 if dt is True else dt


class IirFilterData(BaseDataObj):
    """
    Infinite Impulse Response (IIR) Filter Data object.
    This class stores IIR filter coefficients and provides methods to analyze
    the filter's transfer function, frequency response and stability.
    """
    def __init__(self,
                 ordnum: list,
                 ordden: list,
                 num,
                 den,
                 n_modes=None,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        :class:`~iteras.data_objects.iir_filter_data.IirFilterData` - IIR Filter Data representation.
 
        This class stores IIR filter coefficients in the following format:
        - Coefficients are stored with lowest order terms first
        - num[i, :] contains numerator coefficients for filter i
        - den[i, :] contains denominator coefficients for filter i
        - ordnum[i] and ordden[i] specify the actual order of each filter

        Transfer function: H(z) = (num[0] + num[1]*z + ...) / (den[0] + den[1]*z + ...)
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        # Handle filter setup (ordnum, ordden, num and den) based on n_modes:
        # - If n_modes is provided, it specifies how many modes (channels) to use.
        # - n_modes length must match ordnum, ordden, num and den.
        # - Each ordnum[i], ordden[i], num[i] and den[i] is expanded into a block of size n_modes[i].
        #   Example: n_modes=[2,3], num=[[0.0, 0.5],[0.0, 0.4]]
        #            -> num = [[0.0, 0.5],[0.0, 0.5],[0.0, 0.4],[0.0, 0.4],[0.0, 0.4]]
        # - Raises ValueError if the lengths do not match.
        if n_modes is not None:
            n_modes = np.atleast_1d(n_modes)
            if len(n_modes) != len(ordnum):
                raise ValueError("n_modes must have the same length as ordnum")
            ordnum = np.repeat(ordnum, n_modes)
            if len(n_modes) != len(ordden):
                raise ValueError("n_modes must have the same length as ordden")
            ordden = np.repeat(ordden, n_modes)
            if len(n_modes) != len(num) or len(n_modes) != len(den):
                raise ValueError("n_modes must have the same length as num and den")
            num = np.repeat(cpuArray(num), n_modes, axis=0)
            den = np.repeat(cpuArray(den), n_modes, axis=0)
        self.ordnum = self.to_xp(ordnum, dtype=int)
        self.ordden = self.to_xp(ordden, dtype=int)
        self.zeros = None
        self.poles = None
        self.gain = None
        self.num = None
        self.den = None
        self._num_normalized = None
        self.set_num(cpuArray(num))
        self.set_den(cpuArray(den))

    @property
    def nfilter(self):
        return len(self.num)

    def get_zeros(self):
        '''
        Zeros of each filter, as a complex array of shape (nfilter, width - 1).
        Each row has the ordnum - 1 zeros of its filter, followed by NaN.
        '''
        if self.zeros is None:
            num_cpu = cpuArray(self.num)
            ordnum_cpu = cpuArray(self.ordnum)
            snum1 = num_cpu.shape[1]
            zeros_cpu = np.full((self.nfilter, snum1 - 1), np.nan, dtype=complex)
            for i in range(self.nfilter):
                oi = int(ordnum_cpu[i])
                if oi > 1:
                    # Coefficients are lowest order first, np.roots wants highest first
                    roots = np.roots(num_cpu[i, snum1 - oi:][::-1])
                    zeros_cpu[i, :len(roots)] = roots
            self.zeros = self.to_xp(zeros_cpu, dtype=self.complex_dtype)
        return self.zeros

    def get_poles(self):
        '''
        Poles of each filter, as a complex array of shape (nfilter, width - 1).
        Each row has the ordden - 1 poles of its filter, followed by NaN.
        '''
        if self.poles is None:
            den_cpu = cpuArray(self.den)
            ordden_cpu = cpuArray(self.ordden)
            sden1 = den_cpu.shape[1]
            poles_cpu = np.full((self.nfilter, sden1 - 1), np.nan, dtype=complex)
            for i in range(self.nfilter):
                oi = int(ordden_cpu[i])
                if oi > 1:
                    roots = np.roots(den_cpu[i, sden1 - oi:][::-1])
                    poles_cpu[i, :len(roots)] = roots
            self.poles = self.to_xp(poles_cpu, dtype=self.complex_dtype)
        return self.poles

    def set_num(self, num):
        mynum = cpuArray(num).copy()
        ordnum_cpu = cpuArray(self.ordnum)
        snum1 = mynum.shape[1]
        for i in range(len(mynum)):
            oi = int(ordnum_cpu[i])
            if oi < snum1 and np.sum(np.abs(mynum[i, oi:])) == 0:
                mynum[i, :] = np.roll(mynum[i, :], snum1 - oi)

        gain = mynum[:, -1]
        nonzero_gain = np.abs(gain) > 0
        safe_gain = np.where(nonzero_gain, gain, 1.0)
        num_normalized = mynum / safe_gain[:, None]
        self.gain = self.to_xp(gain, dtype=self.dtype)
        self._num_normalized = self.to_xp(num_normalized, dtype=self.dtype)
        self.zeros = None
        self.num = self.to_xp(mynum, dtype=self.dtype)
        self._match_widths()

    def set_den(self, den):
        myden = cpuArray(den).copy()
        ordden_cpu = cpuArray(self.ordden)
        sden1 = myden.shape[1]
        for i in range(len(myden)):
            oi = int(ordden_cpu[i])
            if oi < sden1 and np.sum(np.abs(myden[i, oi:])) == 0:
                myden[i, :] = np.roll(myden[i, :], sden1 - oi)

        self.den = self.to_xp(myden, dtype=self.dtype)
        self.poles = None
        self._match_widths()

    def _match_widths(self):
        '''
        Give num and den the same number of columns, adding zeros on the
        lowest order side of the shorter one. IirFilter aligns the two arrays
        on their last (highest order) coefficient, while the analysis methods
        (frequency_response(), RTF(), stability...) align them on the first one:
        with the same width the two conventions agree. Multiplying both by the
        same power of z does not change the filter.
        '''
        if self.num is None or self.den is None:
            return
        wnum, wden = self.num.shape[1], self.den.shape[1]
        if wnum < wden:
            self.set_num(np.pad(cpuArray(self.num), ((0, 0), (wden - wnum, 0))))
        elif wden < wnum:
            self.set_den(np.pad(cpuArray(self.den), ((0, 0), (wnum - wden, 0))))

    def set_zeros(self, zeros):
        zeros_cpu = cpuArray(zeros)
        ordnum_cpu = cpuArray(self.ordnum)
        num_cpu = np.zeros((self.nfilter, zeros_cpu.shape[1] + 1))
        snum1 = num_cpu.shape[1]
        for i in range(self.nfilter):
            oi = int(ordnum_cpu[i])
            if oi > 1:
                num_cpu[i, snum1 - oi:] = _real_poly(zeros_cpu[i, :oi - 1], 'zeros')
        self.set_num(num_cpu)  # resets self.zeros = None internally
        self.zeros = self.to_xp(zeros_cpu, dtype=self.complex_dtype)

    def set_poles(self, poles):
        poles_cpu = cpuArray(poles)
        ordden_cpu = cpuArray(self.ordden)
        den_cpu = np.zeros((self.nfilter, poles_cpu.shape[1] + 1))
        sden1 = den_cpu.shape[1]
        for i in range(self.nfilter):
            oi = int(ordden_cpu[i])
            if oi > 1:
                den_cpu[i, sden1 - oi:] = _real_poly(poles_cpu[i, :oi - 1], 'poles')
        self.poles = self.to_xp(poles_cpu, dtype=self.complex_dtype)
        self.den = self.to_xp(den_cpu, dtype=self.dtype)

    def set_gain(self, gain):
        if np.ndim(gain) == 0:
            gain = np.repeat(gain, self.nfilter)
        gain = self.to_xp(gain, dtype=self.dtype)
        self.logger.info(f'Setting gain: {gain}')

        if self._num_normalized is None:
            self._num_normalized = self.to_xp(self.num, dtype=self.dtype)
            if self.gain is not None:
                nonzero_gain = self.xp.abs(self.gain) > 0
                safe_gain = self.xp.where(nonzero_gain, self.gain, 1)
                self._num_normalized = self._num_normalized / safe_gain[:, None]

        if self.xp.size(gain) < self.nfilter:
            nfilter = np.size(gain)
        else:
            nfilter = self.nfilter

        if self.gain is None:
            current_gain = self.xp.zeros(self.nfilter, dtype=self.dtype)
            current_gain[:nfilter] = gain[:nfilter]
        else:
            current_gain = self.to_xp(self.gain, dtype=self.dtype)

        finite_gain = self.xp.isfinite(gain[:nfilter])
        current_gain[:nfilter] = self.xp.where(finite_gain, gain[:nfilter], current_gain[:nfilter])

        self.gain = self.to_xp(current_gain, dtype=self.dtype)
        self.num = self.to_xp(self._num_normalized * self.gain[:, None], dtype=self.dtype)

        self.logger.info(f'new gain: {self.gain}')

    def RTF(self, mode, fs, freq=None, dm=None, nw=None, dw=None,
            title=None, plot=True, overplot=False, delay=None, unity_plant=False,
            **extra):
        """
        Plot Rejection Transfer Function: RTF = 1 / (1 + CP)
        
        Args:
            mode: Filter mode index to use for C coefficients
            fs: Sampling frequency
            freq: Frequency vector for evaluation (if None, auto-generated)
            dm, nw, dw: Plant P = nw / (dm * dw), coefficients lowest order first
            title: Title for the plot
            plot: If True, generate the plot
            overplot: If True, plot on existing figure instead of creating new one
            delay: Plant P = discrete_delay_tf(delay), alternative to dm, nw, dw
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            **extra: Additional plotting parameters (e.g., color)
        
        Returns:
            rtf_mag: Magnitude of the Rejection Transfer Function at specified frequencies    
        """
        cp_num, cp_den = self._loop_tf(mode, delay=delay, dm=dm, nw=nw, dw=dw,
                                       unity_plant=unity_plant)
        # RTF = 1 / (1 + CP) = Cp_den / (Cp_den + Cp_num)
        return self._closed_loop_plot(cp_den, cp_den + cp_num, fs, freq, plot, overplot,
                                      title if title else 'Rejection Transfer Function',
                                      'blue', **extra)

    def NTF(self, mode, fs, freq=None, dm=None, nw=None, dw=None,
            title=None, plot=True, overplot=False, delay=None, unity_plant=False,
            **extra):
        """
        Plot Noise Transfer Function: NTF = CP / (1 + CP)
        
        Args:
            mode: Filter mode index to use for C coefficients
            fs: Sampling frequency
            freq: Frequency vector for evaluation (if None, auto-generated)
            dm, nw, dw: Plant P = nw / (dm * dw), coefficients lowest order first
            title: Title for the plot
            plot: If True, generate the plot
            overplot: If True, plot on existing figure instead of creating new one
            delay: Plant P = discrete_delay_tf(delay), alternative to dm, nw, dw
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            **extra: Additional plotting parameters (e.g., color)
            
        Returns:
            ntf_mag: Magnitude of the Noise Transfer Function at specified frequencies    
        """
        cp_num, cp_den = self._loop_tf(mode, delay=delay, dm=dm, nw=nw, dw=dw,
                                       unity_plant=unity_plant)
        # NTF = CP / (1 + CP) = Cp_num / (Cp_den + Cp_num)
        return self._closed_loop_plot(cp_num, cp_den + cp_num, fs, freq, plot, overplot,
                                      title if title else 'Noise Transfer Function',
                                      'red', **extra)

    def _closed_loop_plot(self, num, den, fs, freq, plot, overplot, title, color, **extra):
        """Magnitude of num / den on the unit circle, optionally plotted (for RTF() and NTF())"""
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)
        self.logger.debug(f"{title} numerator: {num}")
        self.logger.debug(f"{title} denominator: {den}")
        mag = np.abs(self.frequency_response(num, den, fs, freq=freq))

        if plot:
            import matplotlib.pyplot as plt
            if overplot:
                plt.plot(freq, mag, color=extra.pop('color', color), **extra)
            else:
                plt.figure()
                plt.loglog(freq, mag, label=title)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Magnitude')
                plt.title(title)
                plt.grid(True)
                plt.legend()
                plt.show()
        return mag

    def _plant(self, delay=None, dm=None, nw=None, dw=None, unity_plant=False):
        """
        Numerator and denominator of the plant P, lowest order first, with the
        numerator padded to the length of the denominator. The plant must be
        chosen explicitly, with exactly one of: delay (P = discrete_delay_tf(delay)),
        dm, nw and dw (P = nw / (dm * dw)), or unity_plant=True (P = 1).
        """
        has_coeffs = any(x is not None for x in (dm, nw, dw))
        if has_coeffs and any(x is None for x in (dm, nw, dw)):
            raise ValueError('The plant coefficients dm, nw and dw must be given together')
        if sum([delay is not None, has_coeffs, bool(unity_plant)]) != 1:
            raise ValueError('Choose the plant explicitly with exactly one of: delay=<frames>, '
                             'dm/nw/dw=<coefficients>, or unity_plant=True (P = 1)')
        if delay is not None:
            p_num, p_den = self.discrete_delay_tf(delay)
        elif has_coeffs:
            p_num = np.atleast_1d(cpuArray(nw))
            p_den = np.convolve(np.atleast_1d(cpuArray(dm)), np.atleast_1d(cpuArray(dw)))
        else:
            p_num, p_den = np.array([1.0]), np.array([1.0])
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')
        return p_num, p_den

    def _loop_tf(self, mode, gain_factor=1.0, **plant):
        """
        Numerator and denominator of the open loop C P of filter *mode*, lowest
        order first and of the same length, with the controller multiplied by
        *gain_factor*. *plant*: the arguments of _plant().
        """
        p_num, p_den = self._plant(**plant)
        cp_num = np.convolve(cpuArray(self.num[mode, :]) * gain_factor, p_num)
        cp_den = np.convolve(cpuArray(self.den[mode, :]), p_den)
        max_len = max(len(cp_num), len(cp_den))
        cp_num = np.pad(cp_num, (0, max_len - len(cp_num)), mode='constant')
        cp_den = np.pad(cp_den, (0, max_len - len(cp_den)), mode='constant')
        return cp_num, cp_den

    def frequency_response(self, num, den, fs, freq=None):
        """Compute complex frequency response of IIR filter.
        
        Args:
            num: Numerator coefficients
            den: Denominator coefficients
            fs: Sampling frequency
            freq: Frequency vector (if None, auto-generated)
            
        Returns:
            Complex frequency response values at specified frequencies
        """

        # Convert to CPU arrays
        num_cpu = cpuArray(num)
        den_cpu = cpuArray(den)

        # Remove the zero lowest order coefficients common to num and den (a common factor
        # z^k, which does not change the transfer function). Lowest order first.
        while len(num_cpu) > 1 and num_cpu[0] == 0 and len(den_cpu) > 1 and den_cpu[0] == 0:
            num_cpu = num_cpu[1:]
            den_cpu = den_cpu[1:]

        # Ensure we have at least one coefficient
        if len(num_cpu) == 0:
            num_cpu = np.array([0])
        if len(den_cpu) == 0:
            den_cpu = np.array([1])

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-3, np.log10(fs/2), 1000)

        x = freq / (fs/2) * self.dtype(np.pi)
        z = np.exp(self.complex_dtype(1j) * x, dtype=self.complex_dtype)

        complex_tf = np.zeros(len(freq), dtype=complex)
        for i, zi in enumerate(z):
            num_val = np.polyval(num_cpu[::-1], zi)
            den_val = np.polyval(den_cpu[::-1], zi)
            complex_tf[i] = num_val / den_val if abs(den_val) > 1e-15 else np.inf + 1j * np.inf

        return complex_tf

    def closed_loop_denominator(self, c_num, c_den, p_num, p_den):
        """Calculate closed-loop denominator for feedback system
           given numerator and denominator of control and plant."""

        # if p_num is shorter than p_den, pad with zeros
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')

        # Calculate CP = C * P
        cp_num = np.convolve(c_num, p_num)
        cp_den = np.convolve(c_den, p_den)

        # Ensure same length by padding with zeros
        max_len = max(len(cp_num), len(cp_den))
        cp_num = np.pad(cp_num, (0, max_len - len(cp_num)), mode='constant')
        cp_den = np.pad(cp_den, (0, max_len - len(cp_den)), mode='constant')

        # Calculate closed-loop denominator: Cp_den + Cp_num (from RTF/NTF)
        closed_loop_den = cp_den + cp_num

        return closed_loop_den

    def is_stable(self, mode, dm=None, nw=None, dw=None, delay=None, unity_plant=False):
        """Check stability by analyzing poles of the closed-loop system.
        
        Args:
            mode: Filter mode index
            dm, nw, dw: Plant P = nw / (dm * dw), coefficients lowest order first
            delay: Plant P = discrete_delay_tf(delay), alternative to dm, nw, dw
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            
        Returns:
            bool: True if stable, False otherwise
        """
        cp_num, cp_den = self._loop_tf(mode, delay=delay, dm=dm, nw=nw, dw=dw,
                                       unity_plant=unity_plant)
        closed_loop_den = cp_den + cp_num

        self.logger.debug(f"Closed-loop denominator: {closed_loop_den}")

        # Find poles (roots of denominator)
        try:
            if len(closed_loop_den) > 1:
                poles = np.roots(closed_loop_den[::-1])
            else:
                # Constant denominator - system might be unstable
                return False

            self.logger.debug(f"Poles: {poles}")

            # Check stability: for discrete-time systems, all poles must be inside unit circle: |pole| < 1
            stable = np.all(np.abs(poles) < 1.0)
            max_pole_mag = np.max(np.abs(poles)) if len(poles) > 0 else 0

            self.logger.debug(f"Maximum pole magnitude: {max_pole_mag}")
            self.logger.debug(f"Stable (discrete): {stable}")

            return stable

        except Exception as e:
            self.logger.error(f"Error computing poles: {e}")
            return False

    @lru_cache(maxsize=16384)
    def _compute_max_stable_gain_internal(self, num_tuple, den_tuple, p_num_tuple, p_den_tuple,
                                          max_gain=20.0, n_gain=10000, tolerance=1e-6):
        """Internal computation of maximum stable gain."""

        num_coeffs = np.array(num_tuple)
        den_coeffs = np.array(den_tuple)
        p_num = np.array(p_num_tuple)
        p_den = np.array(p_den_tuple)

        # Test range of gains
        gains = np.linspace(tolerance, max_gain, n_gain)
        max_stable = 0.0

        for gain in gains:
            # Scale controller by gain after normalization of last coefficient (i.e. current gain)
            c_num = num_coeffs/num_coeffs[-1] * gain
            c_den = den_coeffs

            closed_loop_den = self.closed_loop_denominator(c_num, c_den, p_num, p_den)

            # Check stability
            try:
                if len(closed_loop_den) > 1:
                    poles = np.roots(closed_loop_den[::-1])
                    if np.all(np.abs(poles) < 1.0):
                        max_stable = gain
                    else:
                        break  # Found unstable gain, no need to test higher gains
                else:
                    # Constant denominator
                    break
            except:
                break

        return max_stable

    def max_stable_gain(self, mode=None, delay=None, dm=None, nw=None, dw=None, 
                              max_gain=20.0, n_gain=10000, tolerance=1e-6, unity_plant=False):
        """Calculate maximum stable gain for closed-loop system.
        
        This function finds the maximum controller gain that maintains stability
        in a closed-loop system with the given plant dynamics.
        
        Args:
            mode: Filter mode index. If None, calculates for all modes
            delay: Plant P = discrete_delay_tf(delay), delay in frames
            dm, nw, dw: Plant P = nw / (dm * dw), alternative to delay
            max_gain: Maximum gain to test (default: 20.0)
            n_gain: Number of gain values to test (default: 10000)
            tolerance: Minimum gain to test (default: 1e-6)
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            
        Returns:
            float or array: Maximum stable gain(s)
            
        Examples:
            # Single mode with delay
            max_gain = filter_data.max_stable_gain(mode=0, delay=2.5)
            
            # All modes with plant dynamics
            max_gains = filter_data.max_stable_gain(dm=dm_coeffs, nw=nw_coeffs, dw=dw_coeffs)
            
            # All modes with delay (useful for integrator-based controllers)
            max_gains = filter_data.max_stable_gain(delay=1.0)
        """
        p_num, p_den = self._plant(delay=delay, dm=dm, nw=nw, dw=dw, unity_plant=unity_plant)
        p_num_tuple, p_den_tuple = tuple(p_num), tuple(p_den)

        if mode is not None and mode >= self.nfilter:
            raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")
        modes = [mode] if mode is not None else range(self.nfilter)
        max_gains = np.zeros(len(modes))
        for k, i in enumerate(modes):
            max_gains[k] = self._compute_max_stable_gain_internal(
                tuple(cpuArray(self.num[i, :])), tuple(cpuArray(self.den[i, :])),
                p_num_tuple, p_den_tuple, max_gain=max_gain, n_gain=n_gain, tolerance=tolerance)
        return max_gains[0] if mode is not None else max_gains

    def resonance_frequency(self, mode, gain_factor=1.0, delay=None, dm=None, nw=None, dw=None,
                                  fs=1000.0, freq=None, unity_plant=False):
        """Calculate resonance frequency of closed-loop system.
        
        Args:
            mode: Filter mode index
            gain_factor: Factor to multiply the filter gain (default: 1.0)
            delay: Plant P = discrete_delay_tf(delay), delay in frames
            dm, nw, dw: Plant P = nw / (dm * dw), alternative to delay
            fs: Sampling frequency in Hz (default: 1000.0)
            freq: Frequency vector for analysis (default: auto-generated)
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            
        Returns:
            tuple: (resonance_frequency, resonance_amplitude)
        """

        if mode >= self.nfilter:
            raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        cp_num, cp_den = self._loop_tf(mode, gain_factor=gain_factor, delay=delay,
                                       dm=dm, nw=nw, dw=dw, unity_plant=unity_plant)

        # Calculate closed-loop transfer function denominator
        closed_loop_den = cp_den + cp_num

        # Calculate frequency response of denominator
        x = freq / (fs/2) * self.dtype(np.pi)
        z = np.exp(self.complex_dtype(1j) * x, dtype=self.complex_dtype)

        denominator_response = np.zeros(len(freq), dtype=self.complex_dtype)
        for i, zi in enumerate(z):
            denominator_response[i] = np.polyval(closed_loop_den[::-1], zi)

        # Find minimum magnitude (resonance point)
        denominator_magnitude = np.abs(denominator_response)
        resonance_idx = np.argmin(denominator_magnitude)

        resonance_freq = freq[resonance_idx]
        resonance_amplitude = 1.0 / denominator_magnitude[resonance_idx]  # Peak amplitude

        return resonance_freq, resonance_amplitude

    def stability_analysis(self, mode=None, delay=None, dm=None, nw=None, dw=None,
                                 fs=1000.0, max_gain=20.0, n_gain=10000, unity_plant=False):
        """Comprehensive stability analysis for controller(s).
        
        Args:
            mode: Filter mode index. If None, analyzes all modes
            delay: Plant P = discrete_delay_tf(delay), delay in frames
            dm, nw, dw: Plant P = nw / (dm * dw), alternative to delay
            fs: Sampling frequency in Hz (default: 1000.0)
            max_gain: Maximum gain to test (default: 20.0)
            n_gain: Number of gain values to test (default: 10000)
            unity_plant: If True, P = 1. Exactly one of delay, (dm, nw, dw) and
                         unity_plant must be given.
            
        Returns:
            dict: Analysis results containing max_stable_gain, resonance_freq, etc.
        """
        plant = dict(delay=delay, dm=dm, nw=nw, dw=dw, unity_plant=unity_plant)
        modes = [mode] if mode is not None else range(self.nfilter)
        max_stable_gains = np.atleast_1d(self.max_stable_gain(mode=mode, max_gain=max_gain,
                                                              n_gain=n_gain, **plant))
        results = []
        for k, i in enumerate(modes):
            # Calculate resonance frequency at 99% of max stable gain
            if max_stable_gains[k] > 0:
                resonance_freq, resonance_amp = self.resonance_frequency(
                    mode=i, gain_factor=max_stable_gains[k] * 0.99, fs=fs, **plant)
            else:
                resonance_freq, resonance_amp = 0.0, 0.0
            results.append({
                'mode': i,
                'max_stable_gain': max_stable_gains[k],
                'resonance_frequency': resonance_freq,
                'resonance_amplitude': resonance_amp,
                'is_stable_at_max': max_stable_gains[k] > 0
            })
        return results[0] if mode is not None else results

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        storage.write_object(filename, hdr, {'ordnum': self.ordnum, 'ordden': self.ordden,
                                             'num': self.num, 'den': self.den})

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            version = f.header['VERSION']
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            ordnum = f.read('ordnum', 1)
            ordden = f.read('ordden', 2)
            num = f.read('num', 3)
            den = f.read('den', 4)
        return IirFilterData(ordnum, ordden, num, den, target_device_idx=target_device_idx)

    def get_fits_header(self):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_header(hdr):
        # TODO
        raise NotImplementedError()

    def get_value(self):
        # TODO
        raise NotImplementedError()

    def set_value(self, v):
        # TODO
        raise NotImplementedError()

    def discrete_delay_tf(self, delay):
        """Generate transfer function for discrete delay.
        
        If not-integer delay TF:
        DelayTF = z^(−l) * ( m * (1−z^(−1)) + z^(−1) )
        where delay = (l+1)*T − mT, T integration time, l integer, 0<m<1
        
        Args:
            delay: Delay value (can be fractional)
            
        Returns:
            tuple: (num, den) - numerator and denominator coefficients
        """

        if delay - np.trunc(delay) != 0:
            d_m = np.ceil(delay)
            den = np.zeros(int(d_m)+1)
            den[int(d_m)] = 1
            num = den*0
            num[0] = delay - np.trunc(delay)
            num[1] = 1. - num[0]
        else:
            d_m = delay
            den = np.zeros(int(d_m)+1)
            den[int(d_m)] = 1
            num = den*0
            num[0] = 1.

        return num, den

    @staticmethod
    def from_gain_and_ff(gain, ff=None, target_device_idx=None):
        '''Build an IirFilterData object from a gain value/vector
        and an optional forgetting factor value/vector'''

        gain = np.array(gain)
        n = len(gain)

        if ff is None:
            ff = np.ones(n)
        elif len(ff) != n:
            ff = np.full(n, ff)
        else:
            ff = np.array(ff)

        # Filter initialization
        num = np.zeros((n, 2))
        ord_num = np.zeros(n)
        den = np.zeros((n, 2))
        ord_den = np.zeros(n)

        for i in range(n):
            # For a first-order IIR filter with gain and forgetting factor ff:
            # H(z) = gain / (1 - ff * z^(-1))
            # or
            # H(z) = gain * z / (z - ff)
            num[i, 0] = 0
            num[i, 1] = gain[i]
            ord_num[i] = 2
            den[i, 0] = -ff[i]
            den[i, 1] = 1
            ord_den[i] = 2

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

    @staticmethod
    def lpf_from_fc(fc, fs, n_ord=2, target_device_idx=None):
        '''Build an IirFilterData object from a cut off frequency value/vector
        and a filter order value (must be even)'''

        if n_ord != 1 and (n_ord % 2) != 0:
            raise ValueError('Filter order must be 1 or even')

        fc = np.atleast_1d(np.array(fc))
        n = len(fc)

        if n_ord == 1:
            n_coeff = 2
        else:
            n_coeff = 2*n_ord + 1

        # Filter initialization
        num = np.zeros((n, n_coeff))
        ord_num = np.zeros(n)
        den = np.zeros((n, n_coeff))
        ord_den = np.zeros(n)

        for i in range(n):
            if fc[i] >= fs / 2:
                raise ValueError('Cut-off frequency must be less than half the sampling frequency')
            fr = fc[i] / fs  # Normalized frequency
            omega = np.tan(np.pi * fr)

            if n_ord == 1:
                # Butterworth filter of order 1: bilinear transform of 1 / (1 + s / wc),
                # with the prewarped wc, so that the -3 dB point is exactly at fc:
                # H(z) = a0 (1 + z) / (z + b0)
                a0 = omega / (1 + omega)
                b0 = (omega - 1) / (omega + 1)

                num_total = np.asarray([a0.item(), a0.item()], dtype=float)
                den_total = np.asarray([b0.item(), 1], dtype=float)
            else:
                #Butterworth filter of order >=2
                num_total = np.array([1.0])
                den_total = np.array([1.0])

                for k in range(n_ord // 2):  # Iterations on poles
                    ck = 1 + 2 * np.cos(np.pi * (2*k+1) / (2*n_ord)) * omega + omega**2

                    a0 = omega**2 / ck
                    a1 = 2 * a0
                    a2 = a0

                    b1 = 2 * (omega**2 - 1) / ck
                    b2 = (1 - 2 * np.cos(np.pi * (2*k+1) / (2*n_ord)) * omega + omega**2) / ck

                    # coefficients of the single filter of order 2
                    num_k = np.asarray([a2.item(), a1.item(), a0.item()], dtype=float)
                    den_k = np.asarray([b2.item(), b1.item(), 1], dtype=float)

                    # ploynomials convolution to get total filter
                    num_total = np.convolve(num_total, num_k)
                    den_total = np.convolve(den_total, den_k)

            # The coefficients must fit in the preallocated array
            if len(num_total) > n_coeff:
                raise ValueError(f"Filter coefficients longer than expected: {len(num_total)} > {n_coeff}")

            # Right-aligned, i.e. zeros on the lowest order side, as IirFilter expects
            num[i, n_coeff - len(num_total):] = num_total
            den[i, n_coeff - len(den_total):] = den_total
            ord_num[i] = len(num_total)
            ord_den[i] = len(den_total)

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

    @staticmethod
    def lpf_from_fc_and_ampl(fc, ampl, fs, target_device_idx=None):
        '''Build an IirFilterData object from a cut off frequency value/vector
        and amplification    value/vector'''

        fc = np.atleast_1d(np.array(fc))
        ampl = np.atleast_1d(np.array(ampl))
        n = len(fc)

        if len(ampl) != n:
            ampl = np.full(n, ampl)
        else:
            ampl = np.array(ampl)

        n_coeff = 3

        # Filter initialization
        num = np.zeros((n, n_coeff))
        ord_num = np.zeros(n)
        den = np.zeros((n, n_coeff))
        ord_den = np.zeros(n)

        for i in range(n):
            if fc[i] >= fs / 2:
                raise ValueError('Cut-off frequency must be less than half the sampling frequency')
            fr = fc[i] / fs
            omega = 2 * np.pi * fr
            alpha = np.sin(omega) / (2 * ampl[i])

            a0 = (1 - np.cos(omega)) / 2
            a1 = 1 - np.cos(omega)
            a2 = (1 - np.cos(omega)) / 2
            b0 = 1 + alpha
            b1 = -2 * np.cos(omega)
            b2 = 1 - alpha

            a0 /= b0
            a1 /= b0
            a2 /= b0
            b1 /= b0
            b2 /= b0

            num_total = np.asarray([a2.item(), a1.item(), a0.item()], dtype=float)
            den_total = np.asarray([b2.item(), b1.item(), 1], dtype=float)

            num[i, :] = num_total
            den[i, :] = den_total
            ord_num[i] = len(num_total)
            ord_den[i] = len(den_total)

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

# -- Additional methods for control library integration - -

    def _check_control_available(self):
        """Check if control library is available and raise error if not."""
        if not CONTROL_AVAILABLE:
            raise ImportError(
                "The 'control' library is required for this functionality. "
                "Install it with: pip install control"
            )

    @property
    def has_control_support(self):
        """Check if control library support is available."""
        return CONTROL_AVAILABLE

    def to_control_tf(self, mode: int = 0, dt=True):
        """Convert a single filter to a control.TransferFunction object.
        
        Args:
            mode: Index of the filter to convert (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)

        Returns:
            control.TransferFunction: The transfer function object

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        if mode >= self.nfilter:
            raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")

        # Extract numerator and denominator for the specified mode
        num_coeffs = cpuArray(self.num[mode, ::-1])
        den_coeffs = cpuArray(self.den[mode, ::-1])

        # Remove the zero lowest order coefficients common to num and den (at the end,
        # since the arrays are now highest order first): a common factor z^k
        while len(num_coeffs) > 1 and num_coeffs[-1] == 0 and len(den_coeffs) > 1 and den_coeffs[-1] == 0:
            num_coeffs = num_coeffs[:-1]
            den_coeffs = den_coeffs[:-1]

        # Ensure we have at least one coefficient
        if len(num_coeffs) == 0:
            num_coeffs = np.array([0])
        if len(den_coeffs) == 0:
            den_coeffs = np.array([1])

        return control.TransferFunction(num_coeffs, den_coeffs, dt=dt)

    def to_control_tf_list(self, dt=True):
        """Convert all filters to a list of control.TransferFunction objects.

        Args:
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)

        Returns:
            list: List of control.TransferFunction objects

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf_list = []
        for i in range(self.nfilter):
            tf_list.append(self.to_control_tf(mode=i, dt=dt))
        return tf_list

    @staticmethod
    def from_control_tf(tf_list, target_device_idx: int = None):
        """Create IirFilterData from control.TransferFunction objects.

        Args:
            tf_list: Single control.TransferFunction or list of control.TransferFunction objects
            target_device_idx: Target device index (default: None)

        Returns:
            IirFilterData: New IirFilterData object
        """
        if not CONTROL_AVAILABLE:
            raise ImportError(
                "The 'control' library is required for this functionality. "
                "Install it with: pip install control"
            )

        # Handle single transfer function
        if isinstance(tf_list, control.TransferFunction):
            tf_list = [tf_list]

        n_filters = len(tf_list)

        # Find maximum coefficient lengths
        max_num_len = max(len(tf.num[0][0]) for tf in tf_list)
        max_den_len = max(len(tf.den[0][0]) for tf in tf_list)

        # Use the maximum of num and den lengths for both arrays
        max_len = max(max_num_len, max_den_len)

        # Initialize arrays with same size
        num = np.zeros((n_filters, max_len))
        den = np.zeros((n_filters, max_len))
        ord_num = np.zeros(n_filters, dtype=int)
        ord_den = np.zeros(n_filters, dtype=int)

        for i, tf in enumerate(tf_list):
            # Get coefficients
            num_coeffs = tf.num[0][0]
            den_coeffs = tf.den[0][0]

            # Store actual orders (length of coefficient arrays)
            ord_num[i] = len(num_coeffs)
            ord_den[i] = len(den_coeffs)

            # python-control is highest order first: reverse, and right-align (zeros on
            # the lowest order side), as IirFilter expects
            num[i, max_len - len(num_coeffs):] = num_coeffs[::-1]
            den[i, max_len - len(den_coeffs):] = den_coeffs[::-1]

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

    def bode_plot(self, mode: int = 0, dt=True, omega: np.ndarray = None,
                  plot: bool = True, **kwargs):
        """Create Bode plot for a specific filter using control library.

        Args:
            mode: Index of the filter to plot (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            omega: Frequency vector (default: auto-generated)
            plot: Whether to display the plot (default: True)
            **kwargs: Additional arguments passed to control.bode_plot

        Returns:
            tuple: (magnitude, phase, frequency) arrays
            or ControlPlot object

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if omega is None:
            # Auto-generate frequency vector
            if dt is not None:
                # Discrete-time system
                omega = np.logspace(-3, np.log10(np.pi/_sampling_time(dt)), 1000)
            else:
                # Continuous-time system
                omega = np.logspace(-2, 4, 1000)

        out = control.bode_plot(tf, omega=omega, plot=plot, **kwargs)

        if hasattr(out, 'mag'):
            return out.mag, out.phase, omega
        else:
            return out

    def nyquist_plot(self, mode: int = 0, dt=True, omega: np.ndarray = None,
                     plot: bool = True, **kwargs):
        """Create Nyquist plot for a specific filter using control library.

        Args:
            mode: Index of the filter to plot (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            omega: Frequency vector (default: auto-generated)
            plot: Whether to display the plot (default: True)
            **kwargs: Additional arguments passed to control.nyquist_plot

        Returns:
            tuple: (real, imaginary, frequency) arrays
            or ControlPlot object

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if omega is None:
            # Auto-generate frequency vector
            if dt is not None:
                # Discrete-time system
                omega = np.logspace(-3, np.log10(np.pi/_sampling_time(dt)), 1000)
            else:
                # Continuous-time system
                omega = np.logspace(-2, 4, 1000)

        # Makes plot and get response data
        out = control.nyquist_plot(tf, omega=omega, plot=plot, **kwargs)

        if hasattr(out, 'response'):
            return out.response.real, out.response.imag, omega
        elif hasattr(out, 'real'):
            return out.real, out.imag, omega
        else:
            return out

    def step_response(self, mode: int = 0, dt=True, T: np.ndarray = None, **kwargs):
        """Compute step response for a specific filter using control library.

        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            T: Time vector (default: auto-generated)
            **kwargs: Additional arguments passed to control.step_response

        Returns:
            tuple: (time, response) arrays

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if T is None:
            if dt is not None:
                # Discrete-time system
                T = np.arange(0, 100) * _sampling_time(dt)
            else:
                # Continuous-time system
                T = np.linspace(0, 10, 1000)

        time, response = control.step_response(tf, T=T, **kwargs)
        return time, response

    def impulse_response(self, mode: int = 0, dt=True, T: np.ndarray = None, **kwargs):
        """Compute impulse response for a specific filter using control library.

        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            T: Time vector (default: auto-generated)
            **kwargs: Additional arguments passed to control.impulse_response

        Returns:
            tuple: (time, response) arrays

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if T is None:
            if dt is not None:
                # Discrete-time system
                T = np.arange(0, 100) * _sampling_time(dt)
            else:
                # Continuous-time system
                T = np.linspace(0, 10, 1000)

        time, response = control.impulse_response(tf, T=T, **kwargs)
        return time, response

    def stability_margins(self, mode: int = 0, dt=True, delay=None, dm=None, nw=None, dw=None,
                          unity_plant=False):
        """Compute the stability margins of the open loop C P, using the control library.

        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            delay: Plant P = discrete_delay_tf(delay), delay in frames
            dm, nw, dw: Plant P = nw / (dm * dw), alternative to delay
            unity_plant: If True, P = 1 (margins of the filter alone). Exactly one of
                         delay, (dm, nw, dw) and unity_plant must be given.

        Returns:
            tuple: (gain_margin, phase_margin, wg, wp) where:
                   - gain_margin: Gain margin in dB
                   - phase_margin: Phase margin in degrees
                   - wg: Frequency at gain margin
                   - wp: Frequency at phase margin

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        if mode >= self.nfilter:
            raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")
        cp_num, cp_den = self._loop_tf(mode, delay=delay, dm=dm, nw=nw, dw=dw,
                                       unity_plant=unity_plant)
        # python-control wants the coefficients highest order first
        tf = control.TransferFunction(np.trim_zeros(cp_num[::-1], 'f'),
                                      np.trim_zeros(cp_den[::-1], 'f'), dt=dt)
        gm, pm, wg, wp = control.margin(tf)

        gm_db = 20 * np.log10(gm) if (gm is not None and gm > 0) else np.inf
        return gm_db, pm, wg, wp

    def pole_zero_map(self, mode: int = 0, dt=True, plot: bool = True, **kwargs):
        """Create pole-zero map for a specific filter using control library.

        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time of the discrete-time system (default: True, discrete time
                with an unspecified sampling time; None for continuous time)
            plot: Whether to display the plot (default: True)
            **kwargs: Additional arguments passed to control.pzmap

        Returns:
            tuple: (poles, zeros) arrays

        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)
        poles, zeros = control.pzmap(tf, plot=plot, **kwargs)
        return poles, zeros
