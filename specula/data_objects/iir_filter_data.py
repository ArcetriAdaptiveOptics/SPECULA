import numpy as np
from functools import lru_cache

from specula import cpuArray
from specula.base_data_obj import BaseDataObj

from astropy.io import fits

# Try to import control library, but make it optional
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

class IirFilterData(BaseDataObj):
    """:class:`~specula.data_objects.iir_filter_data.IirFilterData` - IIR Filter Data representation.
    
    This class stores IIR filter coefficients in the following format:
    - Coefficients are stored with highest order terms first
    - num[i, :] contains numerator coefficients for filter i
    - den[i, :] contains denominator coefficients for filter i
    - ordnum[i] and ordden[i] specify the actual order of each filter
    
    Transfer function: H(z) = (num[0] + num[1]*z^-1 + ...) / (den[0] + den[1]*z^-1 + ...)
    """
    def __init__(self,
                 ordnum: list,
                 ordden: list,
                 num,
                 den,
                 n_modes=None,
                 target_device_idx: int=None,
                 precision: int=None):
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
        self.set_num(self.to_xp(num, dtype=self.dtype))
        self.set_den(self.to_xp(den, dtype=self.dtype))

    @property
    def nfilter(self):
        return len(self.num)

    def get_zeros(self):
        if self.zeros is None:
            snum1 = self.num.shape[1]
            zeros = self.xp.zeros((self.nfilter, snum1 - 1), dtype=self.dtype)
            for i in range(self.nfilter):
                if self.ordnum[i] > 1:
                    roots = self.xp.roots(self.num[i, snum1 - int(self.ordnum[i]):])
                    if np.sum(np.abs(roots)) > 0:
                        zeros[i, :int(self.ordnum[i]) - 1] = roots
            self.zeros = zeros
        return self.zeros

    def get_poles(self):
        if self.poles is None:
            sden1 = self.den.shape[1]
            poles = self.xp.zeros((self.nfilter, sden1 - 1), dtype=self.dtype)
            for i in range(self.nfilter):
                if self.ordden[i] > 1:
                    poles[i, :int(self.ordden[i]) - 1] = self.xp.roots(self.den[i, sden1 - int(self.ordden[i]):])
            self.poles = poles
        return self.poles

    def set_num(self, num):
        snum1 = num.shape[1]
        mynum = num.copy()
        for i in range(len(mynum)):
            if self.ordnum[i] < snum1:
                if np.sum(self.xp.abs(mynum[i, int(self.ordnum[i]):])) == 0:
                    mynum[i, :] = self.xp.roll(mynum[i, :], snum1 - int(self.ordnum[i]))

        gain = self.xp.zeros(len(mynum), dtype=self.dtype)
        for i in range(len(gain)):
            gain[i] = mynum[i, - 1]
        self.gain = gain
        self.zeros = None 
        self.num = self.to_xp(mynum, dtype=self.dtype)

    def set_den(self, den):
        sden1 = den.shape[1]
        myden = den.copy()
        for i in range(len(myden)):
            if self.ordden[i] < sden1:
                if np.sum(self.xp.abs(myden[i, int(self.ordden[i]):])) == 0:
                    myden[i, :] = self.xp.roll(myden[i, :], sden1 - int(self.ordden[i]))

        self.den = self.to_xp(myden, dtype=self.dtype)
        self.poles = None

    def set_zeros(self, zeros):
        self.zeros = self.to_xp(zeros, dtype=self.dtype)
        num = self.xp.zeros((self.nfilter, self.zeros.shape[1] + 1), dtype=self.dtype)
        snum1 = num.shape[1]
        for i in range(self.nfilter):
            if self.ordnum[i] > 1:
                num[i, snum1 - int(self.ordnum[i]):] = self.xp.poly(self.zeros[i, :int(self.ordnum[i]) - 1])
        self.num = num

    def set_poles(self, poles):
        self.poles = self.to_xp(poles, dtype=self.dtype)
        den = self.xp.zeros((self.nfilter, self.poles.shape[1] + 1), dtype=self.dtype)
        sden1 = den.shape[1]
        for i in range(self.nfilter):
            if self.ordden[i] > 1:
                den[i, sden1 - int(self.ordden[i]):] = self.xp.poly(self.poles[i, :int(self.ordden[i]) - 1])
        self.den = den

    def set_gain(self, gain, verbose=False):
        gain = self.to_xp(gain, dtype=self.dtype)
        if verbose:
            print('original gain:', self.gain)
        if self.xp.size(gain) < self.nfilter:
            nfilter = np.size(gain)
        else:
            nfilter = self.nfilter
        if self.gain is None:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self.ordnum[i] > 1:
                        self.num[i, :] *= gain[i]
                    else:
                        self.num[i, - 1] = gain[i]
                else:
                    gain[i] = self.num[i, - 1]
        else:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self.ordnum[i] > 1:
                        self.num[i, :] *= (gain[i] / self.gain[i])
                    else:
                        self.num[i, - 1] = gain[i] / self.gain[i]
                else:
                    gain[i] = self.gain[i]
        self.gain = self.to_xp(gain, dtype=self.dtype)
        if verbose:
            print('new gain:', self.gain)

    def RTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, plot=True, overplot=False, **extra):
        """Plot Rejection Transfer Function: RTF = 1 / (1 - CP)"""
        plotTitle = title if title else 'Rejection Transfer Function'

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        # Get controller coefficients C
        c_num = cpuArray(self.num[mode, :])
        c_den = cpuArray(self.den[mode, :])

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            nw = cpuArray(nw)
            dm = cpuArray(dm)
            dw = cpuArray(dw)
            p_num = nw
            p_den = np.convolve(dm, dw)
        else:
            p_num = np.array([1])  # Unity plant numerator
            p_den = np.array([1])  # Unity plant denominator

        # if p_num is shorter than p_den, pad with zeros
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')

        # Calculate CP = C * P
        Cp_num = np.convolve(c_num, p_num)
        Cp_den = np.convolve(c_den, p_den)

        # Ensure same length by padding with zeros
        max_len = max(len(Cp_num), len(Cp_den))
        Cp_num = np.pad(Cp_num, (max_len - len(Cp_num), 0), mode='constant')
        Cp_den = np.pad(Cp_den, (max_len - len(Cp_den), 0), mode='constant')

        # Calculate RTF = 1 / (1 + CP) = Cp_den / (Cp_den + Cp_num)
        rtf_num = Cp_den
        rtf_den = Cp_den + Cp_num

        if verbose:
            print(f"RTF numerator: {rtf_num}")
            print(f"RTF denominator: {rtf_den}")

        # Calculate frequency response
        rtf_complex = self.frequency_response(rtf_num, rtf_den, fs, freq=freq)
        rtf_mag = np.abs(rtf_complex)

        if plot:
            import matplotlib.pyplot as plt
            if overplot:
                color = extra.get('color', 'blue')
                plt.plot(freq, rtf_mag, color=color, **extra)
            else:
                plt.figure()
                plt.loglog(freq, rtf_mag, label=plotTitle)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Magnitude')
                plt.title(plotTitle)
                plt.grid(True)
                plt.legend()
                plt.show()

        return rtf_mag

    def NTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, plot=True, overplot=False, **extra):
        """Plot Noise Transfer Function: NTF = CP / (1 - CP)"""
        plotTitle = title if title else 'Noise Transfer Function'

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        # Get controller coefficients C
        c_num = cpuArray(self.num[mode, :])
        c_den = cpuArray(self.den[mode, :])

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            nw = cpuArray(nw)
            dm = cpuArray(dm)
            dw = cpuArray(dw)
            p_num = nw
            p_den = np.convolve(dm, dw)
        else:
            p_num = np.array([1])  # Unity plant numerator
            p_den = np.array([1])  # Unity plant denominator

        # if p_num is shorter than p_den, pad with zeros
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')

        # Calculate CP = C * P
        Cp_num = np.convolve(c_num, p_num)
        Cp_den = np.convolve(c_den, p_den)

        # Ensure same length by padding with zeros
        max_len = max(len(Cp_num), len(Cp_den))
        Cp_num = np.pad(Cp_num, (max_len - len(Cp_num), 0), mode='constant')
        Cp_den = np.pad(Cp_den, (max_len - len(Cp_den), 0), mode='constant')

        # Calculate NTF = CP / (1 + CP) = Cp_num / (Cp_den + Cp_num)
        ntf_num = Cp_num
        ntf_den = Cp_den + Cp_num

        if verbose:
            print(f"NTF numerator: {ntf_num}")
            print(f"NTF denominator: {ntf_den}")

        # Calculate frequency response
        ntf_complex = self.frequency_response(ntf_num, ntf_den, fs, freq=freq)
        ntf_mag = np.abs(ntf_complex)

        if plot:
            import matplotlib.pyplot as plt
            if overplot:
                color = extra.get('color', 'red')
                plt.plot(freq, ntf_mag, color=color, **extra)
            else:
                plt.figure()
                plt.loglog(freq, ntf_mag, label=plotTitle)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Magnitude')
                plt.title(plotTitle)
                plt.grid(True)
                plt.legend()
                plt.show()

        return ntf_mag

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

        # Remove initial zeros (coefficients are stored highest order first)
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

        x = freq.copy() / (fs/2) * np.pi
        z = np.exp(1j * x)

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
        cp_num = np.pad(cp_num, (max_len - len(cp_num), 0), mode='constant')
        cp_den = np.pad(cp_den, (max_len - len(cp_den), 0), mode='constant')

        # Calculate closed-loop denominator: Cp_den + Cp_num (from RTF/NTF)
        closed_loop_den = cp_den + cp_num

        return closed_loop_den

    def is_stable(self, mode, dm=None, nw=None, dw=None, verbose=False):
        """Check stability by analyzing poles of the closed-loop system.
        
        Args:
            mode: Filter mode index
            dm, nw, dw: Plant coefficients (optional)
            verbose: Print debug information
            
        Returns:
            bool: True if stable, False otherwise
        """

        # Get controller coefficients C
        c_num = cpuArray(self.num[mode, :])
        c_den = cpuArray(self.den[mode, :])

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            p_num = cpuArray(nw)
            p_den = cpuArray(np.convolve(cpuArray(dm), cpuArray(dw)))
        else:
            p_num = np.array([1])  # Unity plant numerator
            p_den = np.array([1])  # Unity plant denominator

        closed_loop_den = self.closed_loop_denominator(c_num, c_den, p_num, p_den)

        if verbose:
            print(f"Closed-loop denominator: {closed_loop_den}")

        # Find poles (roots of denominator)
        try:
            if len(closed_loop_den) > 1:
                poles = np.roots(closed_loop_den[::-1])
            else:
                # Constant denominator - system might be unstable
                return False

            if verbose:
                print(f"Poles: {poles}")

            # Check stability: for discrete-time systems, all poles must be inside unit circle: |pole| < 1
            stable = np.all(np.abs(poles) < 1.0)
            max_pole_mag = np.max(np.abs(poles)) if len(poles) > 0 else 0

            if verbose:
                print(f"Maximum pole magnitude: {max_pole_mag}")
                print(f"Stable (discrete): {stable}")

            return stable

        except Exception as e:
            if verbose:
                print(f"Error computing poles: {e}")
            return False

    @lru_cache(maxsize=16384)
    def _compute_max_stable_gain_internal(self, num_tuple, den_tuple, delay=None, dm_tuple=None, nw_tuple=None, dw_tuple=None,
                                        max_gain=20.0, n_gain=10000, tolerance=1e-6):
        """Internal computation of maximum stable gain."""

        num_coeffs = np.array(num_tuple)
        den_coeffs = np.array(den_tuple)
        dm = np.array(dm_tuple) if dm_tuple is not None else None
        nw = np.array(nw_tuple) if nw_tuple is not None else None
        dw = np.array(dw_tuple) if dw_tuple is not None else None

        # Create plant transfer function
        if delay is not None:
            # Use discrete delay transfer function
            p_num, p_den = self.discrete_delay_tf(delay)
        elif dm is not None and nw is not None and dw is not None:
            p_num = cpuArray(nw)
            p_den = cpuArray(np.convolve(cpuArray(dm), cpuArray(dw)))
        else:
            # Unity plant
            p_num = np.array([1])
            p_den = np.array([1])

        # Pad p_num if shorter than p_den
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')

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
                    max_gain=20.0, n_gain=10000, tolerance=1e-6):
        """Calculate maximum stable gain for closed-loop system.
        
        This function finds the maximum controller gain that maintains stability
        in a closed-loop system with the given plant dynamics.
        
        Args:
            mode: Filter mode index. If None, calculates for all modes
            delay: Delay in frames (alternative to dm/nw/dw)
            dm, nw, dw: Plant coefficients (alternative to delay)
            max_gain: Maximum gain to test (default: 20.0)
            n_gain: Number of gain values to test (default: 10000)
            tolerance: Minimum gain to test (default: 1e-6)
            
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

        if mode is not None:
            # Single mode calculation
            if mode >= self.nfilter:
                raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")

            num_coeffs = cpuArray(self.num[mode, :])
            den_coeffs = cpuArray(self.den[mode, :])

            # Create hashable tuples for caching
            num_tuple = tuple(num_coeffs)
            den_tuple = tuple(den_coeffs)
            dm_tuple = tuple(cpuArray(dm)) if dm is not None else None
            nw_tuple = tuple(cpuArray(nw)) if nw is not None else None
            dw_tuple = tuple(cpuArray(dw)) if dw is not None else None

            return self._compute_max_stable_gain_internal(
                num_tuple, den_tuple, delay=delay, dm_tuple=dm_tuple, nw_tuple=nw_tuple, dw_tuple=dw_tuple,
                max_gain=max_gain, n_gain=n_gain, tolerance=tolerance
            )
        else:
            # All modes calculation
            max_gains = np.zeros(self.nfilter)

            # Calculate for each mode separately
            for i in range(self.nfilter):
                num_coeffs = cpuArray(self.num[i, :])
                den_coeffs = cpuArray(self.den[i, :])

                num_tuple = tuple(num_coeffs)
                den_tuple = tuple(den_coeffs)
                dm_tuple = tuple(cpuArray(dm)) if dm is not None else None
                nw_tuple = tuple(cpuArray(nw)) if nw is not None else None
                dw_tuple = tuple(cpuArray(dw)) if dw is not None else None

                max_gains[i] = self._compute_max_stable_gain_internal(
                    num_tuple, den_tuple, delay=delay, dm_tuple=dm_tuple, nw_tuple=nw_tuple, dw_tuple=dw_tuple,
                    max_gain=max_gain, n_gain=n_gain, tolerance=tolerance
                )

            return max_gains

    def resonance_frequency(self, mode, gain_factor=1.0, delay=None, dm=None, nw=None, dw=None, 
                           fs=1000.0, freq=None):
        """Calculate resonance frequency of closed-loop system.
        
        Args:
            mode: Filter mode index
            gain_factor: Factor to multiply the filter gain (default: 1.0)
            delay: Delay in frames (alternative to dm/nw/dw)
            dm, nw, dw: Plant coefficients (alternative to delay)
            fs: Sampling frequency in Hz (default: 1000.0)
            freq: Frequency vector for analysis (default: auto-generated)
            
        Returns:
            tuple: (resonance_frequency, resonance_amplitude)
        """

        if mode >= self.nfilter:
            raise ValueError(f"Mode {mode} exceeds number of filters {self.nfilter}")

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        # Get controller coefficients C with gain factor
        c_num = cpuArray(self.num[mode, :]) * gain_factor
        c_den = cpuArray(self.den[mode, :])

        # Create plant transfer function
        if delay is not None:
            p_num, p_den = self.discrete_delay_tf(delay)
        elif dm is not None and nw is not None and dw is not None:
            p_num = cpuArray(nw)
            p_den = cpuArray(np.convolve(cpuArray(dm), cpuArray(dw)))
        else:
            p_num = np.array([1])
            p_den = np.array([1])

        # Pad p_num if shorter than p_den
        if len(p_num) < len(p_den):
            p_num = np.pad(p_num, (0, len(p_den) - len(p_num)), mode='constant')

        # Calculate CP = C * P
        Cp_num = np.convolve(c_num, p_num)
        Cp_den = np.convolve(c_den, p_den)

        # Ensure same length by padding with zeros
        max_len = max(len(Cp_num), len(Cp_den))
        Cp_num = np.pad(Cp_num, (max_len - len(Cp_num), 0), mode='constant')
        Cp_den = np.pad(Cp_den, (max_len - len(Cp_den), 0), mode='constant')

        # Calculate closed-loop transfer function denominator
        closed_loop_den = Cp_den + Cp_num

        # Calculate frequency response of denominator
        x = freq.copy() / (fs/2) * np.pi
        z = np.exp(1j * x)

        denominator_response = np.zeros(len(freq), dtype=complex)
        for i, zi in enumerate(z):
            denominator_response[i] = np.polyval(closed_loop_den[::-1], zi)

        # Find minimum magnitude (resonance point)
        denominator_magnitude = np.abs(denominator_response)
        resonance_idx = np.argmin(denominator_magnitude)

        resonance_freq = freq[resonance_idx]
        resonance_amplitude = 1.0 / denominator_magnitude[resonance_idx]  # Peak amplitude

        return resonance_freq, resonance_amplitude

    def stability_analysis(self, mode=None, delay=None, dm=None, nw=None, dw=None, 
                          fs=1000.0, max_gain=20.0, n_gain=10000):
        """Comprehensive stability analysis for controller(s).
        
        Args:
            mode: Filter mode index. If None, analyzes all modes
            delay: Delay in frames (alternative to dm/nw/dw)
            dm, nw, dw: Plant coefficients (alternative to delay)
            fs: Sampling frequency in Hz (default: 1000.0)
            max_gain: Maximum gain to test (default: 20.0)
            n_gain: Number of gain values to test (default: 10000)
            
        Returns:
            dict: Analysis results containing max_stable_gain, resonance_freq, etc.
        """

        if mode is not None:
            # Single mode analysis
            max_stable = self.max_stable_gain(
                mode=mode, delay=delay, dm=dm, nw=nw, dw=dw,
                max_gain=max_gain, n_gain=n_gain
            )

            # Calculate resonance frequency at 99% of max stable gain
            if max_stable > 0:
                resonance_freq, resonance_amp = self.resonance_frequency(
                    mode=mode, gain_factor=max_stable * 0.99,
                    delay=delay, dm=dm, nw=nw, dw=dw, fs=fs
                )
            else:
                resonance_freq, resonance_amp = 0.0, 0.0

            return {
                'mode': mode,
                'max_stable_gain': max_stable,
                'resonance_frequency': resonance_freq,
                'resonance_amplitude': resonance_amp,
                'is_stable_at_max': max_stable > 0
            }
        else:
            # All modes analysis
            max_stable_gains = self.max_stable_gain(
                delay=delay, dm=dm, nw=nw, dw=dw,
                max_gain=max_gain, n_gain=n_gain
            )

            results = []
            for i in range(self.nfilter):
                # Calculate resonance frequency at 99% of max stable gain
                if max_stable_gains[i] > 0:
                    resonance_freq, resonance_amp = self.resonance_frequency(
                        mode=i, gain_factor=max_stable_gains[i] * 0.99,
                        delay=delay, dm=dm, nw=nw, dw=dw, fs=fs
                    )
                else:
                    resonance_freq, resonance_amp = 0.0, 0.0

                results.append({
                    'mode': i,
                    'max_stable_gain': max_stable_gains[i],
                    'resonance_frequency': resonance_freq,
                    'resonance_amplitude': resonance_amp,
                    'is_stable_at_max': max_stable_gains[i] > 0
                })

            return results

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.ordnum), name='ORDNUM'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.ordden), name='ORDDEN'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.num), name='NUM'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.den), name='DEN'))
        hdul.writeto(filename, overwrite=True)
        hdul.close()  # Force close for Windows

    @staticmethod
    def restore(filename, target_device_idx=None):
        # pylint: disable=no-member # members of HDUList[i] are created dynamically by pyfits
        with fits.open(filename) as hdul:
            hdr = hdul[0].header     
            version = hdr['VERSION']
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            ordnum = hdul[1].data
            ordden = hdul[2].data
            num = hdul[3].data
            den = hdul[4].data
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

        if delay - np.fix(delay) != 0:
            d_m = np.ceil(delay)
            den = np.zeros(int(d_m)+1)
            den[int(d_m)] = 1
            num = den*0
            num[0] = delay - np.fix(delay)
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
                # Butterworth filter of order 1
                a0 = omega / (1 + omega)
                b1 = -(1 - a0)

                num_total = np.asarray([0, a0.item()], dtype=float)
                den_total = np.asarray([b1.item(), 1], dtype=float)
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

            # Assicurati che i coefficienti si adattino all'array pre-allocato
            if len(num_total) > n_coeff:
                raise ValueError(f"Filter coefficients longer than expected: {len(num_total)} > {n_coeff}")

            # Pad with zeros at the beginning (highest order terms first)
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

    def to_control_tf(self, mode: int = 0, dt: float = None):
        """Convert a single filter to a control.TransferFunction object.
        
        Args:
            mode: Index of the filter to convert (default: 0)
            dt: Sampling time for discrete-time system (default: None for continuous-time)
            
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

        # Remove final zeros (highest order first)
        while len(num_coeffs) > 1 and num_coeffs[-1] == 0 and len(den_coeffs) > 1 and den_coeffs[-1] == 0:
            num_coeffs = num_coeffs[:-1]
            den_coeffs = den_coeffs[:-1]

        # Ensure we have at least one coefficient
        if len(num_coeffs) == 0:
            num_coeffs = np.array([0])
        if len(den_coeffs) == 0:
            den_coeffs = np.array([1])

        return control.TransferFunction(num_coeffs, den_coeffs, dt=dt)

    def to_control_tf_list(self, dt: float = None):
        """Convert all filters to a list of control.TransferFunction objects.
        
        Args:
            dt: Sampling time for discrete-time system (default: None for continuous-time)
            
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

            # Pad with zeros at the beginning (highest order terms first)
            num[i, max_len - len(num_coeffs):] = num_coeffs[::-1]
            den[i, max_len - len(den_coeffs):] = den_coeffs[::-1]

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

    def bode_plot(self, mode: int = 0, dt: float = None, omega: np.ndarray = None,
                  plot: bool = True, **kwargs):
        """Create Bode plot for a specific filter using control library.
        
        Args:
            mode: Index of the filter to plot (default: 0)
            dt: Sampling time for discrete-time system (default: None)
            omega: Frequency vector (default: auto-generated)
            plot: Whether to display the plot (default: True)
            **kwargs: Additional arguments passed to control.bode_plot
            
        Returns:
            tuple: (magnitude, phase, frequency) arrays
            
        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if omega is None:
            # Auto-generate frequency vector
            if dt is not None:
                # Discrete-time system
                omega = np.logspace(-3, np.log10(np.pi/dt), 1000)
            else:
                # Continuous-time system
                omega = np.logspace(-2, 4, 1000)

        mag, phase, freq = control.bode_plot(tf, omega=omega, plot=plot, **kwargs)
        return mag, phase, freq

    def nyquist_plot(self, mode: int = 0, dt: float = None, omega: np.ndarray = None,
                     plot: bool = True, **kwargs):
        """Create Nyquist plot for a specific filter using control library.
        
        Args:
            mode: Index of the filter to plot (default: 0)
            dt: Sampling time for discrete-time system (default: None)
            omega: Frequency vector (default: auto-generated)
            plot: Whether to display the plot (default: True)
            **kwargs: Additional arguments passed to control.nyquist_plot
            
        Returns:
            tuple: (real, imaginary, frequency) arrays
            
        Raises:
            ImportError: If control library is not installed
        """
        self._check_control_available()

        tf = self.to_control_tf(mode=mode, dt=dt)

        if omega is None:
            # Auto-generate frequency vector
            if dt is not None:
                # Discrete-time system
                omega = np.logspace(-3, np.log10(np.pi/dt), 1000)
            else:
                # Continuous-time system
                omega = np.logspace(-2, 4, 1000)

        real, imag, freq = control.nyquist_plot(tf, omega=omega, plot=plot, **kwargs)
        return real, imag, freq

    def step_response(self, mode: int = 0, dt: float = None, T: np.ndarray = None, **kwargs):
        """Compute step response for a specific filter using control library.
        
        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time for discrete-time system (default: None)
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
                T = np.arange(0, 100) * dt
            else:
                # Continuous-time system
                T = np.linspace(0, 10, 1000)

        time, response = control.step_response(tf, T=T, **kwargs)
        return time, response

    def impulse_response(self, mode: int = 0, dt: float = None, T: np.ndarray = None, **kwargs):
        """Compute impulse response for a specific filter using control library.
        
        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time for discrete-time system (default: None)
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
                T = np.arange(0, 100) * dt
            else:
                # Continuous-time system
                T = np.linspace(0, 10, 1000)

        time, response = control.impulse_response(tf, T=T, **kwargs)
        return time, response

    def stability_margins(self, mode: int = 0, dt: float = None):
        """Compute stability margins for a specific filter using control library.
        
        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time for discrete-time system (default: None)
            
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

        tf = self.to_control_tf(mode=mode, dt=dt)
        gm, pm, wg, wp = control.margin(tf)
        return gm, pm, wg, wp

    def pole_zero_map(self, mode: int = 0, dt: float = None, plot: bool = True, **kwargs):
        """Create pole-zero map for a specific filter using control library.
        
        Args:
            mode: Index of the filter (default: 0)
            dt: Sampling time for discrete-time system (default: None)
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
