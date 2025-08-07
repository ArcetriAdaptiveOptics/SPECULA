import numpy as np

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
    """IIR Filter Data representation.
    
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
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
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
        C_num = self.num[mode, :]
        C_den = self.den[mode, :]

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            P_num = nw
            P_den = np.convolve(dm, dw)
        else:
            P_num = np.array([1])  # Unity plant numerator
            P_den = np.array([1])  # Unity plant denominator

        # if P_num is shorter than P_den, pad with zeros
        if len(P_num) < len(P_den):
            P_num = np.pad(P_num, (0, len(P_den) - len(P_num)), mode='constant')

        # Calculate CP = C * P
        CP_num = np.convolve(C_num, P_num)
        CP_den = np.convolve(C_den, P_den)

        # Ensure same length by padding with zeros
        max_len = max(len(CP_num), len(CP_den))
        CP_num = np.pad(CP_num, (max_len - len(CP_num), 0), mode='constant')
        CP_den = np.pad(CP_den, (max_len - len(CP_den), 0), mode='constant')

        # Calculate RTF = 1 / (1 + CP) = CP_den / (CP_den + CP_num)
        rtf_num = CP_den
        rtf_den = CP_den + CP_num

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
        C_num = self.num[mode, :]
        C_den = self.den[mode, :]

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            P_num = nw
            P_den = np.convolve(dm, dw)
        else:
            P_num = np.array([1])  # Unity plant numerator
            P_den = np.array([1])  # Unity plant denominator

        # if P_num is shorter than P_den, pad with zeros
        if len(P_num) < len(P_den):
            P_num = np.pad(P_num, (0, len(P_den) - len(P_num)), mode='constant')
        
        # Calculate CP = C * P
        CP_num = np.convolve(C_num, P_num)
        CP_den = np.convolve(C_den, P_den)

        # Ensure same length by padding with zeros
        max_len = max(len(CP_num), len(CP_den))
        CP_num = np.pad(CP_num, (max_len - len(CP_num), 0), mode='constant')
        CP_den = np.pad(CP_den, (max_len - len(CP_den), 0), mode='constant')

        # Calculate NTF = CP / (1 + CP) = CP_num / (CP_den + CP_num)
        ntf_num = CP_num
        ntf_den = CP_den + CP_num

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
        C_num = cpuArray(self.num[mode, :])
        C_den = cpuArray(self.den[mode, :])

        # Get plant coefficients P from dm, nw, dw
        if dm is not None and nw is not None and dw is not None:
            P_num = cpuArray(nw)
            P_den = cpuArray(np.convolve(cpuArray(dm), cpuArray(dw)))
        else:
            P_num = np.array([1])  # Unity plant numerator
            P_den = np.array([1])  # Unity plant denominator

        # if P_num is shorter than P_den, pad with zeros
        if len(P_num) < len(P_den):
            P_num = np.pad(P_num, (0, len(P_den) - len(P_num)), mode='constant')

        # Calculate CP = C * P
        CP_num = np.convolve(C_num, P_num)
        CP_den = np.convolve(C_den, P_den)

        # Ensure same length by padding with zeros
        max_len = max(len(CP_num), len(CP_den))
        CP_num = np.pad(CP_num, (max_len - len(CP_num), 0), mode='constant')
        CP_den = np.pad(CP_den, (max_len - len(CP_den), 0), mode='constant')

        # Calculate closed-loop denominator: CP_den + CP_num (from RTF/NTF)
        closed_loop_den = CP_den + CP_num

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

    @staticmethod
    def restore(filename, target_device_idx=None):
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
    
    def set_value(self, v, force_copy=False):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_header(hdr):
        raise NotImplementedError

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