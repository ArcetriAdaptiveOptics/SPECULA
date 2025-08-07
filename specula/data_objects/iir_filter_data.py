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

    def complexRTF(self, mode, fs, delay, freq=None, verbose=False):
        if delay > 1:
            dm = self.xp.array([0.0, 1.0], dtype=self.dtype)
            nm = self.xp.array([1.0, 0.0], dtype=self.dtype)
            nw, dw = self.discrete_delay_tf(delay - 1)
        else:
            dm = self.xp.array([1.0], dtype=self.dtype)
            nm = self.xp.array([1.0], dtype=self.dtype)
            nw, dw = self.discrete_delay_tf(delay)

        complex_yt_tf = self.plot_iirfilter_tf(
            self.num[mode, :], self.den[mode, :], fs, 
            dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose
        )
        return complex_yt_tf

    def RTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, overplot=False, **extra):
        """Plot Rejection Transfer Function."""
        plotTitle = title if title else 'Rejection Transfer Function'

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        # Get complex transfer function
        complex_tf = self.plot_iirfilter_tf(
            self.num[mode, :], self.den[mode, :], fs, 
            dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose
        )

        # Convert to magnitude
        tf_mag = np.abs(complex_tf)

        import matplotlib.pyplot as plt
        if overplot:
            color = extra.get('color', 'blue')
            plt.plot(freq, tf_mag, color=color, **extra)
        else:
            plt.figure()
            plt.loglog(freq, tf_mag, label=plotTitle)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.title(plotTitle)
            plt.grid(True)
            plt.legend()
            plt.show()

        return tf_mag

    def NTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, overplot=False, **extra):
        """Plot Noise Transfer Function."""
        plotTitle = title if title else 'Noise Transfer Function'

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-1, np.log10(fs/2), 1000)

        # Get complex transfer function
        complex_tf = self.plot_iirfilter_tf(
            self.num[mode, :], self.den[mode, :], fs, 
            dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose
        )

        # Convert to magnitude
        tf_mag = np.abs(complex_tf)

        import matplotlib.pyplot as plt
        if overplot:
            color = extra.get('color', 'red')
            plt.plot(freq, tf_mag, color=color, **extra)
        else:
            plt.figure()
            plt.loglog(freq, tf_mag, label=plotTitle)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.title(plotTitle)
            plt.grid(True)
            plt.legend()
            plt.show()

        return tf_mag

    def is_stable(self, mode, nm=None, dm=None, nw=None, dw=None, gain=None, no_margin=False, verbose=False):
        nm = nm if nm is not None else self.xp.array([1, 0], dtype=self.dtype)
        nw = nw if nw is not None else self.xp.array([1, 0], dtype=self.dtype)
        dm = dm if dm is not None else self.xp.array([0, 1], dtype=self.dtype)
        dw = dw if dw is not None else self.xp.array([0, 1], dtype=self.dtype)

        temp1 = self.xp.polymul(dm, dw)
        while temp1[-1] == 0:
            temp1 = temp1[:-1]
        DDD = self.xp.polymul(temp1, self.den[mode, :])
        while DDD[-1] == 0:
            DDD = DDD[:-1]

        temp2 = self.xp.polymul(nm, nw)
        while temp2[-1] == 0:
            temp2 = temp2[:-1]
        NNN = self.xp.polymul(temp2, self.num[mode, :])
        if self.xp.sum(self.xp.abs(NNN)) != 0:
            while NNN[-1] == 0:
                NNN = NNN[:-1]

        if gain is not None:
            NNN *= gain / self.gain[mode]

        stable, ph_margin, g_margin, mroot, m_one_dist = self.nyquist(NNN, DDD, no_margin=no_margin)

        if verbose:
            print('max root (closed loop) =', mroot)
            print('phase margin =', ph_margin)
            print('gain margin =', g_margin)
            print('min. distance from (-1;0) =', m_one_dist)
        return stable

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
        num_coeffs = cpuArray(self.num[mode, :])
        den_coeffs = cpuArray(self.den[mode, :])

        # Remove trailing zeros
        num_coeffs = num_coeffs[num_coeffs != 0] if np.any(num_coeffs != 0) else np.array([0])
        den_coeffs = den_coeffs[den_coeffs != 0] if np.any(den_coeffs != 0) else np.array([1])

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
            num[i, max_len - len(num_coeffs):] = num_coeffs
            den[i, max_len - len(den_coeffs):] = den_coeffs

        return IirFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)

    def plot_iirfilter_tf(self, num, den, fs, dm=None, nw=None, dw=None, freq=None, noplot=True, verbose=False):
        """Compute IIR filter transfer function using control library or fallback implementation."""
        
        # Convert to CPU arrays for processing
        num_cpu = cpuArray(num)
        den_cpu = cpuArray(den)

        # Remove leading zeros
        num_cpu = num_cpu[num_cpu != 0] if np.any(num_cpu != 0) else np.array([0])
        den_cpu = den_cpu[den_cpu != 0] if np.any(den_cpu != 0) else np.array([1])

        # Apply additional filters if provided
        if dm is not None and nw is not None and dw is not None:
            dm_cpu = cpuArray(dm)
            nw_cpu = cpuArray(nw)
            dw_cpu = cpuArray(dw)

            # Multiply polynomials
            num_total = np.convolve(num_cpu, nw_cpu)
            den_total = np.convolve(den_cpu, np.convolve(dm_cpu, dw_cpu))
        else:
            num_total = num_cpu
            den_total = den_cpu

        # Generate frequency vector if not provided
        if freq is None:
            freq = np.logspace(-3, np.log10(fs/2), 1000)

        # Use control library if available
        if CONTROL_AVAILABLE:
            try:
                # Create discrete-time transfer function
                dt = 1.0 / fs
                tf = control.TransferFunction(num_total, den_total, dt=dt)

                # Convert frequency to angular frequency for discrete systems
                omega = 2 * np.pi * freq / fs
                
                # Limit omega to avoid Nyquist frequency warning
                omega = np.clip(omega, 0, np.pi - 1e-6)

                # Use the new frequency_response method instead of freqresp
                if hasattr(control, 'frequency_response'):
                    response = control.frequency_response(tf, omega)
                    complex_tf = response[0].flatten()  # Get complex response
                else:
                    # Fallback to freqresp for older versions
                    response = control.freqresp(tf, omega)
                    complex_tf = response[0].flatten()

                if verbose:
                    print(f"Transfer function computed using control library")
                    print(f"Numerator: {num_total}")
                    print(f"Denominator: {den_total}")

                return complex_tf

            except Exception as e:
                if verbose:
                    print(f"Control library evaluation failed: {e}")
                    print("Falling back to manual computation")

        # Fallback: manual computation using numpy
        if verbose:
            print("Computing transfer function manually")

        # Convert to z-domain evaluation
        complex_tf = np.zeros(len(freq), dtype=complex)

        for i, f in enumerate(freq):
            # z = exp(j*2*pi*f/fs) for discrete-time systems
            z = np.exp(1j * 2 * np.pi * f / fs)

            # Evaluate polynomials at z
            num_val = np.polyval(num_total, z)
            den_val = np.polyval(den_total, z)

            # Avoid division by zero
            if abs(den_val) > 1e-15:
                complex_tf[i] = num_val / den_val
            else:
                complex_tf[i] = np.inf + 1j * np.inf

        if verbose:
            print(f"Numerator: {num_total}")
            print(f"Denominator: {den_total}")
            print(f"Frequency range: {freq[0]:.3f} - {freq[-1]:.3f} Hz")

        return complex_tf

    def nyquist(self, NNN, DDD, no_margin=False, verbose=False):
        """Nyquist stability analysis using control library or fallback implementation.
        
        Args:
            NNN: Numerator coefficients
            DDD: Denominator coefficients  
            no_margin: If True, skip margin calculations
            verbose: If True, print detailed information

        Returns:
            tuple: (stable, ph_margin, g_margin, mroot, m_one_dist)
        """

        # Convert to CPU arrays
        num_cpu = cpuArray(NNN)
        den_cpu = cpuArray(DDD)

        # Use control library if available
        if CONTROL_AVAILABLE:
            try:
                # Create transfer function (assume discrete-time for stability analysis)
                tf = control.TransferFunction(num_cpu, den_cpu, dt=True)

                # Check stability using poles
                poles = control.pole(tf)
                stable = np.all(np.abs(poles) < 1.0)  # For discrete-time: |poles| < 1

                if no_margin:
                    return stable, 0, 0, np.max(np.abs(poles)), 0

                # Calculate stability margins
                try:
                    gm, pm, wg, wp = control.margin(tf)

                    # Convert gain margin from linear to dB if needed
                    if gm is not None and gm > 0:
                        g_margin = 20 * np.log10(gm)
                    else:
                        g_margin = np.inf if stable else -np.inf

                    ph_margin = pm if pm is not None else (180 if stable else 0)

                except:
                    g_margin = np.inf if stable else -np.inf
                    ph_margin = 180 if stable else 0

                # Calculate distance from (-1, 0) using Nyquist data
                try:
                    # Generate frequency response
                    omega = np.logspace(-3, 3, 1000)
                    _, response = control.freqresp(tf, omega)

                    # Find minimum distance from (-1, 0)
                    distances = np.abs(response + 1)
                    m_one_dist = np.min(distances)

                except:
                    m_one_dist = 1.0 if stable else 0.0

                mroot = np.max(np.abs(poles))

                return stable, ph_margin, g_margin, mroot, m_one_dist
       
            except Exception as e:
                if verbose:
                    print(f"Control library Nyquist analysis failed: {e}")
                    print("Falling back to manual computation")

        # Fallback: manual implementation
        return self._nyquist_manual(num_cpu, den_cpu, no_margin)
    
    def _nyquist_manual(self, NNN, DDD, no_margin=False):
        """Manual Nyquist stability analysis fallback implementation."""

        # Find roots of denominator (poles of closed-loop system)
        try:
            # For closed-loop stability analysis: 1 + G(z) = 0
            # So we need to analyze DDD + NNN = 0
            closed_loop_char = DDD + NNN
            roots = np.roots(closed_loop_char)

            # For discrete-time systems: stable if |roots| < 1
            mroot = np.max(np.abs(roots)) if len(roots) > 0 else 0
            stable = mroot < 1.0

        except:
            # If root finding fails, assume unstable
            stable = False
            mroot = np.inf

        if no_margin:
            return stable, 0, 0, mroot, 0

        # Calculate margins manually
        try:
            # Generate frequency response for open-loop system G(z) = NNN/DDD
            omega = np.logspace(-3, 3, 1000)
            z = np.exp(1j * omega)
            
            # Evaluate transfer function
            G = np.zeros(len(omega), dtype=complex)
            for i, zi in enumerate(z):
                num_val = np.polyval(NNN, zi)
                den_val = np.polyval(DDD, zi)
                if abs(den_val) > 1e-15:
                    G[i] = num_val / den_val
                else:
                    G[i] = np.inf

            # Find gain margin: frequency where phase = -180°
            phases = np.angle(G) * 180 / np.pi
            phase_180_idx = np.argmin(np.abs(phases + 180))

            if abs(phases[phase_180_idx] + 180) < 5:  # Within 5 degrees
                g_margin = -20 * np.log10(np.abs(G[phase_180_idx]))
            else:
                g_margin = np.inf if stable else -np.inf

            # Find phase margin: frequency where |G| = 1 (0 dB)
            magnitudes = np.abs(G)
            unity_gain_idx = np.argmin(np.abs(magnitudes - 1))

            if abs(magnitudes[unity_gain_idx] - 1) < 0.1:  # Within 0.1 of unity
                ph_margin = 180 + phases[unity_gain_idx]
            else:
                ph_margin = 180 if stable else 0

            # Find minimum distance from (-1, 0)
            distances = np.abs(G + 1)
            m_one_dist = np.min(distances)

        except:
            g_margin = np.inf if stable else -np.inf
            ph_margin = 180 if stable else 0
            m_one_dist = 1.0 if stable else 0.0

        return stable, ph_margin, g_margin, mroot, m_one_dist

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