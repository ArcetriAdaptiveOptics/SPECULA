
import numpy as np

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.modal_pushpull_signal import modal_pushpull_signal
from specula.lib.utils import is_scalar, psd_to_signal
from specula.data_objects.simul_params import SimulParams

class Vibrations:
    def __init__(self,
                 nmodes,
                 psd=None,
                 freq=None,
                 time_hist=None,
                 seed=1987,
                 samp_freq=1000,
                 niter=1000,
                 start_from_zero=False,
                 verbose=False,
                 xp=np,
                 dtype=np.float32,
                 complex_dtype=np.complex64):
        self._verbose = verbose
        self._nmodes = nmodes
        self._psd = []
        self._freq = []
        self._seed = seed
        self._start_from_zero = start_from_zero
        self._type = ''
        self._niter = niter
        self._samp_freq = samp_freq
        self.xp = xp
        self.dtype = dtype
        self.complex_dtype = complex_dtype

        # Determine type
        if psd is None or freq is None:
            raise ValueError('psd and freq must be defined and time_hist will be computed.')

        # Store PSD and freq as lists of arrays (one per mode)
        psd = np.asarray(psd)
        for i in range(self._nmodes):
            self._psd.append(psd[i, :])
        freq = np.asarray(freq)
        if freq.ndim == 1:
            freq = np.tile(freq, (self._nmodes, 1)).T
        for i in range(self._nmodes):
            self._freq.append(freq[:, i])

    def get_time_hist(self):
        n = int(np.floor((self._niter + 1) / 2.))
        time_hist = np.zeros((2 * n, self._nmodes))
        for i in range(self._nmodes):
            # Interpolation of the PSD on n points
            freq_mode = self._freq[i]
            psd_mode = self._psd[i]
            freq_bins = np.linspace(freq_mode[0], freq_mode[-1], n)
            psd_interp = np.interp(freq_bins, freq_mode, psd_mode)
            # Generate the signal from the interpolated PSD
            temp, _ = psd_to_signal(psd_interp, self._samp_freq, self.xp, self.dtype,
                                    self.complex_dtype, seed=self._seed + i)
            if self._start_from_zero:
                temp -= temp[0]
            time_hist[:, i] = temp
        return time_hist

class FuncGenerator(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams=None,
                 func_type='SIN',
                 nmodes: int=None,
                 time_hist=None,
                 psd=None,
                 fr_psd=None,
                 constant: list=None,
                 amp: list=None, 
                 freq: list=None,
                 offset: list=None,
                 vect_amplitude: list=None,
                 nsamples: int=1,
                 seed: int=None,
                 ncycles: int=1,
                 vsize: int=1,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if type == 'VIB_PSD' and simul_params is None:
            raise ValueError('SIMUL_PARAMS keyword is mandatory for type VIB_PSD')
        self.simul_params = simul_params

        if nmodes is not None and vsize>1:
            raise ValueError('NMODES and VSIZE cannot be used together. Use NMODES only for PUSHPULL, PUSHPULLREPEAT, VIB_HIST or VIB_PSD types')

        self.type = func_type.upper()
        if self.type == 'PUSHPULLREPEAT':
            repeat_ncycles = True
            self.type = 'PUSHPULL'
        else:
            repeat_ncycles = False

        if nsamples != 1 and self.type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL or PUSHPULLREPEAT types')

        if str(seed).strip() == 'auto':
            self.seed = self.xp.around(self.xp.random.random() * 1e4)
        elif seed is not None:
            self.seed = self.xp.array(seed, dtype=self.dtype)
        else:
            self.seed = 0

        self.constant = self.xp.array(constant, dtype=self.dtype) if constant is not None else 0.0
        self.amp = self.xp.array(amp, dtype=self.dtype) if amp is not None else 0.0
        self.freq = self.xp.array(freq, dtype=self.dtype) if freq is not None else 0.0
        self.offset = self.xp.array(offset, dtype=self.dtype) if offset is not None else 0.0
        self.vect_amplitude = self.xp.array(vect_amplitude, dtype=self.dtype) if vect_amplitude is not None else 0.0

        if self.type in ['SIN', 'SQUARE_WAVE', 'LINEAR', 'RANDOM', 'RANDOM_UNIFORM']:
            # Check if the parameters are scalars or arrays and have coherent sizes
            params = [self.amp, self.freq, self.offset, self.constant]
            param_names = ['amp', 'freq', 'offset', 'constant']
            vector_lengths = [p.shape[0] for p in params if not is_scalar(p, np)]

            if len(vector_lengths) > 0:
                unique_lengths = set(vector_lengths)
                if len(unique_lengths) > 1:
                    # Find the names of the parameters with different lengths
                    details = [f"{name}={p.shape[0]}" for p, name in zip(params, param_names) if not is_scalar(p, np)]
                    raise ValueError(
                        f"Shape mismatch: parameter lengths are {details} (must all be equal if not scalar)"
                    )
                output_size = unique_lengths.pop()
            else:
                output_size = vsize if nmodes is None else vsize * nmodes
        elif self.type in ['PUSH', 'PUSHPULL', 'TIME_HIST']:
            if time_hist is not None:
                output_size = np.array(time_hist).shape[1]
            elif nmodes is not None:
                output_size = nmodes
        else:
            output_size = vsize if nmodes is None else vsize * nmodes
        
        self.output = BaseValue(target_device_idx=target_device_idx, value=self.xp.zeros(output_size, dtype=self.dtype))
        self.vib = None

        if seed is not None:
            self.seed = seed

        # Initialize attributes based on the type
        if self.type == 'SIN':
            pass

        elif self.type == 'SQUARE_WAVE':
            pass

        elif self.type == 'LINEAR':
            self.slope = 0.0

        elif self.type == 'RANDOM' or self.type == 'RANDOM_UNIFORM':
            pass

        elif self.type == 'VIB_HIST':
            raise ValueError('VIB_HIST is not implemented yet')

        elif self.type == 'VIB_PSD':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_PSD')
            if psd is None:
                raise ValueError('PSD keyword is mandatory for type VIB_PSD')
            if fr_psd is None:
                raise ValueError('FR_PSD keyword is mandatory for type VIB_PSD')
            samp_freq = 1/simul_params.time_step
            niter = simul_params.total_time/self.simul_params.time_step
            self.vib = Vibrations(nmodes, psd=psd, freq=fr_psd, seed=seed,
                                  samp_freq=samp_freq, niter=niter, start_from_zero=False, verbose=False,
                                  xp=self.xp, dtype=self.dtype, complex_dtype=self.complex_dtype)
            self.time_hist = self.vib.get_time_hist()

        elif self.type == 'PUSH':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSH')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, only_push=True, ncycles=ncycles)

        elif self.type == 'PUSHPULL':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSHPULL')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSHPULL')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles, repeat_ncycles=repeat_ncycles, nsamples=nsamples)

        elif self.type == 'TIME_HIST':
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type TIME_HIST')
            self.time_hist = self.xp.array(time_hist)

        else:
            raise ValueError(f'Unknown function type: {self.type}')

        self.nmodes = nmodes
        self.outputs['output'] = self.output
        self.iter_counter = 0
        self.current_time_gpu = self.xp.zeros(1, dtype=self.dtype)
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.current_time_gpu[:] = self.current_time_seconds

    def trigger_code(self):

        if self.type == 'SIN':
            phase = self.freq*2 * self.xp.pi * self.current_time_gpu + self.offset
            self.output.value[:] = (self.amp * self.xp.sin(phase, dtype=self.dtype) + self.constant) * self.vsize_array

        elif self.type == 'SQUARE_WAVE':
            phase = self.freq*2 * self.xp.pi*self.current_time_gpu + self.offset
            self.output.value[:] = (self.amp * self.xp.sign(self.xp.sin(phase, dtype=self.dtype)) + self.constant) * self.vsize_array

        elif self.type == 'LINEAR':
            self.output.value[:] = (self.slope * self.current_time_gpu + self.constant) * self.vsize_array

        elif self.type == 'RANDOM':
            self.output.value[:] = (self.xp.random.normal(size=len(self.amp)) * self.amp + self.constant) * self.vsize_array

        elif self.type == 'RANDOM_UNIFORM':
            lowv = self.constant - self.amp/2
            highv = self.constant + self.amp/2
            self.output.value[:] = (self.xp.random.uniform(low=lowv, high=highv)) * self.vsize_array

        elif self.type in ['VIB_HIST', 'VIB_PSD', 'PUSH', 'PUSHPULL', 'TIME_HIST']:
            self.output.value[:] = self.get_time_hist_at_current_time() * self.vsize_array

        else:
            raise ValueError(f'Unknown function generator type: {self.type}')

    def post_trigger(self):

        self.output.generation_time = self.current_time
        self.iter_counter += 1

    def get_time_hist_at_current_time(self):
        return self.xp.array(self.time_hist[self.iter_counter])

    def setup(self):
        super().setup()

#       TODO
#       if self.vib:
#           self.vib.set_niters(self.loop_niters + 1)
#           self.vib.set_samp_freq(1.0 / self.t_to_seconds(self.loop_dt))
#           self.vib.compute()
#           self.time_hist = self.vib.get_time_hist()

        if self.type in ['SIN', 'LINEAR', 'RANDOM']:
            self.build_stream()

