from specula import cp, np
from specula.base_processing_obj import BaseProcessingObj
from specula.processing_objects.base_filter import BaseFilter
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams

class MultirateComplementaryFilter(BaseFilter):
    '''
    Multirate filter for differential sensor fusion (Generalized Barycentric Approach).
    
    Inherits from BaseFilter to support delay, gain_mod, and POLC synchronous outputs.
    '''
    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 g_track: float,
                 weights: list,
                 N_list: list,
                 delay: float = 0,
                 idx_yf=None,
                 idx_ys=None,
                 target_device_idx=None,
                 precision=None):

        self.n_slow_sensors = len(N_list)

        if len(weights) != self.n_slow_sensors + 1:
            raise ValueError("The weights list must contain exactly one element"
                             " for the fast sensor plus one for each slow sensor.")

        if iir_filter_data is None:
            raise ValueError("iir_filter_data cannot be None.")

        # Call BaseFilter init. It creates 'out_comm', 'out_comm_no_delay' and 'output_buffer'
        super().__init__(simul_params=simul_params,
                         nfilter=iir_filter_data.nfilter,
                         delay=delay,
                         target_device_idx=target_device_idx,
                         precision=precision)

        self.iir_filter_data = iir_filter_data
        self.idx_yf = idx_yf
        self.idx_ys = idx_ys
        self.N_list = N_list

        # Remove the default delta_comm input from BaseFilter as we use custom topology
        if 'delta_comm' in self.inputs:
            del self.inputs['delta_comm']

        # Custom Inputs
        self.inputs['in_yf'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_ys'] = InputList(type=BaseValue, optional=True)
        self.inputs['in_vec'] = InputValue(type=BaseValue, optional=True)

        w_array = np.array(weights)
        w_array = w_array / np.sum(w_array)
        w_fast = w_array[0]
        w_slow = w_array[1:]

        # LTI approximation coefficients
        self.c_yf_0 = 1.0 + (g_track * w_fast)
        self.c_yf_1 = -1.0
        c_ys_list = [g_track * w_slow[i] * N_list[i] for i in range(self.n_slow_sensors)]

        self._ist = self.xp.zeros_like(self.iir_filter_data.num)
        self._ost = self.xp.zeros_like(self.iir_filter_data.den)

        self.c_ys_array = np.array(c_ys_list)
        self.N_array = np.array(N_list)

        self._use_vector_input = False
        self._cpu_frame_counter = 0


    def setup(self):
        super().setup()

        self._use_vector_input = self.local_inputs['in_vec'] is not None
        has_yf = self.local_inputs['in_yf'] is not None

        if self._use_vector_input:
            if self.idx_yf is None or self.idx_ys is None:
                raise ValueError("idx_yf and idx_ys must be provided when using in_vec.")

            self._idx_yf = self.to_xp(np.atleast_1d(self.idx_yf), dtype=self.xp.int32)
            self._idx_ys = [self.to_xp(np.atleast_1d(idx), dtype=self.xp.int32) for idx in self.idx_ys]
            yf_shape = self.local_inputs['in_vec'].value[self._idx_yf].shape

        elif has_yf:
            connected_ys = len(self.local_inputs['in_ys'])
            if connected_ys != self.n_slow_sensors:
                raise ValueError(f"Expected {self.n_slow_sensors} slow inputs,"
                                 f" but {connected_ys} connected.")
            yf_shape = self.local_inputs['in_yf'].value.shape
        else:
            raise ValueError("Connect either 'in_vec' or 'in_yf'+'in_ys'.")

        self._yf_prev = self.xp.zeros(yf_shape, dtype=self.dtype)
        self._frame_counter = self.xp.array([0], dtype=self.xp.int64)
        self.c_ys_array = self.to_xp(self.c_ys_array, dtype=self.dtype)
        self.N_array = self.to_xp(self.N_array, dtype=self.xp.int64)


    def prepare_trigger(self, t):
        # Call BaseProcessingObj to set time variables, skipping BaseFilter's delta_comm logic
        BaseProcessingObj.prepare_trigger(self, t)

        # 1. Delay buffer handling (Replicated from BaseFilter)
        if self.delay > 0:
            self.output_buffer[:, 1:] = self.output_buffer[:, :-1]

        # 2. Gain Mod handling (Replicated from BaseFilter)
        if self.local_inputs['gain_mod'] is not None:
            gain_mod = self.local_inputs['gain_mod'].value
            gain_mod_array = self.xp.asarray(gain_mod, dtype=self.dtype)
            self._gain_mod = self.xp.atleast_1d(gain_mod_array).ravel()
            if self._gain_mod.size != self._nfilter:
                if self._gain_mod.size == 1:
                    self._gain_mod = self.xp.full(self._nfilter,
                                                  self._gain_mod[0],
                                                  dtype=self.dtype)
                else:
                    raise ValueError("gain_mod size mismatch.")
        else:
            self._gain_mod = self.xp.ones(self._nfilter, dtype=self.dtype)

        # 3. Synchronization Check (Only in separate input mode)
        if not self._use_vector_input:
            t_fast = self.local_inputs['in_yf'].generation_time
            expected_frame = self._cpu_frame_counter + 1

            for i, n in enumerate(self.N_list):
                t_slow = self.local_inputs['in_ys'][i].generation_time
                if expected_frame % n == 0:
                    if t_slow != t_fast:
                        raise RuntimeError(f"Sync error: Slow sensor {i} (N={n}) missing update at"
                                      f" frame {expected_frame}. t_fast={t_fast}, t_slow={t_slow}")
                else:
                    if t_slow == t_fast:
                        raise RuntimeError(f"Sync error: Slow sensor {i} (N={n})"
                                           f" updated unexpectedly at fast frame {expected_frame}.")

        self._cpu_frame_counter += 1


    def trigger_code(self):

        if self._use_vector_input:
            vec = self.local_inputs['in_vec'].value
            yf = vec[self._idx_yf]
            ys_list = [vec[idx] for idx in self._idx_ys]
        else:
            yf = self.local_inputs['in_yf'].value
            ys_list = [item.value for item in self.local_inputs['in_ys']]

        self._frame_counter += 1

        mixed_input = (self.c_yf_0 * yf) + (self.c_yf_1 * self._yf_prev)

        for i in range(self.n_slow_sensors):
            is_slow_mask = (self._frame_counter % self.N_array[i]) == 0
            ys_stuffed = ys_list[i] * is_slow_mask
            mixed_input += self.c_ys_array[i] * ys_stuffed

        self._yf_prev[:] = yf

        # IIR ENGINE
        sden = self.iir_filter_data.den.shape
        snum = self.iir_filter_data.num.shape
        no = sden[1]
        ni = snum[1]

        self._ost[:, :-1] = self._ost[:, 1:]
        self._ost[:, -1] = 0
        self._ist[:, :-1] = self._ist[:, 1:]
        self._ist[:, -1] = 0

        self._ist[:, ni - 1] = mixed_input

        factor = 1.0 / self.iir_filter_data.den[:, no - 1]
        num_contrib = self.xp.sum(self.iir_filter_data.num \
                     * self._gain_mod[:, None] * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] \
                     * self._ost[:, :no - 1], axis=1)

        output = factor * (num_contrib - den_contrib)
        self._ost[:, no - 1] = output

        # Write to BaseFilter's output buffer (instead of out_comm directly)
        self.output_buffer[:, 0] = output
