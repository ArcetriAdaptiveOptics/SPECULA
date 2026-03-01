from specula import cp, np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.iir_filter_data import IirFilterData

class MultirateComplementaryFilter(BaseProcessingObj):
    '''
    Multirate filter for differential sensor fusion (Generalized Barycentric Approach).
    
    This object implements the shared double-integrator generalized equation 
    for 1 fast sensor and N slow sensors with arbitrary DC weights:
    
    U(z) = [C_f(z) / (1 - z^-1)] * [ (1 + W_f)*yf - yf*z^-1 + sum(W_si * N_i * ys_i_stuffed) ]

    Supports two input modes:
    1. Independent inputs: connect `in_yf` and multiple `in_ys`.
    2. Vector input: connect `in_vec` and provide `idx_yf` and `idx_ys` to extract signals.
    '''
    def __init__(self,
                 iir_filter_data: IirFilterData,
                 g_track: float,
                 weights: list,
                 N_list: list,
                 idx_yf=None,
                 idx_ys=None,
                 target_device_idx=None,
                 precision=None):
        """
        Initialize the Multirate Complementary Filter.

        Parameters:
        iir_filter_data (IirFilterData): Coefficients of the shared LTI engine: C_f(z) / (1 - z^-1).
        g_track (float): Overall tracking gain for the offset rejection.
        weights (list of floats): Normalized DC weights for [fast_sensor, slow_sensor_1, ...].
        N_list (list of ints): List of downsampling ratios for each slow sensor.
        idx_yf (int, list or array): Indices to extract the fast sensor from `in_vec` (if used).
        idx_ys (list of ints/lists): List of indices to extract each slow sensor from `in_vec`
                                     (if used).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.n_slow_sensors = len(N_list)

        if len(weights) != self.n_slow_sensors + 1:
            raise ValueError("The weights list must contain exactly one element"
                             " for the fast sensor plus one for each slow sensor.")

        self.iir_filter_data = iir_filter_data

        self.idx_yf = idx_yf
        self.idx_ys = idx_ys

        # Normalize weights
        w_array = np.array(weights)
        w_array = w_array / np.sum(w_array)

        w_fast = w_array[0]
        w_slow = w_array[1:]

        # LTI approximation coefficients (Generalized Barycentric)
        self.c_yf_0 = 1.0 + (g_track * w_fast)
        self.c_yf_1 = -1.0

        # Calculate the multiplier for each slow sensor: W_si * N_i
        c_ys_list = [g_track * w_slow[i] * N_list[i] for i in range(self.n_slow_sensors)]

        # Inputs and Outputs (All optional to support both routing strategies)
        self.inputs['in_yf'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_ys'] = InputList(type=BaseValue, optional=True)
        self.inputs['in_vec'] = InputValue(type=BaseValue, optional=True)

        self.out_comm = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_comm'] = self.out_comm

        self._ist = self.xp.zeros_like(self.iir_filter_data.num)
        self._ost = self.xp.zeros_like(self.iir_filter_data.den)

        self.c_ys_array = np.array(c_ys_list)
        self.N_array = np.array(N_list)

        self._use_vector_input = False
        self._idx_yf = None
        self._idx_ys = None
        self._yf_prev = None
        self._frame_counter = None


    def setup(self):
        super().setup()

        # Determine Input Mode
        self._use_vector_input = self.local_inputs['in_vec'] is not None
        has_yf = self.local_inputs['in_yf'] is not None

        if self._use_vector_input:
            if self.idx_yf is None or self.idx_ys is None:
                raise ValueError("idx_yf and idx_ys must be provided when using in_vec.")
            if len(self.idx_ys) != self.n_slow_sensors:
                raise ValueError(f"idx_ys must contain {self.n_slow_sensors} index definitions.")

            # Convert indices to GPU arrays for fast slicing during trigger
            self._idx_yf = self.to_xp(np.atleast_1d(self.idx_yf), dtype=self.xp.int32)
            self._idx_ys = [self.to_xp(np.atleast_1d(idx), dtype=self.xp.int32) for idx in self.idx_ys]

            # Infer shape from the vector slice
            vec_val = self.local_inputs['in_vec'].value
            yf_shape = vec_val[self._idx_yf].shape

        elif has_yf:
            connected_ys = len(self.local_inputs['in_ys'])
            if connected_ys != self.n_slow_sensors:
                raise ValueError(f"Expected {self.n_slow_sensors} slow inputs,"
                                 f" but {connected_ys} were connected.")
            yf_shape = self.local_inputs['in_yf'].value.shape

        else:
            raise ValueError("You must connect either 'in_vec' or 'in_yf'+'in_ys'.")

        self.out_comm.value = self.xp.zeros(yf_shape, dtype=self.dtype)
        self._yf_prev = self.xp.zeros(yf_shape, dtype=self.dtype)

        # GPU arrays
        self._frame_counter = self.xp.array([0], dtype=self.xp.int64)
        self.c_ys_array = self.to_xp(self.c_ys_array, dtype=self.dtype)
        self.N_array = self.to_xp(self.N_array, dtype=self.xp.int64)


    def trigger_code(self):

        # --- DATA ROUTING ---
        if self._use_vector_input:
            vec = self.local_inputs['in_vec'].value
            yf = vec[self._idx_yf]
            ys_list = [vec[idx] for idx in self._idx_ys]
        else:
            yf = self.local_inputs['in_yf'].value
            ys_list = [item.value for item in self.local_inputs['in_ys']]

        # --- MULTIRATE COMPUTATION ---
        self._frame_counter += 1

        mixed_input = (self.c_yf_0 * yf) + (self.c_yf_1 * self._yf_prev)

        for i in range(self.n_slow_sensors):
            is_slow_mask = (self._frame_counter % self.N_array[i]) == 0
            ys_stuffed = ys_list[i] * is_slow_mask
            mixed_input += self.c_ys_array[i] * ys_stuffed

        self._yf_prev[:] = yf

        # --- IIR ENGINE ---
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
        num_contrib = self.xp.sum(self.iir_filter_data.num * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] \
                      * self._ost[:, :no - 1], axis=1)

        output = factor * (num_contrib - den_contrib)
        self._ost[:, no - 1] = output

        self.out_comm.value[:] = output
        self.out_comm.generation_time = self.current_time
