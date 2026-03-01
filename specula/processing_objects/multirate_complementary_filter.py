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
    '''
    def __init__(self,
                 iir_filter_data: IirFilterData,
                 g_track: float,
                 weights: list,
                 N_list: list,
                 target_device_idx=None,
                 precision=None):
        """
        Initialize the Multirate Complementary Filter.

        Parameters:
        iir_filter_data (IirFilterData): Coefficients of the shared LTI engine:
                                            C_f(z) / (1 - z^-1).
        g_track (float): Overall tracking gain for the offset rejection.
        weights (list of floats): Normalized DC weights for
                                  [fast_sensor, slow_sensor_1, slow_sensor_2, ...].
        N_list (list of ints): List of downsampling ratios for each slow sensor.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.n_slow_sensors = len(N_list)

        if len(weights) != self.n_slow_sensors + 1:
            raise ValueError("The weights list must contain exactly one element"
                             " for the fast sensor plus one for each slow sensor.")

        self.iir_filter_data = iir_filter_data

        # Normalize weights just to be safe
        w_array = np.array(weights)
        w_array = w_array / np.sum(w_array)

        w_fast = w_array[0]
        w_slow = w_array[1:]

        # LTI approximation coefficients (Generalized Barycentric)
        self.c_yf_0 = 1.0 + (g_track * w_fast)
        self.c_yf_1 = -1.0

        # Calculate the multiplier for each slow sensor: W_si * N_i
        c_ys_list = [g_track * w_slow[i] * N_list[i] for i in range(self.n_slow_sensors)]

        # Inputs and Outputs
        self.inputs['in_yf'] = InputValue(type=BaseValue)
        self.inputs['in_ys'] = InputList(type=BaseValue)

        self.out_u = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_u'] = self.out_u

        self._ist = self.xp.zeros_like(self.iir_filter_data.num)
        self._ost = self.xp.zeros_like(self.iir_filter_data.den)

        self.c_ys_array = np.array(c_ys_list)
        self.N_array = np.array(N_list)


    def setup(self):
        super().setup()

        connected_ys = len(self.local_inputs['in_ys'])
        if connected_ys != self.n_slow_sensors:
            raise ValueError(f"Expected {self.n_slow_sensors} slow inputs,"
                             f" but {connected_ys} were connected.")

        yf_value = self.local_inputs['in_yf'].value

        self.out_u.value = self.xp.zeros_like(yf_value)
        self._yf_prev = self.xp.zeros_like(yf_value)

        # GPU arrays
        self._frame_counter = self.xp.array([0], dtype=self.xp.int64)
        self.c_ys_array = self.to_xp(self.c_ys_array, dtype=self.dtype)
        self.N_array = self.to_xp(self.N_array, dtype=self.xp.int64)


    def trigger_code(self):
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
        num_contrib = self.xp.sum(self.iir_filter_data.num * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] \
                      * self._ost[:, :no - 1], axis=1)

        output = factor * (num_contrib - den_contrib)
        self._ost[:, no - 1] = output

        self.out_u.value[:] = output
        self.out_u.generation_time = self.current_time
