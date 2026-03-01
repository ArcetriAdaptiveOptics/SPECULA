from specula import cp, np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.iir_filter_data import IirFilterData

class MultirateComplementaryFilter(BaseProcessingObj):
    '''
    Multirate filter for differential sensor fusion (LTI Complementary Approach).
    
    This object implements the shared double-integrator generalized equation 
    for 1 fast sensor and N slow sensors operating at different framerates:
    
    U(z) = [C_f(z) / (1 - z^-1)] * [ c_yf_0 * yf - yf*z^-1 + sum(c_ys_i * ys_i_stuffed) ]
    
    where:
    c_yf_0 = 1 + sum(g_s_i / 2*N_i)
    c_ys_i = g_s_i / 2
    '''
    def __init__(self,
                 iir_filter_data: IirFilterData,
                 g_s_list: list,
                 N_list: list,
                 target_device_idx=None,
                 precision=None):
        """
        Initialize the Multirate Complementary Filter.

        Parameters:
        iir_filter_data (IirFilterData): Coefficients of the shared LTI engine:
                                            C_f(z) / (1 - z^-1).
        g_s_list (list of floats): List of integral gains for each slow sensor.
        N_list (list of ints): List of downsampling ratios for each slow sensor.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if len(g_s_list) != len(N_list):
            raise ValueError("g_s_list and N_list must have the same length.")

        self.iir_filter_data = iir_filter_data
        self.g_s_list = g_s_list
        self.N_list = N_list
        self.n_slow_sensors = len(N_list)

        # Pre-compute LTI approximation coefficients based on the LaTeX derivation
        # c_yf_0 = 1.0 + sum_i (g_s_i / 2*N_i)
        sum_g_n = sum([g / (2.0 * n) for g, n in zip(self.g_s_list, self.N_list)])
        self.c_yf_0 = 1.0 + sum_g_n
        self.c_yf_1 = -1.0

        # Inputs and Outputs
        self.inputs['in_yf'] = InputValue(type=BaseValue)
        # Using InputList to accept an arbitrary number of slow sensors
        self.inputs['in_ys'] = InputList(type=BaseValue)

        self.out_u = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_u'] = self.out_u

        # Internal IIR states (same shape logic as standard IirFilter)
        self._ist = self.xp.zeros_like(self.iir_filter_data.num)
        self._ost = self.xp.zeros_like(self.iir_filter_data.den)

    def setup(self):
        super().setup()

        # Check if the number of connected slow inputs matches the parameters length
        connected_ys = len(self.local_inputs['in_ys'])
        if connected_ys != self.n_slow_sensors:
            raise ValueError(f"Expected {self.n_slow_sensors} slow inputs, "
                             f"but {connected_ys} were connected to in_ys.")

        yf_value = self.local_inputs['in_yf'].value

        # Initialize output array
        self.out_u.value = self.xp.zeros_like(yf_value)

        # Memory state for the discrete derivative of the fast sensor: yf[k-1]
        self._yf_prev = self.xp.zeros_like(yf_value)

        # Frame counter must be a GPU array to be fully compatible with CUDA Graphs
        # We use int64 to avoid overflow during very long simulations
        self._frame_counter = self.xp.array([0], dtype=self.xp.int64)

        # Move parameters to target device to avoid CPU-GPU synchronization during trigger
        self.c_ys_array = self.xp.array([g / 2.0 for g in self.g_s_list], dtype=self.dtype)
        self.N_array = self.xp.array(self.N_list, dtype=self.xp.int64)

    def trigger_code(self):
        yf = self.local_inputs['in_yf'].value
        ys_list = [item.value for item in self.local_inputs['in_ys']]

        # 1. INCREMENT FRAME COUNTER (GPU native)
        self._frame_counter += 1

        # 2. COMPUTE FAST SENSOR CONTRIBUTION (Phase Lead FIR)
        mixed_input = (self.c_yf_0 * yf) + (self.c_yf_1 * self._yf_prev)

        # 3. ADD SLOW SENSORS CONTRIBUTIONS (Zero-Stuffing)
        # We loop through the slow sensors. Since self.n_slow_sensors is known at
        # setup time, CUDA Graph capture will correctly "unroll" this Python loop.
        for i in range(self.n_slow_sensors):
            # Create a boolean mask directly on GPU without Python 'if' branches
            is_slow_mask = (self._frame_counter % self.N_array[i]) == 0

            # Zero-stuffing: ys becomes an array of zeros if it's not a slow frame
            ys_stuffed = ys_list[i] * is_slow_mask

            # Add scaled contribution
            mixed_input += self.c_ys_array[i] * ys_stuffed

        # Update fast sensor memory for the next timestep
        self._yf_prev[:] = yf

        # 4. APPLY THE SHARED IIR ENGINE
        # Standard optimized matrix operations matching specula.processing_objects.IirFilter
        sden = self.iir_filter_data.den.shape
        snum = self.iir_filter_data.num.shape
        no = sden[1]
        ni = snum[1]

        # Shift state buffers
        self._ost[:, :-1] = self._ost[:, 1:]
        self._ost[:, -1] = 0
        self._ist[:, :-1] = self._ist[:, 1:]
        self._ist[:, -1] = 0

        # Insert the newly mixed multirate input
        self._ist[:, ni - 1] = mixed_input

        # Compute IIR difference equation
        factor = 1.0 / self.iir_filter_data.den[:, no - 1]

        num_contrib = self.xp.sum(self.iir_filter_data.num * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] \
                                  * self._ost[:, :no - 1], axis=1)

        output = factor * (num_contrib - den_contrib)
        self._ost[:, no - 1] = output

        # 5. WRITE OUTPUT
        self.out_u.value[:] = output
        self.out_u.generation_time = self.current_time
