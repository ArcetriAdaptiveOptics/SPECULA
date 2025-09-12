import numpy as np

from specula.data_objects.iir_filter_data import IirFilterData
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams

class IirFilter(BaseProcessingObj):
    '''Infinite Impulse Response filter based Time Control
    
    Set *integration* to False to disable integration, regardless
    of wha the input IirFilter object contains
    '''
    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 delay: float=0,
                 integration: bool=True,
                 offset: float=None,
                 og_shaper: float=None,
                 target_device_idx=None,
                 precision=None
                 ):

        self.time_step = simul_params.time_step

        self.verbose = True
        self.iir_filter_data = iir_filter_data

        self.integration = integration
        if integration is False:
            raise NotImplementedError('IirFilter: integration=False is not implemented yet')

        if og_shaper is not None:
            raise NotImplementedError('OG Shaper not implementd yet')

        if offset != None:
            raise NotImplementedError('Offset not implemented yet')

        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        self.delay = delay if delay is not None else 0
        self._n = iir_filter_data.nfilter
        self._type = iir_filter_data.num.dtype
        self.set_state_buffer_length(int(np.ceil(self.delay)) + 1)

        # Initialize state vectors
        self._ist = self.xp.zeros_like(iir_filter_data.num)
        self._ost = self.xp.zeros_like(iir_filter_data.den)

        self.out_comm = BaseValue(value=self.xp.zeros(self._n, dtype=self.dtype), target_device_idx=target_device_idx)
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['gain_mod'] = InputValue(type=BaseValue,optional=True)
        self.outputs['out_comm'] = self.out_comm

        self._opticalgain = None  # TODO
        self._og_shaper = None  # TODO
        self._offset = None  # TODO
        self._bootstrap_ptr = None  # TODO
        self._modal_start_time = None  # TODO
        self._skipOneStep = False  # TODO
        self._StepIsNotGood = False  # TODO
        self._start_time = 0  # TODO

    def set_state_buffer_length(self, total_length):
        self._total_length = total_length
        if self._n is not None and self._type is not None:
            self.state = self.xp.zeros((self._n, self._total_length), dtype=self.dtype)

    # TODO not used
    @property
    def last_state(self):
        return self.state[:, 0]

    # TODO not used
    def set_modal_start_time(self, modal_start_time):
        modal_start_time_ = self.to_xp(modal_start_time, dtype=self.dtype)
        for i in range(len(modal_start_time)):
            modal_start_time_[i] = self.seconds_to_t(modal_start_time[i])
        self._modal_start_time = modal_start_time_

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_comm = self.local_inputs['delta_comm'].value

        # Update the state
        if self.delay > 0:
            self.state[:, 1:self._total_length] = self.state[:, 0:self._total_length-1]

        # check if gain_mod is provided
        if self.local_inputs['gain_mod'] is not None:
            self._gain_mod = self.local_inputs['gain_mod'].value
        else:
            # Default gain_mod is an array of ones
            self._gain_mod = self.xp.ones_like(self.delta_comm, dtype=self.dtype)

        return

        ##############################
        # Start of unused code

        if self._opticalgain is not None:
            if self._opticalgain.value > 0:
                self.delta_comm *= 1.0 / self._opticalgain.value
                if self._og_shaper is not None:
                    self.delta_comm *= self._og_shaper
                # should not modify an input, right?
                # self.local_inputs['delta_comm'].value = self.delta_comm
                print(f"WARNING: optical gain compensation has been applied (g_opt = {self._opticalgain.value:.5f}).")
        if self._start_time > 0 and self._start_time > t:
            # self.newc = self.xp.zeros_like(delta_comm.value)
            print(f"delta comm generation time: {self.local_inputs['delta_comm'].generation_time} is not greater than {self._start_time}")

        if self._modal_start_time is not None:
            for i in range(len(self._modal_start_time)):
                if self._modal_start_time[i] > t:
                    self.delta_comm[i] = 0
                    print(f"delta comm generation time: {self.delta_comm.generation_time} is not greater than {self._modal_start_time[i]}")
                    print(f" -> value of mode no. {i} is set to 0.")

        if self._skipOneStep:
            if self._StepIsNotGood:
                self.delta_comm *= 0
                self._StepIsNotGood = False
                print("WARNING: the delta commands of this step is set to 0 because skipOneStep key is active.")
            else:
                self._StepIsNotGood = True

        if self._bootstrap_ptr is not None:
            bootstrap_array = self._bootstrap_ptr
            bootstrap_time = bootstrap_array[:, 0]
            bootstrap_scale = bootstrap_array[:, 1]
            idx = self.xp.where(bootstrap_time <= self.t_to_seconds(t))[0]
            if len(idx) > 0:
                idx = idx[-1]
                if bootstrap_scale[idx] != 1:
                    print(f"ATTENTION: a scale factor of {bootstrap_scale[idx]} is applied to delta commands for bootstrap purpose.")
                    self.delta_comm *= bootstrap_scale[idx]
                else:
                    print("no scale factor applied")

# this is probably useless
#        n_delta_comm = self.delta_comm.size
#        if n_delta_comm < self.iir_filter_data.nfilter:
#            self.delta_comm = self.xp.zeros(self.iir_filter_data.nfilter, dtype=self.dtype)
#            self.delta_comm[:n_delta_comm] = self.local_inputs['delta_comm'].value

        if self._offset is not None:
            self.delta_comm[:self._offset.shape[0]] += self._offset

    def trigger_code(self):
        sden = self.iir_filter_data.den.shape
        snum = self.iir_filter_data.num.shape
        no = sden[1]
        ni = snum[1]

        # Delay the vectors
        self._ost[:, :-1] = self._ost[:, 1:]
        self._ost[:, -1] = 0  # Reset the last column

        self._ist[:, :-1] = self._ist[:, 1:]
        self._ist[:, -1] = 0  # Reset the last column

        # New input
        self._ist[:, ni - 1] = self.delta_comm

        # Precompute the reciprocal of the denominator
        factor = 1 / self.iir_filter_data.den[:, no - 1]

        # Compute new output
        num_contrib = self.xp.sum(self.iir_filter_data.num * self._gain_mod[:, None] * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] * self._ost[:, :no - 1], axis=1)
        self._ost[:, no - 1] = factor * (num_contrib - den_contrib)
        output = self._ost[:, no - 1]

        # Update the state
        self.state[:, 0] = output

    def post_trigger(self):
        super().post_trigger()

        # Calculate output from the state considering the delay
        remainder_delay = self.delay % 1
        if remainder_delay == 0:
            output = self.state[:, int(self.delay)]
        else:
            output = (remainder_delay * self.state[:, int(np.ceil(self.delay))] + \
                     (1 - remainder_delay) * self.state[:, int(np.ceil(self.delay))-1])

        if self._offset is not None and self.xp.all(output == 0):
            output[:self._offset.shape[0]] += self._offset

        self.out_comm.value = output
        self.out_comm.generation_time = self.current_time