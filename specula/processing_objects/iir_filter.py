import numpy as np

import sys
try:
    from numba import jit
except ImportError:
    print("WARNING: numba not found, using non-jitted version of IirFilter")
    def jit(x, *args, **kwargs):
        return x

from specula.data_objects.iir_filter_data import IirFilterData
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.lib.calc_loop_delay import calc_loop_delay

gxp = None
gdtype = None

def trigger_function(input, _outFinite, _ist, _ost, _iir_filter_data_num, _iir_filter_data_den, delay, state, total_length):
    '''
    1) this function uses a global variables for xp and dtype, because NUMBA is unable
    to compile if they are given as parameters instead.
    
    2) the xp.isfinite() is used to detect modes that must not be time-filtered
       during this step (they are supposed to be set to NaN instead),
       This feature is not used by SPECULA yet and it is probably better to find
       an entirely different way, like for example add a mask vector to the modes vector,
       to make it possible an implemention that does not rely on variable-length arrays
       (for easier GPU implementation).
    '''
    global gxp, gdtype
    comm = input*0

    gxp.isfinite(input, _outFinite)

    # Extract modes that need to be time-filtered
    _idx_finite = gxp.where(_outFinite)[0]
    temp_input = input[_idx_finite]
    temp_ist = _ist[_idx_finite]
    temp_ost = _ost[_idx_finite]
    temp_num = _iir_filter_data_num[_idx_finite, :]
    temp_den = _iir_filter_data_den[_idx_finite, :]

    # online_filter_nojit
    sden = temp_den.shape
    snum = temp_num.shape
    n_input = temp_input.size
    no = sden[1]
    ni = snum[1]

    # Delay the vectors
    temp_ost = gxp.concatenate((temp_ost[:, 1:], gxp.zeros((sden[0], 1), dtype=gdtype)), axis=1)
    temp_ist = gxp.concatenate((temp_ist[:, 1:], gxp.zeros((sden[0], 1), dtype=gdtype)), axis=1)

    # New input
    temp_ist[:n_input, ni - 1] = temp_input

    # New output
    factor = 1/temp_den[:, no - 1]
    temp_ost[:, no - 1] = factor * gxp.sum(temp_num * temp_ist, axis=1)
    temp_ost[:, no - 1] -= factor * gxp.sum(temp_den[:, :no - 1] * temp_ost[:, :no - 1], axis=1)
    temp_output = temp_ost[:n_input, no - 1]

    # Put back time-filtered modes
    comm[_idx_finite] = temp_output
    _ost[_idx_finite] = temp_ost
    _ist[_idx_finite] = temp_ist

    finite_mask = gxp.isfinite(comm)
    if gxp.any(finite_mask):
        if delay > 0:
            state[:, 1:total_length] = state[:, 0:total_length-1]

        state[finite_mask, 0] = comm[finite_mask]

    remainder_delay = delay % 1
    if remainder_delay == 0:
        comm = state[:, int(delay)]
    else:
        comm = (remainder_delay * state[:, int(np.ceil(delay))] + (1 - remainder_delay) * state[:, int(np.ceil(delay))-1])

#    TODO offset not implemented yet
#
#    if offset is not None and gxp.all(comm == 0):
#        comm[:self._offset.shape[0]] += offset
#        print("WARNING (IIRCONTROL): self.newc is a null vector, applying offset.")

#        if self._verbose:
#            n_self.newc = self.newc.size
#            print(f"first {min(6, n_delta_comm)} delta_comm values: {self.delta_comm[:min(6, n_delta_comm)]}")
#            print(f"first {min(6, n_newc)} comm values: {self.newc[:min(6, n_newc)]}")
#        else:
#            if self._verbose:
#                print(f"delta comm generation time: {self.local_inputs['delta_comm'].generation_time} is not equal to {t}")
#            self.newc = self.last_state

    return comm, state


class IirFilter(BaseProcessingObj):
    '''Infinite Impulse Response filter based Time Control
    
    Set *integration* to False to disable integration, regardless
    of wha the input IirFilter object contains
    '''
    def __init__(self,
                 iir_filter_data: IirFilterData,
                 delay: int=0,
                 integration: bool=True,
                 offset: float=None,
                 og_shaper=None,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        global gxp, gdtype    

        self._verbose = True
        self.iir_filter_data = iir_filter_data
        
        self.integration = integration
        if integration is False:
            raise NotImplementedError('IirFilter: integration=False is not implemented yet')
        
        if og_shaper is not None:
            raise NotImplementedError('OG Shaper not implementd yet')

        if offset != None:
            raise NotImplementedError('Offset not implemented yet')

        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        gxp = self.xp
        gdtype = self.dtype
        if self.xp==np:
            self.trigger_function = jit(trigger_function, nopython=True, cache=True)
        else:
            self.trigger_function = trigger_function

        self.delay = delay if delay is not None else 0
        self._n = iir_filter_data.nfilter
        self._type = iir_filter_data.num.dtype
        self.set_state_buffer_length(int(np.ceil(self.delay)) + 1)
        
        # Initialize state vectors
        self._ist = self.xp.zeros_like(iir_filter_data.num)
        self._ost = self.xp.zeros_like(iir_filter_data.den)

        self.out_comm = BaseValue(value=self.xp.zeros(self._n, dtype=self.dtype), target_device_idx=target_device_idx)
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self.out_comm

        self._opticalgain = None  # TODO
        self._og_shaper = None  # TODO
        self._offset = None  # TODO
        self._bootstrap_ptr = None  # TODO
        self._modal_start_time = None  # TODO
        self._time_gmt_imm = None  # TODO
        self._gain_gmt_imm = None  # TODO
        self._do_gmt_init_mod_manager = False  # TODO
        self._skipOneStep = False  # TODO
        self._StepIsNotGood = False  # TODO
        self._start_time = 0  # TODO

        self._outFinite = self.xp.zeros(self.iir_filter_data.nfilter, dtype=self.dtype)
        self._idx_finite = self.xp.zeros(self.iir_filter_data.nfilter, dtype=self.dtype)

    def set_state_buffer_length(self, total_length):
        self._total_length = total_length
        if self._n is not None and self._type is not None:
            self.state = self.xp.zeros((self._n, self._total_length), dtype=self.dtype)
            self.comm = self.xp.zeros((self._n, 1), dtype=self.dtype)  # TODO not used?

    def auto_params_management(self, main_params, control_params, detector_params, dm_params, slopec_params):
        result = control_params.copy()

        if str(result['delay']) == 'auto':
            binning = detector_params.get('binning', 1)
            computation_time = slopec_params.get('computation_time', 0) if slopec_params else 0
            delay = calc_loop_delay(1.0 / detector_params['dt'], dm_set=dm_params['settling_time'],
                                    type=detector_params['name'], bin=binning, comp_tim=computation_time)
            if delay == float('inf'):
                raise ValueError("Delay calculation resulted in infinity")
            result['delay'] = delay * (1.0 / main_params['time_step']) - 1

        return result

    @property
    def last_state(self):
        return self.state[:, 0]

    def set_modal_start_time(self, modal_start_time):
        modal_start_time_ = self.xp.array(modal_start_time, dtype=self.dtype)
        for i in range(len(modal_start_time)):
            modal_start_time_[i] = self.seconds_to_t(modal_start_time[i])
        self._modal_start_time = modal_start_time_

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_comm = self.local_inputs['delta_comm'].value

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

        # Avoid warnings        
        def gmt_init_mod_manager(*args, **kwargs):
            pass

        if self._do_gmt_init_mod_manager:
            time_idx = self._time_gmt_imm if self._time_gmt_imm is not None else self.xp.zeros(0, dtype=self.dtype)
            gain_idx = self._gain_gmt_imm if self._gain_gmt_imm is not None else self.xp.zeros(0, dtype=self.dtype)
            self.delta_comm *= gmt_init_mod_manager(self.t_to_seconds(t), len(self.delta_comm), time_idx=time_idx, gain_idx=gain_idx)

# this is probably useless
#        n_delta_comm = self.delta_comm.size
#        if n_delta_comm < self.iir_filter_data.nfilter:
#            self.delta_comm = self.xp.zeros(self.iir_filter_data.nfilter, dtype=self.dtype)
#            self.delta_comm[:n_delta_comm] = self.local_inputs['delta_comm'].value

        if self._offset is not None:
            self.delta_comm[:self._offset.shape[0]] += self._offset

    def trigger_code(self):
        '''
        self.trigger_function is factored out because
        for the CPU case it can be jit-compiled with NUMBA.
        For GPU the graph is not implemented yet.
        '''
        self.out_comm.value[:], self.state = self.trigger_function(self.delta_comm, self._outFinite, self._ist, self._ost, 
                                                       self.iir_filter_data.num, self.iir_filter_data.den, self.delay, 
                                                       self.state, self._total_length)

        self.out_comm.generation_time = self.current_time
    

