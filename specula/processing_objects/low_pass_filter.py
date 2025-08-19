
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams
    
class LowPassFilter(IirFilter):
    def __init__(self,
                 simul_params: SimulParams,
                 cutoff_freq: float,    # TODO =1.0,           
                 amplif_fact: float=None,
                 n_ord: int=None,
                 delay: float=0,
                 offset: list=None,
                 og_shaper=None,
                 target_device_idx: int=None,
                 precision: int=None
                ):

        samp_freq = 1 / simul_params.time_step

        if amplif_fact is not None:
            if n_ord is not None:
                raise ValueError('Only one of amplif_fact and n_ord can be specified')
            iir_filter_data = IirFilterData.lpf_from_fc_and_ampl(cutoff_freq, amplif_fact,
                                               samp_freq, target_device_idx=target_device_idx)
        else:
            iir_filter_data = IirFilterData.lpf_from_fc(cutoff_freq, samp_freq, n_ord=n_ord,
                                                        target_device_idx=target_device_idx)

        # Initialize IirFilter object
        super().__init__(simul_params, iir_filter_data, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
