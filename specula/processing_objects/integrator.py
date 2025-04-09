
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.iir_filter_data import IirFilterData

    
class Integrator(IirFilter):
    def __init__(self, 
                 int_gain: float, #  TODO =1.0,
                 ff: list=None, 
                 delay: float=0,    #  TODO =0.0, 
                 offset: float=None, 
                 og_shaper=None,                 
                 target_device_idx: int=None, 
                 precision: int=None
                ):
        
        iir_filter_data = IirFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IirFilter object
        super().__init__(iir_filter_data, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
