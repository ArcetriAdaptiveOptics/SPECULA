import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.psd import PSD
from specula.connections import InputValue


class PSDCalibrator(BaseProcessingObj):
    """
    PSD calibrator processing object.
    Analyzes a set of inputs to compute the temporal PSD.
    """
    def __init__(self,
                 data_dir: str,         # Set by main simul object
                 recmat: Recmat = None,
                 output_tag: str = '',     
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):    
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._data_dir = data_dir
        self.overwrite = overwrite
        self._filename = output_tag
        if recmat is not None:
            self.rec = self.to_xp(recmat.recmat)
        else:
            self.rec = None
        self.values_list = []
        self._n_iter = 0
        self.inputs['in_values'] = InputValue(type=Slopes)

        self.aliasing_path = os.path.join(self._data_dir, self._filename)
        if not self.aliasing_path.endswith('.fits'):
            self.aliasing_path += '.fits'
        if os.path.exists(self.aliasing_path) and not self.overwrite:
            raise FileExistsError(f'PSDs file {self.aliasing_path} already exists, please remove it')

    def trigger_code(self):
        self.values_list.append(self.local_inputs['in_values'].get_value())
        self._n_iter += 1

    def finalize(self):
        values_timehist = self.to_xp(self.values_list).T
        dt = self.t_to_seconds(self.current_time)/(max(1,self._n_iter-1))
        if self.rec is not None:
            values_timehist = self.rec @ values_timehist
        modes_psd = PSD(values_timehist, dt=dt, nperseg=1024)
        
        os.makedirs(os.path.dirname(self.aliasing_path), exist_ok=True)
        modes_psd.save(self.aliasing_path,overwrite=self.overwrite)