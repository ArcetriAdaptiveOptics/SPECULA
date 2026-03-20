import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.psd import PSD
from specula.connections import InputValue


class AliasingCalibrator(BaseProcessingObj):
    """
    Aliasing PSD calibrator processing object.
    Analyzes a set of slope measurements to compute the temporal PSD.
    """
    def __init__(self,
                 data_dir: str,         # Set by main simul object
                 recmat: Recmat,
                 output_tag: str = '',     
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):    
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._data_dir = data_dir
        self.overwrite = overwrite
        self._filename = output_tag
        self.rec = self.to_xp(recmat.recmat)
        self.slopes_list = []
        self._n_iter = 0
        self.inputs['in_slopes'] = InputValue(type=Slopes)

        self.aliasing_path = os.path.join(self._data_dir, self._filename)
        if not self.aliasing_path.endswith('.fits'):
            self.aliasing_path += '.fits'
        if os.path.exists(self.aliasing_path) and not self.overwrite:
            raise FileExistsError(f'Aliasing PSDs file {self.aliasing_path} already exists, please remove it')

    def trigger_code(self):
        self.slopes_list.append(self.local_inputs['in_slopes'].slopes.copy())
        self._n_iter += 1

    def finalize(self):
        slopes_timehist = self.to_xp(self.slopes_list)
        dt = self.t_to_seconds(self.current_time)/(self._n_iter)
        modes_thist = self.rec @ slopes_timehist.T
        modes_psd = PSD(modes_thist, dt=dt, nperseg=1024)
        
        os.makedirs(os.path.dirname(self.aliasing_path), exist_ok=True)
        modes_psd.save(self.aliasing_path,overwrite=self.overwrite)