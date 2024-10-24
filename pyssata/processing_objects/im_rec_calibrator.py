import os

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.slopes import Slopes
from pyssata.data_objects.intmat import Intmat
from pyssata.base_value import BaseValue
from pyssata.connections import InputValue


class ImRecCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,
                 data_dir: str,         # Set by main simul object
                 output_tag: str = None,
                 tag_template: str = None,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._nmodes = nmodes
        self._data_dir = data_dir
        if tag_template is None and (output_tag is None or output_tag == 'auto'):
            raise ValueError('At least one of tag_template and output_tag must be set')

        if output_tag is None or output_tag == 'auto':
            self._filename = tag_template
        else:
            self._filename = output_tag
        self._im = None
        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_commands'] = InputValue(type=BaseValue)

    def trigger_code(self):
        
        slopes = self.local_inputs['in_slopes'].slopes
        commands = self.local_inputs['in_commands'].value
        
        # First iteration
        if self._im is None:
            self._im = self.xp.zeros((self._nmodes, len(slopes)), dtype=self.dtype)

        idx = self.xp.nonzero(commands)
        if len(idx)>0:
            mode = idx[0]
            self._im[mode] += slopes / commands[idx]
    
    def finalize(self):
        im = Intmat(self._im, target_device_idx=self.target_device_idx, precision=self.precision)
        rec = im.generate_rec(self._nmodes)
        rec.save(os.path.join(self._data_dir, self._filename))

    def run_check(self, time_step):
        return True