import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.connections import InputValue


class SnCalibrator(BaseProcessingObj):
    def __init__(self,
                 data_dir: str,         # Set by main simul object
                 output_tag: str = None,
                 tag_template: str = None,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._data_dir = data_dir
        if tag_template is None and (output_tag is None or output_tag == 'auto'):
            raise ValueError('At least one of tag_template and output_tag must be set')

        if output_tag is None or output_tag == 'auto':
            self._filename = tag_template
        else:
            self._filename = output_tag
        self.inputs['in_slopes'] = InputValue(type=Slopes)

    def trigger_code(self):
        self.slopes = Slopes(slopes=self.local_inputs['in_slopes'].slopes)
        self.slopes.generation_time = self.local_inputs['in_slopes'].generation_time

    def finalize(self):
        filename = self._filename
        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self._data_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.slopes.save(os.path.join(self._data_dir, filename))
        