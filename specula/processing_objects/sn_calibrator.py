import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.connections import InputValue


class SnCalibrator(BaseProcessingObj):
    def __init__(self,
                 data_dir: str,         # Set by main simul object
                 output_tag: str = None,
                 overwrite: bool = False,
                 tag_template: str = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._data_dir = data_dir
        self.overwrite = overwrite

        if tag_template is None and (output_tag is None or output_tag == 'auto'):
            raise ValueError('At least one of tag_template and output_tag must be set')

        if output_tag is None or output_tag == 'auto':
            self._filename = tag_template
        else:
            self._filename = output_tag
        self.slopes = None
        self._n_iter = 0
        self.inputs['in_slopes'] = InputValue(type=Slopes)

        self.sn_path = os.path.join(self._data_dir, self._filename)
        if not self.sn_path.endswith('.fits'):
            self.sn_path += '.fits'
        if os.path.exists(self.sn_path) and not self.overwrite:
            raise FileExistsError(f'Slope null file {self.sn_path} already exists, please remove it')

    def trigger_code(self):
        if self.slopes is None:
            self.slopes = Slopes(slopes=self.local_inputs['in_slopes'].slopes.copy(), target_device_idx=self.target_device_idx)
        else:
            self.slopes.slopes += self.local_inputs['in_slopes'].slopes.copy()
        self._n_iter += 1

    def finalize(self):
        # n_iter is used to normalize a slope null computed as average of several slopes
        self.slopes.slopes /= self._n_iter
        filename = self._filename
        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self._data_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.slopes.save(file_path,overwrite=self.overwrite)