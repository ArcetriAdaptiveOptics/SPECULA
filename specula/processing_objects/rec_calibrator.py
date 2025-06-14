import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputValue


class RecCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,         # TODO =0,
                 data_dir: str,       # TODO = "",         # Set by main simul object
                 rec_tag: str,        # TODO = "",
                 first_mode: int = 0,
                 pupdata_tag: str = None,
                 tag_template: str = None,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._nmodes = nmodes
        self._first_mode = first_mode
        self._data_dir = data_dir
        if tag_template is None and (rec_tag is None or rec_tag == 'auto'):
            raise ValueError('At least one of tag_template and rec_tag must be set')

        if rec_tag is None or rec_tag == 'auto':
            self._rec_filename = tag_template
        else:
            self._rec_filename = rec_tag
        self.inputs['intmat'] = InputValue(type=BaseValue)

    def trigger_code(self):

        # Do nothing, the computation is done in finalize
        self._im = self.local_inputs['intmat'].get(self.target_device_idx)

    def finalize(self):
        im = Intmat(self._im, pupdata_tag = self.pupdata_tag,
                    target_device_idx=self.target_device_idx, precision=self.precision)

        # TODO add to RM the information about the first mode
        if self._rec_filename:
            rec = im.generate_rec(self._nmodes)
            rec.save(os.path.join(self._data_dir, self._rec_filename))

    def setup(self):
        super().setup()

        if self._rec_filename:
            rec_path = os.path.join(self._data_dir, self._rec_filename)
            if not rec_path.endswith('.fits'):
                rec_path += '.fits'
            if os.path.exists(rec_path):
                raise FileExistsError(f'REC file {rec_path} already exists, please remove it')
