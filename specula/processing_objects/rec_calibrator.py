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
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._nmodes = nmodes
        self._first_mode = first_mode
        self._data_dir = data_dir
        if tag_template is None and (rec_tag is None or rec_tag == 'auto'):
            raise ValueError('At least one of tag_template and rec_tag must be set')
        self.pupdata_tag = pupdata_tag
        self._overwrite = overwrite

        if rec_tag is None or rec_tag == 'auto':
            self._rec_filename = tag_template
        else:
            self._rec_filename = rec_tag
        self.inputs['in_intmat'] = InputValue(type=BaseValue)

    def trigger(self):

        # Do nothing, the computation is done in finalize
        self._im = self.local_inputs['in_intmat']

    def finalize(self):
        im = Intmat(self._im.value, pupdata_tag = self.pupdata_tag,
                    target_device_idx=self.target_device_idx, precision=self.precision)

        os.makedirs(self._data_dir, exist_ok=True)
        # TODO add to RM the information about the first mode
        if self._rec_filename:
            rec = im.generate_rec(self._nmodes)
            rec.save(os.path.join(self._data_dir, self._rec_filename), overwrite=self._overwrite)

    def setup(self):
        super().setup()

        if self._rec_filename:
            rec_path = os.path.join(self._data_dir, self._rec_filename)
            if not rec_path.endswith('.fits'):
                rec_path += '.fits'
            if os.path.exists(rec_path) and not self._overwrite:
                raise FileExistsError(f'REC file {rec_path} already exists, please remove it')
