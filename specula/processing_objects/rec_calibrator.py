import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intmat import Intmat
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
        self.nmodes = nmodes
        self.first_mode = first_mode
        self.data_dir = data_dir
        if tag_template is None and (rec_tag is None or rec_tag == 'auto'):
            raise ValueError('At least one of tag_template and rec_tag must be set')
        self.pupdata_tag = pupdata_tag
        self.overwrite = overwrite

        if rec_tag is None or rec_tag == 'auto':
            rec_filename = tag_template
        else:
            rec_filename = rec_tag

        rec_path = os.path.join(self.data_dir, rec_filename)
        if not rec_path.endswith('.fits'):
            rec_path += '.fits'
        if os.path.exists(rec_path) and not self.overwrite:
            raise FileExistsError(f'REC file {rec_path} already exists, please remove it')
        self.rec_path = rec_path

        self.inputs['in_intmat'] = InputValue(type=Intmat)

    def finalize(self):
        im = self.local_inputs['in_intmat']

        os.makedirs(self.data_dir, exist_ok=True)

        # TODO add to RM the information about the first mode
        rec = im.generate_rec(self.nmodes)
        rec.save(self.rec_path, overwrite=self.overwrite)
