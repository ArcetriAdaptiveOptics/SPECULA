import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intmat import Intmat
from specula.connections import InputValue
from specula.connections import InputList


class MultiRecCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,
                 data_dir: str,         # Set by main simul object
                 rec_tag: str = None,
                 rec_tag_template: str = None,
                 full_rec_tag: str = None,
                 full_rec_tag_template: str = None,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._nmodes = nmodes
        self._data_dir = data_dir
        self._rec_filename = self.tag_filename(rec_tag, rec_tag_template, prefix='rec')
        self._full_rec_filename = self.tag_filename(full_rec_tag, full_rec_tag_template, prefix='full_rec')
        self._overwrite = overwrite

        full_rec_path = self.full_rec_path()
        if full_rec_path and os.path.exists(full_rec_path) and not self._overwrite:
            raise FileExistsError(f'Rec file {full_rec_path} already exists, please remove it')

        self.inputs['intmat_list'] = InputList(type=Intmat)
        self.inputs['full_intmat'] = InputValue(type=Intmat)

    def tag_filename(self, tag, tag_template, prefix):
        if tag == 'auto' and tag_template is None:
            raise ValueError(f'{prefix}_tag_template must be set if {prefix}_tag is"auto"')

        if tag == 'auto':
            return tag_template
        else:
            return tag

    def rec_path(self, i):
        if self._rec_filename:
            return os.path.join(self._data_dir, self._rec_filename+str(i) + '.fits')
        else:
            return None

    def full_rec_path(self):
        if self._full_rec_filename:
            return os.path.join(self._data_dir, self._full_rec_filename + '.fits')
        else:
            return None

    def trigger_code(self):
        # Do nothing, the computation is done in finalize
        pass

    def finalize(self):
        self._ims = self.local_inputs['intmat_list']

        for i, intmat in enumerate(self._ims):
            if self.rec_path(i):
                rec = intmat.generate_rec(self._nmodes)
                rec.save(os.path.join(self._data_dir, self.rec_path(i)), overwrite=self._overwrite)

        full_intmat = self.local_inputs['full_intmat']

        os.makedirs(self._data_dir, exist_ok=True)

        full_rec_path = self.full_rec_path()
        if full_rec_path:
            fullrec = full_intmat.generate_rec(self._nmodes)
            fullrec.save(os.path.join(self._data_dir, full_rec_path), overwrite=self._overwrite)

    def setup(self):
        super().setup()

        for i in range(len(self.local_inputs['intmat_list'])):
            rec_path = self.rec_path(i)
            if rec_path and os.path.exists(rec_path) and not self._overwrite:
                raise FileExistsError(f'Rec file {rec_path} already exists, please remove it')
