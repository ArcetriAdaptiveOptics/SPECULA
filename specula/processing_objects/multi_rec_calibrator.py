import os

from specula.base_processing_obj import BaseProcessingObj, InputDesc
from specula.data_objects.intmat import Intmat
from specula.connections import InputValue
from specula.connections import InputList


class MultiRecCalibrator(BaseProcessingObj):
    """
    Multiple reconstructor calibrator processing object.
    Calibrates reconstruction matrices for multiple sources/sensors pairs
    """
    def __init__(self,
                 nmodes: int,
                 data_dir: str,         # Set by main simul object
                 rec_tag: str = None,
                 full_rec_tag: str = None,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._nmodes = nmodes
        self._data_dir = data_dir
        self._rec_filename = rec_tag
        self._full_rec_filename = full_rec_tag
        self._overwrite = overwrite

        full_rec_path = self.full_rec_path()
        if full_rec_path and os.path.exists(full_rec_path) and not self._overwrite:
            raise FileExistsError(f'Rec file {full_rec_path} already exists, please remove it')

        self.inputs['intmat_list'] = InputList(type=Intmat)
        self.inputs['full_intmat'] = InputValue(type=Intmat)

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

    @classmethod
    def input_names(cls):
        return {'intmat_list': InputDesc(Intmat, 'List of per-sensor interaction matrices to invert'),
                'full_intmat': InputDesc(Intmat, 'Full combined interaction matrix to invert')}

    @classmethod
    def output_names(cls):
        return {}

    def trigger_code(self):
        # Do nothing, the computation is done in finalize
        pass

    def finalize(self):
        ims = self.local_inputs['intmat_list']

        for i, intmat in enumerate(ims):
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


