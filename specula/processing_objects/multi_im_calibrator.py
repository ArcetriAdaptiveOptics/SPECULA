import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputList


class MultiImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,
                 data_dir: str,         # Set by main simul object
                 im_tag: str = None,
                 im_tag_template: str = None,
                 full_im_tag: str = None,
                 full_im_tag_template: str = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._nmodes = nmodes
        self._data_dir = data_dir
        self._im_filename = self.tag_filename(im_tag, im_tag_template, prefix='im')
        self._full_im_filename = self.tag_filename(full_im_tag, full_im_tag_template, prefix='full_im')

        self._ims = []
        self.inputs['in_slopes_list'] = InputList(type=Slopes)
        self.inputs['in_commands_list'] = InputList(type=BaseValue)

        self.outputs['intmat_list'] = []
        self.outputs['full_intmat'] = BaseValue('full_intmat', target_device_idx=self.target_device_idx)

    def tag_filename(self, tag, tag_template, prefix):
        if tag == 'auto' and tag_template is None:
            raise ValueError(f'{prefix}_tag_template must be set if {prefix}_tag is"auto"')

        if tag == 'auto':
            return tag_template
        else:
            return tag

    def im_path(self, i):
        if self._im_filename:
            return os.path.join(self._data_dir, self._im_filename+str(i) + '.fits')
        else:
            return None

    def full_im_path(self):
        if self._full_im_filename:
            return os.path.join(self._data_dir, self._full_im_filename + '.fits')
        else:
            return None

    def trigger_code(self):

        slopes = [x.slopes for x in self.local_inputs['in_slopes_list']]
        commands = [x.value for x in self.local_inputs['in_commands_list']]

        for im, ss, cc in zip(self._ims, slopes, commands):
            temp_im = im.value
            idx = self.xp.nonzero(cc)
            if len(idx)>0:
                mode = idx[0]
                temp_im[mode] += ss / cc[idx]
            im.value += temp_im

    def finalize(self):
        for i, im in enumerate(self._ims):
            intmat = Intmat(im, target_device_idx=self.target_device_idx, precision=self.precision)
            if self.im_path(i):
                intmat.save(os.path.join(self._data_dir, self.im_path(i)))

        full_im_path = self.full_im_path()
        if full_im_path:
            full_im = self.xp.hstack(self._ims)
            full_intmat = Intmat(full_im, target_device_idx=self.target_device_idx, precision=self.precision)
            if full_im_path:
                full_intmat.save(os.path.join(self._data_dir, full_im_path))

    def setup(self):
        super().setup()

        self._ims = []
        for i, slopes in enumerate(self.inputs['in_slopes_list'].get(self.target_device_idx)):
            im = BaseValue(f'intmat_{i}', target_device_idx=self.target_device_idx)
            im.value = self.xp.zeros((self._nmodes, len(slopes.slopes)), dtype=self.dtype)
            self._ims.append(im)
            im_path = self.im_path(i)
            full_im_path = self.full_im_path()
            if im_path and os.path.exists(im_path):
                raise FileExistsError(f'IM file {im_path} already exists, please remove it')
            if full_im_path and os.path.exists(full_im_path):
                raise FileExistsError(f'IM file {full_im_path} already exists, please remove it')

        self.outputs['intmat_list'] = self._ims
