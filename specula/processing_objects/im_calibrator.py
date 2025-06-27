import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputValue


class ImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,         # TODO =0,
                 data_dir: str,       # TODO = "",         # Set by main simul object
                 im_tag: str='',
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
        if tag_template is None and (im_tag is None or im_tag == 'auto'):
            raise ValueError('At least one of tag_template and im_tag must be set')
        self.pupdata_tag = pupdata_tag
        self._overwrite = overwrite

        if im_tag is None or im_tag == 'auto':
            self._im_filename = tag_template
        else:
            self._im_filename = im_tag
        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_commands'] = InputValue(type=BaseValue)

        self.output_im = [Slopes(length=2, target_device_idx=self.target_device_idx) for _ in range(nmodes)]
        self.outputs['out_im'] = self.output_im
        self._im = BaseValue('intmat', target_device_idx=self.target_device_idx)
        self.outputs['out_intmat'] = self._im

    def trigger_code(self):

        # Slopes *must* have been refreshed. We could have been triggered
        # just by the commands, but we need to skip it
        if self.local_inputs['in_slopes'].generation_time != self.current_time:
            return

        slopes = self.local_inputs['in_slopes'].slopes
        commands = self.local_inputs['in_commands'].value

        # First iteration initialization
        if self._im.value is None:
            self._im.value = self.xp.zeros((self._nmodes, len(slopes)), dtype=self.dtype)
            for i in range(self._nmodes):
                self.output_im[i].resize(len(self._im.value[i]))
            if self.verbose:
                print(f"Initialized interaction matrix: {self._im.value.shape}")

        idx = self.xp.nonzero(commands)

        if len(idx[0])>0:
            mode = int(idx[0]) - self._first_mode
            if mode < self._nmodes:
                self._im.value[mode] += slopes / commands[idx]

        in_slopes_object = self.local_inputs['in_slopes']

        for i in range(self._nmodes):
            self.output_im[i].slopes[:] = self._im.value[i].copy()
            self.output_im[i].single_mask = in_slopes_object.single_mask
            self.output_im[i].display_map = in_slopes_object.display_map
            self.output_im[i].generation_time = self.current_time

        self._im.generation_time = self.current_time

    def finalize(self):
        im = Intmat(self._im.value, pupdata_tag = self.pupdata_tag,
                    target_device_idx=self.target_device_idx, precision=self.precision)

        os.makedirs(self._data_dir, exist_ok=True)
        # TODO add to IM the information about the first mode
        if self._im_filename:
            im.save(os.path.join(self._data_dir, self._im_filename), overwrite=self._overwrite)

    def setup(self):
        super().setup()

        if self._im_filename:
            im_path = os.path.join(self._data_dir, self._im_filename)
            if not im_path.endswith('.fits'):
                im_path += '.fits'
            if os.path.exists(im_path) and not self._overwrite:
                raise FileExistsError(f'IM file {im_path} already exists, please remove it')