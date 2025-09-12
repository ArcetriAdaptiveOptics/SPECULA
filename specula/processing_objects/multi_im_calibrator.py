import os
import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputList


class MultiImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,
                 n_inputs: int,
                 data_dir: str,         # Set by main simul object
                 im_tag: str = None,
                 im_tag_template: str = None,
                 full_im_tag: str = None,
                 full_im_tag_template: str = None,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.n_inputs = n_inputs
        self.data_dir = data_dir
        self.im_filename = self.tag_filename(im_tag, im_tag_template, prefix='im')
        self.full_im_filename = self.tag_filename(full_im_tag, full_im_tag_template, prefix='full_im')
        self.overwrite = overwrite

        # Add counts tracking for each input, this is used to normalize the IM
        self.count_commands = [np.zeros(nmodes, dtype=int) for _ in range(n_inputs)]

        # Existing file existence checks
        for i in range(self.n_inputs):  # Use self.n_inputs instead of len(...)
            im_path = self.im_path(i)
            if im_path and os.path.exists(im_path) and not self.overwrite:
                raise FileExistsError(f'IM file {im_path} already exists, please remove it')

        full_im_path = self.full_im_path()
        if full_im_path and os.path.exists(full_im_path) and not self.overwrite:
            raise FileExistsError(f'IM file {full_im_path} already exists, please remove it')

        self.inputs['in_slopes_list'] = InputList(type=Slopes)
        self.inputs['in_commands_list'] = InputList(type=BaseValue)

        self.outputs['out_intmat_list'] = []
        for i in range(self.n_inputs):
            im = Intmat(nmodes=nmodes, nslopes=0, target_device_idx=self.target_device_idx)
            self.outputs['out_intmat_list'].append(im)
        self.outputs['out_intmat_full'] = Intmat(nmodes=nmodes, nslopes=0, target_device_idx=self.target_device_idx)

    def tag_filename(self, tag, tag_template, prefix):
        if tag == 'auto' and tag_template is None:
            raise ValueError(f'{prefix}_tag_template must be set if {prefix}_tag is"auto"')

        if tag == 'auto':
            return tag_template
        else:
            return tag

    def im_path(self, i):
        if self.im_filename:
            return os.path.join(self.data_dir, self.im_filename+str(i) + '.fits')
        else:
            return None

    def full_im_path(self):
        if self.full_im_filename:
            return os.path.join(self.data_dir, self.full_im_filename + '.fits')
        else:
            return None

    def trigger_code(self):

        slopes = [x.slopes for x in self.local_inputs['in_slopes_list']]
        commands = [x.value for x in self.local_inputs['in_commands_list']]

        # First iteration
        if self.outputs['out_intmat_list'][0].nslopes == 0:
            for im, ss in zip(self.outputs['out_intmat_list'], slopes):
                im.set_nslopes(len(ss))

        for i, (im, ss, cc) in enumerate(zip(self.outputs['out_intmat_list'], slopes, commands)):
            idx = self.xp.nonzero(cc)[0]

            if len(idx)>0:
                mode = int(idx[0])
                if mode < self.nmodes:
                    im.modes[mode] += ss / cc[idx]
                    self.count_commands[i][mode] += 1
            im.generation_time = self.current_time

    def finalize(self):
        os.makedirs(self.data_dir, exist_ok=True)

        for i, im in enumerate(self.outputs['out_intmat_list']):
            # Normalize by counts before saving
            for mode in range(self.nmodes):
                if self.count_commands[i][mode] > 0:
                    im.modes[mode] /= self.count_commands[i][mode]
            if self.im_path(i):
                im.save(os.path.join(self.data_dir, self.im_path(i)), overwrite=self.overwrite)
            im.generation_time = self.current_time

        full_im_path = self.full_im_path()
        if full_im_path:
            if not self.outputs['out_intmat_list']:
                full_im = self.xp.array([])
            else:
                full_im = self.xp.vstack([im.intmat for im in self.outputs['out_intmat_list']])

            self.outputs['out_intmat_full'].intmat = full_im
            self.outputs['out_intmat_full'].generation_time = self.current_time
            if full_im_path:
                self.outputs['out_intmat_full'].save(os.path.join(self.data_dir, full_im_path), overwrite=self.overwrite)

    def setup(self):
        super().setup()

        # Validate that actual input length matches expected n_inputs
        actual_n_inputs = len(self.local_inputs['in_slopes_list'])
        if actual_n_inputs != self.n_inputs:
            raise ValueError(
                f"Number of input slopes ({actual_n_inputs}) does not match "
                f"expected n_inputs ({self.n_inputs}). "
                f"Please check your configuration."
            )

        # Also validate commands list has the same length
        actual_n_commands = len(self.local_inputs['in_commands_list'])
        if actual_n_commands != self.n_inputs:
            raise ValueError(
                f"Number of input commands ({actual_n_commands}) does not match "
                f"expected n_inputs ({self.n_inputs}). "
                f"Both slopes and commands lists must have the same length."
            )

