import os
import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.processing_objects.im_calibrator import ImCalibrator
from specula.processing_objects.dm import DM
from specula.data_objects.slopes import Slopes
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputList


class MultiImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,
                 n_inputs: int,
                 data_dir: str,         # Set by main simul object
                 im_tag: str = None,
                 full_im_tag: str = None,
                 overwrite: bool = False,
                 pupilstop: Pupilstop = None,
                 source_dict: list = None,
                 dm: DM = None,
                 sensor_dict: list = None,
                 slopec_dict: list = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.n_inputs = n_inputs
        self.data_dir = data_dir

        if im_tag == 'auto':
            im_tag = self._generate_multi_im_tags(pupilstop, source_dict, dm, sensor_dict,
                                                slopec_dict)

        if full_im_tag == 'auto':
            raise NotImplementedError('full_im_tag auto generation not implemented yet')

        self.im_tag = im_tag
        self.full_im_tag = full_im_tag
        self.overwrite = overwrite

        # Path and existing file existence checks
        self.im_paths = []
        for i in range(self.n_inputs):
            path = os.path.join(self.data_dir, f"{self.im_tag[i]}.fits")
            self.im_paths.append(path)
            if os.path.exists(path) and not self.overwrite:
                raise FileExistsError(f'IM file {path} already exists, please remove it')

        self.full_im_path = os.path.join(self.data_dir, f"{self.full_im_tag}.fits")
        if os.path.exists(self.full_im_path) and not self.overwrite:
            raise FileExistsError(f'IM file {self.full_im_path} already exists, please remove it')

        # Add counts tracking for each input, this is used to normalize the IM
        self.count_commands = [np.zeros(nmodes, dtype=int) for _ in range(n_inputs)]

        self.inputs['in_slopes_list'] = InputList(type=Slopes)
        self.inputs['in_commands_list'] = InputList(type=BaseValue)

        self.outputs['out_intmat_list'] = []
        for i in range(self.n_inputs):
            im = Intmat(nmodes=nmodes, nslopes=0, target_device_idx=self.target_device_idx)
            self.outputs['out_intmat_list'].append(im)
        self.outputs['out_intmat_full'] = Intmat(nmodes=nmodes, nslopes=0,
                                                 target_device_idx=self.target_device_idx)

    def _generate_multi_im_tags(self, pupilstop, source_dict, dm, sensor_dict, slopec_dict):
        """Generate IM tags for multi-input configuration using static method."""

        if source_dict is None or len(source_dict) != self.n_inputs:
            raise ValueError(f'source_dict must have {self.n_inputs} elements if im_tag is "auto"')
        if sensor_dict is None or len(sensor_dict) != self.n_inputs:
            raise ValueError(f'sensor_dict must have {self.n_inputs} elements if im_tag is "auto"')
        if slopec_dict is None or len(slopec_dict) != self.n_inputs:
            raise ValueError(f'slopec_dict must have {self.n_inputs} elements if im_tag is "auto"')

        # Generate tag for each input using the static method
        tags = []

        # Get the keys from the dictionaries (should be the same order)
        source_keys = list(source_dict.keys())
        sensor_keys = list(sensor_dict.keys())
        slopec_keys = list(slopec_dict.keys())

        for i in range(self.n_inputs):
            source = source_dict[source_keys[i]]
            sensor = sensor_dict[sensor_keys[i]]
            slopec = slopec_dict[slopec_keys[i]]

            # Use static method
            tag = ImCalibrator.generate_im_tag(pupilstop, source, dm, sensor, slopec, self.nmodes)
            tags.append(tag)

        return tags

    def trigger_code(self):

        slopes = [x.slopes for x in self.local_inputs['in_slopes_list']]
        commands = [x.value for x in self.local_inputs['in_commands_list']]

        # First iteration
        if self.outputs['out_intmat_list'][0].nslopes == 0:
            for im, ss in zip(self.outputs['out_intmat_list'], slopes):
                im.set_nslopes(len(ss))

        for cmd_idx, cc in enumerate(commands):
            idx = self.xp.nonzero(cc)[0]
            if len(idx) > 0:
                mode = int(idx[0])
                if mode < self.nmodes:
                    # Update ALL interaction matrices for this command
                    for i, (im, ss) in enumerate(zip(self.outputs['out_intmat_list'], slopes)):
                        im.modes[mode] += ss / cc[idx]
                        self.count_commands[i][mode] += 1

        # Update generation time for all IMs
        for im in self.outputs['out_intmat_list']:
            im.generation_time = self.current_time

    def finalize(self):
        os.makedirs(self.data_dir, exist_ok=True)

        for i, im in enumerate(self.outputs['out_intmat_list']):
            # Normalize by counts before saving
            for mode in range(self.nmodes):
                if self.count_commands[i][mode] > 0:
                    im.modes[mode] /= self.count_commands[i][mode]
            if self.im_paths[i]:
                im.save(os.path.join(self.data_dir, self.im_paths[i]), overwrite=self.overwrite)
            im.generation_time = self.current_time

        if self.full_im_path:
            if not self.outputs['out_intmat_list']:
                full_im = self.xp.array([])
            else:
                full_im = self.xp.vstack([im.intmat for im in self.outputs['out_intmat_list']])

            self.outputs['out_intmat_full'].intmat = full_im
            self.outputs['out_intmat_full'].generation_time = self.current_time
            if self.full_im_path:
                self.outputs['out_intmat_full'].save(
                    os.path.join(self.data_dir, self.full_im_path), overwrite=self.overwrite)

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
