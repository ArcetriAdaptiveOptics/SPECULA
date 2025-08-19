import os

from specula.base_processing_obj import BaseProcessingObj
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.slopes import Slopes
from specula.data_objects.source import Source
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula import RAD2ASEC

class ImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,         # TODO =0,
                 data_dir: str,       # TODO = "",         # Set by main simul object
                 im_tag: str='',
                 first_mode: int = 0,
                 overwrite: bool = False,
                 source: Source = None,
                 dm: DM = None,
                 sensor: BaseProcessingObj = None,
                 slopec: BaseProcessingObj = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._nmodes = nmodes
        self._first_mode = first_mode
        self._data_dir = data_dir
        if im_tag is None or im_tag == 'auto':
            im_tag = 'im'
            # SOURCE coordinates
            if source.polar_coordinates[0] != 0:
                im_tag += f'_{source.polar_coordinates[0]:.1f}r{source.polar_coordinates[0]:.1f}a'
            if source.height != float('inf'):
                im_tag += f'_{source.height:.1f}h'
            # WFS related
            if isinstance(sensor, SH):
                im_tag += '_sh'
                im_tag += f'_{sensor._wavelengthInNm}nm'
                im_tag += f'_{sensor._lenslet.n_lenses}x{sensor._lenslet.n_lenses}sa'
                im_tag += f'_{sensor._subap_wanted_fov * RAD2ASEC}asec'
            if isinstance(sensor, ModulatedPyramid):
                im_tag += '_pyr'
                im_tag += f'_{sensor.wavelength_in_nm}nm'
                im_tag += f'_{sensor.pup_diam}x{sensor.pup_diam}sa' # TODO THIS IS NOT PRESENT
                im_tag += f'_{sensor.fov}asec' # TODO THIS IS NOT PRESENT
            # SLOPEC related
            if isinstance(slopec, ShSlopec):
                if slopec.quadcell_mode:
                    im_tag += f'_qc'
            if isinstance(slopec, PyrSlopec):
                if slopec.slopes_from_intensity:
                    im_tag += f'_slint'
            # TODO DM related keys
            im_tag = f'_{self._nmodes}modes'
            if self._first_mode != 0:
                im_tag += f'_firstmode{self._first_mode}'
            
        self._overwrite = overwrite

        im_filename = im_tag
        self.im_path = os.path.join(self._data_dir, im_filename)
        if not self.im_path.endswith('.fits'):
            self.im_path += '.fits'
        if os.path.exists(self.im_path) and not self._overwrite:
            raise FileExistsError(f'IM file {self.im_path} already exists, please remove it')

        # Add counts tracking, this is used to normalize the IM
        self.count_commands = self.xp.zeros(nmodes, dtype=self.xp.int32)

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

        idx = self.xp.nonzero(commands)[0]

        if len(idx)>0:
            mode = int(idx[0]) - self._first_mode
            if mode < self._nmodes:
                self._im.value[mode] += slopes / commands[idx]
                self.count_commands[mode] += 1

        in_slopes_object = self.local_inputs['in_slopes']

        for i in range(self._nmodes):
            self.output_im[i].slopes[:] = self._im.value[i].copy()
            self.output_im[i].single_mask = in_slopes_object.single_mask
            self.output_im[i].display_map = in_slopes_object.display_map
            self.output_im[i].generation_time = self.current_time

        self._im.generation_time = self.current_time

    def finalize(self):
        # normalize by counts
        for i in range(self._nmodes):
            if self.count_commands[i] > 0:
                self._im.value[i] /= self.count_commands[i]

        im = Intmat(self._im.value, pupdata_tag = self.pupdata_tag,
                    target_device_idx=self.target_device_idx, precision=self.precision)

        os.makedirs(self._data_dir, exist_ok=True)

        # TODO add to IM the information about the first mode
        im.save(self.im_path, overwrite=self._overwrite)

