import os

from specula.base_processing_obj import BaseProcessingObj
from specula.processing_objects.dm import DM
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.slopes import Slopes
from specula.data_objects.source import Source
from specula.data_objects.intmat import Intmat
from specula.base_value import BaseValue
from specula.connections import InputValue


class ImCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,         # TODO =0,
                 data_dir: str,       # TODO = "",         # Set by main simul object
                 im_tag: str='',
                 first_mode: int = 0,
                 overwrite: bool = False,
                 pupilstop: Pupilstop = None,
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

        self.subapdata_tag = None
        self.pupdata_tag = None
        if slopec is not None:
            if isinstance(slopec, ShSlopec):
                if slopec.subapdata.tag is not None and slopec.subapdata.tag != '':
                    self.subapdata_tag = slopec.subapdata.tag
            if isinstance(slopec, PyrSlopec):
                if slopec.pupdata.tag is not None and slopec.pupdata.tag != '':
                    self.pupdata_tag = slopec.pupdata.tag

        if im_tag is None or im_tag == 'auto':
            im_tag = self._generate_im_tag(pupilstop, source, dm, sensor, slopec)
        self.im_tag = im_tag

        self._overwrite = overwrite

        self.im_path = os.path.join(self._data_dir, self.im_tag)
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

    def _generate_im_tag(self, pupilstop, source, dm, sensor, slopec):
        """Generate automatic im_tag based on configuration parameters retrieved from other objects."""

        if pupilstop is None:
            raise ValueError('Pupilstop must be provided if im_tag is not set')
        if source is None:
            raise ValueError('Source must be provided if im_tag is not set')
        if dm is None:
            raise ValueError('DM must be provided if im_tag is not set')
        if sensor is None:
            raise ValueError('Sensor must be provided if im_tag is not set')
        if slopec is None:
            raise ValueError('SLOPEC must be provided if im_tag is not set')

        im_tag = 'im'

        # WFS related
        if isinstance(sensor, SH):
            im_tag += '_sh'
            im_tag += f'{sensor.subap_on_diameter}x{sensor.subap_on_diameter}sa'
            im_tag += f'_w{sensor.wavelength_in_nm}nm'
            im_tag += f'_f{sensor.subap_wanted_fov}asec'
        if isinstance(sensor, ModulatedPyramid):
            im_tag += '_pyr'
            im_tag += f'{sensor.pup_diam}x{sensor.pup_diam}sa'
            im_tag += f'_w{sensor.wavelength_in_nm}nm'
            im_tag += f'_f{sensor.fov}asec'

        # SLOPEC related
        im_tag += f'_ns{slopec.nsubaps()}'
        if isinstance(slopec, ShSlopec):
            if slopec.quadcell_mode:
                im_tag += f'_qc'
            if slopec.subapdata.tag is not None and slopec.subapdata.tag != '':
                im_tag += f'_{slopec.subapdata.tag}'
        if isinstance(slopec, PyrSlopec):
            if slopec.slopes_from_intensity:
                im_tag += f'_slint'
            if slopec.pupdata.tag is not None and slopec.pupdata.tag != '':
                im_tag += f'_{slopec.pupdata.tag}'

        # no. pixel and pixel pitch
        im_tag += f'_pup{dm.simul_params.pixel_pupil}x{dm.simul_params.pixel_pupil}p{dm.simul_params.pixel_pitch}m'

        # SOURCE coordinates
        if source.polar_coordinates[0] != 0:
            im_tag += f'_coor{source.polar_coordinates[0]:.1f}a{source.polar_coordinates[0]:.1f}d'
        if source.height != float('inf'):
            im_tag += f'_h{source.height:.1f}m'

        # DM related keys
        im_tag += f'_dm'
        if dm.mask.shape[0] != dm.simul_params.pixel_pupil:
            im_tag += f'{dm.mask.shape[0]}x{dm.mask.shape[1]}p'
        if dm.type_str is not None:
            im_tag += '_'+dm.type_str
        elif dm.tag is not None and dm.tag != '':
            im_tag += '_'+dm.tag
        nmodes_dm = dm.ifunc.shape[0]
        im_tag += f'_{min(nmodes_dm,self._nmodes)}mds'
        if self._first_mode != 0:
            im_tag += f'_firstm{self._first_mode}'

        # Pupilstop
        im_tag += f'_stop'
        if pupilstop.tag is not None and pupilstop.tag != '':
            im_tag += f'_{pupilstop.tag}'
        else:
            if pupilstop.mask_diam is not None and pupilstop.mask_diam != 1.0:
                im_tag += f'd{pupilstop.mask_diam:.1f}'
            if pupilstop.obs_diam is not None and pupilstop.obs_diam != 0.0:
                im_tag += f'o{pupilstop.obs_diam:.1f}'
        if pupilstop.shiftXYinPixel.any() != 0.0:
            im_tag += f'_s{pupilstop.shiftXYinPixel[0]:.1f}x{pupilstop.shiftXYinPixel[1]:.1f}pix'
        if pupilstop.rotInDeg is not None and pupilstop.rotInDeg != 0.0:
            im_tag += f'_r{pupilstop.rotInDeg:.1f}deg'
        if pupilstop.magnification is not None and pupilstop.magnification != 1.0:
            im_tag += f'_m{pupilstop.magnification:.1f}'

        return im_tag

    def trigger_code(self):

        # Slopes *must* have been refreshed. We could have been triggered
        # just by the commands, but we need to skip it
        if self.local_inputs['in_slopes'].generation_time != self.current_time:
            return

        slopes = self.local_inputs['in_slopes'].slopes
        commands = self.local_inputs['in_commands'].value

        # First iteration initialization
        if self._im.value is None:
            self._im.value = self.xp.zeros((len(slopes), self._nmodes), dtype=self.dtype)
            for i in range(self._nmodes):
                self.output_im[i].resize(len(slopes))
            if self.verbose:
                print(f"Initialized interaction matrix: {self._im.value.shape}")

        idx = self.xp.nonzero(commands)[0]

        if len(idx)>0:
            mode = int(idx[0]) - self._first_mode
            if mode < self._nmodes:
                self._im.value[:, mode] += slopes / commands[idx]
                self.count_commands[mode] += 1

        in_slopes_object = self.local_inputs['in_slopes']

        for i in range(self._nmodes):
            self.output_im[i].slopes[:] = self._im.value[:, i].copy()
            self.output_im[i].single_mask = in_slopes_object.single_mask
            self.output_im[i].display_map = in_slopes_object.display_map
            self.output_im[i].generation_time = self.current_time

        self._im.generation_time = self.current_time

    def finalize(self):
        # normalize by counts
        for i in range(self._nmodes):
            if self.count_commands[i] > 0:
                self._im.value[:, i] /= self.count_commands[i]

        im = Intmat(self._im.value, pupdata_tag = self.pupdata_tag, subapdata_tag=self.subapdata_tag,
                    target_device_idx=self.target_device_idx, precision=self.precision)

        os.makedirs(self._data_dir, exist_ok=True)

        # TODO add to IM the information about the first mode
        im.save(self.im_path, overwrite=self._overwrite)
