import os
import numpy as np

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
                 dm: DM = None,
                 source: Source = None,
                 sensor: BaseProcessingObj = None,
                 slopec: BaseProcessingObj = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.nmodes = nmodes
        self.first_mode = first_mode
        self.data_dir = data_dir

        if im_tag is None or im_tag == 'auto':
            im_tag = ImCalibrator.generate_im_tag(pupilstop, source, dm, sensor,
                                                  slopec, self.nmodes, self.first_mode)
        self.im_tag = im_tag

        self.overwrite = overwrite

        self.im_path = os.path.join(self.data_dir, self.im_tag)
        if not self.im_path.endswith('.fits'):
            self.im_path += '.fits'
        if os.path.exists(self.im_path) and not self.overwrite:
            raise FileExistsError(f'IM file {self.im_path} already exists, please remove it')

        # Add counts tracking, this is used to normalize the IM
        self.count_commands = np.zeros(nmodes, dtype=int)

        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_commands'] = InputValue(type=BaseValue)

        self.intmat = Intmat(nmodes=nmodes, nslopes=0, target_device_idx=self.target_device_idx)
        self.outputs['out_intmat'] = self.intmat

        self.single_im = [Intmat(nmodes=1, nslopes=0,
                                 target_device_idx=self.target_device_idx) for i in range(nmodes)]
        self.outputs['out_single_im'] = self.single_im

    @staticmethod
    def generate_im_tag(pupilstop, source, dm, sensor, slopec, nmodes, first_mode=0):
        """Generate automatic im_tag based on configuration parameters."""

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
                im_tag += '_qc'
            if slopec.subapdata.tag is not None and slopec.subapdata.tag != '':
                im_tag += f'_{slopec.subapdata.tag}'
        if isinstance(slopec, PyrSlopec):
            if slopec.slopes_from_intensity:
                im_tag += '_slint'
            if slopec.pupdata.tag is not None and slopec.pupdata.tag != '':
                im_tag += f'_{slopec.pupdata.tag}'

        # no. pixel and pixel pitch
        im_tag += f'_pup{dm.simul_params.pixel_pupil}x{dm.simul_params.pixel_pupil}p'
        im_tag += f'{dm.simul_params.pixel_pitch}m'

        # SOURCE coordinates
        if source.polar_coordinates[0] != 0:
            im_tag += f'_coor{source.polar_coordinates[0]:.1f}a{source.polar_coordinates[1]:.1f}d'
        if source.height != float('inf'):
            im_tag += f'_h{source.height:.1f}m'

        # DM related keys
        im_tag += '_dm'
        if dm.mask.shape[0] != dm.simul_params.pixel_pupil:
            im_tag += f'{dm.mask.shape[0]}x{dm.mask.shape[1]}p'
        if dm.type_str is not None:
            im_tag += '_'+dm.type_str
        elif dm.tag is not None and dm.tag != '':
            im_tag += '_'+dm.tag
        nmodes_dm = dm.ifunc.shape[0]
        im_tag += f'_{min(nmodes_dm,nmodes)}mds'
        if first_mode != 0:
            im_tag += f'_firstm{first_mode}'

        # Pupilstop
        im_tag += '_stop'
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
        if self.intmat.nslopes == 0:
            self.intmat.set_nslopes(len(slopes))
            if self.verbose:
                print(f"Initialized interaction matrix: {self.im.value.shape}")
            for i in range(self.nmodes):
                self.single_im[i].set_nslopes(len(slopes))

        idx = self.xp.nonzero(commands)[0]

        if len(idx)>0:
            mode = int(idx[0]) - self.first_mode
            if mode < self.nmodes:
                self.intmat.modes[mode] += slopes / commands[idx]
                self.count_commands[mode] += 1

        in_slopes_object = self.local_inputs['in_slopes']

        for mode in range(self.nmodes):
            self.single_im[mode].modes[0] = self.intmat.modes[mode].copy()
            self.single_im[mode].single_mask = in_slopes_object.single_mask
            self.single_im[mode].display_map = in_slopes_object.display_map
            self.single_im[mode].generation_time = self.current_time

        self.intmat.single_mask = in_slopes_object.single_mask
        self.intmat.display_map = in_slopes_object.display_map
        self.intmat.generation_time = self.current_time

    def finalize(self):
        # normalize by counts
        for mode in range(self.nmodes):
            if self.count_commands[mode] > 0:
                self.intmat.modes[mode] /= self.count_commands[mode]

        os.makedirs(self.data_dir, exist_ok=True)

        # TODO add to IM the information about the first mode
        self.intmat.save(self.im_path, overwrite=self.overwrite)
