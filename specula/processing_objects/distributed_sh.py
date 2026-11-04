import copy
import math

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH
import specula


class DistributedSH(SH):
    """
    SH class that distributes work on multiple devices.

    Internally, it manages a series of hidden SH objects each performing
    a section of the subaperture processing. In post_trigger(), all
    results are gathered into a single Intensity array.
    """
    def __init__(self,
                 wavelengthInNm: float,
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 n_slices: int,
                 squaremask: bool = True,
                 fov_ovs_coeff: float = 2.0,     # as SH
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 target_device_idx: int = None,
                 precision: int = None,
        ):
        # Complete dict of init arguments, without extra ones
        args = copy.copy(locals())
        del args['self']
        del args['__class__']

        del args['n_slices']

        # Initialize base class - we do not use the calculation routines,
        # but it is needed for inputs and outputs
        super().__init__(**args)

        # Subaperture rows of each SH, covering all rows even when
        # subap_on_diameter is not a multiple of n_slices
        bounds = [i * subap_on_diameter // n_slices for i in range(n_slices + 1)]
        self.slices = [slice(bounds[i], bounds[i+1]) for i in range(n_slices)]

        # Initialize internal SH with the other slices.
        # If using GPUs, each one targets a different device.
        self.sub_sh = []

        for i in range(n_slices):
            # The device of this object (target_device_idx may be None: the default one)
            if self.target_device_idx >= 0:  # pragma: gpu
                num_devices = specula.cp.cuda.runtime.getDeviceCount()
                args['target_device_idx'] = (self.target_device_idx + i) % num_devices
            args['subap_rows_slice'] = self.slices[i]
            self.sub_sh.append( SH(**args))
        # Building the slices made their devices current: back to the one of this object
        if self.target_device_idx >= 0:  # pragma: gpu
            self._target_device.use()

    @classmethod
    def devices_used(cls, params):
        return int(params.get('n_slices', 1))

    @classmethod
    def single_device_params(cls, params):
        return {**params, 'n_slices': 1}

    @classmethod
    def repeated_work_fraction(cls, params):
        # Each slice interpolates the whole input field before computing its rows,
        # a larger fraction of the work for sensors with few rows. Measured on
        # Leonardo (A100), 2 and 4 slices, 480x480 input field: f = 0.22 for the
        # MORFEO NGS and reference sensors (2 and 8 rows), 0.05 for the LGS ones
        # (68 rows, with the laser launch telescope). Interpolated in log(rows).
        rows = params.get('subap_on_diameter', 0)
        if not isinstance(rows, int) or rows <= 8:
            return 0.22
        if rows >= 68:
            return 0.05
        return 0.22 + (0.05 - 0.22) * math.log(rows / 8) / math.log(68 / 8)

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def setup(self):
        '''
        Skip the SH method for this object, since we do not perform
        any calculation, but call the BaseProcessingObj one for housekeeping.
        Then set inputs on all sub-SHs
        '''
        BaseProcessingObj.setup(self)

        # Copy our inputs into all sub-SH
        for i, sh in enumerate(self.sub_sh):
            sh.name = f'subsh{i}'
            for k, v in self.inputs.items():
                if len(v.input_values) > 0:
                    sh.inputs[k].set(v.input_values[0].cloned_value)
            sh.setup()

    def check_ready(self, t):
        '''
        Skip the SH method for this object, since we do not perform
        any calculation, but call the BaseProcessingObj one for housekeeping.
        Then call all sub-SHs
        '''
        BaseProcessingObj.check_ready(self, t)
        for sh in self.sub_sh:
            sh.check_ready(t)

    def prepare_trigger(self, t):
        '''
        Skip the SH method for this object, since we do not perform
        any calculation, but call the BaseProcessingObj one for housekeeping.
        The sub-SHs prepare in their own check_ready() (see check_ready()),
        after getting their inputs: preparing them here, before, would use the
        inputs of the previous step, and SH initializes the extrapolation of the
        input field from the pupil at its first prepare_trigger() only.
        '''
        BaseProcessingObj.prepare_trigger(self, t)
   
    def trigger(self):
        '''
        Skip the SH method for this object, since we do not perform
        any calculation, but call the BaseProcessingObj one for housekeeping.
        Then call all sub-SHs
        '''
        BaseProcessingObj.trigger(self)
        for sh in self.sub_sh:
            sh.trigger()

    def trigger_code(self):
        '''
        Nothing to do in the distributed SH. The Sub-SH will run
        the regular SH implementation.
        '''
        return

    def post_trigger(self):
        '''
        Skip the SH method for this object, but call the 
        BaseProcessingObj one for housekeeping.
        Then gather results from the sub-SH and perform
        the final normalization
        '''
        BaseProcessingObj.post_trigger(self)

        # Collect results from the other SHs into our Intensity result
        for s, sh in zip(self.slices, self.sub_sh):
            y1 = self._subap_npx * s.start
            y2 = self._subap_npx * s.stop
            self._out_i.i[y1:y2] = sh._out_i.i[y1:y2]

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i.i *= phot / self._out_i.i.sum()
        self._out_i.generation_time = self.current_time


