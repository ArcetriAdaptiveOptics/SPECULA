import copy

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH


class DistributedSH(SH):
    '''
    SH class that distributes work on multiple devices.

    Internally, it manages a series of hidden SH objects each performing
    a section of the subaperture processing. In post_trigger(), all
    results are gathered into a single Intensity array.
    '''
    def __init__(self,
                 wavelengthInNm: float,
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 n_slices: int,
                 FoVres30mas: bool = False,
                 squaremask: bool = True,
                 fov_ovs_coeff: float = 0,
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 do_not_double_fov_ovs: bool = False,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 target_device_idx: int = None,
                 precision: int = None,
        ):
        # Complete dict of init arguments, without extra ones
        args = copy.copy(locals())
        del args['self']
        del args['__class__']

        # Calculate slices for each SH
        subaps_per_sh = subap_on_diameter // n_slices
        del args['n_slices']

        # Initialize base class - we do not use the calculation routines,
        # but it is needed for inputs and outputs
        super().__init__(**args)

        self.slices = []
        for i in range(n_slices):
            self.slices.append(slice( i * subaps_per_sh, (i+1) * subaps_per_sh))

        # Initialize internal SH with the other slices.
        # If using GPUs, each one targets a different device.
        self.sub_sh = []
        for i in range(n_slices):
            if target_device_idx >= 0:
                args['target_device_idx'] = target_device_idx + i
            args['subap_rows_slice'] = self.slices[i]
            self.sub_sh.append( SH(**args))

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
        Then call all sub-SHs
        '''
        BaseProcessingObj.prepare_trigger(self, t)
        for sh in self.sub_sh:
            sh.prepare_trigger(t)
   
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


