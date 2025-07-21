import copy

from specula import cpuArray
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH


class DistributedSH(BaseProcessingObj):
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
                 aXShiftPhInPixel: float = 0,
                 aYShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 aRotAnglePhInDeg: float = 0,
                 do_not_double_fov_ovs: bool = False,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 target_device_idx: int = None,
                 precision: int = None,
        ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Complete dict of init arguments, without extra ones
        args = copy.copy(locals())
        del args['self']
        del args['__class__']

        # Calculate slices for each SH
        subaps_per_sh = subap_on_diameter // n_slices
        del args['n_slices']

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

        # Same inputs and outputs as a SH
        # Inputs will be copied to the sub-SH
        if laser_launch_tel is not None:
            self.inputs['sodium_altitude'] = InputValue(type=BaseValue, optional=True)
            self.inputs['sodium_intensity'] = InputValue(type=BaseValue, optional=True)

        sh0 = self.sub_sh[0]
        self.out_i = Intensity(sh0._ccd_side, sh0._ccd_side, precision=precision, target_device_idx=target_device_idx)
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self.out_i
        self.outputs['wf1'] = BaseValue(target_device_idx=self.target_device_idx)
        self.subap_npx = subap_npx

    def setup(self):
        super().setup()

        # Copy our inputs into all sub-SH
        for i, sh in enumerate(self.sub_sh):
            sh.name = f'subsh{i}'
            for k, v in self.inputs.items():
                if len(v.input_values) > 0:
                    sh.inputs[k].set(v.input_values[0].output_ref)
            sh.setup()

    def check_ready(self, t):
        super().check_ready(t)
        for sh in self.sub_sh:
            sh.check_ready(t)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        for sh in self.sub_sh:
            sh.prepare_trigger(t)
   
    def trigger(self):
        super().trigger()
        for sh in self.sub_sh:
            sh.trigger()

    def post_trigger(self):
        BaseProcessingObj.post_trigger(self)

        # Collect results from the other SHs into our Intensity result
        for s, sh in zip(self.slices, self.sub_sh):
            y1 = self.subap_npx * s.start
            y2 = self.subap_npx * s.stop
            self.out_i.i[y1:y2] = sh._out_i.i[y1:y2]

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self.out_i.i *= phot / self.out_i.i.sum()
        self.out_i.generation_time = self.current_time

#        import matplotlib.pyplot as plt
#        plt.imshow(cpuArray(self._out_i.i))
#        plt.show()
        # import time
        # time.sleep(0.1)

