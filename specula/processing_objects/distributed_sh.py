import copy

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH


class DistributedSH(SH):
    
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
        args = copy.copy(locals())  # Complete dict of init arguments
        subaps_per_sh = subap_on_diameter // n_slices

        del args['n_slices']
        del args['self']
        del args['__class__']

        self.slices = []
        for i in range(n_slices):
            self.slices.append(slice( i * subaps_per_sh, (i+1) * subaps_per_sh))

        args['subap_rows_slice'] = self.slices[0]
        super().__init__(**args)

        self.sub_sh = []
        for i in range(1, n_slices):
            if target_device_idx >= 0:
                args['target_device_idx'] = target_device_idx + i
            args['subap_rows_slice'] = self.slices[i]
            self.sub_sh.append( SH(**args))

    def setup(self):
        super().setup()
        for i, sh in enumerate(self.sub_sh):
            sh.name = f'subsh{i}'
            for k, v in self.inputs.items():
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

        for s, sh in zip(self.slices[1:], self.sub_sh):
            y1 = self._subap_npx * s.start
            y2 = self._subap_npx * s.stop
            self._out_i.i[y1:y2] = sh._out_i.i[y1:y2]

        phot = self.in_ef.S0 * self.in_ef.masked_area()
        self._out_i.i *= phot / self._out_i.i.sum()

        import matplotlib.pyplot as plt
        plt.imshow(cpuArray(self.outputs['out_i'].i))
        plt.show()


