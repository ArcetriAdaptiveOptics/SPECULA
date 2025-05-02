import numpy as np

from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel
from specula.data_objects.convolution_kernel import ConvolutionKernel

import os

class ShKernel(BaseProcessingObj):

    def __init__(self,
                 subap_on_diameter: int = 0,
                 spot_size: float = 0.0,
                 laser_launcher_pos: list = [],
                 laser_beacon_focus: float = 90e3,
                 laser_beacon_tt: list = [],
                 target_device_idx: int = None, 
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision) 

        self._dim = subap_on_diameter
        self._spot_size = spot_size
        self._laser_launcher_pos = laser_launcher_pos
        self._laser_beacon_focus = laser_beacon_focus
        self._laser_beacon_tt = laser_beacon_tt
    
        if len(self._laser_launcher_pos) == 0:                        
            self._kernelobj = GaussianConvolutionKernel(self._spot_size,
                                                        self._dim, self._dim,
                                                        target_device_idx=self.target_device_idx)
        else:
            self._kernelobj = ConvolutionKernel(self._dim, self._dim,
                                                target_device_idx=self.target_device_idx)
            self._kernelobj.launcher_pos = self._laser_launcher_pos
            self._kernelobj.seeing = 0.0
            self._kernelobj.launcher_size = self._spot_size
            self._kernelobj.zfocus = self._laser_beacon_focus
            if len(self._laser_beacon_tt) != 0:
                self._kernelobj.lgs_tt = self._laser_beacon_tt

        self._kernelobj.oversampling = 1
        self._kernelobj.return_fft = True
        self._kernelobj.positive_shift_tt = True

        self._kernel_fn = None

        self.inputs['sodium_altitude'] = InputValue(type=BaseValue, optional=True)
        self.inputs['sodium_intensity'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_kernels'] = self._kernelobj

    def trigger(self):
        if len(self._laser_launcher_pos) != 0:
            sodium_altitude = self.local_inputs['sodium_altitude']
            sodium_intensity = self.local_inputs['sodium_intensity']
            if sodium_altitude is None or sodium_intensity is None:
                raise ValueError('sodium_altitude and sodium_intensity must be provided')
            self._kernelobj.zlayer = sodium_altitude.value
            self._kernelobj.zprofile = sodium_intensity.value

        # Get the kernel filename hash based on current parameters
        new_kernel_fn = self._kernelobj.build()

        # Only reload or recalculate if the kernel has changed
        if new_kernel_fn != self._kernel_fn:
            self._kernel_fn = new_kernel_fn  # Update the stored kernel filename

            if os.path.exists(self._kernel_fn):
                print(f"Loading kernel from {self._kernel_fn}")
                if len(self._laser_launcher_pos) == 0:
                    self._kernelobj = GaussianConvolutionKernel.restore(self._kernel_fn,
                                                                        kernel_obj=self._kernelobj,
                                                                        target_device_idx=self.target_device_idx,
                                                                        return_fft=True)
                else:
                    self._kernelobj = ConvolutionKernel.restore(self._kernel_fn, 
                                                                kernel_obj=self._kernelobj,
                                                                target_device_idx=self.target_device_idx,
                                                                return_fft=True)
            else:
                print('Calculating kernel...')
                self._kernelobj.calculate_lgs_map()
                self._kernelobj.save(self._kernel_fn)
                print('Done')
        else:
            # Kernel hasn't changed, no need to reload or recalculate
            print("Kernel unchanged, using cached version")

        self._kernelobj.generation_time = self.current_time