import numpy as np
from specula import fuse
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj

@fuse(kernel_name='abs2_cwfs')
def abs2_cwfs(u_fp, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp))

class CurvatureSensor(BaseProcessingObj):
    def __init__(self,
                 wavelengthInNm: float,
                 focal_length: float,
                 defocus_distance: float, # l in the Yao code
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.wavelength_in_nm = wavelengthInNm
        self.focal_length = focal_length
        self.defocus_distance = defocus_distance

        # Output: Two intensities (Intra- and Extra-focal)
        self.inputs['in_ef'] = InputValue(type=ElectricField)

        # The outputs must be initialized in setup when we know the size
        self._out_i1 = None
        self._out_i2 = None
        self.fp_plus = None
        self.fp_minus = None
        self.defoc_term = None
        self.exp_plus = None
        self.exp_minus = None

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']
        size = in_ef.size[0]

        self._out_i1 = Intensity(size, size, precision=self.precision,
                                 target_device_idx=self.target_device_idx)
        self._out_i2 = Intensity(size, size, precision=self.precision,
                                 target_device_idx=self.target_device_idx)
        self.outputs['out_i1'] = self._out_i1
        self.outputs['out_i2'] = self._out_i2

        # Pre-allocate arrays for CUDA graphs (no new allocation in trigger)
        self.fp_plus = self.xp.zeros((size, size), dtype=self.complex_dtype)
        self.fp_minus = self.xp.zeros((size, size), dtype=self.complex_dtype)

        # Calculation of the defocus term (Equivalent to lines 718-719 of yao_wfs.i)
        # defoc = (pi * lambda / (size^2 * (D_tel/D_pup)^2)) * rho^2 * fratio_factor
        # Adapt this equation with the correct physical parameters for Specula
        xx, yy = self.xp.meshgrid(self.xp.arange(-size//2, size//2), 
                                  self.xp.arange(-size//2, size//2))
        r2 = xx**2 + yy**2

        # To be calculated based on focal length and defocus (l)
        defocus_phase_amp = ... # <--- Insert the physical constant here
        self.defoc_term = defocus_phase_amp * r2

        # Pre-calculate exponentials for maximum speed
        self.exp_plus = self.xp.exp(1j * self.defoc_term, dtype=self.complex_dtype)
        self.exp_minus = self.xp.exp(-1j * self.defoc_term, dtype=self.complex_dtype)

    def trigger_code(self):
        # 1. Retrieve input electric field
        ef = self.local_inputs['in_ef'].ef

        # 2. Intrafocal propagation (Phase multiplication + FFT)
        ef_plus = ef * self.exp_plus
        self.fp_plus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_plus))
        abs2_cwfs(self.fp_plus, self._out_i1.i, xp=self.xp)

        # 3. Extrafocal propagation
        ef_minus = ef * self.exp_minus
        self.fp_minus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_minus))
        abs2_cwfs(self.fp_minus, self._out_i2.i, xp=self.xp)

    def post_trigger(self):
        super().post_trigger()
        # Photometric normalization as in sh.py
        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i1.i *= phot / self._out_i1.i.sum()
        self._out_i2.i *= phot / self._out_i2.i.sum()

        self._out_i1.generation_time = self.current_time
        self._out_i2.generation_time = self.current_time
