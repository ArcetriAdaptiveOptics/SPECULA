import numpy as np
from specula import fuse
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.zernike_generator import ZernikeGenerator

@fuse(kernel_name='abs2_cwfs')
def abs2_cwfs(u_fp, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp))

class CurvatureSensor(BaseProcessingObj):
    """
    Curvature Wavefront Sensor (CWFS) propagator.
    
    This class applies a Zernike Focus aberration (defocus) to the input electric field
    and propagates it to generate intra-focal and extra-focal intensity images.
    """
    def __init__(self,
                 wavelengthInNm: float,
                 defocus_rms_nm: float,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.wavelength_in_nm = wavelengthInNm
        self.defocus_rms_nm = defocus_rms_nm

        # Output: Two intensities (Intra- and Extra-focal)
        self.inputs['in_ef'] = InputValue(type=ElectricField)

        # The outputs must be initialized in setup when we know the size
        self._out_i1 = None
        self._out_i2 = None
        self.fp_plus = None
        self.fp_minus = None
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

        # Pre-allocate arrays for CUDA graphs
        self.fp_plus = self.xp.zeros((size, size), dtype=self.complex_dtype)
        self.fp_minus = self.xp.zeros((size, size), dtype=self.complex_dtype)

        # 1. Generate Zernike Focus (Z4 Noll) using ZernikeGenerator
        zgen = ZernikeGenerator(size, self.xp, self.dtype)
        z4 = zgen.getZernike(4) # Index 4 is Focus (Noll)

        # 2. Convert RMS Nanometers to Phase Radians
        k = 2.0 * np.pi / self.wavelength_in_nm
        phase_aberration = z4 * self.defocus_rms_nm * k

        # 3. Pre-calculate exponentials for maximum speed
        self.exp_plus = self.xp.exp(1j * phase_aberration, dtype=self.complex_dtype)
        self.exp_minus = self.xp.exp(-1j * phase_aberration, dtype=self.complex_dtype)


    def trigger_code(self):
        # 1. Retrieve input electric field data components
        in_ef_data = self.local_inputs['in_ef']

        # 2. Construct Complex Electric Field from Amplitude and Phase
        # E = A * exp(i * phase_rad)
        k = 2.0 * self.xp.pi / self.wavelength_in_nm

        # Convert phase from nm to radians
        # Note: We use in_ef_data.phaseInNm directly.
        # Specula ElectricField objects store phase in nm.
        phase_rad = in_ef_data.phaseInNm * k

        # Combine Amplitude (A) and Phase into complex field
        # Ensure we cast to complex_dtype to match the pre-calculated exponentials
        ef = in_ef_data.A * self.xp.exp(1j * phase_rad, dtype=self.complex_dtype)

        # 3. Intrafocal propagation (Phase multiplication + FFT)
        ef_plus = ef * self.exp_plus
        self.fp_plus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_plus))
        abs2_cwfs(self.fp_plus, self._out_i1.i, xp=self.xp)

        # 4. Extrafocal propagation
        ef_minus = ef * self.exp_minus
        self.fp_minus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_minus))
        abs2_cwfs(self.fp_minus, self._out_i2.i, xp=self.xp)


    def post_trigger(self):
        super().post_trigger()
        # Photometric normalization
        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()

        # Avoid division by zero if image is empty
        sum1 = self._out_i1.i.sum()
        sum2 = self._out_i2.i.sum()

        if sum1 > 0:
            self._out_i1.i *= phot / sum1
        if sum2 > 0:
            self._out_i2.i *= phot / sum2

        self._out_i1.generation_time = self.current_time
        self._out_i2.generation_time = self.current_time
