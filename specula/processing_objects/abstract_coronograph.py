from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.toccd import toccd

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_geometry import calc_geometry

from abc import abstractmethod

class Coronograph(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 fov_errinf: float = 0.1,
                 fov_errsup: float = 10,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.fov = fov

        # interpolation settings
        self.mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        result = calc_geometry(self.pixel_pupil,
                               self.pixel_pitch,
                               wavelengthInNm,
                               self.fov,
                               fov_errinf=fov_errinf,
                               fov_errsup=fov_errsup,
                               fft_res=fft_res)

        self.wavelength_in_nm = wavelengthInNm
        self.fov_res = result['fov_res']
        self.fft_res = result['fft_res']
        self.fft_sampling = result['fft_sampling']
        self.fft_padding = result['fft_padding']
        self.fft_totsize = result['fft_totsize']

        # Apodizer, focal plane mask, pupil stop
        self.apodizer = self.make_apodizer()
        self.fp_mask = self.make_focal_plane_mask()
        self.pupil_mask = self.make_pupil_plane_mask()

        self.out_ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                                    precision=self.precision, target_device_idx=self.target_device_idx)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_ef'] = self.out_ef

        self.ef_in = self.xp.zeros((self.fft_sampling, self.fft_sampling), dtype=self.complex_dtype)
        self.ef_out = self.xp.zeros((self.fft_sampling, self.fft_sampling), dtype=self.complex_dtype)

    
    def make_apodizer(self):
        """ Override this method to add an apodizer.
        By default, no apodizer mask is considered """
        return 1.0
    
    @abstractmethod
    def make_focal_plane_mask(self):
        """ Override this method to create the 
        desired focal plane (complex) mask """

    @abstractmethod
    def make_pupil_plane_mask(self):
        """ Override this method to create the 
        desired pupil plane (complex) mask """

    def _pupil_to_focal_plane(self, pup_ef):
        ef_pad = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)
        pad_start = self.fft_padding // 2
        ef_pad[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling] = pup_ef
        return self.xp.fft.fft2(ef_pad)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        self.ef_interpolator.interpolate()
        self.ef_interpolator.interpolated_ef().ef_at_lambda(self.wavelength_in_nm, out=self.ef_in)

    def trigger_code(self):

        # Step 1: Apodize electric field
        apodized_ef = self.ef_in * self.apodizer

        # Step 2: Propagate field to focal plane with FFT
        ef_fp = self._pupil_to_focal_plane(apodized_ef)

        # Step 3: Apply focal plane mask (appropriately shifted)
        fp_mask_centered = self.xp.fft.fftshift(self.fp_mask)
        ef_fp_masked = ef_fp * fp_mask_centered

        # Step 4: Return to the pupil plane with IFFT
        self.ef_pad = self.xp.fft.ifft2(ef_fp_masked)
        pad_start = self.fft_padding // 2
        ef_pp = self.ef_pad[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling]

        # Step 5: Apply pupil stop
        self.ef_out[:] = ef_pp * self.pupil_mask


    def post_trigger(self):
        super().post_trigger()

        # Then rebin if needed
        ef_out = toccd(self.ef_out, self.out_ef.size, xp=self.xp)

        # Calculate transmission
        # PSF before masking vs PSF after masking
        psf_before = self.xp.abs(self._pupil_to_focal_plane(self.ef_in))**2
        psf_after = self.xp.abs(self._pupil_to_focal_plane(self.ef_out))**2
        transmission = self.xp.sum(psf_after) / self.xp.sum(psf_before)

        # Amplitude
        self.out_ef.A[:] = self.xp.abs(ef_out)
        # Phase in nm
        self.out_ef.phaseInNm[:] = (self.xp.angle(ef_out) / (2 * self.xp.pi)) * self.wavelength_in_nm

        # Scale S0 by transmission
        in_ef = self.local_inputs['in_ef']
        self.out_ef.S0 = in_ef.S0 * transmission
        self.out_ef.generation_time = self.current_time


    def setup(self):
        super().setup()

        # Get input electric field
        in_ef = self.local_inputs['in_ef']

        self.ef_interpolator = EFInterpolator(
            in_ef,
            (self.fft_sampling, self.fft_sampling),
            rotAnglePhInDeg=0,
            xShiftPhInPixel=0,
            yShiftPhInPixel=0,
            mask_threshold=self.mask_threshold,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        # Cannot be used if self._pupil_to_focal_plane is called in trigger_code()
        # super().build_stream()