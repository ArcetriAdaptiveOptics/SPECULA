from specula import cpuArray, RAD2ASEC
from specula.lib.extrapolation_2d import calculate_extrapolation_indices_coeffs, apply_extrapolation
from specula.lib.interp2d import Interp2D

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams

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
        self.interp = None
        self._do_interpolation = False
        self._wf_interpolated = None
        self._edge_pixels = None
        self._reference_indices = None
        self._coefficients = None
        self._valid_indices = None
        self._amplitude_is_binary = None
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        result = self.calc_geometry(self.pixel_pupil,
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


    def calc_geometry(self,
        DpupPix,
        pixel_pitch,
        lambda_,
        FoV,
        fov_errinf=0.1,
        fov_errsup=0.5,
        fft_res=3.0):

        fov_internal = lambda_ * 1e-9 / pixel_pitch * RAD2ASEC

        maxfov = FoV * (1 + fov_errsup)
        if fov_internal > maxfov:
            raise ValueError("Error: Calculated FoV is higher than maximum accepted FoV."
                  f" FoV calculated (arcsec): {fov_internal:.2f},"
                  f" maximum accepted FoV (arcsec): {maxfov:.2f}."
                  f"\nPlease revise error margin, or the input phase dimension and/or pitch")

        minfov = FoV * (1 - fov_errinf)
        if fov_internal < minfov:
            fov_res = int(self.xp.ceil(minfov / fov_internal))
            fov_internal_interpolated = fov_internal * fov_res
            print(f"Interpolated FoV (arcsec): {fov_internal_interpolated:.2f}")
            print(f"Warning: reaching the requested FoV requires {fov_res}x interpolation"
                  f" of input phase array.")
            print("Consider revising the input phase dimension and/or pitch to improve"
                  " performance.")
        else:
            fov_res = 1
            fov_internal_interpolated = fov_internal

        fp_masking = FoV / fov_internal_interpolated

        if fp_masking > 1.0:
            if minfov / fov_internal_interpolated > 1.0:
                raise ValueError(f"fp_masking ratio cannot be larger than 1.0.")
            else:
                fp_masking = 1.0

        if fov_internal_interpolated != FoV:
            print(f"FoV reduction from {fov_internal_interpolated:.2f} to {FoV:.2f}"
                  f" will be performed with a focal plane mask")

        DpupPixFov = DpupPix * fov_res
        totsize = self.xp.around(DpupPixFov * fft_res / 2) * 2
        fft_res = totsize / float(DpupPixFov)
        padding = self.xp.around((DpupPixFov * fft_res - DpupPixFov) / 2) * 2

        return {
            'fov_res': fov_res,
            'fp_masking': fp_masking,
            'fft_res': fft_res,
            'fft_sampling': int(DpupPixFov),
            'fft_padding': int(padding),
            'fft_totsize': int(totsize),
            # 'wavelengthInNm': lambda_
        }

    def _pupil_to_focal_plane(self, pup_ef):
        ef_pad = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)
        pad_start = self.fft_padding // 2
        ef_pad[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling] = pup_ef
        return self.xp.fft.fft2(ef_pad)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Update input reference
        in_ef = self.local_inputs['in_ef']

        # Apply interpolation if needed (like SH)
        if self._do_interpolation:

            if self._edge_pixels is None:
                # Compute once indices and coefficients
                (self._edge_pixels,
                self._reference_indices,
                self._coefficients,
                self._valid_indices) = calculate_extrapolation_indices_coeffs(
                    cpuArray(in_ef.A), threshold=self._mask_threshold
                )

                # convert to xp
                self._edge_pixels = self.to_xp(self._edge_pixels)
                self._reference_indices = self.to_xp(self._reference_indices)
                self._coefficients = self.to_xp(self._coefficients)
                self._valid_indices = self.to_xp(self._valid_indices)
                
                # Check if input amplitude is binary (all values close to 0 or 1) with tolerance
                unique_values = self.xp.unique(in_ef.A)
                tol = 1e-3
                is_binary = self.xp.all(
                    self.xp.logical_or(
                        self.xp.abs(unique_values - 0) < tol,
                        self.xp.abs(unique_values - 1) < tol
                    )
                )

                self._amplitude_is_binary = is_binary

            self.phase_extrapolated = in_ef.phaseInNm * \
                (in_ef.A >= self._mask_threshold).astype(int)
            _ = apply_extrapolation(
                in_ef.phaseInNm,
                self._edge_pixels,
                self._reference_indices,
                self._coefficients,
                self._valid_indices,
                out=self.phase_extrapolated,
                xp=self.xp
            )

            # Interpolate amplitude and phase separately
            self.interp.interpolate(in_ef.A, out=self._wf_interpolated.A)
            
            # Apply binary threshold if input amplitude was binary
            if self._amplitude_is_binary:
                self._wf_interpolated.A[:] = (self._wf_interpolated.A > 0.5).astype(self.dtype)

            self.interp.interpolate(self.phase_extrapolated, out=self._wf_interpolated.phaseInNm)

            # Copy other properties
            self._wf_interpolated.S0 = in_ef.S0
            self._wf_interpolated.pixel_pitch = in_ef.pixel_pitch

        # Always use self._wf_interpolated for calculations (like SH uses self._wf1)
        self._wf_interpolated.ef_at_lambda(self.wavelength_in_nm, out=self.ef_in)


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
        if self._do_interpolation and self.fov_res > 1:
            # Rebin back to original sampling
            fov_res_int = int(self.fov_res)
            h, w = self.ef_out.shape
            new_h, new_w = h // fov_res_int, w // fov_res_int
            ef_out = self.ef_out[:new_h*fov_res_int, :new_w*fov_res_int].reshape(
                new_h, fov_res_int, new_w, fov_res_int).mean(axis=(1, 3))
        else:
            ef_out = self.ef_out

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

        # Determine if interpolation is needed (like in SH)
        if self.fov_res != 1:

            self._do_interpolation = True

            # Create the interpolated field (like SH does with self._wf1)
            self._wf_interpolated = ElectricField(
                self.fft_sampling,
                self.fft_sampling,
                in_ef.pixel_pitch,
                target_device_idx=self.target_device_idx,
                precision=self.precision
            )

            # Create the interpolator (like in SH)
            self.interp = Interp2D(
                in_ef.size,
                (self.fft_sampling, self.fft_sampling),
                dtype=self.dtype,
                xp=self.xp
            )
        else:
            self._do_interpolation = False
            # Use the original field directly (like SH does)
            self._wf_interpolated = in_ef

        super().build_stream()