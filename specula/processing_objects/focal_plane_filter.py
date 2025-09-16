from specula import cpuArray, RAD2ASEC
from specula.lib.make_mask import make_mask
from specula.lib.extrapolation_2d import calculate_extrapolation_indices_coeffs, apply_extrapolation
from specula.lib.interp2d import Interp2D

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams

class FocalPlaneFilter(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 fov_errinf: float = 0.1,
                 fov_errsup: float = 10,
                 fft_res: float = 3.0,
                 fp_obs: float = 0.0,
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

        result = self.calc_geometry(self.pixel_pupil,
                                    self.pixel_pitch,
                                    wavelengthInNm,
                                    self.fov,
                                    fov_errinf=fov_errinf,
                                    fov_errsup=fov_errsup,
                                    fft_res=fft_res)

        self.wavelength_in_nm = result['wavelengthInNm']
        self.fp_masking = result['fp_masking']
        self.fov_res = result['fov_res']
        self.fft_res = result['fft_res']
        self.fft_sampling = result['fft_sampling']
        self.fft_padding = result['fft_padding']
        self.fft_totsize = result['fft_totsize']

        # Focal plane mask
        fp_obsratio = fp_obs * self.fp_masking if fp_obs else 0
        self.fp_mask = make_mask(self.fft_totsize, diaratio=self.fp_masking, obsratio=fp_obsratio, xp=self.xp)

        self.out_ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                                    precision=self.precision, target_device_idx=self.target_device_idx)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_ef'] = self.out_ef

        self.ef = self.xp.zeros((self.fft_sampling, self.fft_sampling), dtype=self.complex_dtype)
        self.ef_pad = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)
        self.ef_out = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)


    def calc_geometry(self, DpupPix, pixel_pitch, lambda_, FoV,
                      fov_errinf=0.1, fov_errsup=0.5,  fft_res=3.0):
        D = DpupPix * pixel_pitch
        Fov_internal = lambda_ * 1e-9 / D * (D / pixel_pitch) * RAD2ASEC
        minfov = FoV * (1 - fov_errinf)
        maxfov = FoV * (1 + fov_errsup)
        fov_res = 1.0
        if Fov_internal < minfov:
            fov_res = int(minfov / Fov_internal)
            if Fov_internal * fov_res < minfov:
                fov_res += 1
        if Fov_internal > maxfov:
            raise ValueError(f"FoV too large compared to the diffraction limit "
                            f"(FoV: {FoV}, Fov_internal: {Fov_internal}, "
                            f"fov_errsup: {fov_errsup}) and fov_errinf: {fov_errinf})")
        if fov_res > 1:
            Fov_internal *= fov_res
        fp_masking = FoV / Fov_internal
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
            'wavelengthInNm': lambda_
        }


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
                self._valid_indices) = calculate_extrapolation_indices_coeffs(cpuArray(in_ef.A))

                # convert to xp
                self._edge_pixels = self.to_xp(self._edge_pixels)
                self._reference_indices = self.to_xp(self._reference_indices)
                self._coefficients = self.to_xp(self._coefficients)
                self._valid_indices = self.to_xp(self._valid_indices)

            self.phase_extrapolated[:] = in_ef.phaseInNm
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
            self.interp.interpolate(self.phase_extrapolated, out=self._wf_interpolated.phaseInNm)

            # Copy other properties
            self._wf_interpolated.S0 = in_ef.S0
            self._wf_interpolated.pixel_pitch = in_ef.pixel_pitch

        # Always use self._wf_interpolated for calculations (like SH uses self._wf1)
        self._wf_interpolated.ef_at_lambda(self.wavelength_in_nm, out=self.ef)


    def trigger_code(self):

        # padding
        #   Reset padding array
        self.ef_pad.fill(0)
        #   Center the field in the padded array (like IDL does)
        pad_start = self.fft_padding // 2
        self.ef_pad[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling] = self.ef

        # FFT -> mask -> IFFT (electric field)
        u_fp = self.xp.fft.fft2(self.ef_pad)

        #   Apply centered focal plane mask (shift mask to center it)
        fp_mask_centered = self.xp.fft.fftshift(self.fp_mask)
        u_fp_masked = u_fp * fp_mask_centered

        self.ef_out[:] = self.xp.fft.ifft2(u_fp_masked)


    def post_trigger(self):
        super().post_trigger()

        # Extract the central part (original sampling size)
        pad_start = self.fft_padding // 2
        if self.fft_padding > 0:
            ef_out_cropped = self.ef_out[pad_start:pad_start+self.fft_sampling,
                                pad_start:pad_start+self.fft_sampling]
        else:
            ef_out_cropped = self.ef_out

        # Then rebin if needed
        if self._do_interpolation and self.fov_res > 1:
            # Rebin back to original sampling
            fov_res_int = int(self.fov_res)
            h, w = ef_out_cropped.shape
            new_h, new_w = h // fov_res_int, w // fov_res_int
            ef_out_cropped = ef_out_cropped[:new_h*fov_res_int, :new_w*fov_res_int].reshape(
                new_h, fov_res_int, new_w, fov_res_int).mean(axis=(1, 3))

        # Calculate transmission
        # PSF before masking vs PSF after masking
        psf_before = self.xp.abs(self.xp.fft.fft2(self.ef_pad))**2
        psf_after = self.xp.abs(self.xp.fft.fft2(self.ef_pad) * self.xp.fft.fftshift(self.fp_mask))**2
        transmission = self.xp.sum(psf_after) / self.xp.sum(psf_before)

        # Amplitude
        self.out_ef.A[:] = self.xp.abs(ef_out_cropped)
        # Phase in nm
        self.out_ef.phaseInNm[:] = (self.xp.angle(ef_out_cropped) / (2 * self.xp.pi)) * self.wavelength_in_nm

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
                0, #-self.rotAnglePhInDeg,  # Negative angle for PASSATA compatibility
                0, #self.xShiftPhInPixel,
                0, #self.yShiftPhInPixel,
                dtype=self.dtype,
                xp=self.xp
            )
        else:
            self._do_interpolation = False
            # Use the original field directly (like SH does)
            self._wf_interpolated = in_ef

        super().build_stream()