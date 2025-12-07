from abc import abstractmethod
from specula import cpuArray#, RAD2ASEC
from specula.lib.extrapolation_2d import calculate_extrapolation_indices_coeffs, apply_extrapolation
from specula.lib.interp2d import Interp2D

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams

class Coronograph(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

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

        self.wavelength_in_nm = wavelengthInNm
        self.fft_sampling = self.pixel_pupil
        self.fft_padding = int((fft_res-1)/2*self.fft_sampling)*2
        self.fft_totsize = int(fft_res/2*self.fft_sampling)*2
        self.fft_res = int(fft_res)
        self.fov_res = 1.0

        self.apodizer = self.make_apodizer() # Apodizer (pupil plane complex mask)
        self.fp_mask = self.make_focal_plane_mask() # Focal plane (complex) mask
        self.pp_mask = self.make_pupil_plane_mask() # Pupil plane stop

        self.out_ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                                    precision=self.precision, target_device_idx=self.target_device_idx)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_ef'] = self.out_ef

        self.ef_in = self.xp.zeros((self.fft_sampling, self.fft_sampling), dtype=self.complex_dtype)
        self.ef_out = self.xp.zeros((self.fft_sampling, self.fft_sampling), dtype=self.complex_dtype)


    @abstractmethod
    def make_focal_plane_mask(self):
        """ Override this method to create the 
        desired focal plane (complex) mask """

    @abstractmethod
    def make_pupil_plane_mask(self):
        """ Override this method to create the 
        desired pupil plane stop """

    def make_apodizer(self):
        """ Override this method to add an apodizer.
        By default, no apodizer mask is considered """
        return 1.0
    
    def propagate_to_focal_plane(self, pup_ef):
        """ Compute focal plane electric field 
        using FFT and appropriate padding """
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
        ef_pad = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)
        pad_start = self.fft_padding // 2
        ef_pad[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling] = self.ef_in
        apodized_ef = ef_pad * self.apodizer

        # Step 2: Propagate field to focal plane with FFT
        ef_fp = self.propagate_to_focal_plane(apodized_ef)

        # Step 3: Apply focal plane mask (appropriately shifted)
        fp_mask_centered = self.xp.fft.fftshift(self.fp_mask)
        ef_fp_masked = ef_fp * fp_mask_centered

        # Step 4: Return to the pupil plane with IFFT
        ef_pp_pad = self.xp.fft.ifft2(ef_fp_masked)
        pad_start = self.fft_padding // 2
        ef_pp = ef_pp_pad[pad_start:pad_start+self.fft_sampling,
                                pad_start:pad_start+self.fft_sampling]

        # Step 5: Apply pupil stop
        ef_out = ef_pp * self.pp_mask
        if self._do_interpolation and self.fov_res > 1:
            # Rebin back to original sampling
            fov_res_int = int(self.fov_res)
            h, w = ef_out.shape
            new_h, new_w = h // fov_res_int, w // fov_res_int
            ef_out = ef_out[:new_h*fov_res_int, :new_w*fov_res_int].reshape(
                new_h, fov_res_int, new_w, fov_res_int).mean(axis=(1, 3))
        self.ef_out[:] = ef_out


    def post_trigger(self):
        super().post_trigger()

        # Calculate transmission
        # PSF before masking vs PSF after masking
        psf_before = self.xp.abs(self.propagate_to_focal_plane(self.ef_in))**2
        psf_after = self.xp.abs(self.propagate_to_focal_plane(self.ef_out))**2
        transmission = self.xp.sum(psf_after) / self.xp.sum(psf_before)

        # Amplitude
        self.out_ef.A[:] = self.xp.abs(self.ef_out)
        # Phase in nm
        self.out_ef.phaseInNm[:] = (self.xp.angle(self.ef_out) / (2 * self.xp.pi)) * self.wavelength_in_nm

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