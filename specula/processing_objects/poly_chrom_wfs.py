from specula import RAD2ASEC
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.lib.make_xy import make_xy

class PolyChromWFS(BaseProcessingObj):
    """
    Base (abstract) class for polychromatic wavefront sensors (WFS).
    Handles multiple wavelengths, flux factors, tilts, and output normalization.
    """

    def __init__(self,
                 wavelengthInNm: list,
                 ccd_side: int,
                 flux_factor: list,
                 xy_tilts_in_arcsec: list = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        n_wavelengths = len(wavelengthInNm)
        if len(flux_factor) != n_wavelengths:
            raise ValueError("wavelengthInNm and flux_factor must have the same length")

        if xy_tilts_in_arcsec is None:
            xy_tilts_in_arcsec = [[0.0, 0.0]] * n_wavelengths
        elif len(xy_tilts_in_arcsec) != n_wavelengths:
            raise ValueError("xy_tilts_in_arcsec must have the same length as wavelengthInNm")

        self.wavelengths_in_nm = wavelengthInNm
        self.flux_factor = self.to_xp(flux_factor)
        self.xy_tilts_in_arcsec = xy_tilts_in_arcsec
        self.n_wavelengths = n_wavelengths

        self._unit_tilt_x = None
        self._unit_tilt_y = None
        self.flux_factor_normalized = None
        self._modified_efs = []
        self._wfs_instances = []

        self._out_i = Intensity(ccd_side, ccd_side,
                                precision=self.precision,
                                target_device_idx=self.target_device_idx)
        self.outputs['out_i'] = self._out_i

        self.inputs['in_ef'] = InputValue(type=ElectricField)

    def _create_unit_tilts(self, in_ef_size, in_ef_pixel_pitch):
        """Create unit tilt phase arrays (1 pixel tilt) in nm."""
        # Create coordinate arrays in meters
        xx, yy = make_xy(in_ef_size, in_ef_pixel_pitch, xp=self.xp)

        # Calculate pupil diameter in meters
        pupil_diameter = in_ef_size * in_ef_pixel_pitch

        # Convert arcseconds to nm RMS:
        # 1 nm RMS = 4e-9/diam * RAD2ASEC arcsec
        nm_to_arcsec = 4e-9 / pupil_diameter * RAD2ASEC
        unit_tilt_nm = 1 / nm_to_arcsec

        # Create linear tilt phase across pupil
        # Normalize coordinates to [-2, 2] range
        xx_norm = 2 * xx / max(abs(xx.max()), abs(xx.min()))
        yy_norm = 2 * yy / max(abs(yy.max()), abs(yy.min()))

        # Create unit tilt phases in nm (peak to valley of 4*rms)
        # For linear tilt: peak-valley = 4*rms, so we use 2*rms for amplitude
        unit_tilt_x_nm = unit_tilt_nm * xx_norm
        unit_tilt_y_nm = unit_tilt_nm * yy_norm

        return unit_tilt_x_nm, unit_tilt_y_nm

    def setup(self):
        super().setup()

        # Get input electric field to determine size
        in_ef = self.local_inputs['in_ef']

        # Create unit tilts (will be scaled later)
        self._unit_tilt_x, self._unit_tilt_y = self._create_unit_tilts(
            in_ef.size[0], in_ef.pixel_pitch
        )

        # Create modified EFs for each SH (always create them for consistency)
        self._modified_efs = []
        for i in range(self.n_wavelengths):
            modified_ef = ElectricField(
                in_ef.size[0], in_ef.size[1], in_ef.pixel_pitch,
                target_device_idx=self.target_device_idx,
                precision=self.precision
            )
            self._modified_efs.append(modified_ef)

        # Setup all SH instances with their modified EFs
        for i, wfs in enumerate(self._wfs_instances):
            wfs.inputs['in_ef'].set(self._modified_efs[i])
            wfs.setup()

        # Normalize flux factors
        total_flux = self.xp.sum(self.flux_factor)
        if total_flux > 0:
            self.flux_factor_normalized = self.flux_factor / total_flux
        else:
            self.flux_factor_normalized = self.flux_factor

    def check_ready(self, t):
        super().check_ready(t)

        # Check if all SH are ready
        for wfs in self._wfs_instances:
            wfs.check_ready(t)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        in_ef = self.local_inputs['in_ef']

        # Update all modified EFs
        for i, modified_ef in enumerate(self._modified_efs):
            # Copy basic properties
            modified_ef.A[:] = in_ef.A
            modified_ef.S0 = in_ef.S0
            modified_ef.pixel_pitch = in_ef.pixel_pitch

            # Apply tilt if present
            tilt_x, tilt_y = self.xy_tilts_in_arcsec[i]

            # Scale unit tilts by the desired amounts (no wavelength scaling needed)
            tilt_phase_nm = tilt_x * self._unit_tilt_x + tilt_y * self._unit_tilt_y

            # Add the tilt phase to the original phase
            modified_ef.phaseInNm[:] = in_ef.phaseInNm + tilt_phase_nm

            # update generation time
            modified_ef.generation_time = in_ef.generation_time

        # Prepare all SH instances
        for wfs in self._wfs_instances:
            wfs.prepare_trigger(t)

    def trigger_code(self):
        # Reset output intensity
        self._out_i.i[:] = 0.0

        # Trigger each SH
        for wfs in self._wfs_instances:
            wfs.trigger_code()

    def post_trigger(self):
        super().post_trigger()

        # Post-process all SH instances
        for wfs in self._wfs_instances:
            wfs.post_trigger()

        # Accumulate results on output intensity
        for i, wfs in enumerate(self._wfs_instances):
            # Add weighted contribution
            flux_factor = self.flux_factor_normalized[i]
            self._out_i.i += wfs.outputs['out_i'].i * flux_factor

        # Set generation time
        self._out_i.generation_time = self.current_time

        # Optional: normalize total intensity to match input photon flux
        in_ef = self.local_inputs['in_ef']
        total_input_flux = in_ef.S0 * in_ef.masked_area()
        current_total = self.xp.sum(self._out_i.i)
        if current_total > 0:
            self._out_i.i *= total_input_flux / current_total

    def get_wavelength_contribution(self, index):
        """Get the intensity contribution from a specific wavelength."""
        if 0 <= index < self.n_wavelengths:
            wfs = self._wfs_instances[index]
            flux_factor = self.flux_factor_normalized[index]
            return wfs.outputs['out_i'].i * flux_factor
        else:
            raise IndexError(f"Wavelength index {index} out of range [0, {self.n_wavelengths-1}]")
