from specula import RAD2ASEC
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH
from specula.lib.make_xy import make_xy


class PolyChromSH(BaseProcessingObj):
    """
    Polychromatic Shack-Hartmann sensor that wraps multiple monochromatic SH sensors.
    Each SH can have its own wavelength, QE factor, and differential tilt.
    """

    def __init__(self,
                 wavelengthInNm: list,           # List of wavelengths for each SH
                 flux_factor: list,                 # Flux factor for each wavelength
                 # SH parameters (shared by all SH instances)
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 FoVres30mas: bool = False,
                 squaremask: bool = True,
                 fov_ovs_coeff: float = 0,
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 do_not_double_fov_ovs: bool = False,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 subap_rows_slice = None,
                 xy_tilts_in_arcsec: list = None,   # Optional differential tilts [dx, dy] for each SH
                 target_device_idx: int = None,
                 precision: int = None,
                ):

        super().__init__(target_device_idx=target_device_idx,
                        precision=precision)

        # Validate input lists
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

        # Store SH parameters
        self.subap_wanted_fov = subap_wanted_fov
        self.sensor_pxscale = sensor_pxscale
        self.subap_on_diameter = subap_on_diameter
        self.subap_npx = subap_npx

        # Calculate output size
        self._ccd_side = subap_on_diameter * subap_npx

        # Create output intensity
        self._out_i = Intensity(self._ccd_side, self._ccd_side,
                               precision=self.precision,
                               target_device_idx=self.target_device_idx)

        # Create SH instances
        self._sh_instances = []
        for i, wavelength in enumerate(self.wavelengths_in_nm):
            sh = SH(
                wavelengthInNm=wavelength,
                subap_wanted_fov=subap_wanted_fov,
                sensor_pxscale=sensor_pxscale,
                subap_on_diameter=subap_on_diameter,
                subap_npx=subap_npx,
                FoVres30mas=FoVres30mas,
                squaremask=squaremask,
                fov_ovs_coeff=fov_ovs_coeff,
                xShiftPhInPixel=xShiftPhInPixel,
                yShiftPhInPixel=yShiftPhInPixel,
                rotAnglePhInDeg=rotAnglePhInDeg,
                do_not_double_fov_ovs=do_not_double_fov_ovs,
                set_fov_res_to_turbpxsc=set_fov_res_to_turbpxsc,
                laser_launch_tel=laser_launch_tel,
                subap_rows_slice=subap_rows_slice,
                target_device_idx=target_device_idx,
                precision=precision,
            )
            self._sh_instances.append(sh)

        # Setup inputs and outputs
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self._out_i

        # Initialize lists (will be populated in setup)
        self._unit_tilt_x = None  # Unit tilt in x direction (in nm)
        self._unit_tilt_y = None  # Unit tilt in y direction (in nm)
        self._modified_efs = []
        self._has_tilts = any(tilt[0] != 0.0 or tilt[1] != 0.0 for tilt in xy_tilts_in_arcsec)

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
        if self._has_tilts:
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
        for i, sh in enumerate(self._sh_instances):
            sh.inputs['in_ef'].set(self._modified_efs[i])
            sh.setup()

        # Normalize flux factors
        total_flux = self.xp.sum(self.flux_factor)
        if total_flux > 0:
            self.flux_factor_normalized = self.flux_factor / total_flux
        else:
            self.flux_factor_normalized = self.flux_factor

    def check_ready(self, t):
        super().check_ready(t)

        # Check if all SH are ready
        for sh in self._sh_instances:
            sh.check_ready(t)

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
            if tilt_x != 0.0 or tilt_y != 0.0:
                # Scale unit tilts by the desired amounts (no wavelength scaling needed)
                tilt_phase_nm = tilt_x * self._unit_tilt_x + tilt_y * self._unit_tilt_y

                # Add the tilt phase to the original phase
                modified_ef.phaseInNm[:] = in_ef.phaseInNm + tilt_phase_nm
            else:
                # No tilt, just copy original phase
                modified_ef.phaseInNm[:] = in_ef.phaseInNm

            # update generation time
            modified_ef.generation_time = in_ef.generation_time

        # Prepare all SH instances
        for sh in self._sh_instances:
            sh.prepare_trigger(t)

    def trigger_code(self):
        # Reset output intensity
        self._out_i.i[:] = 0.0

        # Trigger each SH and accumulate results
        for i, sh in enumerate(self._sh_instances):
            sh.trigger_code()

            # Add weighted contribution
            flux_factor = self.flux_factor_normalized[i]
            self._out_i.i += sh.outputs['out_i'].i * flux_factor

    def post_trigger(self):
        super().post_trigger()

        # Post-process all SH instances
        for sh in self._sh_instances:
            sh.post_trigger()

        # Set generation time
        self._out_i.generation_time = self.current_time

        # Optional: normalize total intensity to match input photon flux
        in_ef = self.local_inputs['in_ef']
        if hasattr(in_ef, 'S0') and in_ef.S0 > 0:
            total_input_flux = in_ef.S0 * in_ef.masked_area()
            current_total = self.xp.sum(self._out_i.i)
            if current_total > 0:
                self._out_i.i *= total_input_flux / current_total

    def get_sh_instance(self, index):
        """Get a specific SH instance for debugging or analysis."""
        if 0 <= index < self.n_wavelengths:
            return self._sh_instances[index]
        else:
            raise IndexError(f"SH index {index} out of range [0, {self.n_wavelengths-1}]")

    def get_wavelength_contribution(self, index):
        """Get the intensity contribution from a specific wavelength."""
        if 0 <= index < self.n_wavelengths:
            sh = self._sh_instances[index]
            flux_factor = self.flux_factor_normalized[index]
            return sh.outputs['out_i'].i * flux_factor
        else:
            raise IndexError(f"Wavelength index {index} out of range [0, {self.n_wavelengths-1}]")