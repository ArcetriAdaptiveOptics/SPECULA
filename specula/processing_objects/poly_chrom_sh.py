from specula.processing_objects.poly_chrom_wfs import PolyChromWFS
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH


class PolyChromSH(PolyChromWFS):
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

        # Calculate output size
        self._ccd_side = subap_on_diameter * subap_npx

        super().__init__(
                        wavelengthInNm=wavelengthInNm,
                        ccd_side=self._ccd_side,
                        flux_factor=flux_factor,
                        xy_tilts_in_arcsec=xy_tilts_in_arcsec,
                        target_device_idx=target_device_idx,
                        precision=precision)

        # Create SH instances
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
            self._wfs_instances.append(sh)

