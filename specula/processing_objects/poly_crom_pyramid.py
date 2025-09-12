from specula.processing_objects.poly_chrom_wfs import PolyChromWFS
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.modulated_pyramid import ModulatedPyramid


class PolyCromPyramid(PolyChromWFS):
    """
    Polychromatic Pyramid sensor that wraps multiple monochromatic ModulatedPyramid sensors.
    Each can have its own wavelength, flux factor, and differential tilt.
    """

    def __init__(self,
                 wavelengthInNm: list,
                 flux_factor: list,
                 # Pyramid parameters (shared by all instances)
                 simul_params: SimulParams,
                 fov: float,            # TODO =2.0,
                 pup_diam: int,         # TODO =30,
                 output_resolution: int,# TODO =80,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 mod_type: str = 'circular',  # 'circular', 'vertical', 'horizontal', 'alternating'
                 fov_errinf: float = 0.5,
                 fov_errsup: float = 2,
                 pup_dist: int = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: float = None,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,    # same as SH
                 yShiftPhInPixel: float = 0.0,    # same as SH
                 xy_tilts_in_arcsec: list = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):

        self._ccd_side = output_resolution  # Output size is square

        super().__init__(
                        wavelengthInNm=wavelengthInNm,
                        ccd_side=self._ccd_side,
                        flux_factor=flux_factor,
                        xy_tilts_in_arcsec=xy_tilts_in_arcsec,
                        target_device_idx=target_device_idx,
                        precision=precision)

        # Create pyramid instances
        for wavelength in self.wavelengths_in_nm:
            pyr = ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=wavelength,
                fov=fov,
                pup_diam=pup_diam,
                output_resolution=output_resolution,
                mod_amp=mod_amp,
                mod_step=mod_step,
                mod_type=mod_type,
                fov_errinf=fov_errinf,
                fov_errsup=fov_errsup,
                pup_dist=pup_dist,
                pup_margin=pup_margin,
                fft_res=fft_res,
                fp_obs=fp_obs,
                pup_shifts=pup_shifts,
                pyr_tlt_coeff=pyr_tlt_coeff,
                pyr_edge_def_ld=pyr_edge_def_ld,
                pyr_tip_def_ld=pyr_tip_def_ld,
                pyr_tip_maya_ld=pyr_tip_maya_ld,
                min_pup_dist=min_pup_dist,
                rotAnglePhInDeg=rotAnglePhInDeg,
                xShiftPhInPixel=xShiftPhInPixel,    # same as SH
                yShiftPhInPixel=yShiftPhInPixel,    # same as SH
                target_device_idx=target_device_idx,
                precision=precision
            )
            self._wfs_instances.append(pyr)
