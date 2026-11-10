from specula.processing_objects.poly_chrom_wfs import PolyChromWFS
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.modulated_pyramid import ModulatedPyramid


class PolyCromPyramid(PolyChromWFS):
    """
    Poly-chromatic Pyramid sensor processing object. 
    Wraps multiple monochromatic ModulatedPyramid sensors, each of which 
    can have its own wavelength, flux factor, and differential tilt.

    The parameters in lambda/D units (mod_amp, fp_obs and the pyramid
    defects pyr_*_ld) are given at the reference wavelength
    wavelengthInNm[ref_wavelength_idx], and are rescaled by
    lambda_ref / lambda for the pyramid of each wavelength: the modulation
    radius and the size of the defects are fixed angles on the sky (a real
    modulator tilts the beam by the same angle at all wavelengths, and a
    defect of the prism has the same size at all wavelengths).

    All the pyramids use the same number of modulation steps: mod_step if
    given, otherwise the default of ModulatedPyramid for the largest
    modulation amplitude in lambda/D, the one at the shortest wavelength.
    An explicit mod_step must be large enough for that amplitude.

    Parameters
    ----------
    ref_wavelength_idx : int, optional
        Index in wavelengthInNm of the reference wavelength of mod_amp, fp_obs
        and the pyr_*_ld parameters (default: 0, the first wavelength).
    """

    def __init__(self,
                 wavelengthInNm: list,
                 flux_factor: list,
                 # Pyramid parameters (shared by all instances)
                 simul_params: SimulParams,
                 fov: float,
                 pup_diam: float,
                 output_resolution: int,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 mod_type: str = 'circular',  # 'circular', 'vertical', 'horizontal', 'alternating'
                 fov_errinf: float = 0.5,
                 fov_errsup: float = 2,
                 pup_dist: float = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: float = None,
                 pyr_roof_ld: float = 0.0,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 pyr_max_side_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,    # same as SH
                 yShiftPhInPixel: float = 0.0,    # same as SH
                 xy_tilts_in_arcsec: list = None,
                 ref_wavelength_idx: int = 0,
                 n_devices: int = 1,     # Devices over which the pyramid instances are spread
                 target_device_idx: int = None,
                 precision: int = None
                ):

        self._ccd_side = output_resolution  # Output size is square

        super().__init__(
                        wavelengthInNm=wavelengthInNm,
                        ccd_side=self._ccd_side,
                        flux_factor=flux_factor,
                        xy_tilts_in_arcsec=xy_tilts_in_arcsec,
                        n_devices=n_devices,
                        target_device_idx=target_device_idx,
                        precision=precision)

        n_wavelengths = len(self.wavelengths_in_nm)
        if not -n_wavelengths <= ref_wavelength_idx < n_wavelengths:
            raise ValueError(f'ref_wavelength_idx = {ref_wavelength_idx} is out of range '
                             f'for {n_wavelengths} wavelengths')
        self.ref_wavelength_in_nm = self.wavelengths_in_nm[ref_wavelength_idx]

        # The same number of modulation steps for all wavelengths: by default,
        # the one needed by the largest modulation in lambda/D (the shortest
        # wavelength)
        if mod_step is None:
            max_mod_amp = mod_amp * self.ref_wavelength_in_nm / min(self.wavelengths_in_nm)
            mod_step = ModulatedPyramid.default_mod_step(max_mod_amp, mod_type)

        # Create pyramid instances
        for i, wavelength in enumerate(self.wavelengths_in_nm):
            # lambda/D at the reference wavelength -> lambda/D at this wavelength
            # (fp_obs is in lambda/D too)
            ld_scale = self.ref_wavelength_in_nm / wavelength
            pyr = ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=wavelength,
                fov=fov,
                pup_diam=pup_diam,
                output_resolution=output_resolution,
                mod_amp=mod_amp * ld_scale,
                mod_step=mod_step,
                mod_type=mod_type,
                fov_errinf=fov_errinf,
                fov_errsup=fov_errsup,
                pup_dist=pup_dist,
                pup_margin=pup_margin,
                fft_res=fft_res,
                fp_obs=fp_obs * ld_scale if fp_obs is not None else None,
                pup_shifts=pup_shifts,
                pyr_tlt_coeff=pyr_tlt_coeff,
                pyr_roof_ld=pyr_roof_ld * ld_scale,
                pyr_edge_def_ld=pyr_edge_def_ld * ld_scale,
                pyr_tip_def_ld=pyr_tip_def_ld * ld_scale,
                pyr_tip_maya_ld=pyr_tip_maya_ld * ld_scale,
                pyr_max_side_ld=pyr_max_side_ld * ld_scale,
                min_pup_dist=min_pup_dist,
                rotAnglePhInDeg=rotAnglePhInDeg,
                xShiftPhInPixel=xShiftPhInPixel,    # same as SH
                yShiftPhInPixel=yShiftPhInPixel,    # same as SH
                target_device_idx=self._wfs_device_idx(i),
                precision=precision
            )
            self._wfs_instances.append(pyr)
        # Building the instances made their devices current: back to the one of this object
        self._use_device(self)
