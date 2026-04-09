
from specula.lib.calc_psf import calc_psf, calc_psf_geometry
from specula.lib.radial_profile import (
    compute_radial_profile,
    compute_fwhm_from_profile,
    compute_encircled_energy,
    get_encircled_energy_at_distance,
)

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

import numpy as np


class PSF(BaseProcessingObj):
    """ 
    Point Spread Function (PSF) processing object. 
    Computes PSF, Strehl ratio (SR), integrated PSF and SR, 
    and PSF standard deviation over time from an input ElectricField.    

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object.
    wavelengthInNm : float
        Wavelength at which to compute the PSF [nm].
    nd : float, optional
        Numerical density of the PSF (pixels per lambda/D). If None, it is calculated
        based on the input ElectricField and pixel size.
    pixel_size_mas : float, optional
        Desired pixel size of the PSF in milliarcseconds. If None, it is calculated
        based on the input ElectricField and numerical density.
    start_time : float, optional
        Time (in seconds) after which to start integrating PSF and SR. Default is 0.0.
    compute_profile_metrics : bool, optional
        If True, also compute radial profile, FWHM and encircled-energy outputs.
        By default these summary metrics are evaluated in `finalize()` only.
    compute_metrics_in_trigger : bool, optional
        If True and `compute_profile_metrics` is enabled, also update the same
        metrics after each trigger.
    ee_radius_in_lambda_d : float or array-like, optional
        Radius or radii in units of lambda/D at which to return the encircled energy.
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,    # TODO =500.0,
                 nd: float=None,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 compute_profile_metrics: bool=False,
                 compute_metrics_in_trigger: bool=False,
                 ee_radius_in_lambda_d=None,
                 target_device_idx: int = None,
                 precision: int = None,
                 verbose:bool = True,
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if wavelengthInNm <= 0:
            raise ValueError('PSF wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm
        self.wave_str = f"{int(wavelengthInNm)}nm"

        self.psf_pixel_size, self.nd = calc_psf_geometry(
                                            simul_params.pixel_pupil,
                                            simul_params.pixel_pitch,
                                            wavelengthInNm,
                                            nd,
                                            pixel_size_mas)

        self.verbose = verbose
        self.start_time = start_time
        self.compute_profile_metrics = compute_profile_metrics
        self.compute_metrics_in_trigger = compute_metrics_in_trigger
        self.ee_radius_in_lambda_d = ee_radius_in_lambda_d

        self.sr = BaseValue(target_device_idx=self.target_device_idx,
                            precision=precision)
        self.int_sr = BaseValue(target_device_idx=self.target_device_idx,
                                precision=precision)
        self.psf = BaseValue(target_device_idx=self.target_device_idx,
                             precision=precision)
        self.int_psf = BaseValue(target_device_idx=self.target_device_idx,
                                precision=precision)
        self.std_psf = BaseValue(target_device_idx=self.target_device_idx,
                                precision=precision)
        self.psf_profile = BaseValue(target_device_idx=self.target_device_idx,
                                     precision=precision)
        self.psf_fwhm = BaseValue(target_device_idx=self.target_device_idx,
                                  precision=precision)
        self.encircled_energy = BaseValue(target_device_idx=self.target_device_idx,
                                          precision=precision)
        self.encircled_energy_at_radius = BaseValue(target_device_idx=self.target_device_idx,
                                                    precision=precision)
        self.int_psf_profile = BaseValue(target_device_idx=self.target_device_idx,
                                         precision=precision)
        self.int_psf_fwhm = BaseValue(target_device_idx=self.target_device_idx,
                                      precision=precision)
        self.int_encircled_energy = BaseValue(target_device_idx=self.target_device_idx,
                                              precision=precision)
        self.int_encircled_energy_at_radius = BaseValue(target_device_idx=self.target_device_idx,
                                                        precision=precision)
        self.ref = None
        self.count = 0
        self.first = True
        self._sum_psf_squared = None # For std dev calculation

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self.sr
        self.outputs['out_psf'] = self.psf
        self.outputs['out_int_sr'] = self.int_sr
        self.outputs['out_int_psf'] = self.int_psf
        self.outputs['out_std_psf'] = self.std_psf
        self.outputs['out_psf_profile'] = self.psf_profile
        self.outputs['out_psf_fwhm'] = self.psf_fwhm
        self.outputs['out_encircled_energy'] = self.encircled_energy
        self.outputs['out_encircled_energy_at_radius'] = self.encircled_energy_at_radius
        self.outputs['out_int_psf_profile'] = self.int_psf_profile
        self.outputs['out_int_psf_fwhm'] = self.int_psf_fwhm
        self.outputs['out_int_encircled_energy'] = self.int_encircled_energy
        self.outputs['out_int_encircled_energy_at_radius'] = self.int_encircled_energy_at_radius

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']
        s = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.int_psf.value = self.xp.zeros(s, dtype=self.dtype)
        self._sum_psf_squared = self.xp.zeros(s, dtype=self.dtype)
        self.std_psf.value = self.xp.zeros(s, dtype=self.dtype)
        self.int_sr.value = 0

        self.out_size = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.ref = Intensity(self.out_size[0], self.out_size[1],
                             target_device_idx=self.target_device_idx)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        in_ef = self.local_inputs['in_ef']

        # First time, calculate reference PSF.
        if self.first:
            self.ref.i[:] = calc_psf(in_ef.A * 0.0,
                                     in_ef.A,
                                     imwidth=self.out_size[0],
                                     normalize=True,
                                     xp=self.xp,
                                     complex_dtype=self.complex_dtype)
            self.first = False

    def trigger_code(self):
        in_ef = self.local_inputs['in_ef']
        self.psf.value, self.total_psf = calc_psf(in_ef.phi_at_lambda(self.wavelengthInNm),
                                                  in_ef.A, imwidth=self.out_size[0], normalize=True,
                                                  xp=self.xp, complex_dtype=self.complex_dtype,
                                                  return_total=True)
        self.sr.value = self.psf.value[self.out_size[0] // 2, \
                                       self.out_size[1] // 2] / self.ref.i[self.out_size[0] // 2, \
                                       self.out_size[1] // 2]
        if self.verbose:
            print('SR at ' + self.wave_str + ':', self.sr.value, flush=True)

    def _compute_radial_profile_data(self, psf):
        if psf is None:
            return None

        peak = self.xp.max(psf)
        if float(peak) <= 0.0:
            norm_psf = self.xp.zeros_like(psf)
        else:
            norm_psf = psf / peak

        profile, pix_dist, n_px_in_bin = compute_radial_profile(
            norm_psf,
            xp=self.xp,
            dtype=self.dtype,
            return_counts=True,
        )
        radial_dist = pix_dist / self.nd
        return profile, radial_dist, n_px_in_bin

    def _compute_profile_metrics(self, psf):
        radial_profile_data = self._compute_radial_profile_data(psf)
        if radial_profile_data is None:
            return None

        profile, radial_dist, n_px_in_bin = radial_profile_data
        fwhm = compute_fwhm_from_profile(profile, radial_dist, xp=self.xp, dtype=self.dtype)
        ee = compute_encircled_energy(profile, n_px_in_bin, xp=self.xp, dtype=self.dtype)

        ee_at_radius = None
        if self.ee_radius_in_lambda_d is not None:
            ee_at_radius = get_encircled_energy_at_distance(
                ee,
                radial_dist,
                self.ee_radius_in_lambda_d,
                xp=self.xp,
                dtype=self.dtype,
            )
        return profile, radial_dist, fwhm, ee, ee_at_radius

    def _set_radial_profile_output(self, psf, profile_output):
        radial_profile_data = self._compute_radial_profile_data(psf)
        if radial_profile_data is None:
            return

        profile, radial_dist, _ = radial_profile_data
        profile_output.value = self.xp.vstack([radial_dist, profile])
        profile_output.generation_time = self.current_time

    def _set_profile_outputs(self, psf, profile_output, fwhm_output,
                             ee_output, ee_at_radius_output):
        metrics = self._compute_profile_metrics(psf)
        if metrics is None:
            return

        profile, radial_dist, fwhm, ee, ee_at_radius = metrics
        profile_output.value = self.xp.vstack([radial_dist, profile])
        fwhm_output.value = self.dtype(fwhm)
        ee_output.value = self.xp.vstack([radial_dist, ee])
        ee_at_radius_output.value = ee_at_radius

        profile_output.generation_time = self.current_time
        fwhm_output.generation_time = self.current_time
        ee_output.generation_time = self.current_time
        ee_at_radius_output.generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
        if self.current_time_seconds >= self.start_time:
            self.count += 1
            self.int_sr.value += self.sr.value
            self.int_psf.value += self.psf.value
            self._sum_psf_squared += self.psf.value ** 2
        self.psf.generation_time = self.current_time
        self.sr.generation_time = self.current_time

        if self.compute_profile_metrics and self.compute_metrics_in_trigger:
            self._set_profile_outputs(
                self.psf.value,
                self.psf_profile,
                self.psf_fwhm,
                self.encircled_energy,
                self.encircled_energy_at_radius,
            )

    def finalize(self):
        if self.count > 0:
            self.int_psf.value /= self.count
            self.int_sr.value /= self.count
            variance = self._sum_psf_squared / self.count - self.int_psf.value ** 2
            self.std_psf.value = self.xp.sqrt(self.xp.maximum(variance, 0))
            if self.compute_profile_metrics:
                self._set_profile_outputs(
                    self.int_psf.value,
                    self.int_psf_profile,
                    self.int_psf_fwhm,
                    self.int_encircled_energy,
                    self.int_encircled_energy_at_radius,
                )

        self.int_psf.generation_time = self.current_time
        self.int_sr.generation_time = self.current_time
        self.std_psf.generation_time = self.current_time
