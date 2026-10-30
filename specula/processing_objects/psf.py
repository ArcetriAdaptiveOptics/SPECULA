
from specula.lib.calc_psf import calc_psf, calc_psf_geometry
from specula.lib.radial_profile import (
    compute_radial_profile,
    compute_fwhm_from_profile,
    compute_encircled_energy,
    get_encircled_energy_at_distance,
)

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

import logging

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
    wavelengthInNm : float [nm]
        Wavelength at which to compute the PSF [nm].
    nd : float [1], optional
        Numerical density of the PSF (pixels per lambda/D). If None, it is calculated
        based on the input ElectricField and pixel size.
    pixel_size_mas : float [mas], optional
        Desired pixel size of the PSF in milliarcseconds. If None, it is calculated
        based on the input ElectricField and numerical density.
    start_time : float [s], optional
        Time (in seconds) after which to start integrating PSF and SR. Default is 0.0.
    compute_profile_metrics : bool
        If True, also compute radial profile, FWHM and encircled-energy outputs.
        By default these summary metrics are evaluated in `finalize()` only.
    compute_metrics_in_trigger : bool
        If True and `compute_profile_metrics` is enabled, also update the same
        metrics after each trigger.
    ee_radius_in_lambda_d : float or array-like [lambda/D], optional
        Radius or radii in units of lambda/D at which to return the encircled energy.
    target_device_idx : int [1], optional
        Target device index for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int [1], optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 nd: float=None,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 compute_profile_metrics: bool=False,
                 compute_metrics_in_trigger: bool=False,
                 ee_radius_in_lambda_d=None,
                 target_device_idx: int = None,
                 precision: int = None,
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if wavelengthInNm <= 0:
            raise ValueError('PSF wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm

        self.psf_pixel_size, self.nd = calc_psf_geometry(
                                            simul_params.pixel_pupil,
                                            simul_params.pixel_pitch,
                                            wavelengthInNm,
                                            nd,
                                            pixel_size_mas)

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
        # Strehl ratio being copied to the host for the log: (buffer, event)
        self._sr_log = None
        self._sr_buffer = None
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

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil')}

    @classmethod
    def output_names(cls):
        return {
            'out_sr': OutputDesc(BaseValue, 'Instantaneous Strehl ratio'),
            'out_psf': OutputDesc(BaseValue, 'Instantaneous PSF'),
            'out_int_sr': OutputDesc(BaseValue, 'Time-integrated Strehl ratio'),
            'out_int_psf': OutputDesc(BaseValue, 'Time-integrated PSF'),
            'out_std_psf': OutputDesc(BaseValue, 'Standard deviation of PSF over time'),
            'out_psf_profile': OutputDesc(BaseValue, 'Radial profile of the instantaneous PSF'),
            'out_psf_fwhm': OutputDesc(BaseValue, 'FWHM of the instantaneous PSF'),
            'out_encircled_energy': OutputDesc(BaseValue, 'Encircled energy of the instantaneous PSF'),
            'out_encircled_energy_at_radius': OutputDesc(BaseValue, 'Encircled energy at specified radius for instantaneous PSF'),
            'out_int_psf_profile': OutputDesc(BaseValue, 'Radial profile of the integrated PSF'),
            'out_int_psf_fwhm': OutputDesc(BaseValue, 'FWHM of the integrated PSF'),
            'out_int_encircled_energy': OutputDesc(BaseValue, 'Encircled energy of the integrated PSF'),
            'out_int_encircled_energy_at_radius': OutputDesc(BaseValue, 'Encircled energy at specified radius for integrated PSF'),
        }

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

    def can_accelerate(self):
        # trigger_code() only launches GPU work (the Strehl ratio is logged in post_trigger())
        return True

    def _log_sr(self):
        '''
        Log the Strehl ratio. On the GPU, without waiting for it: the value of this
        step is copied to the host now, and logged at the next step (or at the end).
        '''
        if not self.logger.isEnabledFor(logging.INFO):
            return
        if self.target_device_idx < 0:
            self.logger.info(f'SR at {int(self.wavelengthInNm)}nm : {self.sr.value}')
            return
        self._flush_sr_log()
        if self._sr_buffer is None:
            import cupyx
            self._sr_buffer = cupyx.empty_pinned((), dtype=self.sr.value.dtype)
        stream = self.xp.cuda.get_current_stream()
        self.sr.value.get(stream=stream, out=self._sr_buffer, blocking=False)
        self._sr_log = (self._sr_buffer, stream.record())

    def _flush_sr_log(self):
        if self._sr_log is not None:
            buffer, event = self._sr_log
            event.synchronize()
            self.logger.info(f'SR at {int(self.wavelengthInNm)}nm : {buffer[()]}')
            self._sr_log = None

    def _compute_radial_profile_data(self, psf, peak:float=None):

        if peak is None:
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

    def _set_radial_profile_output(self, psf, profile_output, norm_peak=None):
        radial_profile_data = self._compute_radial_profile_data(psf, peak=norm_peak)

        profile, radial_dist, _ = radial_profile_data
        profile_output.value = self.xp.vstack([radial_dist, profile])
        profile_output.generation_time = self.current_time

    def _set_profile_outputs(self, psf, profile_output, fwhm_output,
                             ee_output, ee_at_radius_output):
        metrics = self._compute_profile_metrics(psf)

        profile, radial_dist, fwhm, ee, ee_at_radius = metrics
        profile_output.value = self.xp.vstack([radial_dist, profile])
        fwhm_output.value = fwhm
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
        self._log_sr()

        if self.compute_profile_metrics and self.compute_metrics_in_trigger:
            self._set_profile_outputs(
                self.psf.value,
                self.psf_profile,
                self.psf_fwhm,
                self.encircled_energy,
                self.encircled_energy_at_radius,
            )

    def finalize(self):
        self._flush_sr_log()
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
