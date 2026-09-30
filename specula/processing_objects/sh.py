import numpy as np

from specula import fuse, RAD2ASEC
from specula.tracing import tracer
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.toccd import toccd
from specula.lib.make_mask import make_mask
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.data_objects.lenslet import Lenslet
from specula.base_value import BaseValue
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel
from specula.data_objects.convolution_kernel import ConvolutionKernel


# numpy 1.x compatibility (cupy sometimes tries to raise this exception)
if hasattr(np, 'exceptions'):
    np.ComplexWarning = np.exceptions.ComplexWarning

@fuse(kernel_name='abs2')
def abs2(u_fp, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp))


@fuse(kernel_name='abs2_masked')
def abs2_masked(u_fp, mask, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp)) * mask


def choose_fov_resolution(turbulence_pxscale, sensor_pxscale, subap_wanted_fov, subap_npx,
                          n_try=10, max_fov_error=0.02):
    '''
    Choose the internal FoV resolution of the SH as turbulence_pxscale / k,
    with k an integer.

    The candidates are the n_try values of k starting from the smallest one
    (at least 2) that gives a resolution finer than the sensor pixel scale.
    For each candidate, the FoV is rounded to an even number of resolution
    elements. Candidates whose FoV error is below max_fov_error are kept (all
    of them, if none is). Among these, the first one is chosen, unless the
    one with the smallest ratio between the L.C.M. used by toccd() and k
    reduces the L.C.M. by more than it increases k.

    Parameters
    ----------
    turbulence_pxscale : float [arcsec]
        Diffraction-limited pixel scale of a subaperture (lambda / d)
    sensor_pxscale : float [arcsec]
        Sensor pixel scale
    subap_wanted_fov : float [arcsec]
        Wanted subaperture FoV
    subap_npx : int
        Number of sensor pixels across a subaperture
    n_try : int, optional
        Number of candidate resolutions
    max_fov_error : float, optional
        Maximum relative FoV error

    Returns
    -------
    float
        FoV resolution in arcsec
    '''
    i_min = 0
    while turbulence_pxscale / (i_min + 2) >= sensor_pxscale:
        i_min += 1

    k = i_min + 2 + np.arange(n_try)
    resolution = turbulence_pxscale / k
    fov_pix = np.round(subap_wanted_fov / resolution / 2.0) * 2
    lcm = np.lcm(int(subap_npx), fov_pix.astype(int))

    fov_error = np.abs(fov_pix * resolution - subap_wanted_fov) / subap_wanted_fov
    idx_good = np.where(fov_error < max_fov_error)[0]
    if len(idx_good) == 0:
        idx_good = np.arange(n_try)

    first = idx_good[0]
    best = idx_good[np.argmin(lcm[idx_good] / k[idx_good])]
    if lcm[first] / lcm[best] > k[best] / k[first]:
        return resolution[best]
    else:
        return resolution[first]


class SH(BaseProcessingObj):
    """
    Shack-Hartmann wavefront sensor processing object.
    Takes an electric field as input and produces an intensity as output.
    
    Parameters
    ----------
    wavelengthInNm : float [nm]
        Wavelength in nanometers
    subap_wanted_fov : float [arcsec]
        Desired subaperture Field of View in arcseconds
    sensor_pxscale : float [arcsec/pixel]
        Sensor pixel scale in arcseconds/pixel
    subap_on_diameter : int [1]
        Subaperture diameter in meters
    subap_npx : int [pixels]
        Number of pixels across the subaperture on the sensor
    squaremask : bool
        If True, use a square mask in the focal plane. Default is True.
    fov_ovs_coeff : float [1], optional
        Coefficient to determine the oversampling of the FoV.
        A value larger than 1 is recommended to avoid FFT wrapping effects.
        Default is 2.0.
    xShiftPhInPixel : float [pixels], optional
        Shift of the phase in the x direction in pixels. Default is 0.
    yShiftPhInPixel : float [pixels], optional
        Shift of the phase in the y direction in pixels. Default is 0.
    rotAnglePhInDeg : float [deg], optional
        Rotation angle of the phase in degrees. Default is 0.
    set_fov_res_to_turbpxsc : bool
        If True, set the FoV resolution to the turbulence pixel scale. Default is False.
    laser_launch_tel : LaserLaunchTelescope
        If provided, use the laser launch telescope parameters for kernel generation.
        Default is None.
    subap_rows_slice : slice [1], optional
        Slice object to specify which rows of subapertures to process.
        Default is None (process all rows).
    data_dir : str
        Directory for data files needed by the kernel object. Default is "".
        Set by simul object if not provided.
    target_device_idx : int [1], optional
        Target device index for GPU processing. Default is None (CPU).
    precision : int [1], optional
        Numerical precision (e.g., 32 or 64). Default is None (use default precision).

    Attributes
    ----------
    sensor_pxscale_effective : float [arcsec/pixel]
        Sensor pixel scale actually simulated, set by setup(). The subaperture FoV
        is rounded to an even number of FFT pixels, so it can differ from
        sensor_pxscale: a warning is logged if the difference is larger than 1%.
    subap_real_fov_arcsec : float [arcsec]
        Subaperture FoV actually simulated (subap_npx * sensor_pxscale_effective),
        set by setup().
    """

    __zeros_cache = {}

    def _zeros_common(self, shape, dtype, key_extra=None):
        """
        Wrapper around self.xp.zeros to enable reuse cache.
        None of the arrays allocated here should be used in 
        prepare_trigger() or post_trigger().
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : dtype
            Data type
        key_extra : hashable, optional
            Additional cache key, for arrays that can only be shared
            by objects that also agree on something else than the shape
            
        Returns
        -------
        array : ndarray
            Array from cache
        """
        key = (self.target_device_idx, shape, dtype, key_extra)
        if key not in self.__zeros_cache:
            self.__zeros_cache[key] = self.xp.zeros(shape, dtype=dtype)
        return self.__zeros_cache[key]

    def __init__(self,
                 wavelengthInNm: float,
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 squaremask: bool = True,
                 fov_ovs_coeff: float = 2.0, # some margin to avoid FFT wrapping
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 subap_rows_slice = None,
                 data_dir: str = "",
                 target_device_idx: int = None,
                 precision: int = None,
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.wavelength_in_nm = wavelengthInNm
        self.subap_wanted_fov = subap_wanted_fov
        self.subap_on_diameter = subap_on_diameter
        self._lenslet = Lenslet(self.subap_on_diameter, target_device_idx=target_device_idx)
        self._subap_wanted_fov_rad = self.subap_wanted_fov / RAD2ASEC
        self._sensor_pxscale_arcsec = sensor_pxscale
        self._subap_npx = subap_npx
        self._fov_ovs_coeff = fov_ovs_coeff
        self._squaremask = squaremask
        self._fov_resolution_arcsec = None
        self.sensor_pxscale_effective = None
        self.subap_real_fov_arcsec = None
        self._rotAnglePhInDeg = rotAnglePhInDeg
        self._xShiftPhInPixel = xShiftPhInPixel
        self._yShiftPhInPixel = yShiftPhInPixel
        self._set_fov_res_to_turbpxsc = set_fov_res_to_turbpxsc
        self._laser_launch_tel = laser_launch_tel
        self.data_dir = data_dir
        self._fft_size = 0
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        self.psf_shifted = None
        self.ef_row = None
        self.ef_interpolator = None
        self._ovs_np_sub = None
        self._ovs_ef_size = None
        self._wf3 = None
        self._cutpixels = None
        self._cutsize = None
        self._psfimage = None
        self._tltf = None
        self._fp_mask = None
        self._cut_slice = None
        self._fp_mask_cut = None
        self._apply_mask = True
        self._wf3_view = None
        self._subap_cube_view = None
        self._psfimage_views = None
        self._kernelobj = None
        self._last_sodium_values = None
        self._fov_ovs = 1

        self._ccd_side = self._subap_npx * self._lenslet.n_lenses
        self._out_i = Intensity(self._ccd_side, self._ccd_side,
                                precision=self.precision,
                                target_device_idx=self.target_device_idx)

        self.subap_rows_slice = subap_rows_slice

        # optional inputs for the kernel object
        if self._laser_launch_tel is not None:
            self.inputs['sodium_altitude'] = InputValue(type=BaseValue, optional=True)
            self.inputs['sodium_intensity'] = InputValue(type=BaseValue, optional=True)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self._out_i

    @classmethod
    def input_names(cls):
        return {
            'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil'),
            'sodium_altitude': InputDesc(BaseValue, 'Sodium layer altitude profile (optional)'),
            'sodium_intensity': InputDesc(BaseValue, 'Sodium layer intensity profile (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_i': OutputDesc(Intensity, 'Output Shack-Hartmann focal-plane intensity image'),
        }

    def _set_in_ef(self, in_ef):
        '''
        Compute the SH geometry from the input electric field.
        All angles are in arcsec.

        Sets _fov_resolution_arcsec, _fov_ovs, _ovs_ef_size, _ovs_np_sub,
        _fft_size, _cutsize, _cutpixels, sensor_pxscale_effective and
        subap_real_fov_arcsec.
        '''
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.size[0]
        sensor_pxscale = self._sensor_pxscale_arcsec

        # Pixels across a subaperture (can be fractional) and diffraction-limited pixel scale
        np_sub = ef_size / n_lenses
        turbulence_pxscale = self.wavelength_in_nm * 1e-9 / (np_sub * in_ef.pixel_pitch) * RAD2ASEC

        # Internal FoV resolution
        if self._set_fov_res_to_turbpxsc:
            if turbulence_pxscale >= sensor_pxscale:
                raise ValueError('set_fov_res_to_turbpxsc property should be set'
                                 ' to one only if turb. pix. sc. is < sensor pix. sc.')
            self._fov_resolution_arcsec = turbulence_pxscale
            self.logger.warning('set_fov_res_to_turbpxsc property is set.')
            self.logger.warning('FoV internal resolution parameter will be set to turb. pix. sc.')
        elif turbulence_pxscale < sensor_pxscale and sensor_pxscale / 2.0 > 0.5:
            self._fov_resolution_arcsec = turbulence_pxscale * 0.5
        else:
            self._fov_resolution_arcsec = choose_fov_resolution(turbulence_pxscale,
                                                                sensor_pxscale,
                                                                self.subap_wanted_fov,
                                                                self._subap_npx)

        # FFT sampling: the FFT pixel scale is turbulence_pxscale / scale_ovs
        scale_ovs = round(turbulence_pxscale / self._fov_resolution_arcsec)
        fft_pxscale = turbulence_pxscale / scale_ovs

        # Sensor subaperture FoV, as an even number of FFT pixels.
        # The resulting sensor pixel scale can differ from the requested one.
        subap_real_fov_pix = round(sensor_pxscale * self._subap_npx / fft_pxscale / 2.0) * 2
        self.subap_real_fov_arcsec = subap_real_fov_pix * fft_pxscale
        self.sensor_pxscale_effective = self.subap_real_fov_arcsec / self._subap_npx
        pxscale_error = abs(self.sensor_pxscale_effective - sensor_pxscale) / sensor_pxscale
        if pxscale_error > 0.01:
            self.logger.warning(f'Effective sensor pixel scale {self.sensor_pxscale_effective:.4f} arcsec'
                                f' differs by {pxscale_error * 100:.1f}% from the requested'
                                f' {sensor_pxscale} arcsec')

        # Oversampling of the electric field. We take the maximum of three constraints:
        # - 1.0: do not downsample (loss of quality);
        # - ratio: the FFT FoV must cover the sensor subaperture FoV;
        # - fov_ovs_coeff: explicit user request for super-sampling.
        turbulence_fov_pix = (scale_ovs * ef_size) // n_lenses
        ratio = subap_real_fov_pix / turbulence_fov_pix if turbulence_fov_pix > 0 else 1.0
        needed_ovs = max(1.0, ratio, self._fov_ovs_coeff)

        # The oversampled size is rounded up to a multiple of 2 * n_lenses, so that
        # each subaperture has an integer and even number of pixels
        modulus = 2 * n_lenses
        self._ovs_ef_size = int(np.ceil(ef_size * needed_ovs / modulus)) * modulus
        self._fov_ovs = self._ovs_ef_size / ef_size
        self._ovs_np_sub = self._ovs_ef_size // n_lenses
        self._fft_size = self._ovs_np_sub * scale_ovs

        # FoV cut: the sensor FoV is kept out of the FFT FoV.
        # Both sizes are even, so the cut is symmetric.
        self._cutsize = subap_real_fov_pix
        self._cutpixels = self._fft_size - self._cutsize

        self.logger.info('-->     FoV resolution [asec], {}'.format(self._fov_resolution_arcsec))
        self.logger.info('-->     turb. pix. sc.,        {}'.format(turbulence_pxscale))
        self.logger.info('-->     sc. over sampl.,       {}'.format(scale_ovs))
        self.logger.info('-->     FoV over sampl.,       {}'.format(self._fov_ovs))
        self.logger.info('-->     FFT pix. sc. [asec],   {}'.format(fft_pxscale))
        self.logger.info('-->     no. elements FoV,      {}'.format(subap_real_fov_pix))
        self.logger.info('-->     sensor pix. sc. [asec],{}'.format(self.sensor_pxscale_effective))
        self.logger.info('-->     FFT size (turb. FoV),  {}'.format(self._fft_size))
        self.logger.info('-->     L.C.M. for toccd,      {}'.format(np.lcm(self._subap_npx, subap_real_fov_pix)))
        self.logger.info('-->     oversampled np_sub,    {}'.format(self._ovs_np_sub))
        self.logger.info('-->     oversampled EF size,   {}'.format(self._ovs_ef_size))

    def _calc_geometry(self, in_ef):
        '''
        Allocate the buffers and compute the arrays used by trigger_code(),
        using the geometry computed by _set_in_ef()
        '''
        fft_size = self._fft_size
        n = self._ovs_np_sub

        # FFT pixel scale [rad] and FFT FoV [rad]. Computed as in the original code
        # rather than from _set_in_ef() values, because the kernel file names are
        # a hash of the exact pixel scale value.
        fp4_pixel_pitch = self.wavelength_in_nm / 1e9 / (in_ef.pixel_pitch / self._fov_ovs * fft_size)
        fov_complete = fft_size * fp4_pixel_pitch

        # Padded subaperture cube extracted from full pupil (one row of subapertures,
        # i.e. dimx of them). Only the top-left _ovs_np_sub x _ovs_np_sub corner of each
        # subap is ever written, so the zero padding is set here once and never touched
        # again. The corner size is part of the cache key: an SH with the same fft_size
        # but a larger corner would otherwise write into the padding of this one.
        self._wf3 = self._zeros_common((self._lenslet.dimx, fft_size, fft_size),
                                       dtype=self.complex_dtype,
                                       key_extra=self._ovs_np_sub)
        self._psfimage = self._zeros_common((self._cutsize * self._lenslet.dimy,
                                             self._cutsize * self._lenslet.dimx),
                                            dtype=self.dtype)

        # 1/2 Px tilt
        self._tltf = self._get_tlt_f(n, fft_size - n)

        # Without a convolution kernel, the FFT output must be fftshift-ed.
        # Since fft_size is even, fft(x * (-1)^(m+n)) == fftshift(fft(x)), so the
        # shift is folded into the tilt as a checkerboard and costs nothing at runtime.
        # The kernel path does not fftshift (the kernels already include it),
        # so it keeps the plain tilt.
        if self._laser_launch_tel is None:
            m = self.xp.arange(n)
            checkerboard = 1 - 2 * ((m[:, None] + m[None, :]) % 2)
            self._tltf *= checkerboard

        self._fp_mask = make_mask(fft_size,
                                  diaratio=self._subap_wanted_fov_rad / fov_complete,
                                  square=self._squaremask, xp=self.xp)

        # FoV cut on each subap: keep cutsize pixels starting at cutpixels // 2.
        # The mask is only ever applied to the kept region, so it is cut here once.
        # If the kept region of the mask is all ones, masking is skipped altogether.
        cut_start = self._cutpixels // 2
        self._cut_slice = slice(cut_start, cut_start + self._cutsize)
        self._fp_mask_cut = self._fp_mask[self._cut_slice, self._cut_slice].astype(self.dtype)
        self._apply_mask = not bool(self.xp.all(self._fp_mask_cut == 1))

        # set up kernel object
        if self._laser_launch_tel is not None:
            if len(self._laser_launch_tel.tel_pos) == 0:
                self._kernelobj = GaussianConvolutionKernel(dimx = self._lenslet.dimx,
                                                            dimy = self._lenslet.dimy,
                                                            pxscale = fp4_pixel_pitch * RAD2ASEC,
                                                            pupil_size_m = in_ef.pixel_pitch * in_ef.size[0],
                                                            dimension = fft_size,
                                                            spot_size = self._laser_launch_tel.spot_size,
                                                            oversampling = 1,
                                                            return_fft = True,
                                                            positive_shift_tt = True,
                                                            data_dir=self.data_dir,
                                                            target_device_idx=self.target_device_idx,
                                                            precision=self.precision)
            else:
                self._kernelobj = ConvolutionKernel(dimx = self._lenslet.dimx,
                                                    dimy = self._lenslet.dimy,
                                                    pxscale = fp4_pixel_pitch * RAD2ASEC,
                                                    pupil_size_m = in_ef.pixel_pitch * in_ef.size[0],
                                                    dimension = fft_size,
                                                    launcher_pos = self._laser_launch_tel.tel_pos,
                                                    seeing = 0.0,
                                                    launcher_size = self._laser_launch_tel.spot_size,
                                                    zfocus = self._laser_launch_tel.beacon_focus,
                                                    theta = self._laser_launch_tel.beacon_tt,
                                                    oversampling = 1,
                                                    return_fft = True,
                                                    positive_shift_tt = True,
                                                    data_dir=self.data_dir,
                                                    target_device_idx=self.target_device_idx,
                                                    precision=self.precision)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        if self._kernelobj is not None:
            self._prepare_kernels()

        # The input field interpolation is done in trigger_code(), so that it is
        # part of the CUDA graph. Its extrapolation data depends on the pupil,
        # which is only valid from the first step on (it is set by the upstream
        # objects in their trigger): it is computed here, before the graph is
        # captured at the first trigger(). Does nothing after the first call.
        self.ef_interpolator.initialize_extrapolation()

    def _prepare_kernels(self):
        if len(self._laser_launch_tel.tel_pos) != 0:
            sodium_altitude = self.local_inputs['sodium_altitude']
            sodium_intensity = self.local_inputs['sodium_intensity']
            if sodium_altitude is None or sodium_intensity is None:
                raise ValueError('sodium_altitude and sodium_intensity must be provided')
            values = (sodium_altitude.value, sodium_intensity.value)
        else:
            values = ()

        # Avoid recomputing kernels if the sodium layer parameters
        # have not changed since the last call. Their values are compared,
        # because generators update the generation time at every step,
        # even if the actual values are unchanged.
        if self._last_sodium_values is not None:
            # Accumulated on the device, so that there is a single sync
            equal = True
            for v, last in zip(values, self._last_sodium_values):
                equal = equal & self.xp.array_equal(v, last)
            if bool(equal):
                return
        self._last_sodium_values = tuple(self.xp.array(v) for v in values)

        if values:
            sodium_altitude = sodium_altitude.value * self._laser_launch_tel.airmass
            sodium_intensity = sodium_intensity.value
        else:
            sodium_altitude = None
            sodium_intensity = None

        self._kernelobj.prepare_for_sh(
            sodium_altitude=sodium_altitude,
            sodium_intensity=sodium_intensity,
            current_time=self.current_time
        )


    def trigger_code(self):
        """
        Compute the SH focal plane image, one row of subapertures at a time.
        Processing the whole lenslet array at once would be faster, but the
        oversampled field can reach 8k x 8k pixels: memory usage is as
        important as speed here, so no full-frame temporaries must be added.

        The input field is first interpolated to the oversampled resolution.
        This is done here rather than in prepare_trigger(), so that it is part
        of the CUDA graph, and so that SH objects with the same geometry can
        share the interpolated field (use_out_ef_cache=True).

        Then, for each row of dimx subapertures:

        1. the row of the oversampled electric field, viewed as a (dimx, n, n)
           subap cube, is multiplied by the half-pixel tilt and written into
           the top-left corner of the zero-padded cube _wf3;
        2. a batched 2D FFT gives the focal plane field of each subap;
        3. without a convolution kernel, |FFT|^2 is cut to the sensor FoV,
           multiplied by the focal plane mask and written directly into the
           row of _psfimage, all in a single fused kernel. With a kernel (LGS),
           |FFT|^2 is first convolved with the subap kernels in Fourier space.

        Finally, _psfimage is rebinned to the CCD pixels with toccd().
        The flux normalization is done in post_trigger().

        The CUDA graph is captured at the first trigger(), because the
        interpolation needs the pupil, and captured again if the interpolation
        parameters are changed with update_interpolator_parameters().

        Main performance points:

        - no fftshift: without kernel, it is folded into _tltf as a
          (-1)^(m+n) checkerboard (see _calc_geometry());
        - the mask is only applied to the pixels kept by the FoV cut, and is
          skipped when it is all ones there;
        - the kernel convolution uses rfft2/irfft2, since both the PSF and
          the kernels are real, and the kernels are stored as half spectra;
        - all views on the preallocated buffers are built once in setup().
        """
        xp = self.xp
        dimx = self._lenslet.dimx
        rows = self.subap_rows_slice
        n = self._ovs_np_sub

        # Interpolation of the input field, if needed
        with tracer('interpolation', self):
            self.ef_interpolator.interpolate()
        wf1 = self.ef_interpolator.interpolated_ef()

        for i, psfimage_view in zip(range(rows.start, rows.stop), self._psfimage_views):

            # Extract 2D subap row
            wf1.ef_at_lambda(self.wavelength_in_nm,
                             slicey=np.s_[i * n: (i + 1) * n],
                             slicex=np.s_[:],
                             out=self.ef_row)

            # Insert tilted subaps into the padded array
            xp.multiply(self._subap_cube_view, self._tltf, out=self._wf3_view)

            fp4 = xp.fft.fft2(self._wf3, axes=(1, 2))

            if self._kernelobj is None:
                fp4_cut = fp4[:, self._cut_slice, self._cut_slice]
                if self._apply_mask:
                    abs2_masked(fp4_cut, self._fp_mask_cut, psfimage_view, xp=xp)
                else:
                    abs2(fp4_cut, psfimage_view, xp=xp)
            else:
                abs2(fp4, self.psf_shifted, xp=xp)

                # Full resolution kernel (real in direct space: ConvolutionKernel
                # stores only the half spectrum used by rfft2/irfft2)
                subap_kern_fft = self._kernelobj.kernels[i * dimx: (i + 1) * dimx]
                psf_fft = xp.fft.rfft2(self.psf_shifted)
                psf_fft *= subap_kern_fft
                psf = xp.fft.irfft2(psf_fft, s=(self._fft_size, self._fft_size), norm='forward')

                psf_cut = psf[:, self._cut_slice, self._cut_slice]
                if self._apply_mask:
                    xp.multiply(psf_cut, self._fp_mask_cut, out=psfimage_view)
                else:
                    psfimage_view[:] = psf_cut

        with tracer('toccd', self):
            # set_total=0: no normalization here, it is done in post_trigger()
            self._out_i.i[:] = toccd(self._psfimage, (self._ccd_side, self._ccd_side),
                                     set_total=0, xp=xp)


    def post_trigger(self):
        super().post_trigger()

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i.i *= phot / self._out_i.i.sum()
        self._out_i.generation_time = self.current_time

        debug_figures = False
        if debug_figures:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self._out_i.i, cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Intensity')
            plt.show()

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']

        self._set_in_ef(in_ef)
        self._calc_geometry(in_ef)

        # Use the integer size: int(size * _fov_ovs) can truncate to size - 1
        shape_ovs = (self._ovs_ef_size, self._ovs_ef_size)

        self.ef_interpolator = EFInterpolator(
            in_ef,
            shape_ovs,
            rotAnglePhInDeg=self._rotAnglePhInDeg,
            xShiftPhInPixel=self._xShiftPhInPixel,
            yShiftPhInPixel=self._yShiftPhInPixel,
            mask_threshold=self._mask_threshold,
            use_out_ef_cache=True,  # the interpolated field is computed and used in trigger_code(),
                                    # so SH objects with the same geometry can share it
                                    # It also needs the allow_parallel=False flag in build_stream()
                                    # to avoid race conditions on the cache.
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        if self.subap_rows_slice is None:
            self.subap_rows_slice = slice(0, self._lenslet.dimy)

        n = self._ovs_np_sub
        dimx = self._lenslet.dimx
        cutsize = self._cutsize
        rows = range(self.subap_rows_slice.start, self.subap_rows_slice.stop)

        # Electric field of a single row of subaps. Only one row at a time is
        # computed, because the whole oversampled field can be very large (8k x 8k)
        self.ef_row = self._zeros_common((n, self._ovs_ef_size), dtype=self.complex_dtype)

        # |FFT|^2 before convolution, only needed by the kernel path
        if self._kernelobj is not None:
            self.psf_shifted = self._zeros_common((dimx, self._fft_size, self._fft_size),
                                                  dtype=self.dtype)

        # Views used by trigger_code(), built once since the buffers never change
        self._wf3_view = self._wf3[:, :n, :n]

        # The field row, as a (dimx, n, n) subap cube
        self._subap_cube_view = self.ef_row.reshape(n, dimx, n).swapaxes(0, 1)

        # Each row of _psfimage, as a (dimx, cutsize, cutsize) subap cube
        self._psfimage_views = [self._psfimage[i * cutsize: (i + 1) * cutsize]
                                .reshape(cutsize, dimx, cutsize).swapaxes(0, 1)
                                for i in rows]

        # Assert that our views are actually views and not temporary allocations
        for view in [self._wf3_view, self._subap_cube_view] + self._psfimage_views:
            assert view.base is not None

        # The CUDA graph is captured at the first trigger(), see prepare_trigger()
        super().build_stream(allow_parallel=False, capture=False)

    def update_interpolator_parameters(self, xShiftPhInPixel=None, yShiftPhInPixel=None,
                                       rotAnglePhInDeg=None, magnification=None):
        '''
        Change the misregistration parameters of the input field interpolation.

        The interpolation is part of the CUDA graph, with its parameters
        frozen at capture time, so the graph is invalidated and captured
        again at the next trigger(). Always use this method instead of
        calling ef_interpolator.update_parameters() directly.

        Parameters are the same as EFInterpolator.update_parameters();
        the ones set to None are left unchanged.
        '''
        self.ef_interpolator.update_parameters(xShiftPhInPixel=xShiftPhInPixel,
                                               yShiftPhInPixel=yShiftPhInPixel,
                                               rotAnglePhInDeg=rotAnglePhInDeg,
                                               magnification=magnification)
        self.invalidate_graph()

    def _get_tlt_f(self, p, c):
        '''
        Half-pixel tilt
        '''
        iu = complex(0, 1)
        xx, yy = self.xp.meshgrid(self.xp.arange(-p // 2, p // 2), self.xp.arange(-p // 2, p // 2))
        tlt_g = xx + yy
        tlt_f = self.xp.exp(-2 * self.xp.pi * iu * tlt_g / (2 * (p + c)), dtype=self.complex_dtype)
        return tlt_f
