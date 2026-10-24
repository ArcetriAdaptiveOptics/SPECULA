import warnings

import numpy as np

from specula import fuse, RAD2ASEC, cpuArray
from specula.tracing import tracer
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.pixel_integration import pixel_integration_matrix, integrate_pixels
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


def kernel_sampling_factor(sample, band_edge, fwhm, tol=1e-6, max_factor=16):
    """
    Smallest integer factor r (up to max_factor) by which the sampling `sample`
    [arcsec] of a Gaussian kernel of the given FWHM [arcsec] must be refined, so
    that the Fourier coefficients of the sampled (and unit-sum normalized) kernel
    differ from the true ones by less than tol at all frequencies up to
    `band_edge` [1/arcsec], i.e. in the band of the PSF it is convolved with.

    Sampling at s / r makes the spectrum periodic with period r / s: the error
    is the sum of the aliased copies. It is small both for kernels much wider
    than the sample (well sampled) and much narrower (close to a delta, whose
    spectrum is flat in the band), and largest in between.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    khat = lambda f: np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
    f = np.linspace(0, band_edge, 65)
    m = np.arange(-64, 65)
    for r in range(1, max_factor + 1):
        period = r / sample
        sampled = khat(f[:, None] - m[None, :] * period).sum(axis=1) / khat(m * period).sum()
        if np.max(np.abs(sampled - khat(f))) < tol:
            return r
    return max_factor


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
        Square field stop of side subap_wanted_fov in the focal plane. Only square
        stops are supported (the pixel integration is separable): must be True.
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
        Deprecated and ignored: the internal FFT sampling is no longer a choice
        (see the Notes).
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
        Sensor pixel scale actually simulated, set by setup(). Always equal to
        sensor_pxscale.
    subap_real_fov_arcsec : float [arcsec]
        Subaperture FoV actually simulated (subap_npx * sensor_pxscale), set by setup().

    Notes
    -----
    Each detector pixel is the exact integral of the subaperture PSF over a pixel of
    sensor_pxscale. The PSF of a subaperture field of n pixels is band limited, so
    its samples from an FFT with 2 * n points (zero padding by 2) determine it
    everywhere, and its integral over any pixel is a weighted sum of them
    (see pixel_integration_matrix()). No finer FFT sampling, FoV rounding or
    rebinning is needed, and the pixel scale is exact.
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
        if not squaremask:
            raise ValueError('SH: only square field stops are supported (squaremask=True)')
        if set_fov_res_to_turbpxsc:
            warnings.warn('SH: set_fov_res_to_turbpxsc is deprecated and ignored: '
                          'the FFT sampling is fixed by the exact pixel integration',
                          FutureWarning, stacklevel=2)
        self.wavelength_in_nm = wavelengthInNm
        self.subap_wanted_fov = subap_wanted_fov
        self.subap_on_diameter = subap_on_diameter
        self._lenslet = Lenslet(self.subap_on_diameter, target_device_idx=target_device_idx)
        self._subap_wanted_fov_rad = self.subap_wanted_fov / RAD2ASEC
        self._sensor_pxscale_arcsec = sensor_pxscale
        self._subap_npx = subap_npx
        self._fov_ovs_coeff = fov_ovs_coeff
        self._fov_resolution_arcsec = None
        self._fft_period_arcsec = None
        self.sensor_pxscale_effective = None
        self.subap_real_fov_arcsec = None
        self._rotAnglePhInDeg = rotAnglePhInDeg
        self._xShiftPhInPixel = xShiftPhInPixel
        self._yShiftPhInPixel = yShiftPhInPixel
        self._laser_launch_tel = laser_launch_tel
        self._pix_int_matrix = None
        self._pix_int_matrix_kernel = None
        self._intensity = None
        self._int_tmp = None
        self._int_img = None
        self._out_views = None
        self.data_dir = data_dir
        self._fft_size = 0
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        self.ef_row = None
        self.ef_interpolator = None
        self._ovs_np_sub = None
        self._ovs_ef_size = None
        self._wf3 = None
        self._wf3_view = None
        self._subap_cube_view = None
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

        Sets _fov_ovs, _ovs_ef_size, _ovs_np_sub, _fft_size, _fft_period_arcsec,
        _fov_resolution_arcsec, sensor_pxscale_effective and subap_real_fov_arcsec.
        '''
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.size[0]
        sensor_pxscale = self._sensor_pxscale_arcsec
        self.sensor_pxscale_effective = sensor_pxscale
        self.subap_real_fov_arcsec = sensor_pxscale * self._subap_npx

        # Oversampling of the electric field. We take the maximum of three constraints:
        # - 1.0: do not downsample (loss of quality);
        # - the FFT period, lambda / pixel pitch of the oversampled field, must cover
        #   the sensor subaperture FoV;
        # - fov_ovs_coeff: margin against the wrapping of the periodic PSF.
        native_period = self.wavelength_in_nm * 1e-9 / in_ef.pixel_pitch * RAD2ASEC
        needed_ovs = max(1.0, self.subap_real_fov_arcsec / native_period, self._fov_ovs_coeff)

        # The oversampled size is rounded up to a multiple of 2 * n_lenses, so that
        # each subaperture has an integer and even number of pixels
        modulus = 2 * n_lenses
        self._ovs_ef_size = int(np.ceil(ef_size * needed_ovs / modulus)) * modulus
        self._fov_ovs = self._ovs_ef_size / ef_size
        self._ovs_np_sub = self._ovs_ef_size // n_lenses

        # Zero padding by 2: enough for the band-limited PSF (see pixel_integration_matrix())
        self._fft_size = 2 * self._ovs_np_sub
        self._fft_period_arcsec = native_period * self._fov_ovs
        # The LGS convolution kernels are not band limited: sampled on the FFT grid,
        # their Fourier coefficients in the PSF band can be wrong by aliasing. The FFT
        # size is multiplied by the smallest integer factor that makes them accurate
        # (any FFT size >= 2 * n is exact for the PSF itself).
        if self._laser_launch_tel is not None and self._laser_launch_tel.spot_size > 0:
            self._fft_size *= kernel_sampling_factor(self._fft_period_arcsec / self._fft_size,
                                                     (self._ovs_np_sub - 1) / self._fft_period_arcsec,
                                                     self._laser_launch_tel.spot_size)
        self._fov_resolution_arcsec = self._fft_period_arcsec / self._fft_size

        self.logger.info('-->     FFT pix. sc. [asec],   {}'.format(self._fov_resolution_arcsec))
        self.logger.info('-->     FFT period [asec],     {}'.format(self._fft_period_arcsec))
        self.logger.info('-->     FoV over sampl.,       {}'.format(self._fov_ovs))
        self.logger.info('-->     FFT size,              {}'.format(self._fft_size))
        self.logger.info('-->     oversampled np_sub,    {}'.format(self._ovs_np_sub))
        self.logger.info('-->     oversampled EF size,   {}'.format(self._ovs_ef_size))

    def _calc_geometry(self, in_ef):
        '''
        Allocate the buffers and compute the arrays used by trigger_code(),
        using the geometry computed by _set_in_ef()
        '''
        n = self._ovs_np_sub
        fft_size = self._fft_size
        dimx = self._lenslet.dimx
        npx = self._subap_npx

        # Pixel integration matrix. The square field stop matters only if it is
        # smaller than the sensor FoV: it then clips the edge pixels.
        stop = self.subap_wanted_fov if self.subap_wanted_fov < self.subap_real_fov_arcsec else None
        g = pixel_integration_matrix(fft_size, self._fft_period_arcsec, self._sensor_pxscale_arcsec,
                                     npx, stop=stop)
        self._pix_int_matrix = self.to_xp(g, dtype=self.dtype)
        # The convolution kernels move the axis from sample 0 to sample fft_size / 2
        # (their impulse response is centred there), so their output uses G rolled by half
        self._pix_int_matrix_kernel = self.to_xp(np.roll(g, fft_size // 2, axis=1), dtype=self.dtype)

        # Padded subaperture cube extracted from full pupil (one row of subapertures,
        # i.e. dimx of them). Only the top-left n x n corner of each subap is ever
        # written, so the zero padding is set here once and never touched again.
        # The corner size is part of the cache key: an SH with the same fft_size
        # but a larger corner would otherwise write into the padding of this one.
        self._wf3 = self._zeros_common((dimx, fft_size, fft_size), dtype=self.complex_dtype,
                                       key_extra=n)
        # |FFT|^2, and the buffers of integrate_pixels()
        self._intensity = self._zeros_common((dimx, fft_size, fft_size), dtype=self.dtype)
        self._int_tmp = self._zeros_common((dimx, npx, fft_size), dtype=self.dtype)
        self._int_img = self._zeros_common((dimx, npx, npx), dtype=self.dtype)

        # FFT pixel scale [rad], computed as in the original code: the kernel file
        # names are a hash of the exact pixel scale value
        fp4_pixel_pitch = self.wavelength_in_nm / 1e9 / (in_ef.pixel_pitch / self._fov_ovs * fft_size)
        self._make_kernel_object(in_ef, fp4_pixel_pitch, fft_size)

    def _make_kernel_object(self, in_ef, fp4_pixel_pitch, fft_size):
        '''Convolution kernel object for a laser launch telescope (LGS), if any'''
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
        # The arrays are small: comparing them on the host is faster than
        # launching several comparison kernels on the device.
        host_values = tuple(cpuArray(v) for v in values)
        if self._last_sodium_values is not None and \
                all(np.array_equal(v, last)
                    for v, last in zip(host_values, self._last_sodium_values)):
            return
        # Copies, because cpuArray() does not copy on the CPU and
        # generators update their output in place
        self._last_sodium_values = tuple(np.array(v) for v in host_values)

        if values:
            sodium_altitude = sodium_altitude.value * self._laser_launch_tel.airmass
            sodium_intensity = sodium_intensity.value
        else:
            sodium_altitude = None
            sodium_intensity = None

        kernels = self._kernelobj.kernels
        self._kernelobj.prepare_for_sh(
            sodium_altitude=sodium_altitude,
            sodium_intensity=sodium_intensity,
            current_time=self.current_time
        )
        # The kernels can be reallocated (e.g. when they start or stop being
        # shared with other SH objects), and the CUDA graph reads them
        # at the address they had at capture time
        if self._kernelobj.kernels is not kernels:
            self.invalidate_graph()


    def trigger_code(self):
        """
        Compute the SH detector image, one row of subapertures at a time.
        Processing the whole lenslet array at once would be faster, but the
        oversampled field can reach 8k x 8k pixels: memory usage is as
        important as speed here, so no full-frame temporaries must be added.

        The input field is first interpolated to the oversampled resolution.
        This is done here rather than in prepare_trigger(), so that it is part
        of the CUDA graph, and so that SH objects with the same geometry can
        share the interpolated field (use_out_ef_cache=True).

        Then, for each row of dimx subapertures:

        1. the row of the oversampled electric field, viewed as a (dimx, n, n)
           subap cube, is written into the top-left corner of the zero-padded
           cube _wf3 (padding by 2, or more with a laser launch telescope);
        2. a batched 2D FFT and |.|^2 give the PSF samples of each subap.
           With a convolution kernel (LGS), they are convolved with the subap
           kernels in Fourier space (rfft2/irfft2, the kernels are stored as
           half spectra);
        3. each detector pixel is the exact integral of the PSF over the pixel,
           G @ PSF @ G.T, written directly into the output image.

        The flux normalization is done in post_trigger().

        The CUDA graph is captured at the first trigger(), because the
        interpolation needs the pupil, and captured again if the interpolation
        parameters are changed with update_interpolator_parameters().
        """
        xp = self.xp
        dimx = self._lenslet.dimx
        rows = self.subap_rows_slice
        n = self._ovs_np_sub

        # Interpolation of the input field, if needed
        with tracer('interpolation', self):
            self.ef_interpolator.interpolate()
        wf1 = self.ef_interpolator.interpolated_ef()

        for i, out_view in zip(range(rows.start, rows.stop), self._out_views):

            # Extract 2D subap row and insert the subaps into the padded array
            wf1.ef_at_lambda(self.wavelength_in_nm,
                             slicey=np.s_[i * n: (i + 1) * n],
                             slicex=np.s_[:],
                             out=self.ef_row)
            self._wf3_view[:] = self._subap_cube_view

            fp4 = xp.fft.fft2(self._wf3, axes=(1, 2))
            abs2(fp4, self._intensity, xp=xp)

            if self._kernelobj is None:
                psf, g = self._intensity, self._pix_int_matrix
            else:
                subap_kern_fft = self._kernelobj.kernels[i * dimx: (i + 1) * dimx]
                psf_fft = xp.fft.rfft2(self._intensity)
                psf_fft *= subap_kern_fft
                psf = xp.fft.irfft2(psf_fft, s=(self._fft_size, self._fft_size), norm='forward')
                g = self._pix_int_matrix_kernel

            # Detector pixels: G @ PSF @ G.T for each subaperture
            out_view[:] = integrate_pixels(g, psf, self._int_tmp, self._int_img, xp)
            # G has oscillating weights: rounding can make dark pixels slightly
            # negative, which a Poisson noise generator would reject
            xp.maximum(out_view, 0, out=out_view)

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
        npx = self._subap_npx
        rows = range(self.subap_rows_slice.start, self.subap_rows_slice.stop)

        # Electric field of a single row of subaps. Only one row at a time is
        # computed, because the whole oversampled field can be very large (8k x 8k)
        self.ef_row = self._zeros_common((n, self._ovs_ef_size), dtype=self.complex_dtype)

        # Views used by trigger_code(), built once since the buffers never change
        self._wf3_view = self._wf3[:, :n, :n]

        # The field row, as a (dimx, n, n) subap cube
        self._subap_cube_view = self.ef_row.reshape(n, dimx, n).swapaxes(0, 1)

        # Each row of the output image, as a (dimx, npx, npx) subap cube
        self._out_views = [self._out_i.i[i * npx: (i + 1) * npx]
                           .reshape(npx, dimx, npx).swapaxes(0, 1)
                           for i in rows]

        # Assert that our views are actually views and not temporary allocations
        for view in [self._wf3_view, self._subap_cube_view] + self._out_views:
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
