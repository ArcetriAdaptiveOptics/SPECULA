import numpy as np

from specula import fuse, show_in_profiler, RAD2ASEC
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.toccd import toccd
from specula.lib.make_mask import make_mask
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj
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


class SH(BaseProcessingObj):

    __zeros_cache = {}

    def _zeros_common(self, *args, **kwargs):
        '''
        Wrapper around self.xp.zeros to enable the reuse cache.
        None of the arrays allocated here should be used in 
        prepare_trigger() or post_trigger().
        '''
        key = (self.target_device_idx, *args, *kwargs.items())
        if key not in self.__zeros_cache:
            self.__zeros_cache[key] = self.xp.zeros(*args, **kwargs)
        return self.__zeros_cache[key]

    def __init__(self,
                 wavelengthInNm: float,
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
        self._sensor_pxscale = sensor_pxscale / RAD2ASEC
        self._subap_npx = subap_npx
        self._fov_ovs_coeff = fov_ovs_coeff
        self._squaremask = squaremask
        self._fov_resolution_arcsec = 0.03 if FoVres30mas else 0
        self._debugOutput = False
        self._noprints = False
        self._rotAnglePhInDeg = rotAnglePhInDeg
        self._xShiftPhInPixel = xShiftPhInPixel
        self._yShiftPhInPixel = yShiftPhInPixel
        self._set_fov_res_to_turbpxsc = set_fov_res_to_turbpxsc
        self._do_not_double_fov_ovs = do_not_double_fov_ovs
        self._laser_launch_tel = laser_launch_tel
        self.data_dir = data_dir
        self._np_sub = 0
        self._fft_size = 0
        self._trigger_geometry_calculated = False
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        # TODO these are fixed but should become parameters
        self._fov_ovs = 1
        self._floatShifts = False

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

    def _set_in_ef(self, in_ef):

        lens = self._lenslet.get(0, 0)
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.size[0]

        self._np_sub = max([1, round((ef_size * lens[2]) / 2.0)])
        if self._np_sub * n_lenses > ef_size:
            self._np_sub -= 1

        # this is the number of pixels per sub-aperture
        np_sub = (ef_size * lens[2]) / 2.0

        sensor_pxscale_arcsec = self._sensor_pxscale * RAD2ASEC
        dSubApInM = np_sub * in_ef.pixel_pitch
        turbulence_pxscale = self.wavelength_in_nm * 1e-9 / dSubApInM * RAD2ASEC
        subap_wanted_fov_arcsec = self.subap_wanted_fov
        subap_real_fov_arcsec = self._sensor_pxscale * self._subap_npx * RAD2ASEC

        if self._fov_resolution_arcsec == 0:
            if not self._noprints:
                print('FoV internal resolution parameter not set.')
            if self._set_fov_res_to_turbpxsc:
                if turbulence_pxscale >= sensor_pxscale_arcsec:
                    raise ValueError('set_fov_res_to_turbpxsc property should be set to one only if turb. pix. sc. is < sensor pix. sc.')
                self._fov_resolution_arcsec = turbulence_pxscale
                if not self._noprints:
                    print('WARNING: set_fov_res_to_turbpxsc property is set.')
                    print('FoV internal resolution parameter will be set to turb. pix. sc.')
            elif turbulence_pxscale < sensor_pxscale_arcsec and sensor_pxscale_arcsec / 2.0 > 0.5:
                self._fov_resolution_arcsec = turbulence_pxscale * 0.5
            else:
                i = 0
                resTry = turbulence_pxscale / (i + 2)
                while resTry >= sensor_pxscale_arcsec:
                    i += 1
                    resTry = turbulence_pxscale / (i + 2)
                iMin = i

                nTry = 10
                resTry = np.zeros(nTry)
                scaleTry = np.zeros(nTry)
                fftScaleTry = np.zeros(nTry)
                subapRealTry = np.zeros(nTry)
                mcmxTry = np.zeros(nTry)

                for i in range(nTry):
                    resTry[i] = turbulence_pxscale / (iMin + i + 2)
                    scaleTry[i] = round(turbulence_pxscale / resTry[i])
                    fftScaleTry[i] = self.wavelength_in_nm / 1e9 * self._lenslet.dimx / (ef_size * in_ef.pixel_pitch * scaleTry[i]) * RAD2ASEC
                    subapRealTry[i] = round(subap_wanted_fov_arcsec / fftScaleTry[i] / 2.0) * 2
                    mcmxTry[i] = np.lcm(int(self._subap_npx), int(subapRealTry[i]))

                # Search for resolution factor with FoV error < 1%
                FoV = subapRealTry * fftScaleTry
                FoVerror = np.abs(FoV - subap_wanted_fov_arcsec) / subap_wanted_fov_arcsec
                idxGood = np.where(FoVerror < 0.02)[0]

                # If no resolution factor gives low error, consider all scale values
                if len(idxGood) == 0:
                    idxGood = np.arange(nTry)

                # Search for index with minimum ratio between M.C.M. and resolution factor
                ratioMcm = mcmxTry[idxGood] / scaleTry[idxGood]
                idxMin = np.argmin(ratioMcm)
                if idxGood[idxMin] != 0 and mcmxTry[idxGood[0]] / mcmxTry[idxGood[idxMin]] > scaleTry[idxGood[idxMin]] / scaleTry[idxGood[0]]:
                    self._fov_resolution_arcsec = resTry[idxGood[idxMin]]
                else:
                    self._fov_resolution_arcsec = resTry[idxGood[0]]

        if not self._noprints:
            print(f'FoV internal resolution parameter set as [arcsec]: {self._fov_resolution_arcsec}')

        # Compute FFT FoV resolution element in arcsec
        scale_ovs = round(turbulence_pxscale / self._fov_resolution_arcsec)

        dTelPaddedInM = ef_size * in_ef.pixel_pitch * scale_ovs
        dSubApPaddedInM = dTelPaddedInM / self._lenslet.dimx
        fft_pxscale_arcsec = self.wavelength_in_nm * 1e-9 / dSubApPaddedInM * RAD2ASEC

        # Compute real FoV
        subap_real_fov_pix = round(subap_real_fov_arcsec / fft_pxscale_arcsec / 2.0) * 2.0
        subap_real_fov_arcsec = subap_real_fov_pix * fft_pxscale_arcsec
        mcmx = np.lcm(int(self._subap_npx), int(subap_real_fov_pix))

        turbulence_fov_pix = int(scale_ovs * np_sub)

        # Avoid increasing the FoV if it's already more than twice the requested one
        if turbulence_fov_pix > 2 * subap_real_fov_pix:
            self._fov_ovs = 1
            if self._fov_ovs_coeff != 0.0:
                self._fov_ovs = self._fov_ovs_coeff
        else:
            ratio = float(subap_real_fov_pix) / float(turbulence_fov_pix)
            np_factor = 1 if abs(np_sub - round(np_sub)) >= 1e-3 else round(np_sub)
            if self._do_not_double_fov_ovs and self._fov_ovs_coeff == 0.0:
                self._fov_ovs_coeff = 1.0
                self._fov_ovs = np.ceil(np_factor * ratio / 2.0) * 2.0 / float(np_factor)
            else:
                if self._fov_ovs_coeff == 0.0:
                    self._fov_ovs_coeff = 2.0
                if ratio < 2:
                    self._fov_ovs = np.ceil(np_factor * self._fov_ovs_coeff) / float(np_factor)
                else:
                    self._fov_ovs = np.ceil(np_factor * ratio * self._fov_ovs_coeff) / float(np_factor)

        self._sensor_pxscale = subap_real_fov_arcsec / self._subap_npx / RAD2ASEC
        self._ovs_np_sub = round(ef_size * self._fov_ovs * lens[2] * 0.5)
        self._fft_size = self._ovs_np_sub * scale_ovs

        if self.verbose:
            print('\n-->     FoV resolution [asec], {}'.format(self._fov_resolution_arcsec))
            print('-->     turb. pix. sc.,        {}'.format(turbulence_pxscale))
            print('-->     sc. over sampl.,       {}'.format(scale_ovs))
            print('-->     FoV over sampl.,       {}'.format(self._fov_ovs))
            print('-->     FFT pix. sc. [asec],   {}'.format(fft_pxscale_arcsec))
            print('-->     no. elements FoV,      {}'.format(subap_real_fov_pix))
            print('-->     FFT size (turb. FoV),  {}'.format(self._fft_size))
            print('-->     L.C.M. for toccd,      {}'.format(mcmx))
            print('-->     oversampled np_sub,    {}'.format(self._ovs_np_sub))


        # Check for valid phase size
        if abs((ef_size * round(self._fov_ovs) * lens[2]) / 2.0 - round((ef_size * round(self._fov_ovs) * lens[2]) / 2.0)) > 1e-4:
            raise ValueError(f'ERROR: interpolated input phase size {ef_size} * {round(self._fov_ovs)} is not divisible by  {self._lenslet.n_lenses} subapertures.')
        elif not self._noprints:
            print(f'GOOD: interpolated input phase size {ef_size} * {round(self._fov_ovs)} is divisible by {self._lenslet.n_lenses} subapertures.')

    def _calc_trigger_geometry(self, in_ef):
        '''
        Calculate the geometry of the SH
        '''

        subap_wanted_fov = self._subap_wanted_fov_rad
        sensor_pxscale = self._sensor_pxscale
        subap_npx = self._subap_npx

        self._xyShiftPhInPixel = np.array([self._xShiftPhInPixel, self._yShiftPhInPixel]) * self._fov_ovs

        if not self._floatShifts:
            self._xyShiftPhInPixel = np.round(self._xyShiftPhInPixel).astype(int)

        ovs_pixel_pitch = in_ef.pixel_pitch / self._fov_ovs

        # Reuse geometry calculated in set_in_ef
        fft_size = self._fft_size

        # Padded subaperture cube extracted from full pupil
        self._wf3 = self._zeros_common((self._lenslet.dimy, fft_size, fft_size), dtype=self.complex_dtype)

        # Focal plane result from FFT
        fp4_pixel_pitch = self.wavelength_in_nm / 1e9 / (ovs_pixel_pitch * fft_size)
        fov_complete = fft_size * fp4_pixel_pitch

        sensor_subap_fov = sensor_pxscale * subap_npx
        fov_cut = fov_complete - sensor_subap_fov

        self._cutpixels = int(np.round(fov_cut / fp4_pixel_pitch) / 2 * 2)
        self._cutsize = fft_size - self._cutpixels
        self._psfimage = self._zeros_common((self._cutsize * self._lenslet.dimx, self._cutsize * self._lenslet.dimy), dtype=self.dtype)
        self._psf_reshaped_2d = self._zeros_common((self._cutsize, self._cutsize * self._lenslet.dimy), dtype=self.dtype)

        # 1/2 Px tilt
        self._tltf = self._get_tlt_f(self._ovs_np_sub, fft_size - self._ovs_np_sub)

        self._fp_mask = make_mask(fft_size, diaratio=subap_wanted_fov / fov_complete, square=self._squaremask, xp=self.xp)

        # set up kernel object
        if self._laser_launch_tel is not None:
            if len(self._laser_launch_tel.tel_pos) == 0:
                self._kernelobj = GaussianConvolutionKernel(dimx = self._lenslet.dimx,
                                                            dimy = self._lenslet.dimy,
                                                            pxscale = fp4_pixel_pitch * RAD2ASEC,
                                                            pupil_size_m = in_ef.pixel_pitch * in_ef.size[0],
                                                            dimension = self._fft_size,
                                                            spot_size = self._laser_launch_tel.spot_size,
                                                            oversampling = 1,
                                                            return_fft = True,
                                                            positive_shift_tt = True,
                                                            data_dir=self.data_dir,
                                                            target_device_idx=self.target_device_idx)
            else:
                if len(self._laser_launch_tel.beacon_tt) != 0:
                    theta = self._laser_launch_tel.beacon_tt
                else:
                    theta = []
                self._kernelobj = ConvolutionKernel(dimx = self._lenslet.dimx,
                                                    dimy = self._lenslet.dimy,
                                                    pxscale = fp4_pixel_pitch * RAD2ASEC,
                                                    pupil_size_m = in_ef.pixel_pitch * in_ef.size[0],
                                                    dimension = self._fft_size,
                                                    launcher_pos = self._laser_launch_tel.tel_pos,
                                                    seeing = 0.0,
                                                    launcher_size = self._laser_launch_tel.spot_size,
                                                    zfocus = self._laser_launch_tel.beacon_focus,
                                                    theta = theta,
                                                    oversampling = 1,
                                                    return_fft = True,
                                                    positive_shift_tt = True,
                                                    data_dir=self.data_dir,
                                                    target_device_idx=self.target_device_idx)
            self._kernel_fn = None
        else:
            self._kernelobj = None

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Interpolation of input array if needed
        with show_in_profiler('interpolation'):
            self.ef_interpolator.interpolate()

        if self._kernelobj is not None:
            self._prepare_kernels()

    def _prepare_kernels(self):
        if len(self._laser_launch_tel.tel_pos) != 0:
            sodium_altitude = self.local_inputs['sodium_altitude']
            sodium_intensity = self.local_inputs['sodium_intensity']
            if sodium_altitude is None or sodium_intensity is None:
                raise ValueError('sodium_altitude and sodium_intensity must be provided')
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

        # Work on SH rows (single-subap code is too inefficient)

        for i in range(self.subap_rows_slice.start, self.subap_rows_slice.stop):

            # Extract 2D subap row
            wf1 = self.ef_interpolator.interpolated_ef()
            wf1.ef_at_lambda(self.wavelength_in_nm,
                             slicey=np.s_[i * self._ovs_np_sub: (i+1) * self._ovs_np_sub],
                             slicex=np.s_[:],
                             out=self.ef_row)

            # Reshape to subap cube (nsubap, npix, npix)
            subap_cube_view = self.ef_row.reshape(self._ovs_np_sub, self._lenslet.dimy, self._ovs_np_sub).swapaxes(0, 1)

            # Insert into padded array
            self._wf3[:, :self._ovs_np_sub, :self._ovs_np_sub] = subap_cube_view * self._tltf[self.xp.newaxis, :, :]

            fp4 = self.xp.fft.fft2(self._wf3, axes=(1, 2))
            abs2(fp4, self.psf_shifted, xp=self.xp)

            # Full resolution kernel
            if self._kernelobj is not None:
                first = i * self._lenslet.dimy
                last = (i + 1) * self._lenslet.dimy
                subap_kern_fft = self._kernelobj.kernels[first:last, :, :]

                psf_fft = self.xp.fft.fft2(self.psf_shifted)
                psf_fft *= subap_kern_fft

                self._scipy_ifft2(psf_fft, overwrite_x=True, norm='forward')
                self.psf[:] = psf_fft.real

                # Assert that our views are actually views and not temporary allocations
                assert subap_kern_fft.base is not None
            else:
                self.psf[:] = self.xp.fft.fftshift(self.psf_shifted, axes=(1, 2))

            # Apply focal plane mask
            self.psf *= self._fp_mask[self.xp.newaxis, :, :]

            cutsize = self._cutsize
            cutpixels = self._cutpixels

            # FoV cut on each subap.
            psf_cut_view = self.psf[:, cutpixels // 2: -cutpixels // 2, cutpixels // 2: -cutpixels // 2]

            # Go back from a subap cube to a 2D frame row.
            # This reshape is too complicated to produce a view,
            # so we use a preallocated array
            self._psf_reshaped_2d[:] = psf_cut_view.swapaxes(0, 1).reshape(-1, self._lenslet.dimy * cutsize)

            # Insert 2D frame row into overall PSF image
            self._psfimage[i * cutsize: (i+1) * cutsize, :] = self._psf_reshaped_2d

            # Assert that our views are actually views and not temporary allocations
            assert psf_cut_view.base is not None
            assert subap_cube_view.base is not None

        with show_in_profiler('toccd'):
            self._out_i.i[:] = toccd(self._psfimage, (self._ccd_side, self._ccd_side), xp=self.xp)


    def post_trigger(self):
        super().post_trigger()

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
       # print(self.name, f'{in_ef.S0=} {self._out_i.i.sum()=}')
        self._out_i.i *= phot / self._out_i.i.sum()
        # self._out_i.i = self.xp.nan_to_num(self._out_i.i, copy=False)
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
        self._calc_trigger_geometry(in_ef)

        fov_oversample = self._fov_ovs
        shape_ovs = (int(in_ef.size[0] * fov_oversample), int(in_ef.size[1] * fov_oversample))

        self.ef_interpolator = EFInterpolator(
            in_ef,
            shape_ovs,
            rotAnglePhInDeg=self._rotAnglePhInDeg,
            xShiftPhInPixel=self._xShiftPhInPixel,
            yShiftPhInPixel=self._yShiftPhInPixel,
            mask_threshold=self._mask_threshold,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        ef_whole_size = int(in_ef.size[0] * self._fov_ovs)
        self.ef_row = self._zeros_common((self._ovs_np_sub, ef_whole_size), dtype=self.complex_dtype)
        self.psf = self._zeros_common((self._lenslet.dimy, self._fft_size, self._fft_size), dtype=self.dtype)
        self.psf_shifted = self._zeros_common((self._lenslet.dimy, self._fft_size, self._fft_size), dtype=self.dtype)

        if self.subap_rows_slice is None:
            self.subap_rows_slice = slice(0, self._lenslet.dimy)


        super().build_stream(allow_parallel=False)

    def _get_tlt_f(self, p, c):
        '''
        Half-pixel tilt
        '''
        iu = complex(0, 1)
        xx, yy = self.xp.meshgrid(self.xp.arange(-p // 2, p // 2), self.xp.arange(-p // 2, p // 2))
        tlt_g = xx + yy
        tlt_f = self.xp.exp(-2 * self.xp.pi * iu * tlt_g / (2 * (p + c)), dtype=self.complex_dtype)
        return tlt_f
