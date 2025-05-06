import numpy as np

from specula import cpuArray, fuse, show_in_profiler, RAD2ASEC
from specula.lib.extrapolate_edge_pixel import extrapolate_edge_pixel
from specula.lib.extrapolate_edge_pixel_mat_define import extrapolate_edge_pixel_mat_define
from specula.lib.toccd import toccd
from specula.lib.interp2d import Interp2D
from specula.lib.make_mask import make_mask
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.lenslet import Lenslet
from specula.base_value import BaseValue
from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel
from specula.data_objects.convolution_kernel import ConvolutionKernel

import os       

@fuse(kernel_name='abs2')
def abs2(u_fp, out, xp):
     out[:] = xp.real(u_fp * xp.conj(u_fp))


class SH(BaseProcessingObj):

    __zeros_cache = {}

    def _zeros_common(self, *args, **kwargs):
        '''
        Wrapper around self.xp.zeros to enable the reuse cache
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
                 aXShiftPhInPixel: float = 0,
                 aYShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 aRotAnglePhInDeg: float = 0,
                 do_not_double_fov_ovs: bool = False,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: dict = None,
                 target_device_idx: int = None, 
                 precision: int = None,
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._wavelengthInNm = wavelengthInNm
        self._lenslet = Lenslet(subap_on_diameter)
        self._subap_wanted_fov = subap_wanted_fov / RAD2ASEC
        self._sensor_pxscale = sensor_pxscale / RAD2ASEC
        self._subap_npx = subap_npx
        self._fov_ovs_coeff = fov_ovs_coeff
        self._squaremask = squaremask
        self._fov_resolution_arcsec = 0.03 if FoVres30mas else 0
        self._debugOutput = False
        self._noprints = False
        self._rotAnglePhInDeg = rotAnglePhInDeg
        self._aRotAnglePhInDeg = aRotAnglePhInDeg  # TODO if intended to change, call again calc_trigger_geometry()
        self._xShiftPhInPixel = xShiftPhInPixel
        self._yShiftPhInPixel = yShiftPhInPixel
        self._aXShiftPhInPixel = aXShiftPhInPixel  # Same TODO as above
        self._aYShiftPhInPixel = aYShiftPhInPixel
        self._set_fov_res_to_turbpxsc = set_fov_res_to_turbpxsc
        self._do_not_double_fov_ovs = do_not_double_fov_ovs
        # first item of laser_launch_tel_dict is the good one
        self._laser_launch_tel = laser_launch_tel
        self._np_sub = 0
        self._fft_size = 0
        self._trigger_geometry_calculated = False
        self._extrapol_mat1 = None
        self._extrapol_mat2 = None
        self._idx_1pix = None
        self._idx_2pix = None
        self._do_interpolation = None

        # TODO these are fixed but should become parameters
        self._fov_ovs = 1
        self._floatShifts = False

        self._ccd_side = self._subap_npx * self._lenslet.n_lenses
        self._out_i = Intensity(self._ccd_side, self._ccd_side, precision=self.precision, target_device_idx=self.target_device_idx)

        self.interp = None

        # Set up the kernel object
        if self._laser_launch_tel is not None:
            if len(self._laser_launch_tel.tel_pos) == 0:                        
                self._kernelobj = GaussianConvolutionKernel(target_device_idx=self.target_device_idx)
                self._kernelobj.spot_size = self._laser_launch_tel.spot_size
            else:
                self._kernelobj = ConvolutionKernel(target_device_idx=self.target_device_idx)
                self._kernelobj.launcher_pos = self._laser_launch_tel.tel_pos
                self._kernelobj.seeing = 0.0
                self._kernelobj.launcher_size = self._laser_launch_tel.spot_size
                self._kernelobj.zfocus = self._laser_launch_tel.beacon_focus
                if len(self._laser_launch_tel.beacon_tt) != 0:
                    self._kernelobj.lgs_tt = self._laser_launch_tel.beacon_tt
            self._kernelobj.oversampling = 1
            self._kernelobj.return_fft = True
            self._kernelobj.positive_shift_tt = True
            self._kernel_fn = None
        else:
            self._kernelobj = None

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self._out_i
        # These are the inputs for the sodium layer used in the kernel
        self.inputs['sodium_altitude'] = InputValue(type=BaseValue, optional=True)
        self.inputs['sodium_intensity'] = InputValue(type=BaseValue, optional=True)

    def set_in_ef(self, in_ef):

        lens = self._lenslet.get(0, 0)
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.size[0]

        self._np_sub = max([1, round((ef_size * lens[2]) / 2.0)])
        if self._np_sub * n_lenses > ef_size:
            self._np_sub -= 1

        np_sub = (ef_size * lens[2]) / 2.0

        sensor_pxscale_arcsec = self._sensor_pxscale * RAD2ASEC
        dSubApInM = np_sub * in_ef.pixel_pitch
        turbulence_pxscale = self._wavelengthInNm * 1e-9 / dSubApInM * RAD2ASEC
        subap_wanted_fov_arcsec = self._subap_wanted_fov * RAD2ASEC
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
                    fftScaleTry[i] = self._wavelengthInNm / 1e9 * self._lenslet.dimx / (ef_size * in_ef.pixel_pitch * scaleTry[i]) * RAD2ASEC
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
        fft_pxscale_arcsec = self._wavelengthInNm * 1e-9 / dSubApPaddedInM * RAD2ASEC

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
            np_factor = 1 if abs(np_sub - int(np_sub)) < 1e-3 else int(np_sub)
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
        self._ovs_np_sub = int(ef_size * self._fov_ovs * lens[2] * 0.5)
        self._fft_size = self._ovs_np_sub * scale_ovs

        if self._verbose:
            print('\n-->     FoV resolution [asec], {}'.format(self._fov_resolution_arcsec))
            print('-->     turb. pix. sc.,        {}'.format(turbulence_pxscale))
            print('-->     sc. over sampl.,       {}'.format(scale_ovs))
            print('-->     FoV over sampl.,       {}'.format(self._fov_ovs))
            print('-->     FFT pix. sc. [asec],   {}'.format(fft_pxscale_arcsec))
            print('-->     no. elements FoV,      {}'.format(subap_real_fov_pix))
            print('-->     FFT size (turb. FoV),  {}'.format(self._fft_size))
            print('-->     L.C.M. for toccd,      {}'.format(mcmx))


        # Check for valid phase size
        if abs((ef_size * round(self._fov_ovs) * lens[2]) / 2.0 - round((ef_size * round(self._fov_ovs) * lens[2]) / 2.0)) > 1e-4:
            raise ValueError(f'ERROR: interpolated input phase size {ef_size} * {round(self._fov_ovs)} is not divisible by  {self._lenslet.n_lenses} subapertures.')
        elif not self._noprints:
            print(f'GOOD: interpolated input phase size {ef_size} * {round(self._fov_ovs)} is divisible by {self._lenslet.n_lenses} subapertures.')

    def calc_trigger_geometry(self, in_ef):
        '''
        Calculate the geometry of the SH
        '''

        subap_wanted_fov = self._subap_wanted_fov
        sensor_pxscale = self._sensor_pxscale
        subap_npx = self._subap_npx

        self._rotAnglePhInDeg = self._rotAnglePhInDeg + self._aRotAnglePhInDeg
        self._xyShiftPhInPixel = np.array([self._xShiftPhInPixel + self._aXShiftPhInPixel, self._yShiftPhInPixel + self._aYShiftPhInPixel]) * self._fov_ovs

        if not self._floatShifts:
            self._xyShiftPhInPixel = np.round(self._xyShiftPhInPixel).astype(int)

        if self._fov_ovs != 1 or self._rotAnglePhInDeg != 0 or np.sum(np.abs(self._xyShiftPhInPixel)) != 0:
            M0 = in_ef.size[0] * self._fov_ovs
            M1 = in_ef.size[1] * self._fov_ovs
            wf1 = ElectricField(M0, M1, in_ef.pixel_pitch / self._fov_ovs, target_device_idx=self.target_device_idx)
        else:
            wf1 = in_ef

        # Reuse geometry calculated in set_in_ef
        fft_size = self._fft_size

        # Padded subaperture cube extracted from full pupil
        self._wf3 = self._zeros_common((self._lenslet.dimy, fft_size, fft_size), dtype=self.complex_dtype)

        # Focal plane result from FFT
        fp4_pixel_pitch = self._wavelengthInNm / 1e9 / (wf1.pixel_pitch * fft_size)
        fov_complete = fft_size * fp4_pixel_pitch

        sensor_subap_fov = sensor_pxscale * subap_npx
        fov_cut = fov_complete - sensor_subap_fov

        self._cutpixels = int(np.round(fov_cut / fp4_pixel_pitch) / 2 * 2)
        self._cutsize = fft_size - self._cutpixels
        self._psfimage = self._zeros_common((self._cutsize * self._lenslet.dimx, self._cutsize * self._lenslet.dimy), dtype=self.dtype)
        self._psf_reshaped_2d = self._zeros_common((self._cutsize, self._cutsize * self._lenslet.dimy), dtype=self.dtype)

        # 1/2 Px tilt
        self._tltf = self.get_tlt_f(self._ovs_np_sub, fft_size - self._ovs_np_sub)

        self._fp_mask = make_mask(fft_size, diaratio=subap_wanted_fov / fov_complete, square=self._squaremask, xp=self.xp)

        # Remember a few things
        self.in_ef = in_ef
        self._wf1 = wf1

        # update kernel parameters
        if self._kernelobj is not None:
            print('Updating kernel parameters')
            self._kernelobj.dimx = self._lenslet.dimx
            self._kernelobj.dimy = self._lenslet.dimy
            self._kernelobj.pxscale = fp4_pixel_pitch * RAD2ASEC
            self._kernelobj.pupil_size_m = in_ef.pixel_pitch * in_ef.size[0]
            self._kernelobj.dimension = self._fft_size

    def prepare_trigger(self, t):
        super().prepare_trigger(t)        
        
        if self._kernelobj is not None:
            if len(self._laser_launch_tel.tel_pos) != 0:
                sodium_altitude = self.local_inputs['sodium_altitude']
                sodium_intensity = self.local_inputs['sodium_intensity']
                if sodium_altitude is None or sodium_intensity is None:
                    raise ValueError('sodium_altitude and sodium_intensity must be provided')
                self._kernelobj.zlayer = sodium_altitude.value
                self._kernelobj.zprofile = sodium_intensity.value

            # Get the kernel filename hash based on current parameters
            new_kernel_fn = self._kernelobj.build()

            # Only reload or recalculate if the kernel has changed
            if new_kernel_fn != self._kernel_fn:
                self._kernel_fn = new_kernel_fn  # Update the stored kernel filename

                if os.path.exists(self._kernel_fn):
                    print(f"Loading kernel from {self._kernel_fn}")
                    if len(self._laser_launch_tel.tel_pos) == 0:
                        self._kernelobj = GaussianConvolutionKernel.restore(self._kernel_fn,
                                                                            kernel_obj=self._kernelobj,
                                                                            target_device_idx=self.target_device_idx,
                                                                            return_fft=True)
                    else:
                        self._kernelobj = ConvolutionKernel.restore(self._kernel_fn, 
                                                                    kernel_obj=self._kernelobj,
                                                                    target_device_idx=self.target_device_idx,
                                                                    return_fft=True)
                else:
                    print('Calculating kernel...')
                    self._kernelobj.calculate_lgs_map()
                    self._kernelobj.save(self._kernel_fn)
                    print('Done')
            else:
                # Kernel hasn't changed, no need to reload or recalculate
                print("Kernel unchanged, using cached version")

            self._kernelobj.generation_time = self.current_time
        
        

    def trigger_code(self):

        # Interpolation of input array if needed
        with show_in_profiler('interpolation'):

            if self._do_interpolation:
                phaseInNmNew = extrapolate_edge_pixel(self.in_ef.phaseInNm, self._extrapol_mat1, self._extrapol_mat2, self._idx_1pix, self._idx_2pix, xp=self.xp)
                self.interp.interpolate(self.in_ef.A, out=self._wf1.A)
                self.interp.interpolate(phaseInNmNew, out=self._wf1.phaseInNm)
            else:
                # self._wf1 already set to in_ef
                pass

        with show_in_profiler('ef_at_lambda'):
            self._wf1.ef_at_lambda(self._wavelengthInNm, out=self.ef_whole)

        # Work on SH rows (single-subap code is too inefficient)

        for i in range(self._lenslet.dimx):

            # Extract 2D subap row
            ef_subap_view = self.ef_whole[i * self._ovs_np_sub: (i+1) * self._ovs_np_sub, :]

            # Reshape to subap cube (nsubap, npix, npix)
            subap_cube_view = ef_subap_view.reshape(self._ovs_np_sub, self._lenslet.dimy, self._ovs_np_sub).swapaxes(0, 1)

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
                self.psf = psf_fft.real

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
            assert ef_subap_view.base is not None
            assert subap_cube_view.base is not None

        with show_in_profiler('toccd'):
            self._out_i.i[:] = toccd(self._psfimage, (self._ccd_side, self._ccd_side), xp=self.xp)

    def post_trigger(self):
        super().post_trigger()

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i.i *= (phot / self._out_i.i.sum())
        self._out_i.generation_time = self.current_time

        debug_figures = False
        if debug_figures:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self._out_i.i, cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Intensity')
            plt.show()

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)

        in_ef = self.inputs['in_ef'].get(target_device_idx=self.target_device_idx)
        
        self.set_in_ef(in_ef)
        self.calc_trigger_geometry(in_ef)

        fov_oversample = self._fov_ovs
        shape_ovs = (int(in_ef.size[0] * fov_oversample), int(in_ef.size[1] * fov_oversample))

        self.interp = Interp2D(in_ef.size, shape_ovs, self._rotAnglePhInDeg, self._xyShiftPhInPixel[0], self._xyShiftPhInPixel[1], dtype=self.dtype, xp=self.xp)

        if fov_oversample != 1 or self._rotAnglePhInDeg != 0 or np.sum(np.abs([self._xyShiftPhInPixel])) != 0:
            sum_1pix_extra, sum_2pix_extra, idx_1pix, idx_2pix = extrapolate_edge_pixel_mat_define(cpuArray(in_ef.A), do_ext_2_pix=True)
            self._extrapol_mat1 = self.xp.array(sum_1pix_extra)
            self._extrapol_mat2 = self.xp.array(sum_2pix_extra)
            self._idx_1pix = tuple(map(self.xp.array, idx_1pix))
            self._idx_2pix = tuple(map(self.xp.array, idx_2pix))
            self._do_interpolation = True
        else:
            self._do_interpolation = False

        ef_whole_size = int(in_ef.size[0] * self._fov_ovs)
        self.ef_whole = self._zeros_common((ef_whole_size, ef_whole_size), dtype=self.complex_dtype)
        self.psf = self._zeros_common((self._lenslet.dimy, self._fft_size, self._fft_size), dtype=self.dtype)
        self.psf_shifted = self._zeros_common((self._lenslet.dimy, self._fft_size, self._fft_size), dtype=self.dtype)

        super().build_stream(allow_parallel=False)

    def get_tlt_f(self, p, c):
        '''
        Half-pixel tilt
        '''
        iu = complex(0, 1)
        xx, yy = self.xp.meshgrid(self.xp.arange(-p // 2, p // 2), self.xp.arange(-p // 2, p // 2))
        tlt_g = xx + yy
        tlt_f = self.xp.exp(-2 * self.xp.pi * iu * tlt_g / (2 * (p + c)), dtype=self.complex_dtype)
        return tlt_f
