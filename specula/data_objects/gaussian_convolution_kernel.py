from specula.base_data_obj import BaseDataObj
from specula.data_objects.convolution_kernel import ConvolutionKernel, lgs_map_sh
from specula import cpuArray

import numpy as np

from astropy.io import fits

class GaussianConvolutionKernel(ConvolutionKernel):
    """
    Kernel processing object for Gaussian kernels.
    """
    
    def __init__(self,
                 convolGaussSpotSize :int,
                 dimx: int,
                 dimy: int,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(dimx, dimy, target_device_idx=target_device_idx, precision=precision)        
        self.spotsize = convolGaussSpotSize

    def build(self):
        """
        Recalculates the Gaussian kernel based on current settings.
        """
        self.orig_dimx = self.dimx
        self.dimx = max(self.dimx, 2)        
        self.lgs_tt = [-0.5, -0.5] if not self.positiveShiftTT else [0.5, 0.5]
        self.lgs_tt = [x * self.pxscale for x in self.lgs_tt]        
        self.hash_arr = [
            self.dimx, self.pupil_size_m, 90e3, self.spotsize,
            self.pxscale, self.dimension, 3, self.lgs_tt, [0, 0, 0], [90e3], [1.0]
        ]
        return 'ConvolutionKernel' + self.generate_hash()        

    def calculate_lgs_map(self):
        real_kernels = lgs_map_sh(
            self.dimx, self.pupil_size_m, 0, 90e3, [0], profz=[1.0], fwhmb=self.spotsize, ps=self.pxscale,
            ssp=self.dimension, overs=1, theta=self.lgs_tt, xp=self.xp )
        self.kernels = self.xp.zeros_like(real_kernels, dtype=self.complex_dtype)
        for i in range(self.dimx):
            for j in range(self.dimy):
                subap_kern = self.xp.array(real_kernels[j * self.dimx + i, :, :])
                subap_kern /= self.xp.sum(subap_kern)
                subap_kern_fft = self.xp.fft.ifft2(subap_kern)
                self.kernels[j * self.dimx + i, :, :] = subap_kern_fft
        
    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['SPOTSIZE'] = self.spotsize
        hdr['DIMX'] = self.dimx
        hdr['DIMY'] = self.dimy
        hdr['PXSCALE'] = self.pxscale
        hdr['DIMENSION'] = self.dimension
        hdr['OVERSAMPLING'] = self.oversampling
        hdr['POSITIVESHIFTTT'] = self.positiveShiftTT
        hdu1 = fits.PrimaryHDU(np.real(cpuArray(self.kernels)), header=hdr)
        hdu2 = fits.ImageHDU(np.imag(cpuArray(self.kernels)), header=hdr)
        hdul = fits.HDUList([hdu1, hdu2])
        hdul.writeto(filename, overwrite=True)
                        

    @staticmethod
    def restore(filename, target_device_idx=None):

        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = int(hdr['VERSION'])
            c = GaussianConvolutionKernel(hdr['SPOTSIZE'], hdr['DIMX'], hdr['DIMY'], target_device_idx=target_device_idx)
            c.pxscale = hdr['PXSCALE']
            c.dimension = hdr['DIMENSION']
            c.oversampling = hdr['OVERSAMPLING']
            c.positiveShiftTT = hdr['POSITIVESHIFTTT']
            dr = c.xp.asarray(hdul[0].data)
            dc = c.xp.asarray(hdul[1].data)
            c.kernels = c.xp.empty_like(dr, dtype=c.complex_dtype)
            c.kernels.real = dr
            c.kernels.imag = dc
            return c