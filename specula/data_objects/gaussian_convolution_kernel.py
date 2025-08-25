from specula.data_objects.convolution_kernel import ConvolutionKernel, lgs_map_sh

from astropy.io import fits

class GaussianConvolutionKernel(ConvolutionKernel):
    """
    Kernel processing object for Gaussian kernels.
    """

    def __init__(self,
                 dimx: int,
                 dimy: int,
                 pxscale: float,
                 dimension: int,
                 spot_size: float,
                 pupil_size_m: 0.0=float,
                 oversampling: int=1,
                 return_fft: bool=True,
                 positive_shift_tt: bool=True,
                 airmass: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(
            dimx=dimx,
            dimy=dimy,
            pxscale=pxscale,
            pupil_size_m=pupil_size_m,
            dimension=dimension,
            airmass=airmass,
            oversampling=oversampling,
            return_fft=return_fft,
            positive_shift_tt=positive_shift_tt,
            target_device_idx=target_device_idx,
            precision=precision
        )
        self.spot_size = spot_size

    def build(self):
        """
        Recalculates the Gaussian kernel based on current settings.
        """
        self.orig_dimx = self.dimx
        self.dimx = max(self.dimx, 2)        
        self.lgs_tt = [-0.5, -0.5] if not self.positive_shift_tt else [0.5, 0.5]
        self.lgs_tt = [x * self.pxscale for x in self.lgs_tt]
        items = [
            self.dimx, self.pupil_size_m, 90e3, self.spot_size, self.dtype,
            self.pxscale, self.dimension, 3, self.lgs_tt, [0, 0, 0], [90e3], [1.0],
        ]
        return 'ConvolutionKernel' + self.generate_hash(items)   

    def calculate_lgs_map(self):
        self.real_kernels = lgs_map_sh(
            self.dimx, self.pupil_size_m, 0, 90e3, [0], profz=[1.0], fwhmb=self.spot_size, ps=self.pxscale,
            ssp=self.dimension, overs=1, theta=self.lgs_tt, xp=self.xp )

        self.process_kernels(return_fft=self.return_fft)

    @staticmethod
    def restore(filename, target_device_idx=None, kernel_obj=None, return_fft=False):
        """
        Restore a :class:`~specula.data_objects.gaussian_convolution_kernel.GaussianConvolutionKernel` object from a FITS file.

        Parameters:
            filename (str): Path to the FITS file
            target_device_idx (int, optional): Target device index for GPU processing
            return_fft (bool, optional): Whether to return FFT of the kernel
    
        Returns:
            :class:`~specula.data_objects.gaussian_convolution_kernel.GaussianConvolutionKernel`: The restored ConvolutionKernel object
        """
        hdr = fits.getheader(filename, ext=0)  # Get header from primary HDU

        version = hdr['VERSION']
        if version != 1.1:
            raise ValueError(f'Unknown version {version}. Only version=1.1 is supported')

        if kernel_obj is None:
            kernel_obj = GaussianConvolutionKernel(
                dimx=hdr['DIMX'],
                dimy=hdr['DIMY'],
                pxscale=hdr['PXSCALE'],
                dimension=hdr['DIM'],
                spot_size=hdr['SPOTSIZE'],
                oversampling=hdr['OVERSAMP'],
                positive_shift_tt=hdr['POSTT'],
                target_device_idx=target_device_idx)
        else:
            # If a kernel object is provided, use it
            # check if the spot size matches
            if kernel_obj.spot_size != hdr['SPOTSIZE']:
                raise ValueError("Provided kernel object spot size does not match the FITS file spot size")
            # check if the dimensions match
            if kernel_obj.dimx != hdr['DIMX'] or kernel_obj.dimy != hdr['DIMY']:
                raise ValueError("Provided kernel object dimensions do not match the FITS file dimensions")
            # Read properties from header
            kernel_obj.spot_size = hdr['SPOTSIZE']
            kernel_obj.pxscale = hdr['PXSCALE']
            kernel_obj.dimension = hdr['DIM']
            kernel_obj.oversampling = hdr['OVERSAMP']
            kernel_obj.positive_shift_tt = hdr['POSTT']

        # This code uses an intermediate array to make sure that endianess is correct (FITS is big-endian)
        data = kernel_obj.xp.array(fits.getdata(filename, ext=1), dtype=kernel_obj.dtype)
        kernel_obj.real_kernels[:] = data
        kernel_obj.process_kernels(return_fft=return_fft)
        return kernel_obj


