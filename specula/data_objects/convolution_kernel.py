import os
import hashlib, json

import numpy as np
from astropy.io import fits

from specula import cpuArray, ASEC2RAD
from specula.base_data_obj import BaseDataObj
from specula.lib.rebin import rebin2d


def lgs_map_sh(nsh, diam, rl, zb, dz, profz, fwhmb, ps, ssp,
               overs=2, theta=[0.0, 0.0], rprof_type=0,
               mask_pupil=False, pupil_weight=None, doCube=True,
               dtype=np.float32, xp=np):
    """
    It returns the pattern of Sodium Laser Guide Star images relayed by a Shack-Hartman lenlet array.
    Only geometrical propagation is taken in account (no diffraction effects).
    The beacon is simulated in the Sodium layer as a cilynder with gaussian radial profile and axially
    discretized in a given set of distances from the telescope entrance pupil with a given relative intensities.
    Currently only zenith telescope pointing is implemented
    Parameters:
        nsh (int): Number of sub-apertures
        diam (float): Telescope entrance pupil diameter [m]
        rl (list): Launcher position in meters [x, y, z]
        zb (float): distance from the telescope pupil of the sodium layer relayed on the SH focal plane [m]
        dz (list): N-elements vector of distances from zb of telescope on-axis sampling points of the sodium layer [m]
        profz (list): Sodium layer profile
        fwhmb (float): full with at high maximum of the section of the sodium beacon orthogonal to the telescope optical axis [on-sky arcsec]
        ps (float): plate scale of the SH foval plane [arcsec/pix]
        ssp (int): Field of view sampling of the SH focal plane (ssp x ssp) [pix]
        overs (int): Oversampling factor
        theta (list): Tip-tilt offsets in arcseconds [x, y]
        rprof_type (int): Radial profile type (0 for Gaussian, 1 for top-hat)
        mask_pupil (bool): Whether to apply a pupil mask
        pupil_weight (ndarray): Pupil mask weight
        doCube (bool): Whether to return a cube of kernels
        xp (module): The numpy or cupy module to use for calculations

    Returns:
        ccd (ndarray): The calculated LGS map
    """

    theta = cpuArray(theta)
    # Oversampling and lenslet grid setup
    ossp = ssp * overs

    xsh, ysh = xp.meshgrid(xp.linspace(-diam / 2, diam / 2, nsh, dtype=dtype),
                           xp.linspace(-diam / 2, diam / 2, nsh, dtype=dtype))
    xfov, yfov = xp.meshgrid(xp.linspace(-ssp * ps / 2, ssp * ps / 2, ossp, dtype=dtype),
                             xp.linspace(-ssp * ps / 2, ssp * ps / 2, ossp, dtype=dtype))
    # Gaussian parameters for the sodium layer
    sigma = (fwhmb * ASEC2RAD * zb) / (2 * xp.sqrt(2 * xp.log(2)))
    one_over_sigma2 = 1.0 / sigma**2
    exp_sigma = -0.5 * one_over_sigma2   
    rb = xp.array([theta[0] * ASEC2RAD * zb, theta[1] * ASEC2RAD * zb, 0], dtype=dtype)
    kv = xp.array([0, 0, 1], dtype=dtype)
    BL = zb * kv + rb - xp.array(rl, dtype=dtype)
    el = BL / BL[2]
    # Create the focal plane field positions (rf) and the sub-aperture positions (rs)
    rs_x, rs_y = xsh, ysh

    rf_x = xp.tile(xfov * ASEC2RAD * zb, (nsh, nsh)).reshape(ossp * nsh, ossp * nsh)
    rf_y = xp.tile(yfov * ASEC2RAD * zb, (nsh, nsh)).reshape(ossp * nsh, ossp * nsh)

    # Distance and direction vectors for calculating intensity maps
    es_x = (rf_x - xp.repeat(xp.repeat(rs_x, ossp, axis=0), ossp, axis=1)) / zb
    es_y = (rf_y - xp.repeat(xp.repeat(rs_y, ossp, axis=0), ossp, axis=1)) / zb
    es_z = 1.0

    # Initialize the field map (fmap) for LGS patterns
    fmap = xp.zeros((nsh * ossp, nsh * ossp), dtype=dtype)
    nz = len(dz)
    # Gaussian or top-hat profile choice for LGS beam
    if rprof_type == 0:
        gnorm = 1.0 / (sigma * xp.pi * xp.sqrt(2.0))  # Gaussian
    elif rprof_type == 1:
        gnorm = 1.0 / (xp.pi / 4 * (fwhmb * ASEC2RAD * zb)**2)  # Top-hat
    else:
        raise ValueError("Unsupported radial profile type")

    # Loop through layers for the sodium layer thickness
    for iz in range(nz):
        if profz[iz] > 0:
            d2 = ((rf_x + dz[iz] * es_x - (rb[0] + dz[iz] * el[0]))**2 +
                  (rf_y + dz[iz] * es_y - (rb[1] + dz[iz] * el[1]))**2 +
                  ( 0.0 + dz[iz] * es_z - (rb[2] + dz[iz] * el[2]))**2)

            if rprof_type == 0:
                fmap += (gnorm * profz[iz]) * xp.exp(d2 * exp_sigma, dtype=dtype)
            elif rprof_type == 1:
                fmap += (gnorm * profz[iz]) * ((d2 * one_over_sigma2) <= 1.0)

    # Resample fmap to match CCD size and apply pupil mask if specified
    # Use rebin2d with sample=True to match IDL's rebin behavior
    ccd = rebin2d(fmap, (ssp*nsh, ssp*nsh), sample=True, xp=xp)

    if mask_pupil:
        ccd *= rebin2d(pupil_weight, (ssp*nsh, ssp*nsh), sample=True, xp=xp)

    if doCube:
        ccd = ccd.reshape(nsh, ssp, nsh, ssp)
        ccd = ccd.transpose((2, 0, 1, 3))
        ccd = ccd.reshape(nsh*nsh, ssp, ssp)

    return ccd

class ConvolutionKernel(BaseDataObj):
    def __init__(self,
                 dimx: int,
                 dimy: int,
                 pxscale: float,
                 pupil_size_m: float,
                 dimension: int,
                 launcher_pos: list=[0.0,0.0,0.0],
                 seeing: float=0.0,
                 launcher_size: float=0.0,
                 zfocus: float=90e3,
                 theta: list=[0.0, 0.0],
                 airmass: float=1.0,
                 oversampling: int=1,
                 return_fft: bool=True,
                 positive_shift_tt: bool=True,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.convolution_kernel.ConvolutionKernel` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.dimx = dimx
        self.dimy = dimy
        self.pxscale = pxscale
        self.pupil_size_m = pupil_size_m
        self.dimension = dimension
        self.seeing = seeing
        self.zlayer = None
        self.zprofile = None
        self.zfocus = zfocus
        self.theta = self.to_xp(theta)
        self.last_zfocus = 0.0
        self.last_theta = self.xp.array([0.0, 0.0], dtype=self.dtype)
        self.return_fft = return_fft
        self.launcher_size = launcher_size
        self.last_seeing = -1.0
        self.airmass = airmass
        self.oversampling = oversampling
        if len(launcher_pos) != 3:
            raise ValueError("Launcher position must be a three-elements vector [m]")
        self.launcher_pos = self.to_xp(launcher_pos)
        self.last_zlayer = -1
        self.last_zprofile = -1
        self.positive_shift_tt = positive_shift_tt
        if self.return_fft:
            dtype = self.complex_dtype
        else:
            dtype = self.dtype
        self.real_kernels = self.xp.zeros((self.dimx*self.dimy, self.dimension, self.dimension), dtype=self.dtype)
        self.kernels = self.xp.zeros((self.dimx*self.dimy, self.dimension, self.dimension), dtype=dtype)
        self._kernel_fn = None

    def build(self):
        if len(self.zlayer) != len(self.zprofile):
            raise ValueError("Number of elements of zlayer and zprofile must be the same")

        zfocus = self.zfocus if self.zfocus != -1 else self.calculate_focus()
        lay_heights = self.to_xp(self.zlayer) * self.airmass
        zfocus *= self.airmass

        self.spot_size = self.xp.sqrt(self.seeing**2 + self.launcher_size**2)
        if not self.positive_shift_tt:
            lgs_tt = self.xp.array([-0.5, -0.5], dtype=self.dtype) * self.pxscale
        else:
            lgs_tt = self.xp.array([0.5, 0.5], dtype=self.dtype) * self.pxscale
        lgs_tt += self.theta

        items = [self.dimx, self.pupil_size_m, self.launcher_pos, zfocus, lay_heights, self.zprofile,
                         self.spot_size, self.pxscale, self.dimension, self.oversampling, lgs_tt, self.dtype]
        return 'ConvolutionKernel' + self.generate_hash(items)

    def calculate_focus(self):
        return self.xp.sum(self.to_xp(self.zlayer) * self.to_xp(self.zprofile)) / self.xp.sum(self.zprofile)

    def calculate_lgs_map(self):
        """
        Calculate the LGS (Laser Guide Star) map based on current parameters.
        This creates convolution kernels for each subaperture.
        """
        if len(self.zlayer) != len(self.zprofile):
            raise ValueError("Number of elements of zlayer and zprofile must be the same")

        if self.spot_size <= 0:
            raise ValueError("Spot size must be greater than zero")

        # Determine focus distance - use calculated focus if zfocus is -1
        zfocus = self.zfocus if self.zfocus != -1 else self.calculate_focus()

        # Apply airmass to heights
        lay_heights = self.to_xp(self.zlayer) * self.airmass
        zfocus *= self.airmass

        # Calculate the spot size (combination of seeing and laser launcher size)
        self.spot_size = self.xp.sqrt(self.seeing**2 + self.launcher_size**2)

        # Determine LGS tip-tilt offsets
        if not self.positive_shift_tt:
            lgs_tt = self.xp.array([-0.5, -0.5], dtype=self.dtype) * self.pxscale
        else:
            lgs_tt = self.xp.array([0.5, 0.5], dtype=self.dtype) * self.pxscale
        lgs_tt += self.theta

        # Calculate normalized layer heights and profiles
        layer_offsets = lay_heights - zfocus

        # Call the LGS map calculation function
        self.real_kernels = lgs_map_sh(
            self.dimx, self.pupil_size_m, self.launcher_pos, zfocus, layer_offsets,
            self.zprofile, self.spot_size, self.pxscale, self.dimension,
            overs=self.oversampling, theta=lgs_tt, doCube=True, xp=self.xp
        )

        # Process the kernels - apply FFT if needed
        self.process_kernels(return_fft=self.return_fft)

        # Save current parameters to avoid unnecessary recalculation
        self.last_zfocus = self.zfocus
        self.last_theta = self.to_xp(self.theta)
        self.last_seeing = self.seeing
        self.last_zlayer = self.zlayer
        self.last_zprofile = self.zprofile

    def generate_hash(self, items):
        """
        Generate a hash for the current kernel settings.
        This is used to check if the kernel needs to be recalculated.

        Returns:
            str: A hash string representing the current kernel settings.
        """
        # Convert all numpy arrays and values to native Python types
        hash_arr = []
        for item in items:
            if isinstance(item, self.xp.ndarray):
                # Convert array to list of native Python types
                hash_arr.append(item.tolist())
            elif isinstance(item, tuple):
                # Convert tuple elements to native Python types
                hash_arr.append([float(x) for x in item])
            elif isinstance(item, type):
                # This matches class types and dtypes as well
                hash_arr.append(str(item))
            elif hasattr(item, 'dtype') and hasattr(item, 'item'):
                # Convert numpy scalars to Python types
                hash_arr.append(item.item())
            else:
                hash_arr.append(item)

        # Placeholder function to compute SHA1 hash
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(hash_arr).encode('utf-8'))
        return sha1.hexdigest()

    def process_kernels(self, return_fft=False):
        # Check for non-finite values
        if self.xp.any(~self.xp.isfinite(self.real_kernels)):
            raise ValueError("Kernel contains non-finite values!")

        # Process the kernels - apply FFT if needed
        for i in range(self.dimx):
            for j in range(self.dimy):
                subap_kern = self.to_xp(self.real_kernels[i * self.dimx + j, :, :])
                total = self.xp.sum(subap_kern)
                if total > 0:  # Avoid division by zero
                    subap_kern /= total
                if return_fft:
                    subap_kern_fft = self.xp.fft.ifft2(subap_kern)
                    self.kernels[j * self.dimx + i, :, :] = subap_kern_fft
                else:
                    self.kernels[j * self.dimx + i, :, :] = subap_kern

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1.1
        hdr['PXSCALE'] = self.pxscale
        hdr['DIM'] = self.dimension
        hdr['OVERSAMP'] = self.oversampling
        hdr['POSTT'] = self.positive_shift_tt
        hdr['SPOTSIZE'] = float(self.spot_size)
        hdr['DIMX'] = self.dimx
        hdr['DIMY'] = self.dimy
        return hdr

    def save(self, filename):
        """
        Save the kernel to a FITS file.
        
        Parameters:
            filename (str): Path to save the FITS file
            hdr (fits.Header, optional): Additional header information
        """
        hdr = self.get_fits_header()

        # Create a primary HDU with just the header
        primary_hdu = fits.PrimaryHDU(header=hdr)

        # Create an HDU with the kernel data
        kernel_data = cpuArray(self.real_kernels)
        kernel_hdu = fits.ImageHDU(data=kernel_data)

        # Create an HDUList and write to file
        hdul = fits.HDUList([primary_hdu, kernel_hdu])
        hdul.writeto(filename, overwrite=True)   
        hdul.close()  # Force close for Windows

    def prepare_for_sh(self, sodium_altitude=None, sodium_intensity=None, current_time=None):
        # Update the kernel parameters if provided
        if sodium_altitude is not None:
            self.zlayer = sodium_altitude
        if sodium_intensity is not None:
            self.zprofile = sodium_intensity

        kernel_fn = self.build()

        # Only reload or recalculate if the kernel has changed
        if kernel_fn != self._kernel_fn:
            self._kernel_fn = kernel_fn  # Update the stored kernel filename

            if os.path.exists(kernel_fn):
                print(f"Loading kernel from {kernel_fn}")
                self.restore(kernel_fn, kernel_obj=self, target_device_idx=self.target_device_idx, return_fft=True)
            else:
                print('Calculating kernel...')
                self.calculate_lgs_map()
                self.save(kernel_fn)
                print('Done')

        if current_time is not None:
            self.generation_time = current_time

    @staticmethod
    def restore(filename, target_device_idx=None, kernel_obj=None, return_fft=False):
        """
        Restore a ConvolutionKernel object from a FITS file.

        Parameters:
            filename (str): Path to the FITS file
            target_device_idx (int, optional): Target device index for GPU processing
            return_fft (bool, optional): Whether to return FFT of the kernel
    
        Returns:
            ConvolutionKernel: The restored ConvolutionKernel object
        """
        hdr = fits.getheader(filename, ext=0)  # Get header from primary HDU

        version = hdr['VERSION']
        if version != 1.1:
            raise ValueError(f'Unknown version {version}. Only version=1.1 is supported')

        if kernel_obj is None:
            kernel_obj = ConvolutionKernel.from_header(hdr, target_device_idx=target_device_idx)
        else:
            # If a kernel object is provided, use it
            # check if the dimensions match
            if kernel_obj.dimx != hdr['DIMX'] or kernel_obj.dimy != hdr['DIMY']:
                raise ValueError("Provided kernel object dimensions do not match the FITS file dimensions")
            # Read properties from header
            kernel_obj.pxscale = hdr['PXSCALE']
            kernel_obj.dimension = hdr['DIM']
            kernel_obj.oversampling = hdr['OVERSAMP']
            kernel_obj.positive_shift_tt = hdr['POSTT']
            kernel_obj.spot_size = hdr['SPOTSIZE']

        # This code uses an intermediate array to make sure that endianess is correct (FITS is big-endian)
        data = kernel_obj.xp.array(fits.getdata(filename, ext=1), dtype=kernel_obj.dtype)
        kernel_obj.real_kernels[:] = data
        kernel_obj.process_kernels(return_fft=return_fft)
        return kernel_obj

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1.1:
            raise ValueError(f'Unknown version {version}. Only version=1.1 is supported')

        kernel_obj = ConvolutionKernel(
            dimx=hdr['DIMX'],
            dimy=hdr['DIMY'],
            pxscale=hdr['PXSCALE'],
            pupil_size_m=0.0,
            dimension=hdr['DIM'],
            launcher_pos=[0.0, 0.0, 0.0],
            launcher_size=hdr['SPOTSIZE'],
            oversampling=hdr['OVERSAMP'],
            positive_shift_tt=hdr['POSTT'],
            target_device_idx=target_device_idx)

        kernel_obj.spot_size = hdr['SPOTSIZE']
        return kernel_obj

    def get_value(self):
        return self.real_kernels
    
    def set_value(self, v):
        '''Set new kernels.
        Arrays are not reallocated.'''
        assert v.shape == self.real_kernels.shape, \
            f"Error: input array shape {v.shape} does not match real_kernels shape {self.real_kernels.shape}"

        self.real_kernels[:] = self.to_xp(v)