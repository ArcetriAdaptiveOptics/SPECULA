from specula.base_data_obj import BaseDataObj

from astropy.io import fits

from specula import cpuArray, ASEC2RAD
from specula.lib.rebin import rebin2d

import numpy as np

import hashlib, json


def lgs_map_sh(nsh, diam, rl, zb, dz, profz, fwhmb, ps, ssp,
               overs=2, theta=[0.0, 0.0], rprof_type=0,
               mask_pupil=False, pupil_weight=None, doCube=True,
               xp=np):
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

    # Oversampling and lenslet grid setup
    ossp = ssp * overs
    xsh, ysh = xp.meshgrid(xp.linspace(-diam / 2, diam / 2, nsh), xp.linspace(-diam / 2, diam / 2, nsh))
    xfov, yfov = xp.meshgrid(xp.linspace(-ssp * ps / 2, ssp * ps / 2, ossp), xp.linspace(-ssp * ps / 2, ssp * ps / 2, ossp))   
    # Gaussian parameters for the sodium layer
    sigma = (fwhmb * ASEC2RAD * zb) / (2 * xp.sqrt(2 * xp.log(2)))
    one_over_sigma2 = 1.0 / sigma**2
    exp_sigma = -0.5 * one_over_sigma2   
    rb = xp.array([theta[0] * ASEC2RAD * zb, theta[1] * ASEC2RAD * zb, 0])
    kv = xp.array([0, 0, 1])
    BL = zb * kv + rb - xp.array(rl)
    el = BL / BL[2]
    # Create the focal plane field positions (rf) and the sub-aperture positions (rs)
    rs_x, rs_y, rs_z = xsh, ysh, xp.zeros((nsh, nsh))

    rf_x = xp.tile(xfov * ASEC2RAD * zb, (nsh, nsh)).reshape(ossp * nsh, ossp * nsh)
    rf_y = xp.tile(yfov * ASEC2RAD * zb, (nsh, nsh)).reshape(ossp * nsh, ossp * nsh)
    rf_z = xp.zeros((ossp * nsh, ossp * nsh))

    # Distance and direction vectors for calculating intensity maps
    fs_x = rf_x - xp.repeat(xp.repeat(rs_x, ossp, axis=0), ossp, axis=1)
    fs_y = rf_y - xp.repeat(xp.repeat(rs_y, ossp, axis=0), ossp, axis=1)
    fs_z = zb + rf_z - xp.repeat(xp.repeat(rs_z, ossp, axis=0), ossp, axis=1)

    es_x = fs_x / fs_z
    es_y = fs_y / fs_z
    es_z = fs_z / fs_z

    # Initialize the field map (fmap) for LGS patterns
    fmap = xp.zeros((nsh * ossp, nsh * ossp))
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
                  (rf_z + dz[iz] * es_z - (rb[2] + dz[iz] * el[2]))**2)

            if rprof_type == 0:
                fmap += (gnorm * profz[iz]) * xp.exp(d2 * exp_sigma)
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
                 airmass: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.real_kernels = None
        self.kernels = None
        self.seeing = None
        self.zlayer = None
        self.zprofile = None
        self.zfocus = 0.0
        self.theta = self.xp.array([0.0, 0.0])
        self.last_zfocus = 0.0
        self.last_theta = self.xp.array([0.0, 0.0])
        self.dimension = 0
        self.pxscale = 0.0        
        self.return_fft = False
        self.launcher_size = 0.0
        self.last_seeing = -1.0
        self.oversampling = 1
        self.launcher_pos = self.xp.zeros(3)
        self.last_zlayer = -1
        self.last_zprofile = -1        
        self.airmass = airmass
        self.positive_shift_tt = False
        self.dimx = 0
        self.dimy = 0

    def set_launcher_pos(self, launcher_pos):
        if len(launcher_pos) != 3:
            raise ValueError("Launcher position must be a three-elements vector [m]")
        self.launcher_pos = self.xp.array(launcher_pos)        

#    def make_grid(self):
#        self.dimx = max(self.dimx, 2)
#        self.pixel_pitch = self.ef_pixel_pitch * (self.ef_size / 2.0)
#        x = self.xp.linspace(-self.pixel_pitch, self.pixel_pitch, self.dimx)
#        y = self.xp.linspace(-self.pixel_pitch, self.pixel_pitch, self.dimx)
#        self.xgrid, self.ygrid = self.xp.meshgrid(x, y)

    def build(self):
        if len(self.zlayer) != len(self.zprofile):
            raise ValueError("Number of elements of zlayer and zprofile must be the same")

        zfocus = self.zfocus if self.zfocus != -1 else self.calculate_focus()
        lay_heights = self.xp.array(self.zlayer) * self.airmass
        zfocus *= self.airmass

        self.spot_size = self.xp.sqrt(self.seeing**2 + self.launcher_size**2)
        lgs_tt = (self.xp.array([-0.5, -0.5]) if not self.positive_shift_tt else self.xp.array([0.5, 0.5])) * self.pxscale + self.theta

        self.hash_arr = [self.dimx, self.pupil_size_m, zfocus, lay_heights, self.zprofile, self.spot_size,
                         self.pxscale, self.dimension, self.oversampling, lgs_tt]
        return 'ConvolutionKernel' + self.generate_hash()

    def calculate_focus(self):
        return self.xp.sum(self.xp.array(self.zlayer) * self.xp.array(self.zprofile)) / self.xp.sum(self.zprofile)

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
        lay_heights = self.xp.array(self.zlayer) * self.airmass
        zfocus *= self.airmass

        # Calculate the spot size (combination of seeing and laser launcher size)
        self.spot_size = self.xp.sqrt(self.seeing**2 + self.launcher_size**2)

        # Determine LGS tip-tilt offsets
        if not self.positive_shift_tt:
            lgs_tt = self.xp.array([-0.5, -0.5]) * self.pxscale
        else:
            lgs_tt = self.xp.array([0.5, 0.5]) * self.pxscale
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
        self.last_theta = self.xp.array(self.theta)
        self.last_seeing = self.seeing
        self.last_zlayer = self.zlayer
        self.last_zprofile = self.zprofile

    def generate_hash(self):
        """
        Generate a hash for the current kernel settings.
        This is used to check if the kernel needs to be recalculated.

        Returns:
            str: A hash string representing the current kernel settings.
        """
        # Convert all numpy arrays and values to native Python types
        hash_arr = []
        for item in self.hash_arr:
            if isinstance(item, self.xp.ndarray):
                # Convert array to list of native Python types
                hash_arr.append(item.tolist())
            elif isinstance(item, tuple):
                # Convert tuple elements to native Python types
                hash_arr.append([float(x) for x in item])
            elif hasattr(item, 'dtype') and hasattr(item, 'item'):
                # Convert numpy scalars to Python types
                hash_arr.append(item.item())
            else:
                hash_arr.append(item)

        # Placeholder function to compute SHA1 hash
        sha1 = hashlib.sha1()
        # converts all numpy arrays to list
        for i, hash_elem in enumerate(self.hash_arr):
            if isinstance(hash_elem, self.xp.ndarray):
                self.hash_arr[i] = hash_elem.tolist()
        sha1.update(json.dumps(self.hash_arr).encode('utf-8'))
        return sha1.hexdigest()

    def process_kernels(self, return_fft=False):
        # Check for non-finite values
        if self.xp.any(~self.xp.isfinite(self.real_kernels)):
            raise ValueError("Kernel contains non-finite values!")

        # Process the kernels - apply FFT if needed
        dtype = self.complex_dtype if return_fft else self.dtype

        self.kernels = self.xp.zeros_like(self.real_kernels, dtype=dtype)
        for i in range(self.dimx):
            for j in range(self.dimy):
                subap_kern = self.xp.array(self.real_kernels[i * self.dimx + j, :, :])
                total = self.xp.sum(subap_kern)
                if total > 0:  # Avoid division by zero
                    subap_kern /= total
                if return_fft:
                    subap_kern_fft = self.xp.fft.ifft2(subap_kern)
                    self.kernels[j * self.dimx + i, :, :] = subap_kern_fft
                else:
                    self.kernels[j * self.dimx + i, :, :] = subap_kern

    def save(self, filename, hdr=None):
        """
        Save the kernel to a FITS file.
        
        Parameters:
            filename (str): Path to save the FITS file
            hdr (fits.Header, optional): Additional header information
        """
        if hdr is None:
            hdr = fits.Header()

        hdr['VERSION'] = 1.1
        hdr['PXSCALE'] = self.pxscale
        hdr['DIM'] = self.dimension
        hdr['OVERSAMP'] = self.oversampling
        hdr['POSTT'] = self.positive_shift_tt
        hdr['SPOTSIZE'] = self.spot_size
        hdr['DIMX'] = self.dimx
        hdr['DIMY'] = self.dimy

        # Create a primary HDU with just the header
        primary_hdu = fits.PrimaryHDU(header=hdr)

        # Create an HDU with the kernel data
        kernel_data = cpuArray(self.real_kernels)
        kernel_hdu = fits.ImageHDU(data=kernel_data)

        # Create an HDUList and write to file
        hdul = fits.HDUList([primary_hdu, kernel_hdu])
        hdul.writeto(filename, overwrite=True)   

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

        version = int(hdr['VERSION'])
        if kernel_obj is None:
            kernel_obj = ConvolutionKernel(target_device_idx=target_device_idx)
        else:
            # If a kernel object is provided, use it
            # check if the dimensions match
            if kernel_obj.dimx != hdr['DIMX'] or kernel_obj.dimy != hdr['DIMY']:
                raise ValueError("Provided kernel object dimensions do not match the FITS file dimensions")

        # Read properties from header
        kernel_obj.dimx = hdr['DIMX']
        kernel_obj.dimy = hdr['DIMY']
        kernel_obj.pxscale = hdr['PXSCALE']
        kernel_obj.dimension = hdr['DIM']
        kernel_obj.oversampling = hdr['OVERSAMP']
        kernel_obj.positive_shift_tt = hdr['POSTT']
        kernel_obj.spot_size = hdr['SPOTSIZE']

        # Read the kernel data from extension 1
        kernel_obj.real_kernels = kernel_obj.xp.array(fits.getdata(filename, ext=1))       
        kernel_obj.process_kernels(return_fft=return_fft)
        return kernel_obj
