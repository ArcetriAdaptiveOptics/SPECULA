from specula.base_data_obj import BaseDataObj

from astropy.io import fits

from specula import cpuArray, ASEC2RAD

import numpy as np

import hashlib, json


def lgs_map_sh(nsh, diam, rl, zb, dz, profz, fwhmb, ps, ssp,
               overs=2, theta=[0.0, 0.0], rprof_type=0,
               mask_pupil=False, doCube=True, xp=np):
   
    if xp==np:
        from scipy.ndimage import zoom
    else:
        from cupyx.scipy.ndimage import zoom

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
    FS_x = rf_x - xp.repeat(rs_x, ossp**2).reshape(ossp * nsh, ossp * nsh)
    FS_y = rf_y - xp.repeat(rs_y, ossp**2).reshape(ossp * nsh, ossp * nsh)
    FS_z = zb + rf_z - xp.repeat(rs_z, ossp**2).reshape(ossp * nsh, ossp * nsh)
    es_x = FS_x / FS_z
    es_y = FS_y / FS_z
    es_z = FS_z / FS_z
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
    ccd = zoom(fmap, ssp / ossp, order=1)  # downsample using zoom
    if mask_pupil:
        mask = xp.sqrt(xsh**2 + ysh**2) <= (diam / 2)
        ccd *= xp.kron(mask, xp.ones((ssp, ssp)))

    if doCube:
        ccd = ccd.reshape(nsh, ssp, nsh, ssp)
        ccd = ccd.transpose((2, 0, 1, 3))
        ccd = ccd.reshape(nsh*nsh, ssp, ssp)

    return ccd

class ConvolutionKernel(BaseDataObj):
    def __init__(self,
                 dimx: int,
                 dimy: int,
                 airmass: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
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
        self.returnFft = False
        self.launcher_size = 0.0
        self.last_seeing = -1.0
        self.oversampling = 1
        self.launcher_pos = self.xp.zeros(3)
        self.last_zlayer = -1
        self.last_zprofile = -1        
        self.airmass = airmass
        self.positiveShiftTT = False
        self.dimx = dimx
        self.dimy = dimy
  
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
        layHeights = self.xp.array(self.zlayer) * self.airmass
        zfocus *= self.airmass

        self.spotsize = self.xp.sqrt(self.seeing**2 + self.launcher_size**2)
        LGS_TT = (self.xp.array([-0.5, -0.5]) if not self.positiveShiftTT else self.xp.array([0.5, 0.5])) * self.pxscale + self.theta
        
        self.hash_arr = [self.dimx, self.pupil_size_m, zfocus, self.spotsize, self.pxscale, self.dimension, self.oversampling, LGS_TT]
        return 'ConvolutionKernel' + self.generate_hash()
        
    def calculate_focus(self):
        return self.xp.sum(self.xp.array(self.zlayer) * self.xp.array(self.zprofile)) / self.xp.sum(self.zprofile)

    def calculate_lgs_map(self):
        raise NotImplementedError('')
        

    def generate_hash(self):
        # Placeholder function to compute SHA1 hash        
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(self.hash_arr).encode('utf-8'))
        return sha1.hexdigest()

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PXSCALE'] = self.pxscale
        hdr['DIMENSION'] = self.dimension
        hdr['OVERSAMPLING'] = self.oversampling
        hdr['POSITIVESHIFTTT'] = self.positiveShiftTT
        hdr['SPOTSIZE'] = self.spotsize
        hdr['DIMX'] = self.dimx
        hdr['DIMY'] = self.dimy        
        fits.append(filename, cpuArray(self.kernels) )        

    def read(self, filename, hdr=None, exten=1):        
        self.kernels = self.xp.array(fits.getdata(filename, ext=exten))
            
    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])
        c = ConvolutionKernel(target_device_idx=target_device_idx)
        c.pxscale = hdr['PXSCALE']
        c.dimension = hdr['DIMENSION']
        c.oversampling = hdr['OVERSAMPLING']
        c.positiveShiftTT = hdr['POSITIVESHIFTTT']
        c.spotsize = hdr['SPOTSIZE']
        c.dimx = hdr['DIMX']
        c.dimy = hdr['DIMY']
        c.read(filename, hdr)
        return c
