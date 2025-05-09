from astropy.io import fits

from specula.base_time_obj import BaseTimeObj
from copy import copy
from specula import cp, np
from functools import lru_cache

# We use lru_cache() instead of cache() for python 3.8 compatibility
@lru_cache(maxsize=None)
def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ]) 
    return result
    # return [attr for attr, value in vars(cls).items() if isinstance(value, property) ]

class BaseDataObj(BaseTimeObj):
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base data object.

        Parameters:
        precision (int, optional):if None will use the global_precision, otherwise pass 0 for double, 1 for single
        """
        super().__init__(target_device_idx, precision)
        self._generation_time = -1

    @property
    def generation_time(self):
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value):
        self._generation_time = value

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseDataObj'
        hdr['GEN_TIME'] = self._generation_time
        hdr['TIME_RES'] = self._time_resolution
        return hdr

    def save(self, filename):
        hdr = self.get_fits_header()
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(filename, overwrite=True)

    def read(self, filename):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            if hdr['VERSION'] != 1:
                raise ValueError(f'Invalid version for BaseDataObj: {hdr["VERSION"]} instead of 1')
            self._generation_time = int(hdr.get('GEN_TIME', 0))
            self._time_resolution = int(hdr.get('TIME_RES', 0))
    
    def transferDataTo(self, destobj):
        excluded = ['_tag']
        #if target_device_idx==self.target_device_idx:
        #    return self
        #else:
        pp = get_properties(type(self))            
        for attr in dir(self):
            if attr not in excluded and attr not in pp:
                concrete_attr = getattr(self, attr)
                aType = type(concrete_attr)
                if destobj.target_device_idx==-1:
                    if aType==cp.ndarray:
                        #print(f'transferDataTo: {attr} to CPU')
                        setattr(destobj, attr, concrete_attr.get(blocking=True) )
                elif self.target_device_idx==-1:
                    if aType==np.ndarray:
                        #print(f'transferDataTo: {attr} to GPU')
                        setattr(destobj, attr, cp.asarray( concrete_attr ) )
        destobj.generation_time = self.generation_time
        return destobj


    def copyTo(self, target_device_idx):
        excluded = ['_tag']
        if target_device_idx==self.target_device_idx:
            return self
        else:
            pp = get_properties(type(self))
            cloned = copy(self)
            for attr in dir(self):
                if attr not in excluded and attr not in pp:
                    concrete_attr = getattr(self, attr)
                    cloned_attr = getattr(cloned, attr)
                    aType = type(concrete_attr)
                    if target_device_idx==-1:
                        if aType==cp.ndarray:
                            #setattr(cloned, attr, cp._cupyx.zeros_like_pinned( cloned_attr ) )
                            setattr(cloned, attr, cp.asnumpy( cloned_attr ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on CPU')
                    elif self.target_device_idx==-1:
                        if aType==np.ndarray:
                            setattr(cloned, attr, cp.asarray( cloned_attr ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on GPU')
            if target_device_idx >= 0:
                cloned.xp = cp
            else:
                cloned.xp = np
            cloned.target_device_idx = target_device_idx
            return cloned
