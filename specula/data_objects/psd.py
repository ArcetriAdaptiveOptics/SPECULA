from specula import np, cpuArray
from astropy.io import fits
from specula.base_data_obj import BaseDataObj

from scipy.integrate import simpson
from scipy.signal import welch


class PSD(BaseDataObj):
    """ 
    PSD (Power Spectral Density) data object for storing and processing PSDs of time-series data.
    Provides methods for interpolation, integration, plotting, and saving/restoring from FITS files.

    """
    def __init__(self, data = None, dt:float = None, fs:float = None, nperseg:int=128,
                 overwrite:bool=False, description='', target_device_idx=None, precision=None):
        """
        Initialize PSD object.

        Parameters
        ----------
        data : ndarray, optional
            Time-series data to compute PSD from. Can be 1D or 2D array. If None, creates an empty PSD object.
            The PSD is computed on the last axis.
        dt : float, optional
            Time step of data (inverse of sampling frequency).
        fs : float, optional
            Sampling frequency of data.
            Either fs or dt must be provided. If both provided, they must be consistent (fs = 1/dt).
        nperseg : int, optional
            Number of samples per segment for Welch method. Default is 128.
        overwrite : bool, optional
            If True, overwrite existing files when saving. Default is False.
        description : str, optional
            Description string for the PSD data. Default is empty string.
        target_device_idx : int, optional
            Target device index for computation (e.g., GPU/CPU selection). Default is None.
        precision : dtype, optional
            Data type precision for computations. Default is None (uses default precision).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.description = description
        self.samplespersegment = nperseg
        self.overwrite = overwrite

        if fs is None and dt is None:
            raise ValueError('At least one of dt and fs inputs must be defined!')
        if fs is not None and dt is not None:
            if fs != 1/dt:
                raise ValueError(f'The input sampling frequency {fs} is not the inverse of the given time step {dt}')
        if fs is None:
            fs = 1/dt
        self.fs = fs

        if data is not None:
            if len(data.shape) > 2:
                raise ValueError(f'Incorrect input data dimensions {data.shape}, only 2D arrays are supported')
            
            if max(data.shape) < nperseg:
                raise ValueError(f'Data has shape {data.shape} but the requested number of samples for Welch method is {nperseg}')

            _,psd_data = welch(cpuArray(data),fs,nperseg=nperseg,scaling='density',axis=-1)
        
            self.psd_data = self.to_xp(psd_data, dtype=self.dtype)
            self.integrated_power = self.integrate_psd(self.psd_data, self.get_freq_vec())
        else:
            self.psd_data = None
            self.integrated_power = None



    def get_integrated_power(self, freq_vec=None):
        """
        Returns integrated power. 
        If freq_vec is None, returns the pre-calculated total power.
        If freq_vec is provided, interpolates and integrates over that specific range.
        """
        if freq_vec is None:
            return self.integrated_power
        interp_psd = self.interpolate(freq_vec)
        return self.integrate_psd(interp_psd, self.to_xp(freq_vec))

    def interpolate(self, new_freq_vec):
        """Interpolates the [N, L] PSD onto a new [new_L] frequency vector."""
        new_freq = self.to_xp(new_freq_vec)
        old_freq = self.get_freq_vec()
        interpolated = self.xp.array([self.xp.interp(new_freq, old_freq, p, right=0, left=0) for p in self.psd_data])
        return interpolated

    def plot(self, mode:int=0, loglog=True, **kwargs):
        """Plots the PSD at index idx."""
        try:
            import matplotlib.pyplot as plt
            freq = cpuArray(self.get_freq_vec())
            data = cpuArray(self.psd_data[mode])
            plt.figure()
            plot_func = plt.loglog if loglog else plt.plot
            plot_func(freq, data, **kwargs)
            plt.xlabel('Frequency')
            plt.ylabel('PSD')
            plt.title(f"{self.description} (Mode {mode})")
            plt.grid(True)
            plt.show()
        except ImportError:
            print('Matplotlib not available for display')

    @staticmethod
    def integrate_psd(psd,freq):
        return simpson(cpuArray(psd.T),cpuArray(freq),axis=0)
    
    def get_value(self):
        return self.psd_data
    
    def set_value(self,v):
        assert v.shape == self.psd_data.shape, \
            f"Error: input array shape {v.shape} does not match PSD shape {self.psd_data.shape}"
        self.psd_data[:]= self.to_xp(v)

    def save(self, filename):
        hdr = self.get_fits_header()
        hdr['DESC'] = self.description
        hdr['NPERSEG'] = self.samplespersegment
        hdr['SAMPFREQ'] = self.fs
        
        primary_hdu = fits.PrimaryHDU(cpuArray(self.psd_data), header=hdr)
        power_hdu = fits.ImageHDU(cpuArray(self.integrated_power), name='INT_PWR')
        
        hdul = fits.HDUList([primary_hdu, power_hdu])
        hdul.writeto(filename, overwrite=self.overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        with fits.open(filename) as hdul:
            psd_data = hdul[0].data
            hdr = hdul[0].header
            pwr = hdul['INT_PWR'].data
            
            v = PSD(target_device_idx=target_device_idx,dt=1.0)
            v = v.from_header(hdr,target_device_idx=target_device_idx)
            v.psd_data = v.to_xp(psd_data)
            v.integrated_power = v.to_xp(pwr) # Loaded directly from file
        return v
    
    def get_freq_vec(self):
        L = self.samplespersegment//2+1
        fvec = self.xp.linspace(0,self.fs/2,L)
        return fvec

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['OBJ_TYPE'] = 'PSD'
        hdr['VERSION'] = 1
        return hdr
    
    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        description = hdr['DESC']
        nperseg = hdr['NPERSEG']
        fs = hdr['SAMPFREQ']
        psd = PSD(description=description,fs=fs,nperseg=nperseg,target_device_idx=target_device_idx)
        return psd
        
