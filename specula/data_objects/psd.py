from specula import np, cpuArray
from astropy.io import fits
from specula.base_data_obj import BaseDataObj

# try:
from scipy.integrate import simpson
from scipy.signal import welch
#     SCIPY_AVAILABLE = True
# else:
#     SCIPY_AVAILABLE = False


class PSD(BaseDataObj):
    def __init__(self, data = None, dt:float = None, fs:float = None, description='', target_device_idx=None, precision=None):
        """
        Initialize PSD object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.description = description

        if data is not None:
            if fs is None and dt is None:
                raise ValueError('At least one of dt and fs inputs must be defined!')
            if fs is not None and dt is not None:
                if fs != 1/dt:
                    raise ValueError(f'The input sampling frequency {fs} is not the inverse of the given time step {dt}: choose the correct one!')
            
            if fs is None:
                fs = 1/dt

            freq_vec,psd_data = welch(cpuArray(data),fs,nperseg=256,scaling='density',axis=-1)
        
            self.freq_vec = self.to_xp(freq_vec, force_copy=True, dtype=self.dtype)
            self.psd_data = self.to_xp(psd_data, force_copy=True, dtype=self.dtype)
            self.integrated_power = self.integrate_psd(self.psd_data, self.freq_vec)
        else:
            self.freq_vec = None
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
        # Linear interpolation for each of the N PSDs
        interpolated = self.xp.array([self.xp.interp(new_freq, self.freq_vec, p, right=0, left=0) for p in self.psd_data])
        return interpolated

    def plot(self, mode:int=0, loglog=True, **kwargs):
        """Plots the PSD at index idx."""
        try:
            import matplotlib.pyplot as plt
            freq = cpuArray(self.freq_vec)
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

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        hdr['DESC'] = self.description
        
        primary_hdu = fits.PrimaryHDU(cpuArray(self.psd_data), header=hdr)
        freq_hdu = fits.ImageHDU(cpuArray(self.freq_vec), name='FREQ')
        power_hdu = fits.ImageHDU(cpuArray(self.integrated_power), name='INT_PWR')
        
        hdul = fits.HDUList([primary_hdu, freq_hdu, power_hdu])
        hdul.writeto(filename, overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        with fits.open(filename) as hdul:
            psd_data = hdul[0].data
            hdr = hdul[0].header
            freq = hdul['FREQ'].data
            pwr = hdul['INT_PWR'].data
            
            v = PSD(target_device_idx=target_device_idx)
            v.description = hdr.get('DESC', '')
            v.psd_data = v.to_xp(psd_data)
            v.freq_vec = v.to_xp(freq)
            v.integrated_power = v.to_xp(pwr) # Loaded directly from file
        return v

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['OBJ_TYPE'] = 'PSD'
        hdr['VERSION'] = 1
        return hdr