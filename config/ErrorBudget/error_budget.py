import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os.path as op
from astropy.io import fits

from scipy.integrate import simpson
from specula.mmlib.utils import radial_order, von_karman_power, get_pupil_mask
from specula.data_objects.iir_filter_data import IirFilterData

class AOErrorBudgetMachine:
    """
    A Semianalytical Error Budget Machine for AO systems with Pyramid WFS.
    Based on Agapito & Pinna (JATIS 2019).
    """
    def __init__(self, base_path:str, controller: IirFilterData = None, L0:float=25, V:float=20,
                 telescope_diameter=8.2, dm_type:str='asm', slopes_from_intensity:bool=False,
                 throughput=0.3, delay_frames=2.0, obsratio=0.0, dm_cutoff_hz=None,
                 RON:float=0.0, F_excess:float=1.0, dark_curr:float=0.0, sky_bkg:float=0.0 ):
        
        self.root_dir = base_path

        # Physical Parameters
        self.D = telescope_diameter
        self.dm_type = dm_type
        self.area = np.pi/4 * (self.D**2- (obsratio*self.D)**2)

        # Atmo parameters
        self.L0 = L0
        
        # Control Loop & Hardware
        self.delay_frames = delay_frames 
        self.dm_cutoff_hz = dm_cutoff_hz # Hz (None for ideal DM)
        self.controller = controller

        # Detector parameters
        self.RON = RON 
        self.throughput = throughput  
        self.F_excess = F_excess
        self.sky_bkg = sky_bkg
        self.dark_curr = dark_curr
        self.slopes_from_intensity = slopes_from_intensity
        


    def get_rtf(self, mode:int, fs:float):
        freq = self.get_freq_vec(fs)
        nw_delay, dw_delay = self.controller.discrete_delay_tf(self.delay_frames)
        if self.dm_cutoff_hz is not None:
            lpf_obj = IirFilterData.lpf_from_fc(fc=self.dm_cutoff_hz, fs=fs, n_ord=4)
            lpf_num, lpf_den = lpf_obj.num[0], lpf_obj.den[0]
            nw = np.convolve(nw_delay, lpf_num)
            dm = np.convolve(dw_delay, lpf_den)
            rtf = self.controller.RTF(mode=mode, fs=fs, freq=freq, dm=dm, nw=nw, dw=1.0, plot=False)
        else:
            rtf = self.controller.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        return rtf
    
    def get_ntf(self, mode:int, fs:float):
        freq = self.get_freq_vec(fs)
        nw_delay, dw_delay = self.controller.discrete_delay_tf(self.delay_frames)
        if self.dm_cutoff_hz is not None:
            lpf_obj = IirFilterData.lpf_from_fc(fc=self.dm_cutoff_hz, fs=fs, n_ord=4)
            lpf_num, lpf_den = lpf_obj.num[0], lpf_obj.den[0]
            nw = np.convolve(nw_delay, lpf_num)
            dm = np.convolve(dw_delay, lpf_den)
            ntf = self.controller.NTF(mode=mode, fs=fs, freq=freq, dm=dm, nw=nw, dw=1.0, plot=False)
        else:
            ntf = self.controller.NTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        return ntf

    @staticmethod
    def get_freq_vec(fs:float):
        return np.logspace(-2, np.log10(fs/2), 4000)
    
    @staticmethod
    def rad2nm(rad, lambdaInM):
        return rad*lambdaInM/(2*np.pi)*1e+9
    
    @staticmethod
    def r02seeing(r0):
        return 0.98 * 500e-9/r0
    
    def analytical_atmo_psd(self,mode_id:int,r0:float,V:float,freq):
        n = radial_order(i_mode=mode_id)
        f_cut = 0.3 * (n+1) * V / self.D 
        psd = np.ones_like(freq)
        if n == 1:
            psd[freq<=f_cut] = (freq[freq<=f_cut] / f_cut) ** (-2.0/3.0)
        psd[freq>f_cut] = (freq[freq>f_cut] / f_cut) ** (-17.0/3.0)
        vkp = von_karman_power(n/self.D, r0, self.L0, self.D) * (2*np.pi*500e-9)**2 # in m
        psd *= vkp/simpson(psd,freq)
        return psd

    def n_photons(self, frequency, magnitude):
        B0 = 1e+10
        flux = B0 * 10**(-magnitude/2.5) * self.area
        return flux * self.throughput / frequency
    
    def total_error(self, r0:float, n_modes:int, fs:float, V:float, n_subap:float, rMod:float, magnitude:float):
        ogs = self.get_optical_gains(r0=r0,n_subap=n_subap,rMod=rMod)
        fitInNm = self.fitting_error(r0=r0,n_modes=n_modes)
        aliasInNm = self.aliasing_error(r0=r0,fs=fs,n_modes=n_modes,n_subap=n_subap,rMod=rMod)
        WFSnoiseInNm = self.wfs_noise_error(fs=fs, magnitude=magnitude, n_subap=n_subap, rMod=rMod, ogs=ogs)
        lagInNm2 = 0.0
        for i in range(n_modes):
            lagInNm2 += self.servo_lag_error(r0=r0, fs=fs, V=V, mode_id=i)**2
        lagInNm = np.sqrt(lagInNm2)
        totInNm = np.sqrt(fitInNm**2 + lagInNm**2 + WFSnoiseInNm**2 + aliasInNm**2)
        error_budget = {'Total error [nm]': totInNm, 'Servo-lag error [nm]': lagInNm, 
                        'Fitting error [nm]': fitInNm, 'Aliasing error [nm]': aliasInNm, 
                        'WFS noise error [nm]': WFSnoiseInNm}
        return totInNm, error_budget


    def fitting_error(self, r0:float, n_modes:int):
        d_over_r0 = self.D / r0
        if self.dm_type == "asm":
            sigma2_fit = 0.2778 * (n_modes**-0.9) * d_over_r0**(5/3)
        else:
            sigma2_fit = 0.2944 * n_modes**(-np.sqrt(3)/2) * d_over_r0**(5/3)
        return self.rad2nm(np.sqrt(sigma2_fit))

    def servo_lag_error(self, r0:float, fs:float, V:float, mode_id:int):
        freq = self.get_freq_vec(fs)
        atmo_psd = self.analytical_atmo_psd(mode_id, r0, V, freq)
        rtf = self.get_rtf(mode=mode_id,fs=fs)
        atmoResInM = simpson(atmo_psd * rtf**2, freq)
        return np.sqrt(atmoResInM)*1e+9
    
    def wfs_noise_error(self, fs:float, magnitude:float, n_subap:int, rMod:float, n_modes:int, ogs=None):
        frame = fits.getdata(op.join(self.root_dir,'frames',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_frame_null.fits'))
        pyr_mask = get_pupil_mask(npix=max(frame.shape),filepath=op.join(self.root_dir,'pupils',f'pyr_pupdata_{n_subap:1.0f}x{n_subap:1.0f}.fits'))
        sn = frame[pyr_mask]
        slope_var = self.slope_noise_variance(self, sn, mag=magnitude, fs=fs, rMod=rMod, n_subap=n_subap)
        rec = self.get_rec(rMod=rMod, n_subap=n_subap, n_modes=n_modes)    
        flux = np.sum(frame)
        norm = np.mean(frame[pyr_mask.astype(bool)])/4
        norm_rec = rec / (norm / flux)
        sig2 = norm_rec @ slope_var @ norm_rec.T
        return np.sqrt(sig2)

    def aliasing_error(self, r0:float, fs:float, n_modes:int, n_subap:int, rMod:float, mode_id:int=None):
        freq = self.get_freq_vec(fs)
        alias_psd = self.get_alias_psd(r0=r0,n_subap=n_subap,rMod=rMod,n_mdoes=n_modes)
        if mode_id is not None:
            ntf = self.get_ntf(mode=mode_id,fs=fs)
            aliasResInM2 = simpson(alias_psd[mode_id] * ntf**2, freq)
        else:
            aliasResInM2 = 0.0        
            for mode_id in n_modes:
                ntf = self.get_ntf(mode=mode_id,fs=fs)
                aliasResInM2 += simpson(alias_psd[mode_id] * ntf**2, freq)
        return np.sqrt(aliasResInM2)*1e+9

    def get_optical_gains(self,r0:float, n_subap:float, rMod:float):
        try:
            seeing = self.r02seeing(r0)
            ogs = fits.getdata(op.join(self.root_dir,'optgains',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_s{seeing:1.1f}_og.fits'))
        except FileNotFoundError:
            ogs = 1.0
        return ogs
    
    def get_alias_psd(self, r0:float, n_subap:float, rMod:float, n_modes:float):
        seeing = self.r02seeing(r0)
        alias_psd = fits.getdata(op.join(self.root_dir,'aliasing',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_s{seeing:1.1f}_{n_modes}modes_alias_PSD.fits'))
        return alias_psd
        
    def get_pyr_thrp(self, rMod:float, n_subap:int):
        try:
            thrp = fits.getdata(op.join(self.root_dir, 'slopenulls', f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_throughput.fits'))
        except FileNotFoundError:
            print('Pyramid throughput not found for this configuration')
            thrp = 1.0
        return thrp
    
    def get_rec(self, rMod:float, n_subap:float, n_modes:int):
        im = fits.getdata(op.join(self.root_dir,'im',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_im.fits'))
        D = im[:,:n_modes]
        U,S,Vt = np.linalg.svd(D,full_matrices=False)
        rec = (Vt.T * 1/S) @ U.T
        return rec

    def slope_noise_variance(self, sn_ri, mag:float, fs:float, rMod:float, n_subap:int):
        n_subaps = int(len(sn_ri)/4)
        n_phot = self.n_photons(frequancy=fs, magnitude=mag)*self.get_pyr_thrp(rMod,n_subap)
        phot_per_pix = sn_ri*n_phot/n_subaps/4
        pixel_variance = self.F_excess ** 2 * (phot_per_pix + self.sky_bkg + self.dark_curr) + self.RON
        if self.slopes_from_intensity is False:
            weights = np.array([[1,1,-1,-1],[-1,1,1,-1]])
            weights = weights / np.sum(abs(weights), axis=1)[:,None]
            pixel_variance = pixel_variance.reshape([4,n_subaps])
            slope_variance = weights**2 @ pixel_variance / n_phot ** 2   
        else:
            slope_variance = pixel_variance / n_phot ** 2                       
        return slope_variance.flatten()

    # def load_interpolation_grid(self, param_name, grid_data, mod_radii, residuals):
    #     """
    #     Loads a 2D lookup table for parameters like 'rho'.
    #     grid_data: array of shape (len(mod_radii), len(residuals), n_modes)
    #     """
    #     self.interpolators[param_name] = RegularGridInterpolator(
    #         (mod_radii, residuals), 
    #         grid_data, 
    #         bounds_error=False, 
    #         fill_value=None # Allows extrapolation
    #     )

    # def _update_optical_gains(self, current_residual_nm):
    #     """Internal: Updates modal rho vector via interpolation."""
    #     if 'rho' in self.interpolators:
    #         point = np.array([self.modulation_radius, current_residual_nm])
    #         self.rho = self.interpolators['rho'](point)

    # def iterate_to_convergence(self, magnitude, tolerance=0.1, max_iter=10):
    #     """
    #     Performs the iterative procedure to find the steady-state error budget.
    #     The WFS sensitivity (rho) is updated as the residual changes.
    #     """
    #     res_guess = 100.0 # Starting guess in nm
        
    #     for i in range(max_iter):
    #         # 1. Refresh optical gains based on current guess
    #         self._update_optical_gains(res_guess)
            
    #         # 2. Compute error terms (Stubs for full integration of Eq 8, 10, 15)
    #         # In practice, you would pass 'frequencies' and integrate the PSDs here.
    #         fit = self.compute_fitting_error(mode="asm")
    #         temp = 40.0   # Integrated Temporal Error Placeholder
    #         noise = 30.0  # Integrated Noise Error Placeholder
    #         alias = 15.0  # Integrated Aliasing Error Placeholder
            
    #         # 3. Calculate total residual (Eq 6)
    #         new_res = np.sqrt(fit**2 + temp**2 + noise**2 + alias**2)
            
    #         if abs(new_res - res_guess) < tolerance:
    #             return {
    #                 "total": new_res, "fitting": fit, 
    #                 "temp": temp, "noise": noise, "alias": alias,
    #                 "iterations": i+1, "photons_per_frame": self.compute_photons(magnitude)
    #             }
            
    #         res_guess = new_res
            
    #     return {"total": res_guess, "status": "max_iterations_reached"}