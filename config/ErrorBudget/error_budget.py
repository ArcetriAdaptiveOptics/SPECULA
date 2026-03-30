import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os.path as op
from astropy.io import fits

from specula import cpuArray
from scipy.integrate import simpson
from specula.mmlib.utils import radial_order, von_karman_power, get_pupil_mask
from specula.data_objects.iir_filter_data import IirFilterData

class AOErrorBudgetMachine:
    """
    A Semianalytical Error Budget Machine for AO systems with Pyramid WFS.
    Based on Agapito & Pinna (JATIS 2019).
    """
    def __init__(self, base_path:str, telescope_diameter=8.2, xp=np,
                 L0:float=25, throughput=0.3, obsratio=0.0, dm_type:str='asm'):
        
        self.root_dir = base_path
        self.xp = xp

        # Physical Parameters
        self.D = telescope_diameter
        self.dm_type = dm_type
        self.area = self.xp.pi/4 * (self.D**2- (obsratio*self.D)**2)
        self.throughput = throughput  

        self.L0 = L0 #TODO
        
        # Control Loop & Hardware
        self.delay_frames = None
        self.dm_cutoff_hz = None
        self.controller = None

        # Detector parameters
        self.RON = None
        self.F_excess = None
        self.sky_bkg = None
        self.dark_curr = None
        self.slopes_from_intensity = None
        

    def set_control_parameters(self, controller:IirFilterData, delay_frames:float, dm_cutoff_hz:float=None):
        self.controller = controller
        self.delay_frames = delay_frames
        self.dm_cutoff_hz = dm_cutoff_hz

    def set_detector_parameters(self, RON:float, slopes_from_intensity:bool=False, 
                                F_excess:float=1.0, sky_bkg:float=0.0, dark_curr:float=0.0):
        self.RON = RON 
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
            nw = self.xp.convolve(nw_delay, lpf_num)
            dm = self.xp.convolve(dw_delay, lpf_den)
            rtf = self.controller.RTF(mode=mode, fs=fs, freq=freq, dm=dm, nw=nw, dw=1.0, plot=False)
        else:
            rtf = self.controller.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        return self.xp.array(rtf)
    
    def get_ntf(self, mode:int, fs:float):
        freq = self.get_freq_vec(fs)
        nw_delay, dw_delay = self.controller.discrete_delay_tf(self.delay_frames)
        if self.dm_cutoff_hz is not None:
            lpf_obj = IirFilterData.lpf_from_fc(fc=self.dm_cutoff_hz, fs=fs, n_ord=4)
            lpf_num, lpf_den = lpf_obj.num[0], lpf_obj.den[0]
            nw = self.xp.convolve(nw_delay, lpf_num)
            dm = self.xp.convolve(dw_delay, lpf_den)
            ntf = self.controller.NTF(mode=mode, fs=fs, freq=freq, dm=dm, nw=nw, dw=1.0, plot=False)
        else:
            ntf = self.controller.NTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        return self.xp.array(ntf)

    def get_freq_vec(self, fs:float):
        return self.xp.logspace(-2, self.xp.log10(fs/2), 4000)
    
    @staticmethod
    def r02seeing(r0):
        return 0.98 * 500e-9/r0

    def integrate_psd(self,psd,freq):
        return self.xp.array(simpson(cpuArray(psd),cpuArray(freq)))
    
    def analytical_atmo_psd(self,mode_id:int,r0:float,V:float,freq):
        n = radial_order(i_mode=mode_id)
        f_cut = 0.3 * (n+1) * V / self.D 
        psd = self.xp.ones_like(freq)
        if n == 1:
            psd[freq<=f_cut] = (freq[freq<=f_cut] / f_cut) ** (-2.0/3.0)
        psd[freq>f_cut] = (freq[freq>f_cut] / f_cut) ** (-17.0/3.0)
        vkp = von_karman_power(n/self.D, r0, self.L0, self.D) * (500e-9/(2*self.xp.pi))**2 # in m
        psd *= vkp/self.integrate_psd(psd,freq)
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
        lagInNm = self.xp.sqrt(lagInNm2)
        totInNm = self.xp.sqrt(fitInNm**2 + lagInNm**2 + WFSnoiseInNm**2 + aliasInNm**2)
        error_budget = {'Total error [nm]': totInNm, 'Servo-lag error [nm]': lagInNm, 
                        'Fitting error [nm]': fitInNm, 'Aliasing error [nm]': aliasInNm, 
                        'WFS noise error [nm]': WFSnoiseInNm}
        return totInNm, error_budget


    def fitting_error(self, r0:float, n_modes:int):
        d_over_r0 = self.D / r0
        if self.dm_type == "asm":
            sigma2_fit = 0.2778 * (n_modes**-0.9) * d_over_r0**(5/3)
        else:
            sigma2_fit = 0.2944 * n_modes**(-self.xp.sqrt(3)/2) * d_over_r0**(5/3)
        return (self.xp.sqrt(sigma2_fit)*500e-9/(2*np.pi))*1e+9

    def servo_lag_error(self, r0:float, fs:float, V:float, mode_id:int):
        freq = self.get_freq_vec(fs)
        atmo_psd = self.analytical_atmo_psd(mode_id, r0, V, freq)
        rtf = self.get_rtf(mode=mode_id,fs=fs)
        atmoResInM = self.xp.sqrt(self.integrate_psd(atmo_psd * rtf**2, freq))
        return atmoResInM*1e+9
    
    def wfs_noise_error(self, fs:float, magnitude:float, n_subap:int, rMod:float, n_modes:int, ogs=None):
        frame = fits.getdata(op.join(self.root_dir,'frames',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_frame_null.fits'))[0]
        pyr_mask = get_pupil_mask(npix=max(frame.shape),filepath=op.join(self.root_dir,'pupils',f'pyr_pupdata_{n_subap:1.0f}x{n_subap:1.0f}.fits'))
        sn = self.xp.array(frame[pyr_mask])
        slope_var = self.slope_noise_variance(sn_ri=sn, mag=magnitude, fs=fs, rMod=rMod, n_subap=n_subap)
        rec = self.get_rec(rMod=rMod, n_subap=n_subap, n_modes=n_modes)    
        flux = self.xp.sum(frame)
        norm = self.xp.mean(frame[pyr_mask.astype(bool)])/4
        norm_rec = rec / (norm / flux)
        sig2 = self.xp.diag(norm_rec @ self.xp.diag(slope_var) @ norm_rec.T)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(np.arange(len(sig2))+1, sig2.get(),'-.')
        plt.grid(which='both',alpha=0.4)
        return self.xp.sqrt(self.xp.sum(sig2)) # IM is already in nm, no need to convert

    def aliasing_error(self, r0:float, fs:float, n_modes:int, n_subap:int, rMod:float, mode_id:int=None):
        freq = self.get_freq_vec(fs)
        alias_psd = self.get_alias_psd(r0=r0,n_subap=n_subap,rMod=rMod,n_modes=n_modes)
        if mode_id is not None:
            ntf = self.get_ntf(mode=mode_id,fs=fs)
            aliasResInM2 = self.integrate_psd(alias_psd[mode_id] * ntf**2, freq)
        else:
            aliasResInM2 = 0.0        
            for mode_id in n_modes:
                ntf = self.get_ntf(mode=mode_id,fs=fs)
                aliasResInM2 += self.integrate_psd(alias_psd[mode_id] * ntf**2, freq)
        return self.xp.sqrt(aliasResInM2)*1e+9

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
        return self.xp.array(alias_psd)
        
    def get_pyr_thrp(self, rMod:float, n_subap:int):
        try:
            thrp = fits.getdata(op.join(self.root_dir, 'slopenulls', f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_throughput.fits'))[0]
        except FileNotFoundError:
            print('Pyramid throughput not found for this configuration')
            thrp = 1.0-self.xp.exp(-(0.35+0.72*rMod))
        return thrp
    
    def get_rec(self, rMod:float, n_subap:float, n_modes:int):
        im = fits.getdata(op.join(self.root_dir,'im',f'pyr{rMod:1.1f}_{n_subap:1.0f}x{n_subap:1.0f}_im.fits'))
        D = self.xp.array(im[:,:n_modes])
        U,S,Vt = self.xp.linalg.svd(D,full_matrices=False)
        rec = (Vt.T * 1/S) @ U.T
        return rec

    def slope_noise_variance(self, sn_ri, mag:float, fs:float, rMod:float, n_subap:int):
        n_subaps = int(len(sn_ri)/4)
        n_phot = self.n_photons(frequency=fs, magnitude=mag)*self.get_pyr_thrp(rMod,n_subap)
        phot_per_pix = sn_ri*n_phot/n_subaps/4
        print(n_phot, phot_per_pix)
        pixel_variance = self.F_excess ** 2 * (phot_per_pix + self.sky_bkg + self.dark_curr) + self.RON
        if self.slopes_from_intensity is False:
            weights = self.xp.array([[1,1,-1,-1],[-1,1,1,-1]])
            weights = weights / self.xp.sum(abs(weights), axis=1)[:,None]
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
    #         point = self.xp.array([self.modulation_radius, current_residual_nm])
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
    #         new_res = self.xp.sqrt(fit**2 + temp**2 + noise**2 + alias**2)
            
    #         if abs(new_res - res_guess) < tolerance:
    #             return {
    #                 "total": new_res, "fitting": fit, 
    #                 "temp": temp, "noise": noise, "alias": alias,
    #                 "iterations": i+1, "photons_per_frame": self.compute_photons(magnitude)
    #             }
            
    #         res_guess = new_res
            
    #     return {"total": res_guess, "status": "max_iterations_reached"}