import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scipy.integrate import simpson
from specula.data_objects.iir_filter_data import IirFilterData

class AOErrorBudgetMachine:
    """
    A Semianalytical Error Budget Machine for AO systems with Pyramid WFS.
    Based on Agapito & Pinna (JATIS 2019).
    """
    def __init__(self, base_path:str, controller: IirFilterData = None, 
                 telescope_diameter=8.2, dm_type:str='asm', slopes_from_intensity:bool=False,
                 throughput=0.3, delay_frames=2.0, obsratio=0.0, dm_cutoff_hz=None,
                 RON:float=0.0, F_excess:float=1.0, dark_curr:float=0.0, sky_bkg:float=0.0 ):
        
        self.root_dir = base_path

        # Physical Parameters
        self.D = telescope_diameter
        self.dm_type = dm_type
        self.area = np.pi/4 * (self.D**2- (obsratio*self.D)**2)
        
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
        freq = self.get_freq_vec()
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

    def n_photons(self, frequency, magnitude):
        B0 = 1e+10
        flux = B0 * 10**(-magnitude/2.5) * self.area
        return flux * self.throughput / frequency
    
    def pyr_thrp(self, rMod:float, n_subap:int):
        raise NotImplementedError

    def fitting_error(self, r0:float, n_modes:int):
        d_over_r0 = (self.D / r0)**(5/3)
        if self.dm_type == "asm":
            sigma2_fit = 0.2778 * (n_modes**-0.9) * d_over_r0
        else:
            sigma2_fit = 0.2944 * (n_modes**(-5/6)) * d_over_r0
        return self.rad2nm(np.sqrt(sigma2_fit))

    def servo_lag_error(self, r0:float, frequency:float):
        raise NotImplementedError
    
    def wfs_noise_error(self, frequency:float, magnitude:float, n_subaps:int, rMod:float):
        raise NotImplementedError

    def aliasing_error(self, r0:float, n_modes:int, n_subaps:int, rMod:float):
        raise NotImplementedError


    def slope_noise_variance(self, sn_ri, mag:float, fs:float, rMod:float, n_subap:int):
        n_subaps = int(len(sn_ri)/4)
        n_phot = self.n_photons(frequancy=fs, magnitude=mag)*self.pyr_thrp(rMod,n_subap)
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