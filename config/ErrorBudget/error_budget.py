import numpy as np
from scipy.interpolate import RegularGridInterpolator
from specula.data_objects.iir_filter_data import IirFilterData

class AOErrorBudgetMachine:
    """
    A Semianalytical Error Budget Machine for AO systems with Pyramid WFS.
    Based on Agapito & Pinna (JATIS 2019).
    """
    def __init__(self, controller: IirFilterData = None, telescope_diameter=8.2, dm_type:str='asm', 
                 throughput=0.3, delay_frames=2.0, obsratio=0.0, dm_cutoff_hz=None):
        # Physical Parameters
        self.D = telescope_diameter
        self.dm_type = dm_type
        self.throughput = throughput  
        self.obsratio = obsratio
        
        # Control Loop & Hardware
        self.delay_frames = delay_frames 
        self.dm_cutoff_hz = dm_cutoff_hz # Hz (None for ideal DM)
        self.controller = controller
        
        # Calibration Data & Interpolation
        self.rho = np.ones(self.n_modes) # Sensitivity loss vector
        self.pi = np.ones(self.n_modes)  # Noise propagation vector
        self.interpolators = {}

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
        freq = self.get_freq_vec()
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

    def n_photons(self, frequency, magnitude):
        B0 = 1e+10
        exposure_time = 1.0 / frequency
        area = np.pi/4 * (self.D**2- (self.obsratio*self.D)**2)
        flux_density = B0 * 10**(-magnitude/2.5)
        return flux_density * area * self.throughput * exposure_time


    def fitting_error(self, r0:float, n_modes:int):
        d_over_r0 = (self.D / r0)**(5/3)
        if self.dm_type == "asm":
            return np.sqrt(0.2778 * (n_modes**-0.9) * d_over_r0)
        else:
            alpha = 0.3 
            return np.sqrt(alpha * (n_modes**(-5/6)) * d_over_r0)

    def servo_lag_error(self, r0:float, frequency:float):
        raise NotImplementedError
    
    def wfs_noise_error(self, frequency:float, magnitude:float, n_subaps:int, rMod:float):
        raise NotImplementedError

    def aliasing_error(self, r0:float, n_modes:int, n_subaps:int, rMod:float):
        raise NotImplementedError




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