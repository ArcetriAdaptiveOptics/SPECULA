import numpy as np
from scipy.interpolate import RegularGridInterpolator

class AOErrorBudgetMachine:
    """
    A Semianalytical Error Budget Machine for AO systems with Pyramid WFS.
    Based on Agapito & Pinna (JATIS 2019).
    """
    def __init__(self, telescope_diameter=8.2, r0=0.15, wind_speed=10.0, 
                 throughput=0.3, delay_frames=1.5, dm_cutoff_hz=None):
        # Physical Parameters
        self.D = telescope_diameter
        self.r0 = r0
        self.V = wind_speed
        self.throughput = throughput  
        self.zero_point = 1e10        # Photons/s/m^2 for Mag 0
        
        # Control Loop & Hardware
        self.sampling_freq = 1000.0   # Hz
        self.delay_frames = delay_frames 
        self.dm_cutoff_hz = dm_cutoff_hz # Hz (None for ideal DM)
        
        # System Configuration
        self.n_modes = 660
        self.modulation_radius = 3.0   # lambda/D
        self.gain = 0.5
        
        # Calibration Data & Interpolation
        self.rho = np.ones(self.n_modes) # Sensitivity loss vector
        self.pi = np.ones(self.n_modes)  # Noise propagation vector
        self.interpolators = {}

    def update_params(self, **kwargs):
        """Dynamically update any class attribute."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def compute_photons(self, magnitude):
        """Computes detected photons per frame based on star magnitude."""
        exposure_time = 1.0 / self.sampling_freq
        area = np.pi * (self.D / 2)**2
        flux_density = self.zero_point * 10**(-0.4 * magnitude)
        return flux_density * area * self.throughput * exposure_time

    def compute_fitting_error(self, mode="asm"):
        """
        Eq 7: Computes fitting error in nm RMS.
        'asm': LBT-ASM formula (0.2778 * N^-0.9)
        'general': Standard analytical formula (alpha * N^-5/6)
        """
        d_over_r0 = (self.D / self.r0)**(5/3)
        if mode == "asm":
            return np.sqrt(0.2778 * (self.n_modes**-0.9) * d_over_r0)
        else:
            alpha = 0.3 
            return np.sqrt(alpha * (self.n_modes**(-5/6)) * d_over_r0)

    def _get_cltfs(self, frequencies):
        """
        Computes Rejection (Hr) and Noise (Hn) Transfer Functions.
        Includes modal sensitivity (rho), pure delay, and DM low-pass filtering.
        """
        omega = 2 * np.pi * frequencies
        T = 1.0 / self.sampling_freq
        
        # Delay and Controller
        tau = self.delay_frames * T
        D_omega = np.exp(-1j * omega * tau)
        C_omega = self.gain / (1.0 - np.exp(-1j * omega * T) + 1e-12)
        
        # DM Low-pass Filter (M_omega)
        if self.dm_cutoff_hz:
            M_omega = 1.0 / (1.0 + 1j * (frequencies / self.dm_cutoff_hz))
        else:
            M_omega = np.ones_like(frequencies)
            
        # Open Loop Gain G = rho * D * M * C
        G = self.rho[:, np.newaxis] * D_omega * M_omega * C_omega
        
        Hr = 1.0 / (1.0 + G)
        Hn = G / (1.0 + G)
        return Hr, Hn

    def load_interpolation_grid(self, param_name, grid_data, mod_radii, residuals):
        """
        Loads a 2D lookup table for parameters like 'rho'.
        grid_data: array of shape (len(mod_radii), len(residuals), n_modes)
        """
        self.interpolators[param_name] = RegularGridInterpolator(
            (mod_radii, residuals), 
            grid_data, 
            bounds_error=False, 
            fill_value=None # Allows extrapolation
        )

    def _update_optical_gains(self, current_residual_nm):
        """Internal: Updates modal rho vector via interpolation."""
        if 'rho' in self.interpolators:
            point = np.array([self.modulation_radius, current_residual_nm])
            self.rho = self.interpolators['rho'](point)

    def iterate_to_convergence(self, magnitude, tolerance=0.1, max_iter=10):
        """
        Performs the iterative procedure to find the steady-state error budget.
        The WFS sensitivity (rho) is updated as the residual changes.
        """
        res_guess = 100.0 # Starting guess in nm
        
        for i in range(max_iter):
            # 1. Refresh optical gains based on current guess
            self._update_optical_gains(res_guess)
            
            # 2. Compute error terms (Stubs for full integration of Eq 8, 10, 15)
            # In practice, you would pass 'frequencies' and integrate the PSDs here.
            fit = self.compute_fitting_error(mode="asm")
            temp = 40.0   # Integrated Temporal Error Placeholder
            noise = 30.0  # Integrated Noise Error Placeholder
            alias = 15.0  # Integrated Aliasing Error Placeholder
            
            # 3. Calculate total residual (Eq 6)
            new_res = np.sqrt(fit**2 + temp**2 + noise**2 + alias**2)
            
            if abs(new_res - res_guess) < tolerance:
                return {
                    "total": new_res, "fitting": fit, 
                    "temp": temp, "noise": noise, "alias": alias,
                    "iterations": i+1, "photons_per_frame": self.compute_photons(magnitude)
                }
            
            res_guess = new_res
            
        return {"total": res_guess, "status": "max_iterations_reached"}