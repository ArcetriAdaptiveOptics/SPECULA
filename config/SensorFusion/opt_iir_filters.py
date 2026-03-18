import numpy as np
from scipy.optimize import minimize
from scipy.signal import freqz

def get_turbulent_psd(f, f_cutoff, noise_floor):
    """Generates the target magnitude: sqrt(PSD)."""
    psd = np.ones_like(f)
    # Apply f^(-17/3) power law after cutoff
    idx_power = f > f_cutoff
    psd[idx_power] = (f[idx_power] / f_cutoff)**(-17/3)
    # Apply noise floor
    psd = np.maximum(psd, noise_floor)
    return np.sqrt(psd)

def solve_iir_coefficients(fs, Nframes, iir_order, f_cutoff, noise_floor):
    # 1. Setup frequency grid (0 to Nyquist)
    freqs = np.linspace(0.01, fs/2, 512)
    w = 2 * np.pi * freqs / fs
    target_mag = get_turbulent_psd(freqs, f_cutoff, noise_floor)
    
    # 2. Plant frequency response P(w) = exp(-j * w * N)
    P_w = np.exp(-1j * w * Nframes)
    
    # 3. Objective function: Minimize sum of squared errors
    def objective(params):
        b = params[:iir_order + 1]
        a = np.concatenate(([1], params[iir_order + 1:]))
        
        # Calculate IIR filter response C(w)
        _, C_w = freqz(b, a, worN=w)
        
        # Calculate resulting magnitude |1 + C*P|
        current_mag = np.abs(1 + C_w * P_w)
        return np.sum((current_mag - target_mag)**2)

    # 4. Run optimization
    initial_guess = np.zeros(2 * iir_order + 1)
    initial_guess[0] = 0.1  # Initial small gain
    
    res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-6)
    
    b_res = res.x[:iir_order + 1]
    a_res = np.concatenate(([1], res.x[iir_order + 1:]))
    return b_res, a_res

# Example Parameters
fs = 2000          # Sampling freq (Hz)
Nframes = 2.5      # Delay (frames)
order = 2          # IIR Filter order
f_cut = 15         # Cutoff freq (Hz)
noise = 1e-5       # PSD noise floor

b, a = solve_iir_coefficients(fs, Nframes, order, f_cut, noise)
print(f"Numerator coefficients (b): {b}")
print(f"Denominator coefficients (a): {a}")