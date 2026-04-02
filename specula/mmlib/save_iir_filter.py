# import specula

import os
import numpy as np
from specula.mmlib.utils import radial_order
from specula.data_objects.iir_filter_data import IirFilterData

import matplotlib.pyplot as plt
import scipy.signal as signal

def design_f3_controller(fs, f1, f2, N, phase_margin_deg=45):
    """
    Designs a 3rd-order digital controller to achieve an f^3 sensitivity rise.
    
    Parameters:
    fs : float - Sampling frequency in Hz
    f1 : float - Frequency (Hz) where the f^3 rise begins
    f2 : float - Crossover frequency (Hz) where sensitivity flattens to 0 dB
    N  : float - Pure delay of the plant in timesteps (can be fractional)
    phase_margin_deg : float - Target phase margin in degrees
    
    Returns:
    b, a : ndarrays - Numerator and denominator coefficients of the digital filter C(z)
    """
    w1 = 2 * np.pi * f1
    wc = 2 * np.pi * f2
    Td = N / fs  # Continuous time delay in seconds
    
    # 1. Phase Calculations at Crossover (wc)
    target_phase_rad = -np.pi + np.radians(phase_margin_deg)
    
    # Phase lag from the 3 poles at f1
    phase_poles = -3 * np.arctan(wc / w1)
    
    # Phase lag from the pure plant delay
    phase_delay = -wc * Td
    
    # Calculate required phase lead from the 3 zeros to hit target phase margin
    phase_zeros_total = target_phase_rad - phase_poles - phase_delay
    phase_per_zero = phase_zeros_total / 3.0
    
    # Cap phase per zero to avoid infinite frequencies (max 89 degrees)
    if phase_per_zero >= np.radians(89):
        phase_per_zero = np.radians(89)
        print("Warning: Zeros pushed to maximum allowable limit.")
        
    wz = wc / np.tan(phase_per_zero)
    fz = wz / (2 * np.pi)
    
    # 2. Gain Calculation
    # We need |L(jwc)| = 1 (0 dB crossover)
    # L(s) = K * [ (1 + s/wz)^3 / (1 + s/w1)^3 ] * e^(-s*Td)
    mag_poles = (1 + (wc/w1)**2)**1.5
    mag_zeros = (1 + (wc/wz)**2)**1.5
    K = mag_poles / mag_zeros
    
    # 3. Continuous-time Controller Transfer Function C(s)
    # C(s) = K * (1/wz^3 s^3 + 3/wz^2 s^2 + 3/wz s + 1) / (1/w1^3 s^3 + 3/w1^2 s^2 + 3/w1 s + 1)
    num_s = K * np.array([1/wz**3, 3/wz**2, 3/wz, 1])
    den_s = np.array([1/w1**3, 3/w1**2, 3/w1, 1])
    
    # 4. Discretize using the Bilinear Transform
    b, a = signal.bilinear(num_s, den_s, fs=fs)
    
    print(f"Designed Zeros at: {fz:.2f} Hz")
    print(f"Calculated Gain K: {K:.2f}")
    
    return b, a

def optimal_ff(V, fs, D, n):
    # Calculate the optimal ff for a given V, fs, D, and n
    ff = 1 - 0.6 * np.pi * V / D * (n + 1) / fs
    return ff

def save_and_test_restore(filter_data_complex:IirFilterData, file_name:str):
    filter_data_complex.save(file_name)
    print(f"Saved with native method: {file_name}")
    try:
        loaded_filter_native = IirFilterData.restore(file_name)
        print(f"Loaded: {loaded_filter_native.nfilter} filters")
        coeffs_match = np.allclose(loaded_filter_native.num, filter_data_complex.num) and \
                      np.allclose(loaded_filter_native.den, filter_data_complex.den)
        print(f"Matching filters: {coeffs_match}")
    except FileNotFoundError:
        print("File FITS not found")

def create_stepped_t(n_filters:int, excluded_filters:int=None):
    n = radial_order(n_filters)
    if excluded_filters is not None:
        n_min = radial_order(excluded_filters)
    else:
        n_min = 0
    rad_t = np.zeros(int(n))
    rad_t[n_min:] = np.linspace(0, 1, int(n - n_min))
    t = np.hstack([np.repeat(rad_t[i-2],i) for i in range(2, len(rad_t)+2)])
    t = t[:n_filters]  # Ensure we only have n_filters elements
    return t

def guidos_standard_iir(n_filters:int, excluded_filters:int,
                        start_pole = [1.0, 0.995],
                        end_pole = [0.9, 0.75],
                        start_zero = [0.85, 0.45],
                        end_zero = [0.55, 0.30],
                        power_exponent = 2.0, iir_gain=1.0):
    t = create_stepped_t(n_filters,excluded_filters=excluded_filters)
    t_powered = t**power_exponent 

    zero_values = start_zero[0] + (end_zero[0] - start_zero[0]) * t_powered
    zero2_values = start_zero[1] + (end_zero[1] - start_zero[1]) * t_powered
    pole_values = start_pole[0] + (end_pole[0] - start_pole[0]) * t_powered
    pole2_values = start_pole[1] + (end_pole[1] - start_pole[1]) * t_powered

    num_list = []
    den_list = []
    for i in range(n_filters):
        num_list.append([zero_values[i]*zero2_values[i],
                         -1*(zero_values[i]+zero2_values[i]), 
                         1.0])
        den_list.append([pole_values[i]*pole2_values[i], 
                         -pole_values[i]-pole2_values[i], 1.0])

    num_array = np.array(num_list)
    den_array = np.array(den_list)

    num_array *= iir_gain
    return num_array, den_array


# def opt_iir(V:float, fs:float, D:float):
#     fz = fs/20
#     fp = 0.3 * V/D * (n+1)
#     z = (2*np.pi*np.array([0.95*fz, fz, 1.05*fz])).tolist()
#     p = (2*np.pi*np.array([0.95*fp, fp, 1.05*fp])).tolist()



def plot_iir_tfs(filter_data_complex:IirFilterData, fs:float, n_filters:int, delay_frames:float=2.0):
    nw_delay, dw_delay = filter_data_complex.discrete_delay_tf(delay_frames)
    freq = np.logspace(-2, np.log10(fs/2), 2000)    
    plt.figure(figsize=(16,5))
    for mode in np.linspace(0,n_filters-1,5,dtype=int):
        rtf = filter_data_complex.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        ntf = filter_data_complex.NTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        plt.subplot(1,2,1)
        plt.loglog(freq,rtf,label=f'Mode {mode}')
        plt.subplot(1,2,2)
        plt.loglog(freq,ntf,label=f'Mode {mode}')
    plt.subplot(1,2,1)
    plt.legend()
    plt.grid(which='both',alpha=0.3)
    plt.xlim([1e-2,fs/2])
    plt.xlabel('Frequency [Hz]')
    plt.title('RTF')
    plt.subplot(1,2,2)
    plt.legend()
    plt.grid(which='both',alpha=0.3)
    plt.xlim([1e-2,fs/2])
    plt.xlabel('Frequency [Hz]')
    plt.title('NTF')

    plt.show()


if __name__ == "__main__":

    root_dir = '/raid1/mmenessini/calibration/XAO'
    path = os.path.join(root_dir,'filter')
    os.makedirs(path,exist_ok=True)

    fs = 2000  # sampling frequency
    n_filters = 300
    excluded_filters = 2
    make_tiled = False
    file_name = os.path.join(path,f'iirfilter_{n_filters}modes.fits')

    num_array,den_array=guidos_standard_iir(n_filters=n_filters,
                                            excluded_filters=excluded_filters,
                                            power_exponent=1.0)
    
    # b,a=design_f3_controller(fs=2000,f1=10,f2=300,N=2)

    filter_data_complex = IirFilterData(
        ordnum=[3] * n_filters,
        ordden=[3] * n_filters,
        num=num_array,
        den=den_array
    )
    save_and_test_restore(filter_data_complex,file_name)

    if make_tiled:
        tiled_file_name = path + f'tiled_iirfilter_{n_filters}modes.fits'
        tiled_filter_data_complex = IirFilterData(
            ordnum=[3] * n_filters*2,
            ordden=[3] * n_filters*2,
            num=np.tile(num_array,[2,1]),
            den=np.tile(den_array,[2,1])
        )
        save_and_test_restore(tiled_filter_data_complex,tiled_file_name)
