import numpy as np
import matplotlib.pyplot as plt
import specula
specula.init(-1)  # CPU execution
from specula.data_objects.iir_filter_data import IirFilterData

# --- Configuration ---
fs = 1000.0         # Sampling frequency [Hz]
gain = 0.35         # Loop gain
ff = 0.999          # Forgetting factor (leaky integrator)
delay_frames = 2.4  # Total loop delay [frames]
fc_lpf = 400.0      # Mirror LPF cutoff [Hz]

# 1. Define Controller (C)
controller = IirFilterData.from_gain_and_ff(gain=[gain], ff=[ff])

# 2. Define Plant Components (P)
# Delay TF: interpolates between samples for fractional values
nw_delay, dw_delay = controller.discrete_delay_tf(delay_frames)

# Mirror dynamics (2nd order Butterworth)
lpf_obj = IirFilterData.lpf_from_fc(fc=fc_lpf, fs=fs, n_ord=2)
lpf_num, lpf_den = lpf_obj.num[0], lpf_obj.den[0]

# 3. Assemble the Open Loop (CP)
# Open Loop Numerator = C_num * P_delay_num * LPF_num
nw_plant = np.convolve(nw_delay, lpf_num)
dm_plant = np.convolve(dw_delay, lpf_den)

# --- Performance Plotting ---
freq = np.logspace(-1, np.log10(fs/2), 1000)
rtf_mag = controller.RTF(mode=0, fs=fs, freq=freq, dm=dm_plant, nw=nw_plant, dw=1.0, plot=False)
ntf_mag = controller.NTF(mode=0, fs=fs, freq=freq, dm=dm_plant, nw=nw_plant, dw=1.0, plot=False)

plt.figure(figsize=(10, 5))
plt.loglog(freq, rtf_mag, label='RTF (Rejection)', linewidth=2)
plt.loglog(freq, ntf_mag, label='NTF (Noise)', linewidth=2, color='red')
idx_cross = np.argmin(np.abs(rtf_mag - 1))
plt.axvline(freq[idx_cross], color='green', linestyle='--', linewidth=1.0,
            label=f'Bandwidth (0 dB): {freq[idx_cross]:.1f} Hz')
plt.title(f'Frequency Response (Gain={gain}, Delay={delay_frames}f)')
plt.legend(loc='lower left')
plt.grid(True, which="both", alpha=0.2)
plt.legend()

# --- Stability Analysis (Nyquist) ---
ol_num = np.convolve(controller.num[0], nw_plant)
ol_den = np.convolve(controller.den[0], dm_plant)

open_loop_obj = IirFilterData(ordnum=[len(ol_num)], ordden=[len(ol_den)], num=[ol_num], den=[ol_den])

plt.figure(figsize=(6, 6))
open_loop_obj.nyquist_plot(dt=1/fs, unit_circle=True)
plt.show()