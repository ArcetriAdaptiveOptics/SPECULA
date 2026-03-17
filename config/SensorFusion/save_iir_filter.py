import specula
specula.init(-1)  # Default target device

import os
import numpy as np
from specula.data_objects.iir_filter_data import IirFilterData

def create_stepped_t(n_filters, excluded_filters=None):
    """
    Crea un vettore t a gradini con gruppi di dimensione crescente
    Primo gruppo: 2 elementi costanti
    Secondo gruppo: 3 elementi costanti
    Terzo gruppo: 4 elementi costanti
    E così via...
    """
    t = np.zeros(n_filters)
    current_idx = 0
    group_size = 2  # Inizia con gruppi di 2

    # Calcola quanti gruppi servono
    total_elements = 0
    groups_needed = 0
    temp_group_size = 2
    while total_elements < n_filters:
        total_elements += temp_group_size
        groups_needed += 1
        temp_group_size += 1

    # Calcola quanti gruppi sono esclusi
    groups_excluded = 0
    if excluded_filters is not None:
        total_elements = 0
        temp_group_size = 2
        while total_elements < excluded_filters:
            total_elements += temp_group_size
            groups_excluded += 1
            temp_group_size += 1

    group_t_values = np.zeros(groups_needed)

    # Crea i valori t per ogni gruppo (da 0 a 1)
    if groups_needed > 1:
        group_t_values[groups_excluded:] = np.linspace(0, 1, groups_needed-groups_excluded)
    else:
        group_t_values = [0]

    # Riempi il vettore t
    current_idx = 0
    for group_idx in range(groups_needed):
        if current_idx >= n_filters:
            break

        elements_to_fill = min(group_size, n_filters - current_idx)
        t[current_idx:current_idx + elements_to_fill] = group_t_values[group_idx]
        current_idx += elements_to_fill
        group_size += 1

    return t

if __name__ == "__main__":
    root_dir = '/home/matte/git/SPECULA/config/SensorFusion/calibration/'
    path = root_dir + 'filter/'
    if not os.path.exists(path):
        os.mkdir(path)
    fs = 2000  # sampling frequency
    excluded_filters = 2 # was 20 
    n_filters = 1300
    file_name = path + f'iirfilter_{n_filters}modes.fits'
    tiled_file_name = path + f'tiled_iirfilter_{n_filters}modes.fits'

    start_pole = [1.0, 0.995]
    end_pole = [0.9, 0.75]
    start_zero = [0.85, 0.45]
    end_zero = [0.55, 0.30]

    power_exponent = 2.0

    mode = np.linspace(0, n_filters-1, n_filters)
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

    iir_gain = 1.0
    num_array *= iir_gain

    filter_data_complex = IirFilterData(
        ordnum=[3] * n_filters,
        ordden=[3] * n_filters,
        num=num_array,
        den=den_array
    )

    tiled_filter_data_complex = IirFilterData(
        ordnum=[3] * n_filters*2,
        ordden=[3] * n_filters*2,
        num=np.tile(num_array,[2,1]),
        den=np.tile(den_array,[2,1])
    )

    plot = True
    if plot:
        delay_frames = 2.0
        import matplotlib.pyplot as plt
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


    filter_data_complex.save(file_name)
    print(f"Saved with native method: {file_name}")
    tiled_filter_data_complex.save(tiled_file_name)
    print(f"Saved with native method: {tiled_file_name}")

    try:
        loaded_filter_native = IirFilterData.restore(file_name)
        print(f"Loaded: {loaded_filter_native.nfilter} filters")
        coeffs_match = np.allclose(loaded_filter_native.num, filter_data_complex.num) and \
                      np.allclose(loaded_filter_native.den, filter_data_complex.den)
        print(f"Matching filters: {coeffs_match}")
        loaded_filter_native = IirFilterData.restore(tiled_file_name)
        print(f"Loaded: {loaded_filter_native.nfilter} filters")
        coeffs_match = np.allclose(loaded_filter_native.num, tiled_filter_data_complex.num) and \
                      np.allclose(loaded_filter_native.den, tiled_filter_data_complex.den)
        print(f"Matching filters: {coeffs_match}")

    except FileNotFoundError:
        print("File FITS not found")
