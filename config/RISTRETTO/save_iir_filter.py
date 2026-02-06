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
    root_dir = '/home/matte/git/SPECULA/config/RISTRETTO/calibration/'
    path = root_dir + 'filter/'
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = path + 'ristretto_iirfilter.fits'
    fs = 2000  # Frequenza di campionamento [Hz]
    excluded_filters = 20 
    n_filters = 1200
    iir_gain = 0.4

    start_pole = [1.0, 0.995]
    end_pole = [0.85, 0.70] #[0.9, 0.75]
    start_zero = [0.85, 0.45]
    end_zero = [0.50, 0.25] #[0.55, 0.30]

    # Parametro di potenza per controllare l'andamento
    power_exponent = 2.0  # Potenza > 1 concentra più valori verso l'inizio
                          # Potenza < 1 concentra più valori verso la fine
                          # Potenza = 1 equivale a linspace

    # Crea la progressione con andamento a potenza
    mode = np.linspace(0, n_filters-1, n_filters)
    t = create_stepped_t(n_filters,excluded_filters=excluded_filters)
    t_powered = t**power_exponent     # Applica la potenza

    # Mappa i valori powered sui range desiderati
    zero_values = start_zero[0] + (end_zero[0] - start_zero[0]) * t_powered
    zero2_values = start_zero[1] + (end_zero[1] - start_zero[1]) * t_powered
    pole_values = start_pole[0] + (end_pole[0] - start_pole[0]) * t_powered
    pole2_values = start_pole[1] + (end_pole[1] - start_pole[1]) * t_powered

    plot_poles = False
    if plot_poles:
        import matplotlib.pyplot as plt
        #plot pole2_values
        plt.figure(figsize=(10, 5))
        plt.plot(mode, pole2_values, label='Pole 2 Values', color='blue')
        plt.plot(mode, pole_values, label='Pole 1 Values', color='red')
        plt.xlabel('Mode')
        plt.ylabel('Pole Value')
        plt.title('Pole Values Progression')
        plt.legend()
        plt.grid()
        plt.show()

    # Calcola i coefficienti
    num_list = []
    den_list = []
    for i in range(n_filters):
        num_list.append([zero_values[i]*zero2_values[i]*iir_gain,
                         -1*(zero_values[i]+zero2_values[i])*iir_gain, 
                         1.00000*iir_gain])
        den_list.append([pole_values[i]*pole2_values[i], 
                         -pole_values[i]-pole2_values[i], 1.0])

    # Crea l'array dei coefficienti
    num_array = np.array(num_list)
    den_array = np.array(den_list)

    # Crea l'oggetto IirFilterData
    filter_data_complex = IirFilterData(
        ordnum=[3] * n_filters,
        ordden=[3] * n_filters,
        num=num_array,
        den=den_array
    )

    print("=== SALVATAGGIO ===")

    # Salva usando il metodo nativo FITS
    filter_data_complex.save(file_name)
    print(f"Salvato con metodo nativo: {file_name}")

    print("\n=== CARICAMENTO E VERIFICA ===")

    # Test di caricamento con metodo nativo
    try:
        loaded_filter_native = IirFilterData.restore(file_name)
        print(f"Caricato: {loaded_filter_native.nfilter} filtri")

        # Verifica che i coefficienti siano identici
        coeffs_match = np.allclose(loaded_filter_native.num, filter_data_complex.num) and \
                      np.allclose(loaded_filter_native.den, filter_data_complex.den)
        print(f"Coefficienti corrispondenti: {coeffs_match}")

    except FileNotFoundError:
        print("File FITS non trovato per il test di caricamento")
