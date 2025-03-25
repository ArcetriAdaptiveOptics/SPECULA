import numpy as np

def calc_detector_noise(fs, ccd_type, binning=1, CCD_speed=None):
    """
    Calculate the detector noise.

    Parameters:
    fs : float
        Sampling frequency [Hz].
    ccd_type : str
        Type of CCD ('CCD39', 'OCAM2', 'CCD220', 'CCID75', etc.).
    binning : int, optional
        Binning mode (default is 1).
    CCD_speed : float, optional
        Detector read-out speed [s]. If not provided, it will be calculated based on the CCD type.

    Returns:
    tuple:
        - RON (float): Read-Out-Noise.
        - dark (float): Dark current.
    """

    T = 1.0 / fs  # Integration time

    if CCD_speed is None:
        CCD_speed = 0.0

    if ccd_type == 'CCD39':
        ReadOutSpeed = [0.95e-3, 1.56e-3, 0.89e-3, 0.68e-3]
        ReadOutNoise = [10.5, 4.2, 4.5, 4.6]
        CCD_speed = ReadOutSpeed[binning - 1] if CCD_speed == 0.0 else CCD_speed
        RON = ReadOutNoise[binning - 1]
        dark = 0.0

    elif ccd_type == 'OCAM2k':
        CCD_speed = 0.5e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 150.0 / 400.0
        dark = (0.444 / fs + 0.00556) * binning ** 2

    elif ccd_type == 'OCAM2kSW':
        CCD_speed = 0.24e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 150.0 / 400.0
        dark = (0.444 / fs + 0.00556) * binning ** 2

    elif ccd_type == 'OCAM2kSWspec':
        CCD_speed = 0.24e-3 if CCD_speed == 0.0 else CCD_speed
        if binning == 1:
            RON = 0.40
        elif binning == 2:
            RON = 0.38
        elif binning == 3:
            RON = 0.45
        elif binning == 4:
            RON = 0.51
        dark = 1.38 / fs * binning ** 2

    elif ccd_type == 'OCAM2':
        if fs > 1.5e3:
            CCD_speed = 0.5e-3 if CCD_speed == 0.0 else CCD_speed
            RON = 150.0 / 400.0
        else:
            CCD_speed = 0.667e-3 if CCD_speed == 0.0 else CCD_speed
            RON = 50.0 / 400.0
        dark = (0.444 / fs + 0.00556) * binning ** 2

    elif ccd_type == 'HnU':
        CCD_speed = 0.995e-3
        RON = 0.1
        dark = (0.0006 / fs + 0.005) * binning ** 2

    elif ccd_type == 'CCD220':
        CCD_speed = 0.667e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 80.0 / 400.0 * (binning ** 0.5)
        if fs >= 900:
            dark = 3.6 * binning ** 2 / fs
        elif fs >= 500:
            dark = 1.8 * binning ** 2 / fs
        elif fs >= 360:
            dark = 1.4 * binning ** 2 / fs
        elif fs >= 180:
            dark = 1.08 * binning ** 2 / fs
        else:
            dark = 0.84 * binning ** 2 / fs

    elif ccd_type == 'CCD220goodBin':
        CCD_speed = 0.667e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 80.0 / 400.0
        if fs >= 900:
            dark = 3.6 * binning ** 2 / fs
        elif fs >= 500:
            dark = 1.8 * binning ** 2 / fs
        elif fs >= 360:
            dark = 1.4 * binning ** 2 / fs
        elif fs >= 180:
            dark = 1.08 * binning ** 2 / fs
        else:
            dark = 0.84 * binning ** 2 / fs

    elif ccd_type == 'CCD220_ESO':
        CCD_speed = 0.667e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 1.0 * (binning ** 0.5)
        dark = 1.2 * binning ** 2 / fs

    elif ccd_type == 'C-RED':
        CCD_speed = 0.500e-3 if CCD_speed == 0.0 else CCD_speed
        RON = 24.0 / 30.0
        dark = 100.0 / fs

    elif ccd_type == 'CCID75':
        if binning > 1:
            temp_fs = max(fs) if isinstance(fs, list) else fs
            ReadOutSpeed = [0.588e-3, 0.294e-3]
            ReadOutNoise = [1.7, 2.1]
            idx = next(i for i, x in enumerate(ReadOutSpeed) if 1.0 / temp_fs >= x)
            CCD_speed = ReadOutSpeed[idx] if CCD_speed == 0.0 else CCD_speed
            RON = ReadOutNoise[idx]
        else:
            ReadOutSpeed = [2.0e-3, 1.0e-3, 0.588e-3, 0.435e-3, 0.294e-3]
            ReadOutNoise = [1.7, 2.1, 2.6, 3.6, 4.4]
            temp_fs = max(fs) if isinstance(fs, list) else fs
            idx = next(i for i, x in enumerate(ReadOutSpeed) if 1.0 / np.around(temp_fs) >= x)
            CCD_speed = ReadOutSpeed[idx] if CCD_speed == 0.0 else CCD_speed
            RON = ReadOutNoise[idx]
        dark = 26.0 * binning ** 2 / fs

    else:
        raise ValueError(f"Unknown CCD type: {ccd_type}")

    return RON, dark, CCD_speed
