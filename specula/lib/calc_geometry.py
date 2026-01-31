import numpy as np
from specula import RAD2ASEC


def calc_geometry(
    DpupPix: int,
    pixel_pitch: float,
    wavelengthInNm: float,
    FoV: float,
    fov_errinf: float=0.1,
    fov_errsup: float=0.5,
    fft_res: float=3.0,
    xp=np):
    '''
    Calculate FFT geometry parameters for a given pupil size, pixel pitch,
    wavelength and field of view.

    Parameters
    ----------
    DpupPix : int
        Pupil diameter in pixels.
    pixel_pitch : float
        Pixel pitch in meters.
    wavelengthInNm : float
        Wavelength in nanometers.
    FoV : float
        Desired field of view in arcseconds.
    fov_errinf : float, optional
        Acceptable lower error margin for the FoV (default: 0.1).
    fov_errsup : float, optional
        Acceptable upper error margin for the FoV (default: 0.5).
    fft_res : float, optional
        Initial FFT resolution in pixels per lambda/D (default: 3.0).
    xp : module, optional
        Array module to use (default: numpy).

    Returns 
    -------
    dict
        A dictionary containing the calculated parameters:
        - 'fov_res': Field of view interpolation factor (int)
        - 'fp_masking': Focal plane masking ratio (float)
        - 'fft_res': Final FFT resolution in pixels per lambda/D.
        - 'fft_sampling': FFT sampling size in pixels.
        - 'fft_padding': FFT padding size in pixels.
        - 'fft_totsize': Total FFT size in pixels.
    '''

    fov_internal = wavelengthInNm * 1e-9 / pixel_pitch * RAD2ASEC

    maxfov = FoV * (1 + fov_errsup)
    if fov_internal > maxfov:
        raise ValueError("Error: Calculated FoV is higher than maximum accepted FoV."
                f" FoV calculated (arcsec): {fov_internal:.2f},"
                f" maximum accepted FoV (arcsec): {maxfov:.2f}."
                f"\nPlease revise error margin, or the input phase dimension and/or pitch")

    minfov = FoV * (1 - fov_errinf)
    if fov_internal < minfov:
        fov_res = int(xp.ceil(minfov / fov_internal))
        fov_internal_interpolated = fov_internal * fov_res
        print(f"Interpolated FoV (arcsec): {fov_internal_interpolated:.2f}")
        print(f"Warning: reaching the requested FoV requires {fov_res}x interpolation"
                f" of input phase array.")
        print("Consider revising the input phase dimension and/or pitch to improve"
                " performance.")
    else:
        fov_res = 1
        fov_internal_interpolated = fov_internal

    fp_masking = FoV / fov_internal_interpolated

    if fp_masking > 1.0:
        if minfov / fov_internal_interpolated > 1.0:
            raise ValueError(f"fp_masking ratio cannot be larger than 1.0.")
        else:
            fp_masking = 1.0

    if fov_internal_interpolated != FoV:
        print(f"FoV reduction from {fov_internal_interpolated:.2f} to {FoV:.2f}"
                f" will be performed with a focal plane mask")

    DpupPixFov = DpupPix * fov_res
    totsize = xp.around(DpupPixFov * fft_res / 2) * 2

    # Update fft_res and padding based on the rounded totsize
    fft_res = totsize / float(DpupPixFov)
    padding = xp.around((DpupPixFov * fft_res - DpupPixFov) / 2) * 2

    return {
        'fov_res': fov_res,
        'fp_masking': fp_masking,
        'fft_res': fft_res,
        'fft_sampling': int(DpupPixFov),
        'fft_padding': int(padding),
        'fft_totsize': int(totsize),
    }