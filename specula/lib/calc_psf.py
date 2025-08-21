import numpy as np
from collections import namedtuple

from specula import fuse


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))


PsfGeometry = namedtuple('PsfGeometry', 'pixel_size_mas nd')


def calc_psf(phase, amp, xp=np, complex_dtype=np.complex64, imwidth=None, normalize=False, nocenter=False, return_total=False):
    """
    Calculates a PSF from an electrical field phase and amplitude.

    Parameters:
    phase : ndarray
        2D phase array.
    amp : ndarray
        2D amplitude array (same dimensions as phase).
    xp : module, optional
       numpy-like module to use (usually np or cp). Default to standard numpy
    complex_dtype: dtype
       dtype for complex numbers, default to single-precision (np.complex64)
    imwidth : int, optional
        Width of the output image. If provided, the output will be of shape (imwidth, imwidth),
        otherwise the output will have the same shape as the amp and phase parameters.
    normalize : bool, optional
        If set, the PSF is normalized to total(psf).
    nocenter : bool, optional
        If set, avoids centering the PSF and leaves the maximum pixel at [0,0].
    return_total: bool, optional
        If set, return a (psf, total) tuple instead of just the psf.

    Returns:
    psf : ndarray
        2D PSF (same dimensions as phase)
    (psf, total): tuple of (ndarray, scalar)
        2D PSF  (same dimensions as phase) and PSF total intensity as a scalar number
    """

    # Set up the complex array based on input dimensions and data type
    if imwidth is not None:
        u_ef = xp.zeros((imwidth, imwidth), dtype=complex_dtype)
        result = amp * xp.exp(1j * phase, dtype=complex_dtype)
        s = result.shape
        u_ef[:s[0], :s[1]] = result
    else:
        u_ef = amp * xp.exp(1j * phase, dtype=complex_dtype)
    # Compute FFT (forward)
    u_fp = xp.fft.fft2(u_ef)
    # Center the PSF if required
    if not nocenter:
        u_fp = xp.fft.fftshift(u_fp)
    # Compute the PSF as the square modulus of the Fourier transform
    psf = psf_abs2(u_fp, xp=xp)

    # Normalize if required
    total_psf = xp.sum(psf)
    if normalize:
        psf /= total_psf

    if return_total:
        return psf, total_psf
    else:
        return psf


def calc_psf_geometry(pixel_pupil: int,
                      pixel_pitch: float,
                      wavelengthInNm: float,
                      nd: float=None,
                      pixel_size_mas: float=None):
    """
    Calculate PSF sampling parameters ensuring constraints are met

    Args:
        pixel_pupil: Number of pixels across the pupil
        pixel_pitch: Physical size of each pixel in meters
        wavelength_nm: Wavelength in nanometers
        nd: desired PSF oversampling factor
        psf_pixel_size_mas: Desired PSF pixel size in milliarcseconds

    Returns:
        psf_sampling: The calculated sampling factor
    """
    if nd is not None:
        if pixel_size_mas is not None:
            raise ValueError('Cannot set both nd and pixel_size_mas. Use one or the other.')
    elif pixel_size_mas is not None:
        nd = calc_psf_sampling(
            pixel_pupil, 
            pixel_pitch, 
            wavelengthInNm, 
            pixel_size_mas,
        )
    else:
        # Default case, use nd as a scaling factor
        nd = 1.0
        
    psf_pixel_size = calc_psf_pixel_size(wavelengthInNm, pixel_pupil * pixel_pitch, nd)
    return PsfGeometry(pixel_size_mas=psf_pixel_size, nd=nd)


def calc_psf_sampling(pixel_pupil: int, pixel_pitch: float, wavelength_nm: float, psf_pixel_size_mas: float):
    """
    Calculate PSF sampling parameters ensuring constraints are met

    Args:
        pixel_pupil: Number of pixels across the pupil
        pixel_pitch: Physical size of each pixel in meters
        wavelength_nm: Wavelength in nanometers
        psf_pixel_size_mas: Desired PSF pixel size in milliarcseconds

    Returns:
        psf_sampling: The calculated sampling factor
    """
    
    # Calculate pupil diameter in meters
    dim_pup_in_m = pixel_pupil * pixel_pitch

    # Calculate theoretical maximum pixel size (Nyquist limit)
    max_pixel_size_mas = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000

    if psf_pixel_size_mas > max_pixel_size_mas:
        raise ValueError(
            f"Requested PSF pixel size ({psf_pixel_size_mas:.2f} mas) is larger than "
            f"the theoretical maximum ({max_pixel_size_mas:.2f} mas) for this wavelength and pupil size."
        )

    # Calculate required sampling
    required_sampling = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000 / psf_pixel_size_mas

    # Find nearest valid sampling (pixel_pupil * sampling must be integer)
    # Try different integer values for the final PSF size
    best_sampling = required_sampling
    best_error = float('inf')

    for psf_size in range(int(pixel_pupil * required_sampling) - 5, 
                        int(pixel_pupil * required_sampling) + 6):
        if psf_size > 0:
            candidate_sampling = psf_size / pixel_pupil
            candidate_pixel_size = max_pixel_size_mas / candidate_sampling
            error = abs(candidate_pixel_size - psf_pixel_size_mas)

            if error < best_error:
                best_error = error
                best_sampling = candidate_sampling

    actual_psf_sampling = best_sampling
    actual_pixel_size_mas = max_pixel_size_mas / actual_psf_sampling

    # Warning if approximation is significant
    error_percent = abs(actual_pixel_size_mas - psf_pixel_size_mas) / psf_pixel_size_mas * 100
    if error_percent > 1.0:
        print(f"Warning: Actual pixel size ({actual_pixel_size_mas:.2f} mas) differs from "
            f"requested ({psf_pixel_size_mas:.2f} mas) by {error_percent:.1f}% due to "
            f"integer sampling constraint.")

    return actual_psf_sampling


def calc_psf_pixel_size(wavelength_nm: float, dim_pup_in_m: float, nd: float):
    """
    Calculate PSF pixel size
    
    Parameters
    ----------
    wavelength_nm: float
        wavelengt in nanometers
    dim_pup_in_m: float
         pupil dimension in meters
    nd: float
         PSF oversampling factor
    
    Returns:
        pixel_size_mas 
    """
    
    pixel_size_mas = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000 / nd
    
    return pixel_size_mas
