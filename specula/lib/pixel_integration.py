"""
Exact integration of a band-limited periodic intensity over detector pixels.

The intensity |DTFT(E)|^2 of a field E with n samples is band limited: it has
frequencies up to n - 1 per period T (T = lambda / pixel pitch of the field).
Its samples on an FFT grid with N >= 2n - 1 points per period therefore give
its Fourier coefficients exactly, and its integral over any interval is a
weighted sum of the samples. This replaces cutting and rebinning a finer FFT
when a detector has pixels that are not a multiple of the FFT sampling.
"""
import numpy as np

from specula.lib.graph_matmul import left_mul


def pixel_integration_matrix(n_fft, period, pxscale, npx, stop=None):
    """
    Real (npx, n_fft) matrix G that integrates a band-limited periodic
    intensity over the detector pixels, from its samples on the FFT grid.

    I(theta) has period `period` [arcsec] and frequencies up to
    n_fft / 2 - 1 per period (true for |FFT|^2 of a field with
    n_fft / 2 pixels zero-padded to n_fft). Its samples I_j at
    theta_j = j * period / n_fft (FFT order, axis at j = 0) give its
    Fourier coefficients exactly, c = ifft(I), with
    I(theta) = sum_k c_k exp(-2 pi i k theta / period). The integral over
    [a, b] is then sum_k c_k B_k, with B_k the integral of the exponential,
    and, along one axis, sum_j G_j I_j with G = ifft(B). In 2D the detector
    image is G @ I @ G.T. Pixel i spans [(i - npx/2), (i + 1 - npx/2)] * pxscale,
    clipped to [-stop/2, stop/2] if a square field stop is given.
    """
    k = np.fft.fftfreq(n_fft) * n_fft
    a = (np.arange(npx) - npx / 2) * pxscale
    b = a + pxscale
    if stop is not None:
        a = np.clip(a, -stop / 2, stop / 2)
        b = np.clip(b, -stop / 2, stop / 2)
    w = -2 * np.pi * k / period
    bm = np.empty((npx, n_fft), dtype=complex)
    nz = k != 0
    bm[:, nz] = (np.exp(1j * np.outer(b, w[nz])) - np.exp(1j * np.outer(a, w[nz]))) / (1j * w[nz])
    bm[:, ~nz] = (b - a)[:, None]
    if n_fft % 2 == 0:
        # Zero by the band limit: dropping it makes G real
        bm[:, n_fft // 2] = 0
    g = np.fft.ifft(bm, axis=1)
    return g.real


def integrate_pixels(g, intensity, tmp, work, xp):
    """
    Integrate a batch of 2D intensity samples over square detector pixels,
    g @ intensity[b] @ g.T for each b, with g from pixel_integration_matrix().

    Parameters
    ----------
    g : array (npx, N)
        Pixel integration matrix (the same along both axes).
    intensity : array (nb, N, N)
        Intensity samples on the FFT grid (FFT order along both axes).
    tmp : array (nb, npx, N)
        Work buffer, contiguous.
    work : array (nb, npx, npx)
        Work buffer, contiguous.
    xp : module
        numpy or cupy.

    Returns
    -------
    image : array (nb, npx, npx)
        A view on `work` (transposed): copy it to keep it beyond the next call.
    """
    # tmp[b] = g @ intensity[b]; work[b] = g @ tmp[b].T = (tmp[b] @ g.T).T
    left_mul(g, intensity, tmp, xp)
    left_mul(g, tmp.swapaxes(1, 2), work, xp)
    return work.swapaxes(1, 2)
