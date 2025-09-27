from specula import cpuArray, float_dtype_list
from specula import complex_dtype_list
from specula.lib.calc_spatialfrequency import calc_spatialfrequency


def calc_phasescreen(L0, dimension, pixel_pitch, xp, precision, seed=0, verbose=False):
    if verbose:
        print("Phase-screen computation")

    # Ensure that the dimension is a multiple of 2
    n = int(xp.ceil(xp.log2(float(dimension))))
    if dimension != 2**n:
        # Force dimension to be a multiple of 2^n
        dimension = 2**n
        if verbose:
            print(f"Dimension is not a multiple of 2, it has been set to {dimension}")

    # Data type based on precision
    dtype = float_dtype_list[precision]
    complex_dtype = complex_dtype_list[precision]

    # Dimension in meters
    m_dimension = dimension * pixel_pitch

    # Create random Gaussian matrices for the real and imaginary parts
    half_dim = dimension // 2

    if verbose:
        print("Compute random matrices")

    # "seed" must be a numpy array even when using CuPY 
    rng = xp.random.RandomState(cpuArray(seed))
    u1 = rng.random((half_dim + 1, 2 * half_dim + 1))
    v1 = rng.random((half_dim + 1, 2 * half_dim + 1))
    # Box-Muller transform method
    re_gauss = (xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * xp.pi * v1)).astype(dtype=dtype)
    u2 = rng.random((half_dim + 1, 2 * half_dim + 1))
    v2 = rng.random((half_dim + 1, 2 * half_dim + 1))
    # Box-Muller transform method
    im_gauss = (xp.sqrt(-2.0 * xp.log(u2)) * xp.cos(2.0 * xp.pi * v2)).astype(dtype=dtype)

    # Check for non-finite elements and handle them
    if not xp.isfinite(re_gauss).all():
        temp = xp.isfinite(re_gauss)
        idx_inf = xp.where(~temp)[0]
        idx_fin = xp.where(temp)[0]
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        re_gauss[idx_inf] = xp.mean(re_gauss[idx_fin])

    if not xp.isfinite(im_gauss).all():
        temp = xp.isfinite(im_gauss)
        idx_inf = xp.where(~temp)[0]
        idx_fin = xp.where(temp)[0]
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        im_gauss[idx_inf] = xp.mean(im_gauss[idx_fin])

    # Initialize the phasescreen
    phasescreen = xp.zeros((dimension, dimension), dtype=complex_dtype)

    if verbose:
        print("Compute noise matrix")

    # Fill in the noise matrix
    phasescreen[half_dim:2 * half_dim, 0:2 * half_dim] = re_gauss[1:half_dim + 1, 1:2 * half_dim + 1] + 1j * im_gauss[1:half_dim + 1, 1:2 * half_dim + 1]
    phasescreen[0:half_dim-1, 0:2 * half_dim] = xp.rot90(re_gauss,2)[1:half_dim, 1:2 * half_dim + 1] - 1j * xp.rot90(im_gauss,2)[1:half_dim, 1:2 * half_dim + 1]
    phasescreen[half_dim, 0:half_dim] = re_gauss[0, 1:half_dim+1] + 1j * im_gauss[0, 1:half_dim+1]
    phasescreen[half_dim, half_dim:2 * half_dim] = xp.flipud(re_gauss)[0, 0:half_dim] - 1j * xp.flipud(im_gauss)[0, 0:half_dim]
    phasescreen[2*half_dim-1, :] = 0
    phasescreen[:, 2*half_dim-1] = 0

    if verbose:
        print("Compute spatial frequency matrix")

    # Compute spatial frequency matrix
    spatial_frequency = calc_spatialfrequency(dimension, xp=xp, precision=precision)
    spatial_frequency = spatial_frequency / m_dimension**2

    # Check for non-finite elements and handle them
    if not xp.isfinite(phasescreen).all():
        temp = xp.isfinite(phasescreen)
        idx_inf = xp.where(~temp)[0]
        idx_fin = xp.where(temp)[0]
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        phasescreen[idx_inf] = xp.mean(phasescreen[idx_fin])

    # Apply spatial frequency
    phasescreen *= (spatial_frequency + 1. / L0**2)**(-11./12.)
    phasescreen *= xp.sqrt(0.033/2./m_dimension**2) * (2 * xp.pi)**(2./3.) * xp.sqrt(0.06) * (1 / pixel_pitch)**(5./6.)

    phasescreen = xp.roll(phasescreen, (-half_dim+1, -half_dim+1), axis=(0, 1))
    phasescreen = xp.fft.ifft2(phasescreen, norm='forward')
    phasescreen = xp.roll(phasescreen, (half_dim-1, half_dim-1), axis=(0, 1))

    phasescreen = xp.real(phasescreen)

    return phasescreen
