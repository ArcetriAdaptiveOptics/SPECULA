import numpy as np
from specula.lib.utils import make_orto_modes
from specula.lib.zernike_generator import ZernikeGenerator

def generate_phase_spectrum(f, r0, L0, xp=np, dtype=np.float32):
    """
    Generate the phase spectrum of the turbulence

    Parameters:
    -----------
    f : 2D array
        Frequency grid
    r0 : float
        Fried parameter
    L0 : float
        Outer scale
    xp : module, optional
        Array processing module (numpy or cupy)
    dtype : data type, optional
        Data type for arrays

    Returns:
    --------
    out : 2D array
        Phase spectrum
    """

    """Performs operations with NumPy or CuPy depending on the backend passed as an argument."""
    if xp.__name__ == "cupy":
        from cupyx.scipy.special import gamma
    else:
        from scipy.special import gamma

    cst = (gamma(11.0/6.0)**2/(2.0*np.pi**(11.0/3.0)))*(24.0*gamma(6.0/5.0)/5.0)**(5.0/6.0)
    out = cst * r0**(-5.0/3.0)*(f**2+(1.0/L0)**2)**(-11.0/6.0)
    return xp.asarray(out, dtype=dtype)

def generate_distance_grid(N, M=None, xp=np, dtype=np.float32):
    """
    Generate a 2D distance grid

    Parameters:
    -----------
    N : int
        Size of the grid
    M : int
        Size of the grid
    xp : module, optional
        Array processing module (numpy or cupy)
    dtype : data type, optional
        Data type for arrays

    Returns:
    --------
    R : 2D array
        Distance grid
    """

    if M is None:
        M = N

    # "wrap" style indices like FFT: 0..N/2, -(N/2-1)..-1
    kx = xp.abs(xp.fft.fftfreq(N) * N).astype(dtype)  # (N,)
    ky = xp.abs(xp.fft.fftfreq(M) * M).astype(dtype)  # (M,)

    KX, KY = xp.meshgrid(kx, ky, indexing='xy')
    R = xp.sqrt(KX*KX + KY*KY, dtype=dtype)

    return R

def compute_ifs_covmat(pupil_mask, diameter, influence_functions, r0, L0,
                       oversampling=2, verbose=False, xp=np, dtype=np.float32):
    """"
    Compute the covariance matrix of the influence functions

    Parameters:
    -----------
    pupil_mask : 2D array
        Pupil mask
    diameter : float
        Telescope diameter in meters
    influence_functions : 2D array
        Influence functions (n_actuators, npupil)
    r0 : float
        Fried parameter in meters
    L0 : float
        Outer scale in meters
    oversampling : int
        Oversampling factor
    verbose : bool
        Verbose mode
    xp : module, optional
        Array processing module (numpy or cupy)
    dtype : data type, optional
        Data type for arrays

    Returns:
    --------
    ifft_covariance : 2D array
        Covariance matrix (n_actuators, n_actuators)
    """

    if verbose:
        print("Computing turbulence covariance matrix...")

    if oversampling < 2:
        raise ValueError("Oversampling factor must be at least 2"
                         " to avoid errors in FFT computations.")

    if dtype == xp.float32:
        cdtype = xp.complex64
    elif dtype == xp.float64:
        cdtype = xp.complex128
    else:
        cdtype = complex

    idx_mask = xp.where(pupil_mask.ravel())[0]
    npupil_mask = int(xp.sum(pupil_mask))
    n_actuators = influence_functions.shape[0]
    mask_shape = pupil_mask.shape

    mask_size = max(mask_shape)

    if verbose:
        print("Step 1: Computing Fourier transforms of influence functions...")

    # Fourier transform of the influence functions 3D array
    ft_shape = (oversampling * mask_size, oversampling * mask_size)

    ft_influence_functions = xp.zeros((ft_shape[0], ft_shape[1], n_actuators), dtype=cdtype)

    for act_idx in range(n_actuators):
        if_flat = influence_functions[act_idx, :]

        if_2d = xp.zeros(mask_shape, dtype=dtype)
        if_2d_flat = if_2d.ravel()
        if_2d_flat[idx_mask] = if_flat
        if_2d = if_2d_flat.reshape(mask_shape)

        support = xp.zeros(ft_shape, dtype=dtype)
        support[:mask_shape[0], :mask_shape[1]] = if_2d

        ft_support = xp.fft.fft2(support)
        ft_influence_functions[:, :, act_idx] = ft_support

    if verbose:
        print("Step 2: Generating phase spectrum and computing covariance matrix...")

    # Generation of Phase Spectrum
    sp_freq        = generate_distance_grid(
        oversampling*mask_size, xp=xp, dtype=dtype
    )/(oversampling*diameter)
    phase_spectrum = generate_phase_spectrum(sp_freq, r0, L0, xp=xp, dtype=dtype)
    norm_factor    = npupil_mask**2 * (oversampling * diameter)**2

    prod_ft_shape = ft_shape[0] * ft_shape[1]

    if verbose:
        print("Step 3: Computing covariance matrix in Fourier domain...")

    # Fourier transform of the influence functions
    ft_weighted = ft_influence_functions * phase_spectrum[:, :, xp.newaxis]
    if_ft = ft_weighted.reshape(prod_ft_shape, n_actuators).astype(cdtype, copy=False)

    if verbose:
        print("Step 4: Computing covariance matrix in spatial domain...")

    # # Fourier transform of the influence functions conjugate
    b_2d = ft_influence_functions.reshape(prod_ft_shape, n_actuators)  # (P, M), complex
    # One complex matmul, then take real part (exactly matches OLD math)
    cov_complex = xp.matmul(if_ft.T, xp.conj(b_2d))
    ifft_covariance = xp.real(cov_complex) / norm_factor

    return ifft_covariance

def make_modal_base_from_ifs_fft(pupil_mask, diameter, influence_functions, r0, L0,
                            zern_modes=0, oversampling=2, filt_modes=None,
                            if_max_condition_number=None, verbose=False,
                            xp=np, dtype=np.float32):
    """"
    Generate a modal basis from the influence functions

    Parameters:
    -----------
    pupil_mask : 2D array
        Pupil mask
    diameter : float
        Telescope diameter in meters
    influence_functions : 2D array
        Influence functions
    r0 : float
        Fried parameter in meters
    L0 : float
        Outer scale in meters
    zern_modes : int
        Number of Zernike modes to be used as first modes
    oversampling : int
        Oversampling factor
    filt_modes : 2D array
        Modes to be removed from the influence functions
    if_max_condition_number : float
        Maximum condition number for the influence functions
    verbose : bool
        Verbose mode
    xp : module, optional
        Array processing module (numpy or cupy)
    dtype : data type, optional
        Data type for arrays

    Returns:
    --------
    kl_basis : 2D array
        Modal basis
    m2c : 2D array
        Modes-to-command matrix
    singular_values : dict
        Singular values of the covariance matrices
    """

    """Performs operations with NumPy or CuPy depending on the backend passed as an argument."""
    if xp.__name__ == "cupy":
        from cupy.linalg import svd, pinv
    else:
        from scipy.linalg import svd, pinv

    if verbose:
        print("Starting modal basis generation...")
        print(f"Input shapes: pupil_mask={pupil_mask.shape}, influence_functions={influence_functions.shape}")

    idx_mask = xp.where(pupil_mask.ravel())[0]
    npupil_mask = int(xp.sum(pupil_mask))
    mask_shape = pupil_mask.shape

    if influence_functions.shape[1] != npupil_mask:
        raise ValueError(f"influence_functions should have shape (n_actuators, {npupil_mask})")

    n_actuators = influence_functions.shape[0]

    if verbose:
        print("Step 1: Removing modes from influence functions...")

    number_of_modes_to_be_removed = 1 + zern_modes
    if filt_modes is not None:
        number_of_modes_to_be_removed += filt_modes.shape[0]

    modes_to_be_removed = xp.zeros((number_of_modes_to_be_removed, npupil_mask), dtype=dtype)
    modes_to_be_removed[0, :] = 1.0

    if zern_modes > 0:
        zg = ZernikeGenerator(mask_shape[0], xp=xp, dtype=dtype)
        zern_modes_cube = xp.stack([zg.getZernike(z) for z in range(2, zern_modes + 2)])

        if verbose:
            print(f"Generated Zernike modes shape: {zern_modes_cube.shape}")

        for i in range(zern_modes):
            modes_to_be_removed[i+1, :] = zern_modes_cube[i].ravel()[idx_mask]

        # Orthonormalize Zernike modes
        modes_to_be_removed = make_orto_modes(modes_to_be_removed, xp=xp, dtype=dtype)
        # Normalize Zernike modes
        for i in range(zern_modes):
            modes_to_be_removed[i+1, :] -= xp.mean(modes_to_be_removed[i+1, :])
            modes_to_be_removed[i+1, :] /= xp.sqrt(xp.mean(modes_to_be_removed[i+1, :]**2))

    if zern_modes > 0:
        coef_zern = xp.matmul(modes_to_be_removed, pinv(influence_functions))
        modes_to_be_removed = xp.matmul(coef_zern, influence_functions)

    coef = xp.zeros((number_of_modes_to_be_removed, n_actuators), dtype=dtype)
    filtered_ifs = influence_functions.copy()

    for mode_idx in range(number_of_modes_to_be_removed):
        mode = modes_to_be_removed[mode_idx, :]
        mode_norm = xp.sum(mode * mode)

        if mode_norm > 0:
            for act_idx in range(n_actuators):
                coef[mode_idx, act_idx] = xp.sum(filtered_ifs[act_idx, :] * mode) / mode_norm
                filtered_ifs[act_idx, :] -= mode * coef[mode_idx, act_idx]

    if verbose:
        print("Step 2: Calculating geometric covariance matrix...")

    if_covariance = xp.matmul(filtered_ifs, filtered_ifs.T) / npupil_mask

    if verbose:
        print("Step 3: SVD decomposition of covariance matrix...")

    U1, S1, Vt1 = svd(if_covariance, full_matrices=True)
    V1 = Vt1.T

    S1 = xp.real(S1)
    U1 = xp.real(U1)
    V1 = xp.real(V1)

    if verbose:
        print(f"-- IF covariance matrix SVD ---")
        cond_number = S1[0] / S1[n_actuators-number_of_modes_to_be_removed-1]
        print(f"    initial condition number is: {cond_number}")

    if if_max_condition_number is not None:
        if cond_number > if_max_condition_number:
            min_cond_number = S1[0] / if_max_condition_number
            idx_cond_number = xp.where(S1[:n_actuators-number_of_modes_to_be_removed] < min_cond_number)[0]
            count_cond_number = len(idx_cond_number)

            if count_cond_number > 0:
                number_of_modes_to_be_removed += count_cond_number
                if verbose:
                    final_cond = S1[0] / S1[n_actuators-number_of_modes_to_be_removed-1]
                    print(f"    final condition number is: {final_cond}")
                    print(f"    no. of cut modes: {count_cond_number}")

    M = xp.zeros((n_actuators, n_actuators), dtype=dtype)
    for i in range(n_actuators):
        if i < n_actuators - number_of_modes_to_be_removed:
            M[:, i] = U1[:, i] / xp.sqrt(S1[i])

    if verbose:
        print("Step 4: Calculating turbulence covariance matrix...")

    ifft_covariance = compute_ifs_covmat(pupil_mask, diameter, filtered_ifs, r0, L0, 
                                         oversampling, verbose, xp=xp, dtype=dtype)

    if verbose:
        print("Step 5: Calculating modal basis...")

    hp = xp.matmul(xp.matmul(M.T, ifft_covariance), M)

    U2, S2, Vt2 = svd(hp, full_matrices=True)
    V2 = Vt2.T
    
    S2 = xp.real(S2)
    U2 = xp.real(U2)
    V2 = xp.real(V2)

    Bp = xp.matmul(M, U2)

    kl_modes = xp.matmul(filtered_ifs.T, Bp[:, :n_actuators-number_of_modes_to_be_removed])

    if zern_modes > 0:
        if verbose:
            print("Step 6: Adding Zernike modes to basis...")

        zern_basis = modes_to_be_removed[1:zern_modes+1, :]
        kl_basis = xp.vstack((zern_basis, kl_modes.T))

        K = xp.eye(n_actuators, dtype=dtype)
        projection = xp.matmul(coef_zern[1:zern_modes+1, :].T, coef[1:zern_modes+1, :])
        K -= projection

        m2c_zern = coef_zern[1:zern_modes+1, :].T
        m2c_kl = xp.matmul(K, Bp[:, :n_actuators-number_of_modes_to_be_removed])
        m2c = xp.hstack((m2c_zern, m2c_kl))
    else:
        kl_basis = kl_modes.T

        K = xp.eye(n_actuators, dtype=dtype)
        projection = xp.outer(coef[0, :], coef[0, :])
        K -= projection

        m2c = xp.matmul(K, Bp[:, :n_actuators-number_of_modes_to_be_removed])

    singular_values = {"S1": S1, "S2": S2}

    if verbose:
        print(f"Final shapes: kl_basis={kl_basis.shape}, m2c={m2c.shape}")

    return kl_basis, m2c, singular_values
