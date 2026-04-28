from specula.log import get_specula_logger

def compute_mmse_reconstructor(interaction_matrix, c_atm, xp, dtype,
                               noise_variance=None, c_noise=None,
                               c_inverse=False, log_level=None):
    """
    Compute the Minimum Mean Square Error (MMSE) reconstructor.
    
    W_mmse = (A' Cz^(-1) A + Cx^(-1))^(-1) A' Cz^(-1)
    
    Args:
        interaction_matrix (numpy.ndarray): Interaction matrix A relating modes to slopes
        c_atm (numpy.ndarray): Covariance matrix of atmospheric modes (Cx)
        noise_variance (list, optional): List of noise variances per WFS. 
                                        Used to build c_noise if c_noise is None.
        c_noise (numpy.ndarray, optional): Covariance matrix of measurement noise (Cz).
                                         If None, it's built from noise_variance.
        c_inverse (bool, optional): If True, c_atm and c_noise are already inverted.
        verbose (bool, optional): Whether to print detailed information during computation.
        
    Returns:
        numpy.ndarray: MMSE reconstructor matrix
    """
    logger = get_specula_logger(__name__)
    if log_level is not None:
        logger.setLevel(log_level)
    logger.info("Starting MMSE reconstructor computation")

    # Setup matrices
    A = interaction_matrix
    n_slopes, n_modes = A.shape

    # Handle noise covariance matrix
    if c_noise is None and noise_variance is not None:
        n_wfs = len(noise_variance)
        n_slopes_per_wfs = n_slopes // n_wfs

        logger.debug(f"Building noise covariance matrix for {n_wfs}"
                     f" WFSs with {n_slopes_per_wfs} slopes each")

        c_noise = xp.zeros((n_slopes, n_slopes), dtype=dtype)
        for i in range(n_wfs):
            # Set the diagonal elements for this WFS
            start_idx = i * n_slopes_per_wfs
            end_idx = (i + 1) * n_slopes_per_wfs
            c_noise[start_idx:end_idx, start_idx:end_idx] = \
                noise_variance[i] * xp.eye(n_slopes_per_wfs, dtype=dtype)

    # Check dimensions
    if A.shape[1] != c_atm.shape[0]:
        raise ValueError(f"A ({A.shape}) and c_atm ({c_atm.shape})"
                         f" must have compatible dimensions")

    if c_noise is not None and A.shape[0] != c_noise.shape[0]:
        raise ValueError(f"A ({A.shape}) and c_noise ({c_noise.shape})"
                         f" must have compatible dimensions")

    # Compute inverses if needed
    if not c_inverse:
        # Check if matrices are diagonal
        if c_noise is not None:
            is_diag_noise = xp.all(xp.abs(xp.diag(xp.diag(c_noise)) - c_noise) < 1e-10)

            if is_diag_noise:
                logger.debug("c_noise is diagonal, using optimized inversion")
                c_noise_inv = xp.diag(1.0 / xp.diag(c_noise))
            else:
                logger.debug("Inverting c_noise matrix")
                try:
                    c_noise_inv = xp.linalg.inv(c_noise)
                except xp.linalg.LinAlgError:
                    logger.debug("Warning: c_noise inversion failed, using pseudo-inverse")
                    c_noise_inv = xp.linalg.pinv(c_noise)
        else:
            # Default: identity matrix (no noise)
            logger.debug("No c_noise provided, using identity matrix")
            c_noise_inv = xp.eye(A.shape[1], dtype=dtype)

        is_diag_atm = xp.all(xp.abs(xp.diag(xp.diag(c_atm)) - c_atm) < 1e-10)

        if is_diag_atm:
            logger.debug("c_atm is diagonal, using optimized inversion")
            c_atm_inv = xp.diag(1.0 / xp.diag(c_atm))
        else:
            logger.debug("Inverting c_atm matrix")
            try:
                c_atm_inv = xp.linalg.inv(c_atm)
            except xp.linalg.LinAlgError:
                logger.debug("Warning: c_atm inversion failed, using pseudo-inverse")
                c_atm_inv = xp.linalg.pinv(c_atm)
    else:
        # Matrices are already inverted
        c_atm_inv = c_atm
        c_noise_inv = c_noise if c_noise is not None else xp.eye(A.shape[1], dtype=dtype)

    # Compute H = A' Cz^(-1) A + Cx^(-1)
    logger.debug("Computing H = A' Cz^(-1) A + Cx^(-1)")

    # Check if c_noise_inv is scalar
    if isinstance(c_noise_inv, (int, float)) or \
        (hasattr(c_noise_inv, 'size') and c_noise_inv.size == 1):
        H = c_noise_inv * xp.dot(A.T, A) + c_atm_inv
    else:
        H = xp.dot(A.T, xp.dot(c_noise_inv, A)) + c_atm_inv

    # Compute H^(-1)
    logger.debug("Inverting H")
    try:
        H_inv = xp.linalg.inv(H)
    except xp.linalg.LinAlgError:
        logger.warning("H inversion failed, using pseudo-inverse")
        H_inv = xp.linalg.pinv(H)

    # Compute W = H^(-1) A' Cz^(-1)
    logger.debug("Computing W = H^(-1) A' Cz^(-1)")

    # Check if c_noise_inv is scalar
    if isinstance(c_noise_inv, (int, float)) or \
        (hasattr(c_noise_inv, 'size') and c_noise_inv.size == 1):
        W_mmse = c_noise_inv * xp.dot(H_inv, A.T)
    else:
        W_mmse = xp.dot(H_inv, xp.dot(A.T, c_noise_inv))

    logger.info("MMSE reconstruction matrix computed")
    logger.info(f"Matrix shape: {W_mmse.shape}")

    return W_mmse
