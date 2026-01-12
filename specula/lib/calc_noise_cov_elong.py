import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from specula import cpuArray
from specula.data_objects.convolution_kernel import lgs_map_sh

def calc_noise_cov_elong(diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
                         sub_aps_index, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
                         t_g_parameter, h_in_m=None, user_pofile_xy=None, theta=None,
                         only_diag=False, eta_is_not_one=False, display=False, verbose=False):
    """
    Computes noise covariance matrix considering WFS sub-aperture,
    laser launcher and sodium layer geometry.
    
    Parameters:
        diameter_in_m (float): Telescope diameter in meters
        zenith_angle_in_deg (float): Zenith angle in degrees
        na_thickness_in_m (float): Sodium layer FWHM in meters
        launcher_coord_in_m (list): Laser launcher coordinates in meters [x,y,z]
        sub_aps_index (array): Index of valid sub-apertures
        n_sub_aps (int): Number of sub-apertures across diameter
        sub_aps_fov (float): Sub-aperture FOV in arcsec
        sh_spot_fwhm (float): FWHM of short axis
        sigma_noise2 (float): Noise variance (round spot)
        t_g_parameter (float): Used to set part of the sub-aperture to "truncated" condition
        h_in_m (float, optional): Altitude of sodium layer in meters
        user_pofile_xy (list, optional): Sodium profile altitude and intensity fits files
        theta (list, optional): Additional TT angle of laser launcher
        only_diag (bool, optional): If True, return a diagonal matrix
        eta_is_not_one (bool, optional): If True, eta is computed considering flux loss
        display (bool, optional): If True, display debug plots
        verbose (bool, optional): If True, print additional information
    
    Returns:
        ndarray: The inverse covariance matrix
        
    Reference:
        Bechet et al., "Optimal reconstruction for closed-loop ground-layer
                        adaptive optics with elongated spots" 
        JOSA A, Vol. 27, No. 11 (2010)
    """

    # Convert inputs to CPU arrays for GPU processing
    diameter_in_m = float(cpuArray(diameter_in_m))
    zenith_angle_in_deg = float(cpuArray(zenith_angle_in_deg))
    na_thickness_in_m = float(cpuArray(na_thickness_in_m))
    launcher_coord_in_m = cpuArray(launcher_coord_in_m)
    sub_aps_index = np.asarray(cpuArray(sub_aps_index), dtype=np.int64)
    n_sub_aps = int(cpuArray(n_sub_aps))
    sub_aps_fov = float(cpuArray(sub_aps_fov))
    sh_spot_fwhm = float(cpuArray(sh_spot_fwhm))
    sigma_noise2 = float(cpuArray(sigma_noise2))
    t_g_parameter = float(cpuArray(t_g_parameter))
    h_in_m = float(cpuArray(h_in_m)) if h_in_m is not None else None
    # Keep theta as list/array, don't convert to scalar
    if theta is not None:
        theta = list(cpuArray(theta)) if hasattr(theta, '__len__') \
            else [float(theta), float(theta)]

    if only_diag and verbose:
        print('onlyDiag is set')
    if eta_is_not_one and verbose:
        print('etaIsNotOne is set')

    if h_in_m is None:
        h_in_m = 90e3  # sodium average altitude

    rad2arcsec = (3600.0 * 360.0) / (2 * np.pi)
    airmass = 1 / np.cos(zenith_angle_in_deg / 180.0 * np.pi)
    h_in_ma = h_in_m * airmass
    na_thickness_in_ma = na_thickness_in_m * airmass

    if user_pofile_xy is not None or eta_is_not_one:
        pix_for_sa = round(7 * sub_aps_fov / sh_spot_fwhm)

        if user_pofile_xy is not None:
            dz = fits.getdata(user_pofile_xy[0]) * airmass - h_in_ma
            profz = fits.getdata(user_pofile_xy[1])
        else:
            n_levels = 30
            dz = np.arange(n_levels) * (4.0 * na_thickness_in_ma / n_levels) \
                 - 2 * na_thickness_in_ma
            sigma = na_thickness_in_ma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            profz = np.exp(-(dz**2) / (2 * sigma**2))

        # Set default theta if not provided
        theta_val = [0.0, 0.0] if theta is None else theta

        # Call lgs_map_sh to generate spots
        spots_temp = lgs_map_sh(n_sub_aps, diameter_in_m,
                                launcher_coord_in_m, h_in_ma,
                                dz, profz * 1e6,
                                sh_spot_fwhm, sub_aps_fov / pix_for_sa,
                                pix_for_sa, overs=2,
                                theta=theta_val, doCube=True)

        # Calculate coord_sub_aps for use in dist0_xy later
        sub_aps_index_2d = np.array(np.unravel_index(sub_aps_index, (n_sub_aps, n_sub_aps))).T
        coord_sub_aps = sub_aps_index_2d.astype(float)
        coord_sub_aps[:, 0] -= float(n_sub_aps / 2)
        coord_sub_aps[:, 1] -= float(n_sub_aps / 2)
        coord_sub_aps *= diameter_in_m / n_sub_aps
        coord_sub_aps[:, 0] -= launcher_coord_in_m[0]
        coord_sub_aps[:, 1] -= launcher_coord_in_m[1]

        beta1 = np.zeros(len(sub_aps_index))
        beta2 = np.zeros(len(sub_aps_index))
        eta = np.zeros(len(sub_aps_index))

        # Calculate max flux (note the different array ordering between IDL and Python)
        max_flux = np.max(np.sum(np.sum(spots_temp, axis=1), axis=1))

        for i, sub_ap_index_i in enumerate(sub_aps_index):
            spot_i = spots_temp[sub_ap_index_i, :, :]

            # Create 2D coordinate grid for fitting
            y, x = np.mgrid[:spot_i.shape[0], :spot_i.shape[1]]
            # Initial guess: center at the middle, max amplitude, estimated sigma
            p_init = models.Gaussian2D(
                amplitude=np.max(spot_i),
                x_mean=spot_i.shape[1]/2,
                y_mean=spot_i.shape[0]/2,
                x_stddev=sh_spot_fwhm/(2.355 * sub_aps_fov/spot_i.shape[1]),
                y_stddev=sh_spot_fwhm/(2.355 * sub_aps_fov/spot_i.shape[0]),
                theta=0
            )
            fit_p = fitting.LevMarLSQFitter()
            try:
                # Fit the 2D Gaussian model to the spot
                p = fit_p(p_init, x, y, spot_i)
                # Compute FWHM from stddev
                fwhm_x = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.abs(p.x_stddev.value)
                fwhm_y = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.abs(p.y_stddev.value)
                beta1[i] = np.sqrt(max(0, fwhm_x**2 - sh_spot_fwhm**2))
                beta2[i] = np.sqrt(max(0, fwhm_y**2 - sh_spot_fwhm**2))
            except Exception as e:
                beta1[i] = 0
                beta2[i] = 0
                print(f"Warning: 2D Gaussian fit failed for sub-aperture {i}: {e}")

            # Compute eta (flux normalization)
            if eta_is_not_one:
                eta[i] = np.sum(spot_i) / max_flux
            else:
                eta[i] = 1.0
    else:
        eta = np.ones(len(sub_aps_index))

        # Convert sub-aperture indices to 2D coordinates
        # In IDL: subApsIndex2D = array_indices(fltarr(nSubAps,nSubAps),subApsIndex)
        sub_aps_index_2d = np.array(np.unravel_index(sub_aps_index, (n_sub_aps, n_sub_aps))).T

        # Coordinates with respect to center
        coord_sub_aps = sub_aps_index_2d.astype(float)
        coord_sub_aps[:, 0] -= float(n_sub_aps / 2)
        coord_sub_aps[:, 1] -= float(n_sub_aps / 2)
        coord_sub_aps *= diameter_in_m / n_sub_aps

        # Coordinates with respect to launcher
        coord_sub_aps[:, 0] -= launcher_coord_in_m[0]
        coord_sub_aps[:, 1] -= launcher_coord_in_m[1]

        # Calculate beta1 and beta2 from geometry, handling zero coordinates
        with np.errstate(divide='ignore', invalid='ignore'):
            beta1_temp = (np.arctan((h_in_ma - na_thickness_in_ma/2.0) / coord_sub_aps[:, 1]) -
                         np.arctan((h_in_ma + na_thickness_in_ma/2.0) / coord_sub_aps[:, 1])) \
                             * rad2arcsec
            beta2_temp = (np.arctan((h_in_ma - na_thickness_in_ma/2.0) / coord_sub_aps[:, 0]) -
                         np.arctan((h_in_ma + na_thickness_in_ma/2.0) / coord_sub_aps[:, 0])) \
                             * rad2arcsec

        # Replace inf/nan with 0 (physically: when aligned with launcher,
        # elongation in that direction is undefined/zero)
        beta1 = np.nan_to_num(beta1_temp, nan=0.0, posinf=0.0, neginf=0.0)
        beta2 = np.nan_to_num(beta2_temp, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print('launcher coordinates [m]:', launcher_coord_in_m)
        print('altitude [m]', h_in_ma)
        print('thickness [m]', na_thickness_in_ma)
        print('min max coordinate X', np.min(coord_sub_aps[:, 0]), np.max(coord_sub_aps[:, 0]))
        print('min max coordinate Y', np.min(coord_sub_aps[:, 1]), np.max(coord_sub_aps[:, 1]))
        print('min max beta 1', np.min(beta1), np.max(beta1))
        print('min max beta 2', np.min(beta2), np.max(beta2))

    sigma2 = sh_spot_fwhm**2

    if only_diag:
        # For diagonal-only covariance matrix
        diag_xy = np.concatenate([
            1/sigma_noise2 * sigma2/(sigma2 + beta1**2),
            1/sigma_noise2 * sigma2/(sigma2 + beta2**2)
        ])

        dist0_xy = np.abs(np.concatenate([coord_sub_aps[:, 0], coord_sub_aps[:, 1]]))

        if t_g_parameter > 0:
            n_truncated = int(t_g_parameter * 2 * len(sub_aps_index))
            idx_sort = np.argsort(dist0_xy)
            idx_truncated = idx_sort[2*len(sub_aps_index)-n_truncated:2*len(sub_aps_index)]
            idx_not_truncated = idx_sort[:2*len(sub_aps_index)-n_truncated]
        else:
            n_truncated = 0
            idx_truncated = np.array([])
            idx_not_truncated = np.arange(2*len(sub_aps_index))

        if verbose:
            print('no. of truncated sub-apertures', n_truncated)

        if display:
            plt.figure(0)
            plt.plot(beta1)
            plt.plot(beta2, 'r')
            plt.ylim([min(np.min(beta1), np.min(beta2)), max(np.max(beta1), np.max(beta2))])
            plt.title("Beta values")

            plt.figure(1)
            plt.plot(eta)
            plt.title("Eta values")

            if n_truncated > 0:
                a = np.full(2*len(sub_aps_index), -1)
                a[idx_truncated] = 1
                plt.figure(2)
                plt.plot(a)
                plt.ylim([-2, 2])
                plt.title("Truncated sub-apertures")

            plt.show()

        if n_truncated > 0:
            diag_xy[idx_truncated] *= 0.25

        cov_mat_inv = np.diag(diag_xy)

    else:
        # Full covariance matrix
        beta_tot = np.sqrt(beta1**2 + beta2**2)

        cov_mat_inv = np.zeros((2*len(sub_aps_index), 2*len(sub_aps_index)))
        dist0_xy = np.max(np.abs(coord_sub_aps), axis=1)

        if t_g_parameter > 0:
            n_truncated = int(t_g_parameter * len(sub_aps_index))
            idx_sort = np.argsort(dist0_xy)
            idx_truncated = idx_sort[len(sub_aps_index)-n_truncated:len(sub_aps_index)]
            idx_not_truncated = idx_sort[:len(sub_aps_index)-n_truncated]
        else:
            n_truncated = 0
            idx_not_truncated = np.arange(len(sub_aps_index))

        if verbose:
            print('no. of truncated sub-apertures', n_truncated)

        n_not_truncated = len(sub_aps_index) - n_truncated

        if display:
            plt.figure(0)
            plt.plot(beta1)
            plt.plot(beta2, 'r')
            plt.title("Beta values")

            plt.figure(1)
            plt.plot(eta)
            plt.title("Eta values")

            if n_truncated > 0:
                a = np.full(len(sub_aps_index), -1)
                a[idx_truncated] = 1
                plt.figure(2)
                plt.plot(a)
                plt.ylim([-2, 2])
                plt.title("Truncated sub-apertures")

            plt.show()

        # Process non-truncated sub-apertures
        if n_not_truncated > 0:
            idx_not_truncated_x = idx_not_truncated
            idx_not_truncated_y = idx_not_truncated + len(sub_aps_index)

            for j in range(n_not_truncated):
                # x diagonal
                cov_mat_inv[idx_not_truncated_x[j], idx_not_truncated_x[j]] = (
                    1/sigma_noise2 * eta[idx_not_truncated[j]] /
                    (1 + beta_tot[idx_not_truncated[j]]**2 / sigma2) *
                    (1 + beta2[idx_not_truncated[j]]**2 / sigma2)
                )

                # y diagonal
                cov_mat_inv[idx_not_truncated_y[j], idx_not_truncated_y[j]] = (
                    1/sigma_noise2 * eta[idx_not_truncated[j]] /
                    (1 + beta_tot[idx_not_truncated[j]]**2 / sigma2) *
                    (1 + beta1[idx_not_truncated[j]]**2 / sigma2)
                )

                # xy and yx cross-terms
                cov_mat_inv[idx_not_truncated_x[j], idx_not_truncated_y[j]] = (
                    1/sigma_noise2 * eta[idx_not_truncated[j]] /
                    (1 + beta_tot[idx_not_truncated[j]]**2 / sigma2) *
                    (-beta1[idx_not_truncated[j]] * beta2[idx_not_truncated[j]] / sigma2)
                )

                cov_mat_inv[idx_not_truncated_y[j], idx_not_truncated_x[j]] = (
                    1/sigma_noise2 * eta[idx_not_truncated[j]] /
                    (1 + beta_tot[idx_not_truncated[j]]**2 / sigma2) *
                    (-beta1[idx_not_truncated[j]] * beta2[idx_not_truncated[j]] / sigma2)
                )

        # Process truncated sub-apertures
        if n_truncated > 0:
            idx_truncated_x = idx_truncated
            idx_truncated_y = idx_truncated + len(sub_aps_index)

            for j in range(n_truncated):
                # x diagonal
                cov_mat_inv[idx_truncated_x[j], idx_truncated_x[j]] = (
                    eta[idx_truncated[j]] / 
                    (sigma_noise2 * beta_tot[idx_truncated[j]]**2) *
                    beta2[idx_truncated[j]]**2
                )

                # y diagonal
                cov_mat_inv[idx_truncated_y[j], idx_truncated_y[j]] = (
                    eta[idx_truncated[j]] / 
                    (sigma_noise2 * beta_tot[idx_truncated[j]]**2) *
                    beta1[idx_truncated[j]]**2
                )

                # xy and yx cross-terms
                cov_mat_inv[idx_truncated_x[j], idx_truncated_y[j]] = (
                    eta[idx_truncated[j]] / 
                    (sigma_noise2 * beta_tot[idx_truncated[j]]**2) *
                    (-beta1[idx_truncated[j]] * beta2[idx_truncated[j]])
                )

                cov_mat_inv[idx_truncated_y[j], idx_truncated_x[j]] = (
                    eta[idx_truncated[j]] / 
                    (sigma_noise2 * beta_tot[idx_truncated[j]]**2) *
                    (-beta1[idx_truncated[j]] * beta2[idx_truncated[j]])
                )

    return cov_mat_inv
