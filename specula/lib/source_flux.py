from specula.lib.n_phot import n_phot


def phot_density_from_source_params(magnitude: float,
                                    wavelengthInNm: float,
                                    band: str = '',
                                    zero_point: float = 0) -> float:
    """
    Compute the photon density of a source from its scalar parameters.

    Parameters
    ----------
    magnitude : float [1]
        Source magnitude.
    wavelengthInNm : float [nm]
        Central wavelength.
    band : str, optional
        Photometric band (e.g. 'V', 'H', 'Na'). If empty (default), the band
        is the one of the n_phot table closest to ``wavelengthInNm``.
    zero_point : float [J/s/m^2/um], optional
        0-magnitude brightness. If <= 0 (default), the n_phot table value
        for the band is used.

    Returns
    -------
    float [photons/s/m^2/nm]
        Photon density over a 1 nm bandwidth.
    """
    e0 = zero_point if zero_point > 0 else None
    used_band = band if band else None
    res = n_phot(magnitude, band=used_band, lambda_=wavelengthInNm / 1e9, width=1e-9, e0=e0)
    return res[0]


def flux_per_pixel(magnitude: float,
                   wavelengthInNm: float,
                   collecting_area_m2: float,
                   bandwidth_nm: float,
                   integration_time_s: float,
                   band: str = '',
                   zero_point: float = 0,
                   n_pixels: int = 1,
                   throughput: float = 1.0,
                   quantum_efficiency: float = 1.0,
                   fraction_on_pixel: float = 1.0) -> float:
    """
    Estimate the detected flux per pixel.

    Simple scalar estimate: detector noise, gain, non-linearity, saturation
    and atmospheric extinction are not modeled. The photon density at
    ``wavelengthInNm`` is assumed flat over ``bandwidth_nm``, and the
    0-magnitude brightness is that of a single n_phot band: for bandwidths
    that are a sizable fraction of, or straddle, Johnson bands the result
    is only indicative.

    Parameters
    ----------
    magnitude : float [1]
        Source magnitude.
    wavelengthInNm : float [nm]
        Central wavelength.
    collecting_area_m2 : float [m^2]
        Effective collecting area. Take it from the pupilstop actually used
        by the simulation, net of central obstruction and spiders, rather
        than the geometric disk, e.g.::

            from specula.data_objects.pupilstop import Pupilstop
            from specula.calib_manager import CalibManager

            cm = CalibManager(main.root_dir)
            pupilstop = Pupilstop.restore(cm.filename('pupilstop', pupilstop.tag))
            collecting_area_m2 = float(pupilstop.masked_area())

    bandwidth_nm : float [nm]
        Spectral bandwidth.
    integration_time_s : float [s]
        Integration time.
    band : str, optional
        See :func:`phot_density_from_source_params`.
    zero_point : float [J/s/m^2/um], optional
        See :func:`phot_density_from_source_params`.
    n_pixels : int, optional
        Number of pixels the flux is evenly split over (default: 1).
    throughput : float, optional
        Optical throughput, in [0, 1] (default: 1).
    quantum_efficiency : float, optional
        Detector QE, in [0, 1] (default: 1). The ``quantum_eff`` parameter
        of CCD-like objects often already includes the total throughput:
        in that case leave ``throughput=1`` to avoid double-counting it.
    fraction_on_pixel : float, optional
        Fraction of the flux falling on the considered pixels, in [0, 1]
        (default: 1).

    Returns
    -------
    float [photo-electrons/pixel]
        Detected flux per pixel over ``integration_time_s``.
    """
    if collecting_area_m2 <= 0:
        raise ValueError('collecting_area_m2 must be > 0')
    if bandwidth_nm <= 0:
        raise ValueError('bandwidth_nm must be > 0')
    if integration_time_s <= 0:
        raise ValueError('integration_time_s must be > 0')
    if n_pixels <= 0:
        raise ValueError('n_pixels must be > 0')
    if not 0 <= throughput <= 1:
        raise ValueError('throughput must be in [0, 1]')
    if not 0 <= quantum_efficiency <= 1:
        raise ValueError('quantum_efficiency must be in [0, 1]')
    if not 0 <= fraction_on_pixel <= 1:
        raise ValueError('fraction_on_pixel must be in [0, 1]')

    photons_per_s_m2_nm = phot_density_from_source_params(
        magnitude=magnitude,
        wavelengthInNm=wavelengthInNm,
        band=band,
        zero_point=zero_point,
    )
    photons_total = photons_per_s_m2_nm * collecting_area_m2 * bandwidth_nm * integration_time_s
    photons_total *= throughput

    photons_per_pixel = photons_total / n_pixels
    photons_per_pixel *= fraction_on_pixel
    return photons_per_pixel * quantum_efficiency
