from specula.lib.n_phot import n_phot


def phot_density_from_source_params(magnitude, wavelengthInNm, band='', zero_point=0):
    """
    Compute photometric density [photons/s/m^2/nm] from source scalar parameters.
    """
    e0 = zero_point if zero_point > 0 else None
    used_band = band if band else None
    res = n_phot(magnitude, band=used_band, lambda_=wavelengthInNm / 1e9, width=1e-9, e0=e0)
    return res[0]


def flux_per_pixel(magnitude,
                   wavelengthInNm,
                   collecting_area_m2,
                   bandwidth_nm,
                   integration_time_s,
                   band='',
                   zero_point=0,
                   n_pixels=1,
                   throughput=1.0,
                   quantum_efficiency=1.0,
                   fraction_on_pixel=1.0):
    """
    Estimate detected flux per pixel (photo-electrons/pixel).

    This is a simple scalar estimate and does not model detector noise,
    gain, non-linearity or saturation.
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
