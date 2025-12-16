import numpy as np

def calc_paa(object_speed, xp=np):
    """
       Computes the point ahead angle (PAA) between uplink and downlink

       Parameters:
           object_speed (float): velocity of satellite/space object in m/s

       Returns:
           float: PAA between uplink and downlink in arcsec
       """
    rad2arcsec = 3600.*180./xp.pi
    speed_light = 299792458.0
    paa = 2*(object_speed/speed_light)*rad2arcsec
    return paa


def calc_effective_wind_speed(atmo_heights, wind_speed, wind_dir, object_height, object_speed, object_dir, xp=np):
    """
       Computes the effective wind speed of an e.g. LEO satellite, including telescope slewing

       Parameters:
           atmo_heights (ndarray): heights of atmospheric layers in m
           wind_speed (ndarray): wind velocity at each layer in m/s
           wind_dir (ndarray): wind direction at each layer in degree
           object_height (float): height of satellite/space object in m
           object_speed (float): velocity of satellite/space object in m/s
           object_dir (float): direction in which satellite/space object moves

       Returns:
           float: effective wind speed from slewing in m/s
           float: effective wind direction combining the layer wind directions and satellite moving direction in degree
       """
    arcsec2rad = xp.pi / (3600. * 180.)
    speed_light = 299792458.0
    wind_dir_rad = wind_dir * xp.pi / 180.
    object_dir_rad = object_dir * xp.pi / 180.

    paa = calc_paa(object_speed, xp)
    t_light = object_height / speed_light
    speed_at_layer = xp.tan(paa*arcsec2rad) * atmo_heights / t_light

    effective_wind_1 = wind_speed * xp.sin(wind_dir_rad) + speed_at_layer * xp.sin(object_dir_rad)
    effective_wind_2 = wind_speed * xp.cos(wind_dir_rad) + speed_at_layer * xp.cos(object_dir_rad)
    effective_wind_speed = xp.sqrt(effective_wind_1 ** 2 + effective_wind_2 ** 2)
    effective_wind_dir = xp.arctan(effective_wind_1, effective_wind_2) * 180. / xp.pi

    return effective_wind_speed, effective_wind_dir


def calc_timing_uplink_downlink(zenith_angle, object_height, atmo_heights, xp=np):
    """
       Computes the delay of uplink and downlink propagation that can be set in atmo_evolution as extra_delta_time

       Parameters:
           zenith_angle (float): zenith angle in degree
           object_height (float): height of satellite/space object in m
           atmo_heights (ndarray): heights of atmospheric layers in m

       Returns:
           float: extra delay for uplink propagation in s
           float: extra delay for downlink propagation in s
       """
    speed_light = 299792458.0
    airmass = 1.0 / xp.cos(zenith_angle / 180. * np.pi)

    delta_time_up = (2 * object_height - atmo_heights) * airmass / speed_light
    delta_time_down = atmo_heights * airmass / speed_light

    return delta_time_up, delta_time_down
