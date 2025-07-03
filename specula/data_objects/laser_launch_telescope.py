from specula.base_data_obj import BaseDataObj

class LaserLaunchTelescope(BaseDataObj):
    '''
    Laser Launch Telescope
    
    args:
    ----------
    spot_size : float
        The size of the laser spot in arcsec.
    tel_position : list
        The x, y and z position of the launch telescope w.r.t. the telescope in m.
    beacon_focus : float
        The distance from the telescope pupil to beacon focus in m.
    beacon_tt : list
        The tilt and tip of the beacon in arcsec.
    '''

    def __init__(self,
                 spot_size: float = 0.0,
                 tel_position: list = [],
                 beacon_focus: float = 90e3,
                 beacon_tt: list = [0.0, 0.0],
                 target_device_idx: int = None, 
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.spot_size = spot_size
        self.tel_pos = tel_position
        self.beacon_focus = beacon_focus
        self.beacon_tt = beacon_tt
