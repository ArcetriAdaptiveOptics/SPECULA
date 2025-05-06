
from specula.base_processing_obj import BaseProcessingObj

class LaserLaunchTelescope(BaseProcessingObj):
    '''Laser Launch Telescope'''

    def __init__(self,
                 spot_size: float = 0.0,
                 position: list = [],
                 beacon_focus: float = 90e3,
                 beacon_tt: list = [],
                 target_device_idx: int = None, 
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision) 

        self.spot_size = spot_size
        self.tel_pos = position
        self.beacon_focus = beacon_focus
        self.beacon_tt = beacon_tt
