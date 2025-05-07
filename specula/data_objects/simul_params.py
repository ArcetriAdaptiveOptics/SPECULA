from specula.base_data_obj import BaseDataObj

class SimulParams(BaseDataObj):
    '''
    Simulation Parameters 
    
    args:
    ----------
    root_dir : str
        The root dir for the simulation
    pixel_pupil : int
        The diameter in pixels of the simulation pupil
    pixel_pitch : float
        The dimension in meters of a pixel (telescope diameter = pixel_pupil * pixel_pitch)
    total_time : float
        The total time duration of the simulation
    time_step : float
        The duration of a single timestep in seconds (number of timesteps = int(total_time/time_step) )
    zenithAngleInDeg : float
        The zenith angle of the telescope
    display_server : bool

    '''

    def __init__(self,
                 root_dir: str = '',                 
                 pixel_pupil: int = 120,
                 pixel_pitch: float = 0.05,
                 total_time: float = 0.1, 
                 time_step: float = 0.001, 
                 zenithAngleInDeg: float = 0,
                 display_server: bool = False,
                 target_device_idx: int = None, 
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.root_dir = root_dir        
        self.pixel_pupil = pixel_pupil
        self.pixel_pitch = pixel_pitch
        self.total_time = total_time
        self.time_step = time_step
        self.zenithAngleInDeg = zenithAngleInDeg
        self.display_server = display_server
    