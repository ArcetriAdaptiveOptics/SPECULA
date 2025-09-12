
from specula.base_data_obj import BaseDataObj
from typing import List

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
        The total time duration of the simulation in seconds
    time_step : float
        The duration of a single timestep in seconds (number of timesteps = int(total_time/time_step) )
    zenithAngleInDeg : float
        The zenith angle of the telescope in degrees
    display_server : bool
        Activate web server for simulation display
    '''
    def __init__(self,
                pixel_pupil: int = None,
                pixel_pitch: float = None,
                root_dir: str = '.',
                total_time: float = 0.1,
                time_step: float = 0.001,
                zenithAngleInDeg: float = 0,
                display_server: bool = False,
                add_modules: List[str] = [],
    ):
        super().__init__()

        if not isinstance(add_modules, list):
            raise ValueError('add_modules parameter must be a list of strings')
        for module_name in add_modules:
            if not isinstance(module_name, str):
                raise ValueError('add_modules parameter must be a list of strings')
        self.pixel_pupil = pixel_pupil
        self.pixel_pitch = pixel_pitch
        self.root_dir = root_dir
        self.total_time = total_time
        self.time_step = time_step
        self.zenithAngleInDeg = zenithAngleInDeg
        self.display_server = display_server
        self.add_modules = add_modules
