from dataclasses import dataclass

@dataclass
class SimulParams:
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
    pixel_pupil: int = None
    pixel_pitch: float = None
    root_dir: str = '.'
    total_time: float = 0.1
    time_step: float = 0.001
    zenithAngleInDeg: float = 0
    display_server: bool = False
