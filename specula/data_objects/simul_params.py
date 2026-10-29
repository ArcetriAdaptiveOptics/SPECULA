
from iteras.data_objects.sim_params import SimParams
from typing import List

class SimulParams(SimParams):
    """
    Simulation Parameters data object.
    This class holds the parameters of the simulation, such as pixel size,
    time step, total time, zenith angle, etc.
    The generic parameters are defined by iteras's SimParams.
    """
    def __init__(self,
                pixel_pupil: int = None,
                pixel_pitch: float = None,
                root_dir: str = '.',
                total_time: float = 0.1,
                time_step: float = 0.001,
                zenithAngleInDeg: float = 0,
                display_server: bool = False,
                stepping: bool = False,
                add_modules: List[str] = [],
    ):
        """
        Parameters
        ----------
        root_dir : str
            The root dir for the simulation
        pixel_pupil : int [pixel]
            The diameter in pixels of the simulation pupil
        pixel_pitch : float [m]
            The dimension in meters of a pixel (telescope diameter = pixel_pupil * pixel_pitch)
        total_time : float [s]
            The total time duration of the simulation
        time_step : float [s]
            The duration of a single timestep in seconds (number of timesteps = int(total_time/time_step) )
        zenithAngleInDeg : float [deg]
            The zenith angle of the telescope
        display_server : bool
            Activate web server for simulation display
        stepping: bool
            Activate interactive single-stepping mode
        add_modules: list of str
            Optional additional modules to add to the search path when importing processing objects
        """
        super().__init__(root_dir=root_dir,
                         total_time=total_time,
                         time_step=time_step,
                         display_server=display_server,
                         stepping=stepping,
                         add_modules=add_modules)

        self.pixel_pupil = pixel_pupil
        self.pixel_pitch = pixel_pitch
        self.zenithAngleInDeg = zenithAngleInDeg
