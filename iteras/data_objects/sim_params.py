from typing import List

from iteras.base_data_obj import BaseDataObj


class SimParams(BaseDataObj):
    """
    Simulation Parameters data object.
    This class holds the main parameters of a simulation: time step,
    total time, root directory, etc.

    Simul uses the section of the parameter file whose class is SimParams,
    or a subclass of it, as the main parameters of the simulation. Plugins
    can subclass it to add domain-specific parameters.
    """
    def __init__(self,
                root_dir: str = '.',
                total_time: float = 0.1,
                time_step: float = 0.001,
                display_server: bool = False,
                stepping: bool = False,
                add_modules: List[str] = [],
    ):
        """
        Parameters
        ----------
        root_dir : str
            The root dir for the simulation
        total_time : float [s]
            The total time duration of the simulation
        time_step : float [s]
            The duration of a single timestep in seconds (number of timesteps = int(total_time/time_step) )
        display_server : bool
            Activate web server for simulation display
        stepping: bool
            Activate interactive single-stepping mode
        add_modules: list of str
            Optional additional modules to add to the search path when importing processing objects
        """
        super().__init__()

        if not isinstance(add_modules, list):
            raise ValueError('add_modules parameter must be a list of strings')
        for module_name in add_modules:
            if not isinstance(module_name, str):
                raise ValueError('add_modules parameter must be a list of strings')
        self.root_dir = root_dir
        self.total_time = total_time
        self.time_step = time_step
        self.display_server = display_server
        self.stepping = stepping
        self.add_modules = add_modules
