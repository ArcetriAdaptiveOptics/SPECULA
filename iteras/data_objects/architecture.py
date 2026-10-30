'''
Description of the computational resources available to a simulation.

A section of the parameter file whose class is Architecture makes Simul run
the allocator (see :mod:`iteras.allocator`), which places the objects on
these resources before building them.
'''

OBJECTIVES = ('speed',)
MODES = ('apply', 'plan')
MEMORY_SOURCES = ('declared', 'probe', 'slurm')
SCHEDULES = ('fixed',)
PROCESS_POLICIES = ('auto', 'per_gpu', 'single')

# Default link speeds, used when the architecture does not give them
# and they cannot be measured: only the order of magnitude matters.
DEFAULT_LINKS = {
    'gpu_gpu_gb_s': 10.0,    # between GPUs of a node, through the host (PCIe)
    'host_gpu_gb_s': 12.0,   # host <-> GPU (PCIe)
    'shm_gb_s': 4.0,         # MPI between processes of the same node
    'shm_latency_us': 30.0,
    'shm_single_copy': True, # the receiver can pull large MPI messages by itself (CMA, XPMEM)
    'mpi_message_us': 30.0,  # host time to send or receive one MPI message, besides the copies
}
DEFAULT_INTER_NODE = {
    'bandwidth_gb_s': 10.0,
    'latency_us': 10.0,
    'single_copy': False,    # the receiver can pull large MPI messages by itself
}


class Architecture:
    '''
    Computational resources and allocator options.

    Parameters
    ----------
    nodes : str or list of dict, optional
        'local' (default): the GPUs and CPU of this computer, detected at run time.
        Otherwise a list of node types, each a dictionary with:

        - ``name``: label of the node type;
        - ``count``: number of identical nodes (default 1);
        - ``cpu``: ``{cores: int, memory_gb: float}``;
        - ``gpus``: ``{count: int, memory_gb: float, model: str, relative_speed: float}``,
          where ``relative_speed`` is the GPU speed relative to the GPU used for
          profiling, used only when no profile exists for that model (default 1);
        - ``links``: optional, ``{gpu_gpu_gb_s, host_gpu_gb_s, shm_gb_s, shm_latency_us,
          shm_single_copy, mpi_message_us}`` (see DEFAULT_LINKS).
    inter_node : dict, optional
        ``{bandwidth_gb_s: float, latency_us: float, single_copy: bool}`` between nodes.
    memory_source : str, optional
        'declared' (default): the memory given above is fully available (like on
        HPC nodes granted by slurm). 'probe': the memory free right now on this
        computer (only with ``nodes: local``). 'slurm': the resources of the
        current slurm allocation.
    memory_margin : float, optional
        Fraction of the memory of each device that is kept free (default 0.1).
    context_overhead_gb : float, optional
        GPU memory used by each process on each GPU it uses, outside the
        arrays (CUDA context, libraries). Default 0.6.
    objective : str, optional
        'speed' (default): minimize the time per step.
    mode : str, optional
        'apply' (default): allocate, then run the simulation. 'plan': only
        write the allocation and its report.
    override_placement : bool, optional
        If True, the target_rank and target_device_idx given in the parameter
        file are ignored; otherwise (default) those objects are not moved.
    allocation_file : str, optional
        Where the allocation is written, and read back in later runs.
        Default 'allocation.yml'.
    profile_file : str, optional
        Where measured object costs are kept between runs. Default: the
        allocation file name with a '_profile.json' suffix.
    recompute : bool, optional
        If True, ignore a saved allocation (like --reallocate).
    profile_cpu : bool, optional
        If True, also measure all objects on the CPU. Otherwise the CPU is
        measured only when the objects do not fit the GPUs.
    profile_device : int, optional
        GPU used for profiling. Default: the local GPU with most free memory.
    profile_steps, warmup_steps : int, optional
        Number of steps measured during profiling, and of steps run before them.
    processes : str, optional
        'auto' (default): choose between one process for each GPU and one
        process per node driving all its GPUs; 'per_gpu'; 'single' (one process
        per node: no MPI with a single node).
    node_tolerance : float, optional
        With several identical nodes, the placement on fewer nodes is preferred
        if its estimated time per step is within this fraction of the fastest
        one (default 0.05).
    device_tolerance : float, optional
        The placement of all the objects on one GPU (except those that must run
        elsewhere) is preferred to the placement found by the search if its
        estimated time per step is within this fraction of it (default 0.05):
        the estimate of the copies between devices is less reliable than that
        of the work on one device.
    pin_ranks_per_node : int, optional
        Number of MPI ranks per node of the launch for which the target_rank of
        the objects placed in the parameter files were written (with slurm's
        block distribution, rank r is on node r // pin_ranks_per_node). Default:
        as in each process layout considered (one per GPU, or one per node).
    schedule : str, optional
        Execution order: only 'fixed' (the trigger order) for now.
    search_time : float, optional
        Maximum time in seconds spent improving the allocation (default 60).
    '''

    def __init__(self,
                 nodes='local',
                 inter_node: dict=None,
                 memory_source: str='declared',
                 memory_margin: float=0.1,
                 context_overhead_gb: float=0.6,
                 objective: str='speed',
                 mode: str='apply',
                 override_placement: bool=False,
                 allocation_file: str='allocation.yml',
                 profile_file: str=None,
                 recompute: bool=False,
                 profile_cpu: bool=False,
                 profile_device: int=None,
                 profile_steps: int=5,
                 warmup_steps: int=3,
                 processes: str='auto',
                 pin_ranks_per_node: int=None,
                 node_tolerance: float=0.05,
                 device_tolerance: float=0.05,
                 schedule: str='fixed',
                 search_time: float=60.0,
                 ):
        self._check('objective', objective, OBJECTIVES)
        self._check('mode', mode, MODES)
        self._check('memory_source', memory_source, MEMORY_SOURCES)
        self._check('schedule', schedule, SCHEDULES)
        self._check('processes', processes, PROCESS_POLICIES)
        if not 0 <= memory_margin < 1:
            raise ValueError(f'memory_margin must be in [0, 1), got {memory_margin}')
        if memory_source == 'probe' and nodes != 'local':
            raise ValueError("memory_source: 'probe' needs nodes: 'local'")
        if nodes != 'local':
            if not isinstance(nodes, list) or not nodes:
                raise ValueError("nodes must be 'local' or a non-empty list of node types")
            for node in nodes:
                self._check_node(node)

        self.nodes = nodes
        self.inter_node = {**DEFAULT_INTER_NODE, **(inter_node or {})}
        self.memory_source = memory_source
        self.memory_margin = memory_margin
        self.context_overhead_gb = context_overhead_gb
        self.objective = objective
        self.mode = mode
        self.override_placement = override_placement
        self.allocation_file = allocation_file
        if profile_file is None:
            stem = allocation_file.rsplit('.', 1)[0] if '.' in allocation_file else allocation_file
            profile_file = stem + '_profile.json'
        self.profile_file = profile_file
        self.recompute = recompute
        self.profile_cpu = profile_cpu
        self.profile_device = profile_device
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.processes = processes
        self.pin_ranks_per_node = pin_ranks_per_node
        self.node_tolerance = node_tolerance
        self.device_tolerance = device_tolerance
        self.schedule = schedule
        self.search_time = search_time

    @staticmethod
    def _check(name, value, allowed):
        if value not in allowed:
            raise ValueError(f'Architecture: {name} must be one of {allowed}, got {value!r}')

    @staticmethod
    def _check_node(node):
        if not isinstance(node, dict):
            raise ValueError(f'Architecture: each node type must be a dictionary, got {node!r}')
        unknown = set(node) - {'name', 'count', 'cpu', 'gpus', 'links'}
        if unknown:
            raise ValueError(f'Architecture: unknown node keys {sorted(unknown)}')
        if 'cpu' not in node:
            raise ValueError(f'Architecture: node {node.get("name")} has no "cpu" entry')
        for key in ('cores', 'memory_gb'):
            if key not in node['cpu']:
                raise ValueError(f'Architecture: node {node.get("name")}: cpu.{key} is missing')
        gpus = node.get('gpus')
        if gpus:
            for key in ('count', 'memory_gb'):
                if key not in gpus:
                    raise ValueError(f'Architecture: node {node.get("name")}: gpus.{key} is missing')
        unknown = set(node.get('links', {})) - set(DEFAULT_LINKS)
        if unknown:
            raise ValueError(f'Architecture: unknown link keys {sorted(unknown)}')

    def as_dict(self):
        '''Parameters that change the result of the allocation (for the allocation file check)'''
        return {k: getattr(self, k) for k in ('nodes', 'inter_node', 'memory_source', 'memory_margin',
                                              'context_overhead_gb', 'objective', 'override_placement',
                                              'processes', 'pin_ranks_per_node', 'node_tolerance',
                                              'device_tolerance', 'schedule')}
